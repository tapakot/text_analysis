from __future__ import annotations

import argparse
import math
import random
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple


# ============================================================
# 1. PREPROCESSING
# ============================================================

TOKEN_RE = re.compile(
    r"[а-яa-z]+(?:[-'][а-яa-z]+)*",
    re.IGNORECASE
)

URL_RE = re.compile(
    r"https?://\S+|www\.\S+",
    re.IGNORECASE
)

SPACE_RE = re.compile(r"\s+")


# Небольшой stop-list.
# Для коротких текстов не стоит использовать слишком агрессивный список.
STOP_WORDS = {
    "и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как", "а",
    "то", "все", "она", "так", "его", "но", "да", "ты", "к", "у", "же",
    "вы", "за", "бы", "по", "только", "ее", "мне", "было", "вот", "от",
    "меня", "еще", "нет", "о", "из", "ему", "теперь", "когда", "даже",
    "ну", "вдруг", "ли", "если", "уже", "или", "ни", "быть", "был",
    "него", "до", "вас", "нибудь", "опять", "уж", "вам", "ведь", "там",
    "потом", "себя", "ничего", "ей", "может", "они", "тут", "где", "есть",
    "надо", "ней", "для", "мы", "тебя", "их", "чем", "была", "сам",
    "чтоб", "без", "будто", "чего", "раз", "тоже", "себе", "под", "будет",
    "ж", "тогда", "кто", "этот", "того", "потому", "этого", "какой",
    "совсем", "ним", "здесь", "этом", "один", "почти", "мой", "тем",
    "чтобы", "нее", "сейчас", "были", "куда", "зачем", "сказать", "всех",
    "никогда", "сегодня", "сегодняшний",
}


def clean_text(text: str) -> str:
    """
    Нормализация текста.
    """
    text = text.lower()
    text = text.replace("ё", "е")

    # URL убираем целиком.
    text = URL_RE.sub(" ", text)

    # Оставляем кириллицу, латиницу, пробелы и дефисы.
    text = re.sub(r"[^а-яa-z\s-]", " ", text)

    # Дефисы превращаем в разделители.
    text = re.sub(r"-+", " ", text)

    text = SPACE_RE.sub(" ", text).strip()

    return text


def get_tokens(text: str) -> List[str]:
    """
    Токенизация без внешних NLP-библиотек.
    """
    cleaned = clean_text(text)

    return [
        token
        for token in TOKEN_RE.findall(cleaned)
        if len(token) > 1 and token not in STOP_WORDS
    ]


# ============================================================
# 2. N-GRAMS
# ============================================================

def word_ngrams(
    tokens: Sequence[str],
    min_n: int,
    max_n: int
) -> Iterable[str]:

    for n in range(min_n, max_n + 1):
        for i in range(len(tokens) - n + 1):
            yield "w:" + "_".join(tokens[i:i + n])


def char_ngrams(
    cleaned_text: str,
    min_n: int,
    max_n: int
) -> Iterable[str]:

    # Пробелы сохраняем, чтобы буквенные n-граммы
    # учитывали границы слов.
    s = " " + cleaned_text + " "

    for n in range(min_n, max_n + 1):
        for i in range(len(s) - n + 1):

            gram = s[i:i + n]

            # Не нужны n-граммы, состоящие только из пробелов.
            if any("а" <= ch <= "я" for ch in gram):
                yield "c:" + gram


# ============================================================
# 3. MODEL
# ============================================================

@dataclass
class SparseTfidf:

    vocabulary: Dict[str, int]

    idf: List[float]

    # Для каждого класса:
    # {feature_id: centroid_weight}
    centroids: List[Dict[int, float]]

    centroid_norms: List[float]

    labels: List[str]

    class_counts: Dict[str, int]

    word_range: Tuple[int, int]

    char_range: Tuple[int, int]

    # Инвертированный индекс центроидов:
    #
    # feature_id -> [(cluster_id, weight), ...]
    #
    # Это существенно ускоряет классификацию.
    inverted_centroids: Dict[
        int,
        List[Tuple[int, float]]
    ]

    # --------------------------------------------------------

    def transform_one(self, text: str) -> Dict[int, float]:
        """
        Преобразовать один документ в sparse TF-IDF вектор.
        """

        counts = Counter()

        cleaned = clean_text(text)

        tokens = [
            token
            for token in TOKEN_RE.findall(cleaned)
            if len(token) > 1 and token not in STOP_WORDS
        ]

        # Словесные n-граммы
        for gram in word_ngrams(
            tokens,
            *self.word_range
        ):
            feature_id = self.vocabulary.get(gram)

            if feature_id is not None:
                counts[feature_id] += 1

        # Буквенные n-граммы
        for gram in char_ngrams(
            cleaned,
            *self.char_range
        ):
            feature_id = self.vocabulary.get(gram)

            if feature_id is not None:
                counts[feature_id] += 1

        if not counts:
            return {}

        # -----------------------------
        # TF-IDF
        # -----------------------------

        vector = {}

        norm2 = 0.0

        for feature_id, tf in counts.items():

            # Sublinear TF.
            tf_value = 1.0 + math.log(tf)

            value = tf_value * self.idf[feature_id]

            vector[feature_id] = value

            norm2 += value * value

        # L2 normalization.
        norm = math.sqrt(norm2)

        if norm:

            inverse_norm = 1.0 / norm

            for feature_id in list(vector):
                vector[feature_id] *= inverse_norm

        return vector

    # --------------------------------------------------------

    def transform(
        self,
        texts: Sequence[str]
    ) -> List[Dict[int, float]]:

        return [
            self.transform_one(text)
            for text in texts
        ]

    # --------------------------------------------------------

    def predict_one(self, text: str) -> str:
        """
        Классификация одного документа.
        """

        vector = self.transform_one(text)

        # Совсем пустой документ.
        if not vector:
            return self.labels[0]

        scores = [0.0] * len(self.centroids)

        # Благодаря inverted_centroids не приходится
        # проверять каждый признак против каждого из 12 классов.
        for feature_id, value in vector.items():

            for cluster_id, centroid_value in (
                self.inverted_centroids.get(
                    feature_id,
                    ()
                )
            ):
                scores[cluster_id] += (
                    value * centroid_value
                )

        # Документ и центроиды нормированы,
        # поэтому dot product = cosine similarity.
        best_cluster = max(
            range(len(scores)),
            key=scores.__getitem__
        )

        return self.labels[best_cluster]

    # --------------------------------------------------------

    def predict(
        self,
        texts: Sequence[str],
        batch_size: int = 4096
    ) -> List[str]:
        """
        Классификация большого корпуса.

        Важный момент:
        здесь нет построения гигантской матрицы документов.
        """

        result = []

        for start in range(
            0,
            len(texts),
            batch_size
        ):

            batch = texts[
                start:start + batch_size
            ]

            result.extend(
                self.predict_one(text)
                for text in batch
            )

        return result

    # ========================================================
    # 4. TYPICAL N-GRAMS
    # ========================================================

    def typical_ngrams(
        self,
        label: str,
        top_k: int = 20
    ) -> List[Tuple[str, float]]:

        if label not in self.labels:
            raise ValueError(
                f"Неизвестный кластер: {label!r}. "
                f"Доступны: {self.labels}"
            )

        cluster_id = self.labels.index(label)

        inverse_vocabulary = [
            None
        ] * len(self.vocabulary)

        for term, feature_id in self.vocabulary.items():
            inverse_vocabulary[feature_id] = term

        items = [
            (
                inverse_vocabulary[feature_id],
                value
            )
            for feature_id, value
            in self.centroids[cluster_id].items()
        ]

        items.sort(
            key=lambda x: x[1],
            reverse=True
        )

        return items[:top_k]

    # --------------------------------------------------------

    def print_typical_ngrams(
        self,
        label: str,
        top_k: int = 20
    ) -> None:

        print()
        print(
            f"Типичные n-граммы кластера "
            f"{label}:"
        )

        for term, score in self.typical_ngrams(
            label,
            top_k
        ):
            print(
                f"{term:35s} {score:.5f}"
            )

    # ========================================================
    # 5. VISUALIZATION
    # ========================================================

    def plot_centroids(
        self,
        top_k: int = 12
    ) -> None:
        """
        Для каждого кластера показывает top n-грамм
        по весу в центроиде.

        matplotlib подключается только при вызове метода.
        """

        try:
            import matplotlib.pyplot as plt

        except ImportError as exc:
            raise RuntimeError(
                "Для визуализации нужен matplotlib."
            ) from exc

        inverse_vocabulary = [
            None
        ] * len(self.vocabulary)

        for term, feature_id in self.vocabulary.items():
            inverse_vocabulary[feature_id] = term

        for cluster_id, label in enumerate(
            self.labels
        ):

            items = sorted(
                (
                    (
                        inverse_vocabulary[feature_id],
                        value
                    )
                    for feature_id, value
                    in self.centroids[cluster_id].items()
                ),
                key=lambda x: x[1],
                reverse=True
            )[:top_k]

            if not items:
                continue

            names = [
                x[0]
                for x in reversed(items)
            ]

            values = [
                x[1]
                for x in reversed(items)
            ]

            plt.figure(
                figsize=(9, 5)
            )

            plt.barh(
                range(len(names)),
                values
            )

            plt.yticks(
                range(len(names)),
                names,
                fontsize=8
            )

            plt.title(
                f"Кластер {label}: "
                f"типичные n-граммы"
            )

            plt.tight_layout()

            plt.show()

    # --------------------------------------------------------

    def plot_projection(
        self,
        texts: Sequence[str],
        labels: Sequence[str] | None = None,
        max_points: int = 1500
    ) -> None:
        """
        Быстрая диагностическая 2D-проекция.

        Это НЕ PCA.
        Документ отображается по сходству
        с двумя первыми центроидами.

        Ограничение max_points специально нужно,
        чтобы не пытаться рисовать 100000+ точек.
        """

        try:
            import matplotlib.pyplot as plt

        except ImportError as exc:
            raise RuntimeError(
                "Для визуализации нужен matplotlib."
            ) from exc

        if not texts:
            return

        texts = texts[:max_points]

        if labels is not None:
            labels = labels[:max_points]

        axis_x = self.centroids[0]
        axis_y = self.centroids[1]

        xs = []
        ys = []

        for text in texts:

            vector = self.transform_one(text)

            x = sum(
                value * axis_x.get(
                    feature_id,
                    0.0
                )
                for feature_id, value
                in vector.items()
            )

            y = sum(
                value * axis_y.get(
                    feature_id,
                    0.0
                )
                for feature_id, value
                in vector.items()
            )

            xs.append(x)
            ys.append(y)

        plt.figure(
            figsize=(8, 6)
        )

        if labels is None:

            plt.scatter(
                xs,
                ys,
                s=8,
                alpha=0.45
            )

        else:

            label_to_id = {
                label: i
                for i, label
                in enumerate(self.labels)
            }

            ids = [
                label_to_id.get(
                    label,
                    -1
                )
                for label in labels
            ]

            plt.scatter(
                xs,
                ys,
                c=ids,
                s=8,
                alpha=0.45
            )

        plt.xlabel(
            f"cosine(cluster={self.labels[0]})"
        )

        plt.ylabel(
            f"cosine(cluster={self.labels[1]})"
        )

        plt.title(
            "Диагностическая проекция документов"
        )

        plt.tight_layout()

        plt.show()


# ============================================================
# 6. CLASSIFIER
# ============================================================

class TfidfNearestCentroid:

    def __init__(
        self,
        word_range: Tuple[int, int] = (1, 2),
        char_range: Tuple[int, int] = (3, 5),
        min_df: int = 2,
        max_features: int = 60000
    ):

        self.word_range = word_range
        self.char_range = char_range
        self.min_df = min_df
        self.max_features = max_features

    # --------------------------------------------------------

    def _extract_counts(
        self,
        text: str
    ) -> Counter:

        counts = Counter()

        cleaned = clean_text(text)

        tokens = [
            token
            for token in TOKEN_RE.findall(cleaned)
            if len(token) > 1
            and token not in STOP_WORDS
        ]

        counts.update(
            word_ngrams(
                tokens,
                *self.word_range
            )
        )

        counts.update(
            char_ngrams(
                cleaned,
                *self.char_range
            )
        )

        return counts

    # --------------------------------------------------------

    def fit(
        self,
        texts: Sequence[str],
        labels: Sequence[str]
    ) -> SparseTfidf:

        if len(texts) != len(labels):
            raise ValueError(
                "texts и labels должны "
                "иметь одинаковую длину."
            )

        if not texts:
            raise ValueError(
                "Пустой обучающий корпус."
            )

        unique_labels = sorted(
            set(labels)
        )

        if len(unique_labels) < 2:
            raise ValueError(
                "Нужно минимум два класса."
            )

        n_docs = len(texts)

        # ====================================================
        # PASS 1
        # ====================================================
        #
        # Считаем только document frequency.
        #
        # Не храним feature map каждого документа.
        #

        df = Counter()

        for text in texts:

            features = self._extract_counts(
                text
            )

            # Для DF важен сам факт наличия
            # признака, а не его TF.
            df.update(
                features.keys()
            )

        # Отбрасываем слишком редкие признаки.
        candidates = [
            (term, frequency)
            for term, frequency
            in df.items()
            if frequency >= self.min_df
        ]

        # Сначала самые частые.
        candidates.sort(
            key=lambda x: (
                -x[1],
                x[0]
            )
        )

        candidates = candidates[
            :self.max_features
        ]

        vocabulary = {
            term: feature_id
            for feature_id, (term, _)
            in enumerate(candidates)
        }

        # ====================================================
        # IDF
        # ====================================================

        idf = [
            math.log(
                (n_docs + 1.0)
                /
                (df[term] + 1.0)
            )
            + 1.0

            for term, _
            in candidates
        ]

        # ====================================================
        # PASS 2
        # ====================================================
        #
        # Строим TF-IDF и сразу складываем его
        # в центроид соответствующего класса.
        #

        class_to_id = {
            label: i
            for i, label
            in enumerate(unique_labels)
        }

        class_counts = Counter(labels)

        centroid_sums = [
            defaultdict(float)
            for _ in unique_labels
        ]

        for text, label in zip(
            texts,
            labels
        ):

            raw_counts = self._extract_counts(
                text
            )

            vector = {}

            norm2 = 0.0

            for term, tf in raw_counts.items():

                feature_id = vocabulary.get(
                    term
                )

                if feature_id is None:
                    continue

                value = (
                    1.0 + math.log(tf)
                ) * idf[feature_id]

                vector[feature_id] = value

                norm2 += value * value

            if not norm2:
                continue

            inverse_norm = (
                1.0 / math.sqrt(norm2)
            )

            cluster_id = class_to_id[label]

            for feature_id, value in vector.items():

                centroid_sums[
                    cluster_id
                ][feature_id] += (
                    value * inverse_norm
                )

        # ====================================================
        # CENTROIDS
        # ====================================================

        centroids = []
        centroid_norms = []

        for cluster_id, label in enumerate(
            unique_labels
        ):

            divisor = float(
                class_counts[label]
            )

            centroid = {
                feature_id:
                    value / divisor

                for feature_id, value
                in centroid_sums[cluster_id].items()
            }

            norm = math.sqrt(
                sum(
                    value * value
                    for value in centroid.values()
                )
            )

            if norm:

                inverse_norm = 1.0 / norm

                centroid = {
                    feature_id:
                        value * inverse_norm

                    for feature_id, value
                    in centroid.items()
                }

            centroids.append(
                centroid
            )

            centroid_norms.append(
                norm
            )

        # ====================================================
        # INVERTED CENTROID INDEX
        # ====================================================
        #
        # Вместо:
        #
        #   document_feature × 12 clusters
        #
        # рассматриваем только те кластеры,
        # где признак реально есть.
        #

        inverted_centroids = defaultdict(list)

        for cluster_id, centroid in enumerate(
            centroids
        ):

            for feature_id, value in centroid.items():

                inverted_centroids[
                    feature_id
                ].append(
                    (
                        cluster_id,
                        value
                    )
                )

        self.model_ = SparseTfidf(
            vocabulary=vocabulary,
            idf=idf,
            centroids=centroids,
            centroid_norms=centroid_norms,
            labels=unique_labels,
            class_counts=dict(
                class_counts
            ),
            word_range=self.word_range,
            char_range=self.char_range,
            inverted_centroids=dict(
                inverted_centroids
            )
        )

        return self.model_


# ============================================================
# 7. METRICS
# ============================================================

def classification_report(
    y_true: Sequence[str],
    y_pred: Sequence[str]
) -> str:

    labels = sorted(
        set(y_true) | set(y_pred)
    )

    confusion = {
        label: Counter()
        for label in labels
    }

    for actual, predicted in zip(
        y_true,
        y_pred
    ):
        confusion[actual][predicted] += 1

    lines = [
        "class                 "
        "precision recall f1    support"
    ]

    f1_values = []

    for label in labels:

        tp = confusion[label][label]

        fp = sum(
            confusion[other][label]
            for other in labels
            if other != label
        )

        fn = sum(
            confusion[label][other]
            for other in labels
            if other != label
        )

        support = sum(
            confusion[label].values()
        )

        precision = (
            tp / (tp + fp)
            if tp + fp
            else 0.0
        )

        recall = (
            tp / (tp + fn)
            if tp + fn
            else 0.0
        )

        f1 = (
            2 * precision * recall
            /
            (precision + recall)
            if precision + recall
            else 0.0
        )

        f1_values.append(f1)

        lines.append(
            f"{label:20s} "
            f"{precision:9.3f} "
            f"{recall:6.3f} "
            f"{f1:5.3f} "
            f"{support:8d}"
        )

    accuracy = (
        sum(
            a == b
            for a, b
            in zip(y_true, y_pred)
        )
        /
        len(y_true)
    )

    macro_f1 = (
        sum(f1_values)
        /
        len(f1_values)
    )

    lines.append("")

    lines.append(
        f"accuracy={accuracy:.4f}  "
        f"macro_f1={macro_f1:.4f}"
    )

    return "\n".join(lines)


# ============================================================
# 8. STRATIFIED TRAIN/TEST SPLIT
# ============================================================

def train_test_split(
    texts: Sequence[str],
    labels: Sequence[str],
    test_ratio: float = 0.2,
    seed: int = 42
):

    by_class = defaultdict(list)

    for i, label in enumerate(labels):
        by_class[label].append(i)

    rng = random.Random(seed)

    train_indices = []
    test_indices = []

    for indices in by_class.values():

        indices = indices[:]

        rng.shuffle(indices)

        test_size = max(
            1,
            int(len(indices) * test_ratio)
        )

        test_indices.extend(
            indices[:test_size]
        )

        train_indices.extend(
            indices[test_size:]
        )

    return (
        [
            texts[i]
            for i in train_indices
        ],
        [
            labels[i]
            for i in train_indices
        ],
        [
            texts[i]
            for i in test_indices
        ],
        [
            labels[i]
            for i in test_indices
        ],
    )


# ============================================================
# 9. SYNTHETIC TEST CORPUS
# ============================================================

def make_synthetic(
    seed: int = 42,
    per_class: int = 500
):

    rng = random.Random(seed)

    topics = [
        (
            "спорт",
            [
                "матч",
                "гол",
                "тренер",
                "команда",
                "турнир",
                "игрок",
                "стадион"
            ]
        ),
        (
            "погода",
            [
                "дождь",
                "ветер",
                "температура",
                "облако",
                "прогноз",
                "снег",
                "погода"
            ]
        ),
        (
            "финансы",
            [
                "банк",
                "кредит",
                "акция",
                "биржа",
                "рубль",
                "доход",
                "инвестиция"
            ]
        ),
        (
            "кино",
            [
                "фильм",
                "актер",
                "режиссер",
                "сцена",
                "премьера",
                "сюжет",
                "кино"
            ]
        ),
        (
            "еда",
            [
                "рецепт",
                "суп",
                "мясо",
                "сыр",
                "овощи",
                "ресторан",
                "вкус"
            ]
        ),
        (
            "транспорт",
            [
                "поезд",
                "автобус",
                "метро",
                "дорога",
                "билет",
                "вокзал",
                "рейс"
            ]
        ),
        (
            "технологии",
            [
                "сервер",
                "код",
                "данные",
                "программа",
                "процессор",
                "сеть",
                "система"
            ]
        ),
        (
            "медицина",
            [
                "врач",
                "пациент",
                "лечение",
                "анализ",
                "клиника",
                "симптом",
                "лекарство"
            ]
        ),
        (
            "образование",
            [
                "школа",
                "студент",
                "экзамен",
                "курс",
                "учеба",
                "преподаватель",
                "лекция"
            ]
        ),
        (
            "туризм",
            [
                "отель",
                "поездка",
                "билет",
                "море",
                "город",
                "экскурсия",
                "турист"
            ]
        ),
        (
            "недвижимость",
            [
                "квартира",
                "дом",
                "аренда",
                "ипотека",
                "район",
                "стройка",
                "жилье"
            ]
        ),
        (
            "музыка",
            [
                "песня",
                "альбом",
                "группа",
                "концерт",
                "музыка",
                "гитара",
                "сцена"
            ]
        ),
    ]

    common = [
        "новый",
        "важный",
        "сегодня",
        "люди",
        "данные",
        "сообщили",
        "после",
        "вопрос"
    ]

    texts = []
    labels = []

    for name, vocabulary in topics:

        for _ in range(per_class):

            selected = rng.choices(
                vocabulary,
                k=rng.randint(5, 12)
            )

            # Добавляем морфологический шум.
            if rng.random() < 0.45:

                selected.append(
                    rng.choice(vocabulary)
                    +
                    rng.choice(
                        [
                            "ы",
                            "а",
                            "ом",
                            "е",
                            "ов"
                        ]
                    )
                )

            selected += rng.choices(
                common,
                k=rng.randint(1, 4)
            )

            rng.shuffle(selected)

            text = " ".join(selected)

            text += rng.choice(
                [
                    ".",
                    "!",
                    "...",
                    ""
                ]
            )

            texts.append(text)
            labels.append(name)

    order = list(
        range(len(texts))
    )

    rng.shuffle(order)

    return (
        [
            texts[i]
            for i in order
        ],
        [
            labels[i]
            for i in order
        ]
    )


# ============================================================
# 10. SELF TESTS
# ============================================================

def run_self_tests():

    print("=== SELF-TEST ===")

    # 6000 документов:
    # 500 на каждый из 12 классов.
    texts, labels = make_synthetic(
        per_class=500
    )

    (
        train_texts,
        train_labels,
        test_texts,
        test_labels
    ) = train_test_split(
        texts,
        labels,
        test_ratio=0.2
    )

    # -----------------------------
    # Training
    # -----------------------------

    start = time.perf_counter()

    classifier = TfidfNearestCentroid(
        word_range=(1, 2),
        char_range=(3, 5),
        min_df=2,
        max_features=60000
    )

    model = classifier.fit(
        train_texts,
        train_labels
    )

    fit_time = (
        time.perf_counter()
        - start
    )

    # -----------------------------
    # Test
    # -----------------------------

    start = time.perf_counter()

    predictions = model.predict(
        test_texts
    )

    prediction_time = (
        time.perf_counter()
        - start
    )

    print(
        f"train={len(train_texts)}, "
        f"test={len(test_texts)}, "
        f"vocabulary={len(model.vocabulary)}"
    )

    print(
        f"fit_time={fit_time:.3f}s, "
        f"predict_time={prediction_time:.3f}s"
    )

    print()

    print(
        classification_report(
            test_labels,
            predictions
        )
    )

    accuracy = (
        sum(
            a == b
            for a, b
            in zip(
                test_labels,
                predictions
            )
        )
        /
        len(test_labels)
    )

    # Для данного искусственного корпуса
    # ожидаем практически идеальное разделение.
    assert accuracy > 0.90

    # Проверяем API типичных n-грамм.
    typical = model.typical_ngrams(
        model.labels[0],
        top_k=10
    )

    assert typical

    assert all(
        isinstance(item[0], str)
        for item in typical
    )

    # -----------------------------
    # 100 000 документов
    # -----------------------------

    large_corpus = (
        test_texts
        *
        (
            100_000
            //
            len(test_texts)
            + 1
        )
    )[:100_000]

    start = time.perf_counter()

    large_predictions = model.predict(
        large_corpus,
        batch_size=4096
    )

    large_time = (
        time.perf_counter()
        - start
    )

    assert len(
        large_predictions
    ) == 100_000

    print()

    print(
        f"100k prediction time="
        f"{large_time:.3f}s "
        f"("
        f"{large_time / 100_000 * 1000:.3f}"
        f" ms/text)"
    )

    print(
        "large-corpus test: PASS"
    )

    print(
        "=== ALL TESTS PASSED ==="
    )


# ============================================================
# 11. SIMPLE INTERACTIVE MODE
# ============================================================

def interactive(
    model: SparseTfidf
):

    print()
    print(
        "Интерактивный режим."
    )

    print(
        "Команды:"
    )

    print(
        "  classify <текст>"
    )

    print(
        "  typical <кластер> [top_k]"
    )

    print(
        "  quit"
    )

    while True:

        try:
            line = input("> ").strip()

        except EOFError:
            break

        if not line:
            continue

        if line == "quit":
            break

        parts = line.split(
            maxsplit=2
        )

        # -----------------------------------------
        # classify
        # -----------------------------------------

        if (
            parts[0] == "classify"
            and len(parts) >= 2
        ):

            text = (
                parts[1]
                if len(parts) == 2
                else parts[2]
            )

            print(
                model.predict_one(text)
            )

        # -----------------------------------------
        # typical
        # -----------------------------------------

        elif (
            parts[0] == "typical"
            and len(parts) >= 2
        ):

            top_k = (
                int(parts[2])
                if len(parts) == 3
                else 20
            )

            model.print_typical_ngrams(
                parts[1],
                top_k
            )

        else:

            print(
                "Неизвестная команда."
            )


# ============================================================
# 12. MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--self-test",
        action="store_true"
    )

    args = parser.parse_args()

    if args.self_test:

        run_self_tests()

        return

    # ========================================================
    # ЗДЕСЬ ПОДКЛЮЧАЕТСЯ РЕАЛЬНЫЙ DATASET
    # ========================================================
    #
    # Например:
    #
    # train_texts = open(
    #     "train_texts.txt",
    #     encoding="utf-8"
    # ).read().splitlines()
    #
    # train_labels = open(
    #     "train_labels.txt",
    #     encoding="utf-8"
    # ).read().splitlines()
    #
    # classifier = TfidfNearestCentroid(
    #     word_range=(1, 2),
    #     char_range=(3, 5),
    #     min_df=2,
    #     max_features=60000
    # )
    #
    # model = classifier.fit(
    #     train_texts,
    #     train_labels
    # )
    #
    # unlabeled_texts = ...
    #
    # predictions = model.predict(
    #     unlabeled_texts,
    #     batch_size=4096
    # )
    #
    # ========================================================

    raise SystemExit(
        "Запустите --self-test или подключите "
        "свои train_texts/train_labels в main()."
    )


if __name__ == "__main__":
    main()
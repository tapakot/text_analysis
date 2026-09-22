import math
import re
import random
from collections import Counter


# =========================================================
# НАСТРОЙКИ
# =========================================================

# Количество кластеров задаёт пользователь
K = 4

# Диапазон буквенных n-грамм
NGRAM_MIN = 2
NGRAM_MAX = 5

# Максимальное количество итераций кластеризации
MAX_ITERATIONS = 100

# Сколько раз запускать алгоритм с разными начальными центрами
N_INIT = 10


# =========================================================
# 1. ОЧИСТКА ТЕКСТА
# =========================================================

def clean_text(text):
    """
    Очистка и нормализация текста.
    """

    # Переводим в нижний регистр
    text = text.lower()

    # Заменяем всё, кроме букв и пробелов, на пробел.
    # Поддерживаются Unicode-буквы, включая русский язык.
    text = re.sub(r'[^^\W\d_]+', ' ', text, flags=re.UNICODE)

    # Убираем лишние пробелы
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# =========================================================
# 2. ТОКЕНИЗАЦИЯ
# =========================================================

def tokenize(text):
    """
    Разбивает текст на слова.
    """

    text = clean_text(text)

    return re.findall(
        r'[^\W\d_]+',
        text,
        flags=re.UNICODE
    )


# =========================================================
# 3. БУКВЕННЫЕ N-ГРАММЫ
# =========================================================

def make_character_ngrams(text, n_min=2, n_max=5):
    """
    Создаёт буквенные n-граммы.

    Например:

    "доллар"

    при n=3:

    "дол"
    "олл"
    "лла"
    "лар"
    """

    text = clean_text(text)

    # Убираем пробелы, чтобы получать непрерывные
    # буквенные последовательности.
    text = text.replace(' ', '')

    grams = []

    for n in range(n_min, n_max + 1):

        for i in range(len(text) - n + 1):

            gram = text[i:i + n]

            grams.append(gram)

    return grams


# =========================================================
# 4. WORD N-GRAMS
# =========================================================

def make_word_ngrams(tokens, n_min=1, n_max=2):
    """
    Дополнительно создаёт n-граммы слов.

    Например:

    ["курс", "доллара", "вырос"]

    unigram:
        курс
        доллара
        вырос

    bigram:
        курс доллара
        доллара вырос
    """

    grams = []

    for n in range(n_min, n_max + 1):

        for i in range(len(tokens) - n + 1):

            gram = " ".join(
                tokens[i:i + n]
            )

            grams.append(gram)

    return grams


# =========================================================
# 5. СОЗДАНИЕ ВСЕХ FEATURES
# =========================================================

def extract_features(
    text,
    char_n_min=2,
    char_n_max=5,
    word_n_min=1,
    word_n_max=2
):
    """
    Получаем признаки документа:

    - буквенные n-граммы
    - n-граммы слов
    """

    tokens = tokenize(text)

    char_grams = make_character_ngrams(
        text,
        char_n_min,
        char_n_max
    )

    word_grams = make_word_ngrams(
        tokens,
        word_n_min,
        word_n_max
    )

    return char_grams + word_grams


# =========================================================
# 6. TF-IDF
# =========================================================

def build_tfidf(
    texts,
    char_n_min=2,
    char_n_max=5,
    word_n_min=1,
    word_n_max=2
):

    documents = []

    # -----------------------------------------------------
    # Получаем признаки каждого документа
    # -----------------------------------------------------

    for text in texts:

        features = extract_features(
            text,
            char_n_min,
            char_n_max,
            word_n_min,
            word_n_max
        )

        documents.append(features)

    # -----------------------------------------------------
    # Создаём словарь всех признаков
    # -----------------------------------------------------

    vocabulary = sorted(
        set(
            feature
            for document in documents
            for feature in document
        )
    )

    feature_to_id = {
        feature: i
        for i, feature in enumerate(vocabulary)
    }

    # -----------------------------------------------------
    # Document Frequency
    # -----------------------------------------------------

    document_frequency = Counter()

    for document in documents:

        # Важен set:
        # если n-грамма встретилась в одном документе
        # 10 раз, она всё равно считается как 1 документ.
        for feature in set(document):

            document_frequency[feature] += 1

    N = len(documents)

    # -----------------------------------------------------
    # Создаём TF-IDF-векторы
    # -----------------------------------------------------

    vectors = []

    for document in documents:

        counts = Counter(document)

        total_features = len(document)

        vector = [
            0.0
            for _ in vocabulary
        ]

        if total_features == 0:

            vectors.append(vector)

            continue

        for feature, count in counts.items():

            # -----------------------------
            # TF
            # -----------------------------

            tf = count / total_features

            # -----------------------------
            # IDF
            # -----------------------------

            idf = (
                math.log(
                    (N + 1) /
                    (document_frequency[feature] + 1)
                )
                + 1
            )

            # -----------------------------
            # TF-IDF
            # -----------------------------

            vector[
                feature_to_id[feature]
            ] = tf * idf

        vectors.append(vector)

    return vectors, vocabulary


# =========================================================
# 7. НОРМАЛИЗАЦИЯ ВЕКТОРА
# =========================================================

def normalize_vector(vector):
    """
    L2-нормализация.

    После нормализации длина вектора становится 1.
    """

    norm = math.sqrt(
        sum(
            value * value
            for value in vector
        )
    )

    if norm == 0:
        return vector[:]

    return [
        value / norm
        for value in vector
    ]


# =========================================================
# 8. КОСИНУСНОЕ СХОДСТВО
# =========================================================

def cosine_similarity(a, b):

    norm_a = math.sqrt(
        sum(
            value * value
            for value in a
        )
    )

    norm_b = math.sqrt(
        sum(
            value * value
            for value in b
        )
    )

    if norm_a == 0 or norm_b == 0:
        return 0.0

    dot = sum(
        x * y
        for x, y in zip(a, b)
    )

    return dot / (norm_a * norm_b)


# =========================================================
# 9. СУММА ВЕКТОРОВ
# =========================================================

def sum_vectors(vectors):

    if not vectors:
        return []

    dimension = len(vectors[0])

    result = [
        0.0
        for _ in range(dimension)
    ]

    for vector in vectors:

        for i, value in enumerate(vector):

            result[i] += value

    return result


# =========================================================
# 10. СФЕРИЧЕСКИЙ K-MEANS
# =========================================================

def spherical_kmeans(
    vectors,
    k,
    max_iterations=100,
    n_init=10
):
    """
    Кластеризация текстов с использованием
    косинусного сходства.

    k — количество кластеров.
    """

    if k <= 0:
        raise ValueError(
            "Количество кластеров должно быть больше 0."
        )

    if k > len(vectors):
        raise ValueError(
            "Количество кластеров не может быть "
            "больше количества документов."
        )

    # -----------------------------------------------------
    # Нормализуем документы
    # -----------------------------------------------------

    normalized_vectors = [
        normalize_vector(vector)
        for vector in vectors
    ]

    best_clusters = None
    best_centroids = None
    best_score = float('-inf')

    # =====================================================
    # Несколько запусков с разными начальными центрами
    # =====================================================

    for _ in range(n_init):

        # -------------------------------------------------
        # Начальные центроиды
        # -------------------------------------------------

        initial_indices = random.sample(
            range(len(vectors)),
            k
        )

        centroids = [
            normalized_vectors[i][:]
            for i in initial_indices
        ]

        previous_clusters = None

        # =================================================
        # Основной цикл
        # =================================================

        for iteration in range(max_iterations):

            # ---------------------------------------------
            # Создаём пустые кластеры
            # ---------------------------------------------

            clusters = [
                []
                for _ in range(k)
            ]

            # ---------------------------------------------
            # Назначаем документы к ближайшему центроиду
            # ---------------------------------------------

            for document_id, vector in enumerate(
                normalized_vectors
            ):

                similarities = [
                    cosine_similarity(
                        vector,
                        centroid
                    )
                    for centroid in centroids
                ]

                cluster_id = similarities.index(
                    max(similarities)
                )

                clusters[cluster_id].append(
                    document_id
                )

            # ---------------------------------------------
            # Проверяем, изменились ли кластеры
            # ---------------------------------------------

            if clusters == previous_clusters:
                break

            previous_clusters = [
                cluster[:]
                for cluster in clusters
            ]

            # ---------------------------------------------
            # Пересчитываем центроиды
            # ---------------------------------------------

            new_centroids = []

            for cluster_id in range(k):

                document_ids = clusters[cluster_id]

                # -----------------------------------------
                # Если кластер пустой
                # -----------------------------------------

                if not document_ids:

                    # Берём случайный документ
                    random_id = random.randrange(
                        len(normalized_vectors)
                    )

                    new_centroids.append(
                        normalized_vectors[random_id][:]
                    )

                    continue

                # -----------------------------------------
                # Суммируем векторы документов
                # -----------------------------------------

                cluster_vectors = [
                    normalized_vectors[i]
                    for i in document_ids
                ]

                centroid = sum_vectors(
                    cluster_vectors
                )

                # -----------------------------------------
                # Нормализуем центроид
                # -----------------------------------------

                centroid = normalize_vector(
                    centroid
                )

                new_centroids.append(
                    centroid
                )

            centroids = new_centroids

        # =================================================
        # Оцениваем получившуюся кластеризацию
        # =================================================

        score = 0.0

        for cluster_id, document_ids in enumerate(
            clusters
        ):

            centroid = centroids[cluster_id]

            for document_id in document_ids:

                score += cosine_similarity(
                    normalized_vectors[document_id],
                    centroid
                )

        # -------------------------------------------------
        # Сохраняем лучший результат
        # -------------------------------------------------

        if score > best_score:

            best_score = score

            best_clusters = [
                cluster[:]
                for cluster in clusters
            ]

            best_centroids = [
                centroid[:]
                for centroid in centroids
            ]

    return (
        best_clusters,
        best_centroids,
        best_score
    )


# =========================================================
# 11. ПРИМЕР ДАННЫХ
# =========================================================

texts = [
    # --- Логин / пароль ---
    "Клиент не может войти в личный кабинет после смены пароля",
    "Не получается авторизоваться в личном кабинете, пароль не принимается",
    "Пользователь забыл пароль и не может войти в аккаунт",
    "После изменения пароля клиент не может выполнить вход",
    "Ошибка авторизации при входе в личный кабинет",

    # --- Оплата / карта ---
    "Не проходит оплата банковской картой",
    "Клиент не может оплатить заказ картой",
    "Ошибка при оплате банковской картой",
    "Платеж картой отклонен при оформлении заказа",
    "Не удается провести оплату по карте",

    # --- Доставка ---
    "Клиент не получил заказ в указанный срок",
    "Заказ задерживается, доставка еще не выполнена",
    "Посылка не пришла в ожидаемую дату",
    "Задержка доставки заказа клиенту",
    "Когда будет доставлен мой заказ",

    # --- Возврат ---
    "Как вернуть товар и получить деньги обратно",
    "Клиент хочет оформить возврат товара",
    "Необходимо вернуть заказ и оформить возврат средств",
    "Как оформить возврат покупки",
    "Возврат товара после получения заказа",

    # --- Другие обращения ---
    "Хочу изменить адрес доставки для заказа",
    "Как поменять номер телефона в профиле",
    "Приложение закрывается при запуске",
    "Не приходит код подтверждения на телефон",
]


# =========================================================
# 12. ПОЛЬЗОВАТЕЛЬ ЗАДАЁТ КОЛИЧЕСТВО КЛАСТЕРОВ
# =========================================================

k = K


# =========================================================
# 13. TF-IDF
# =========================================================

vectors, vocabulary = build_tfidf(
    texts,

    char_n_min=NGRAM_MIN,
    char_n_max=NGRAM_MAX,

    word_n_min=1,
    word_n_max=2
)


# =========================================================
# 14. КЛАСТЕРИЗАЦИЯ
# =========================================================

clusters, centroids, score = spherical_kmeans(
    vectors,
    k=k,
    max_iterations=MAX_ITERATIONS,
    n_init=N_INIT
)


# =========================================================
# 15. ВЫВОД РЕЗУЛЬТАТОВ
# =========================================================

for cluster_id, cluster in enumerate(clusters):

    print()
    print("=" * 60)
    print(
        f"КЛАСТЕР {cluster_id + 1}"
    )
    print("=" * 60)

    for document_id in cluster:

        print(
            f"[{document_id}] "
            f"{texts[document_id]}"
        )


print()
print("=" * 60)
print(f"Количество кластеров: {k}")
print(f"Итоговый score: {score:.4f}")
print("=" * 60)
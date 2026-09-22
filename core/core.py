import math
import re
from collections import Counter


# ---------------------------------------------------------
# 1. Подготовка текста
# ---------------------------------------------------------

def tokenize(text):
    """Разбивает русский/английский текст на слова."""
    return re.findall(r'\b\w+\b', text.lower())


# ---------------------------------------------------------
# 2. Построение n-грамм
# ---------------------------------------------------------

def make_ngrams(tokens, n_min=1, n_max=2):
    """Создаёт word n-граммы от n_min до n_max."""
    grams = []

    for n in range(n_min, n_max + 1):
        for i in range(len(tokens) - n + 1):
            gram = " ".join(tokens[i:i + n])
            grams.append(gram)

    return grams


# ---------------------------------------------------------
# 3. TF-IDF
# ---------------------------------------------------------

def build_tfidf(texts, n_min=1, n_max=2):
    """
    Возвращает:
        vectors — TF-IDF-векторы документов
        vocabulary — словарь n-грамм
    """

    documents = []

    for text in texts:
        tokens = tokenize(text)
        grams = make_ngrams(tokens, n_min, n_max)
        documents.append(grams)

    # Словарь всех n-грамм
    vocabulary = sorted(
        set(gram for document in documents for gram in document)
    )

    word_to_id = {
        word: i
        for i, word in enumerate(vocabulary)
    }

    # В скольких документах встречается n-грамма
    document_frequency = Counter()

    for document in documents:
        for gram in set(document):
            document_frequency[gram] += 1

    N = len(documents)

    vectors = []

    for document in documents:

        counts = Counter(document)
        total = len(document)

        vector = [0.0] * len(vocabulary)

        for gram, count in counts.items():

            # TF
            tf = count / total

            # IDF
            idf = math.log(
                (N + 1) / (document_frequency[gram] + 1)
            ) + 1

            vector[word_to_id[gram]] = tf * idf

        vectors.append(vector)

    return vectors, vocabulary


# ---------------------------------------------------------
# 4. Косинусное сходство
# ---------------------------------------------------------

def cosine_similarity(a, b):

    dot = sum(x * y for x, y in zip(a, b))

    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)


# ---------------------------------------------------------
# 5. Косинусное расстояние
# ---------------------------------------------------------

def cosine_distance(a, b):
    return 1.0 - cosine_similarity(a, b)


# ---------------------------------------------------------
# 6. Расстояние между кластерами
# ---------------------------------------------------------

def cluster_distance(cluster_a, cluster_b, vectors):
    """
    Среднее расстояние между всеми парами документов
    двух кластеров.
    """

    distances = []

    for i in cluster_a:
        for j in cluster_b:
            distances.append(
                cosine_distance(vectors[i], vectors[j])
            )

    return sum(distances) / len(distances)


# ---------------------------------------------------------
# 7. Агломеративная кластеризация
# ---------------------------------------------------------

def agglomerative_clustering(
    vectors,
    threshold=0.65
):
    """
    Начинаем с одного кластера на каждый текст.

    Объединяем два наиболее похожих кластера,
    пока расстояние между ними не станет больше threshold.

    threshold — максимальное допустимое
    cosine distance.
    """

    clusters = [
        [i]
        for i in range(len(vectors))
    ]

    while True:

        best_distance = float("inf")
        best_pair = None

        # Ищем два наиболее близких кластера
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):

                distance = cluster_distance(
                    clusters[i],
                    clusters[j],
                    vectors
                )

                if distance < best_distance:
                    best_distance = distance
                    best_pair = (i, j)

        # Больше объединять нельзя
        if best_pair is None:
            break

        if best_distance > threshold:
            break

        i, j = best_pair

        # Объединяем
        new_cluster = clusters[i] + clusters[j]

        # Удаляем старые кластеры
        clusters.pop(j)
        clusters.pop(i)

        # Добавляем новый
        clusters.append(new_cluster)

    return clusters


# Пример

texts = [
    "курс доллара вырос",
    "доллар подорожал сегодня",
    "курс американской валюты вырос",

    "акции компании резко упали",
    "цена акций компании снизилась",

    "команда выиграла матч",
    "футболист забил два гола",
    "команда победила в матче",

    "новый смартфон поступил в продажу",
    "компания представила новый смартфон",
]


# TF-IDF + n-граммы
vectors, vocabulary = build_tfidf(
    texts,
    n_min=0,
    n_max=1
)


# Кластеризация
clusters = agglomerative_clustering(
    vectors,
    threshold=0.5
)

for cluster_id, cluster in enumerate(clusters):

    print(f"\nКластер {cluster_id + 1}:")

    for document_id in cluster:
        print(
            f"  [{document_id}] {texts[document_id]}"
        )
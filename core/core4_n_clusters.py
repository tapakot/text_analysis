import math
import re
import random
from collections import Counter


# =========================================================
# НАСТРОЙКИ
# =========================================================

# Количество кластеров задаёт пользователь
K = 4

# Буквенные n-граммы
CHAR_NGRAM_MIN = 2
CHAR_NGRAM_MAX = 5

# N-граммы слов
WORD_NGRAM_MIN = 1
WORD_NGRAM_MAX = 2

# Максимальное количество итераций
MAX_ITERATIONS = 100

# Количество запусков алгоритма
N_INIT = 20

# Для воспроизводимости
RANDOM_SEED = 42


# =========================================================
# 1. ОЧИСТКА ТЕКСТА
# =========================================================

def clean_text(text):

    text = text.lower()

    # Оставляем только буквы и пробелы
    text = re.sub(
        r'[^\W\d_]+',
        ' ',
        text,
        flags=re.UNICODE
    )

    # Убираем лишние пробелы
    text = re.sub(
        r'\s+',
        ' ',
        text
    ).strip()

    return text


# =========================================================
# 2. ТОКЕНИЗАЦИЯ
# =========================================================

def tokenize(text):

    text = clean_text(text)

    return re.findall(
        r'[^\W\d_]+',
        text,
        flags=re.UNICODE
    )


# =========================================================
# 3. БУКВЕННЫЕ N-ГРАММЫ
# =========================================================

def make_character_ngrams(
    text,
    n_min=2,
    n_max=5
):

    tokens = tokenize(text)

    grams = []

    # N-граммы строим отдельно внутри каждого слова
    for token in tokens:

        for n in range(
            n_min,
            n_max + 1
        ):

            if len(token) < n:
                continue

            for i in range(
                len(token) - n + 1
            ):

                grams.append(
                    token[i:i + n]
                )

    return grams


# =========================================================
# 4. N-GRAMS СЛОВ
# =========================================================

def make_word_ngrams(
    tokens,
    n_min=1,
    n_max=2
):

    grams = []

    for n in range(
        n_min,
        n_max + 1
    ):

        for i in range(
            len(tokens) - n + 1
        ):

            grams.append(
                " ".join(
                    tokens[i:i + n]
                )
            )

    return grams


# =========================================================
# 5. ПРИЗНАКИ ДОКУМЕНТА
# =========================================================

def extract_features(
    text,
    char_n_min=2,
    char_n_max=5,
    word_n_min=1,
    word_n_max=2
):

    tokens = tokenize(text)

    char_ngrams = make_character_ngrams(
        text,
        char_n_min,
        char_n_max
    )

    word_ngrams = make_word_ngrams(
        tokens,
        word_n_min,
        word_n_max
    )

    return char_ngrams + word_ngrams


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
    # Получаем признаки документов
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
    # Словарь
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

        for feature in set(document):

            document_frequency[feature] += 1

    N = len(documents)

    # -----------------------------------------------------
    # TF-IDF
    # -----------------------------------------------------

    vectors = []

    for document in documents:

        counts = Counter(document)

        total = len(document)

        vector = [
            0.0
            for _ in vocabulary
        ]

        if total == 0:

            vectors.append(vector)
            continue

        for feature, count in counts.items():

            tf = count / total

            idf = (
                math.log(
                    (N + 1)
                    /
                    (document_frequency[feature] + 1)
                )
                + 1
            )

            vector[
                feature_to_id[feature]
            ] = tf * idf

        vectors.append(vector)

    return vectors, vocabulary


# =========================================================
# 7. НОРМАЛИЗАЦИЯ
# =========================================================

def normalize_vector(vector):

    norm = math.sqrt(
        sum(
            x * x
            for x in vector
        )
    )

    if norm == 0:
        return vector[:]

    return [
        x / norm
        for x in vector
    ]


# =========================================================
# 8. КОСИНУСНОЕ СХОДСТВО
# =========================================================

def cosine_similarity(a, b):

    dot = sum(
        x * y
        for x, y in zip(a, b)
    )

    norm_a = math.sqrt(
        sum(
            x * x
            for x in a
        )
    )

    norm_b = math.sqrt(
        sum(
            y * y
            for y in b
        )
    )

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (
        norm_a * norm_b
    )


# =========================================================
# 9. ИНИЦИАЛИЗАЦИЯ ЦЕНТРОИДОВ
# =========================================================

def initialize_centroids(
    vectors,
    k
):
    """
    Выбираем первый центр случайно.

    Каждый следующий центр — это документ,
    максимально удалённый от ближайшего
    уже выбранного центра.

    Это вариант farthest-point initialization.
    """

    n = len(vectors)

    first = random.randrange(n)

    selected = [first]

    while len(selected) < k:

        best_document = None
        best_distance = -1.0

        for document_id, vector in enumerate(
            vectors
        ):

            if document_id in selected:
                continue

            # Расстояние до ближайшего центра
            min_distance = float("inf")

            for center_id in selected:

                similarity = cosine_similarity(
                    vector,
                    vectors[center_id]
                )

                distance = 1.0 - similarity

                if distance < min_distance:
                    min_distance = distance

            # Берём наиболее удалённый документ
            if min_distance > best_distance:

                best_distance = min_distance
                best_document = document_id

        selected.append(
            best_document
        )

    return [
        vectors[i][:]
        for i in selected
    ]


# =========================================================
# 10. НАЗНАЧЕНИЕ ДОКУМЕНТОВ
# =========================================================

def assign_documents(
    vectors,
    centroids
):

    assignments = []

    for vector in vectors:

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

        assignments.append(
            cluster_id
        )

    return assignments


# =========================================================
# 11. СОЗДАНИЕ КЛАСТЕРОВ
# =========================================================

def make_clusters(
    assignments,
    k
):

    clusters = [
        []
        for _ in range(k)
    ]

    for document_id, cluster_id in enumerate(
        assignments
    ):

        clusters[cluster_id].append(
            document_id
        )

    return clusters


# =========================================================
# 12. ИСПРАВЛЕНИЕ ПУСТЫХ КЛАСТЕРОВ
# =========================================================

def repair_empty_clusters(
    vectors,
    assignments,
    centroids,
    k
):
    """
    Очень важный этап.

    Если кластер пустой, мы НЕ создаём
    случайный центр.

    Вместо этого берём документ, который
    хуже всего соответствует своему
    текущему кластеру, и переносим его
    в пустой кластер.

    При этом не позволяем донорскому
    кластеру стать пустым.
    """

    clusters = make_clusters(
        assignments,
        k
    )

    # -----------------------------------------------------
    # Ищем пустые кластеры
    # -----------------------------------------------------

    empty_clusters = [
        cluster_id
        for cluster_id, cluster in enumerate(clusters)
        if not cluster
    ]

    for empty_cluster in empty_clusters:

        best_document = None
        worst_similarity = float("inf")

        # -------------------------------------------------
        # Ищем наиболее "чужой" документ
        # -------------------------------------------------

        for cluster_id in range(k):

            # В кластере должен остаться хотя бы
            # один документ
            if len(
                clusters[cluster_id]
            ) <= 1:
                continue

            centroid = centroids[
                cluster_id
            ]

            for document_id in clusters[
                cluster_id
            ]:

                similarity = cosine_similarity(
                    vectors[document_id],
                    centroid
                )

                if similarity < worst_similarity:

                    worst_similarity = similarity
                    best_document = document_id

        # -------------------------------------------------
        # Если нашли документ — переносим
        # -------------------------------------------------

        if best_document is not None:

            old_cluster = assignments[
                best_document
            ]

            assignments[
                best_document
            ] = empty_cluster

            clusters[
                old_cluster
            ].remove(
                best_document
            )

            clusters[
                empty_cluster
            ].append(
                best_document
            )

    return assignments


# =========================================================
# 13. ПЕРЕСЧЁТ ЦЕНТРОИДОВ
# =========================================================

def calculate_centroids(
    vectors,
    assignments,
    k
):

    dimension = len(vectors[0])

    centroids = []

    for cluster_id in range(k):

        document_ids = [
            i
            for i, assigned_cluster in enumerate(
                assignments
            )
            if assigned_cluster == cluster_id
        ]

        # На этом этапе пустых кластеров уже
        # быть не должно.
        if not document_ids:

            centroids.append(
                [0.0] * dimension
            )

            continue

        centroid = [
            0.0
            for _ in range(dimension)
        ]

        # Среднее
        for document_id in document_ids:

            vector = vectors[
                document_id
            ]

            for i in range(dimension):

                centroid[i] += vector[i]

        count = len(document_ids)

        centroid = [
            value / count
            for value in centroid
        ]

        # Нормализация
        centroid = normalize_vector(
            centroid
        )

        centroids.append(
            centroid
        )

    return centroids


# =========================================================
# 14. ОЦЕНКА
# =========================================================

def calculate_score(
    vectors,
    assignments,
    centroids
):

    score = 0.0

    for document_id, cluster_id in enumerate(
        assignments
    ):

        similarity = cosine_similarity(
            vectors[document_id],
            centroids[cluster_id]
        )

        score += similarity

    return score


# =========================================================
# 15. SPHERICAL K-MEANS
# =========================================================

def spherical_kmeans(
    vectors,
    k,
    max_iterations=100,
    n_init=20
):

    if k <= 0:
        raise ValueError(
            "K должно быть больше 0."
        )

    if k > len(vectors):
        raise ValueError(
            "K не может быть больше "
            "количества документов."
        )

    # -----------------------------------------------------
    # Нормализуем документы
    # -----------------------------------------------------

    vectors = [
        normalize_vector(vector)
        for vector in vectors
    ]

    best_assignments = None
    best_centroids = None
    best_score = float("-inf")

    # =====================================================
    # Несколько запусков
    # =====================================================

    for run in range(n_init):

        # -------------------------------------------------
        # Начальные центроиды
        # -------------------------------------------------

        centroids = initialize_centroids(
            vectors,
            k
        )

        assignments = None

        # =================================================
        # Итерации
        # =================================================

        for iteration in range(
            max_iterations
        ):

            # ---------------------------------------------
            # Назначаем документы
            # ---------------------------------------------

            new_assignments = assign_documents(
                vectors,
                centroids
            )

            # ---------------------------------------------
            # Исправляем пустые кластеры
            # ---------------------------------------------

            new_assignments = repair_empty_clusters(
                vectors,
                new_assignments,
                centroids,
                k
            )

            # ---------------------------------------------
            # Проверяем сходимость
            # ---------------------------------------------

            if (
                assignments is not None
                and
                new_assignments == assignments
            ):

                break

            assignments = new_assignments

            # ---------------------------------------------
            # Пересчитываем центроиды
            # ---------------------------------------------

            centroids = calculate_centroids(
                vectors,
                assignments,
                k
            )

        # =================================================
        # Финальное назначение
        # =================================================

        assignments = assign_documents(
            vectors,
            centroids
        )

        # -------------------------------------------------
        # Ещё раз проверяем пустые кластеры
        # -------------------------------------------------

        assignments = repair_empty_clusters(
            vectors,
            assignments,
            centroids,
            k
        )

        # После переноса документа центроид этого
        # кластера изменился — пересчитываем его.
        centroids = calculate_centroids(
            vectors,
            assignments,
            k
        )

        # -------------------------------------------------
        # Score
        # -------------------------------------------------

        score = calculate_score(
            vectors,
            assignments,
            centroids
        )

        # -------------------------------------------------
        # Сохраняем лучший результат
        # -------------------------------------------------

        if score > best_score:

            best_score = score

            best_assignments = assignments[:]

            best_centroids = [
                centroid[:]
                for centroid in centroids
            ]

    # -----------------------------------------------------
    # Формируем кластеры
    # -----------------------------------------------------

    best_clusters = make_clusters(
        best_assignments,
        k
    )

    return (
        best_clusters,
        best_centroids,
        best_score
    )


# =========================================================
# 16. ДАННЫЕ
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
# 17. ПАРАМЕТР K
# =========================================================

K = 4


# =========================================================
# 18. TF-IDF
# =========================================================

vectors, vocabulary = build_tfidf(
    texts,

    char_n_min=CHAR_NGRAM_MIN,
    char_n_max=CHAR_NGRAM_MAX,

    word_n_min=WORD_NGRAM_MIN,
    word_n_max=WORD_NGRAM_MAX
)


# =========================================================
# 19. КЛАСТЕРИЗАЦИЯ
# =========================================================

random.seed(
    RANDOM_SEED
)

clusters, centroids, score = spherical_kmeans(
    vectors,
    k=K,

    max_iterations=MAX_ITERATIONS,
    n_init=N_INIT
)


# =========================================================
# 20. ВЫВОД
# =========================================================

print()
print("=" * 70)
print("РЕЗУЛЬТАТ")
print("=" * 70)

for cluster_id, cluster in enumerate(
    clusters
):

    print()
    print(
        f"КЛАСТЕР {cluster_id + 1}"
    )

    print(
        "-" * 70
    )

    for document_id in cluster:

        print(
            f"[{document_id}] "
            f"{texts[document_id]}"
        )


print()
print("=" * 70)

print(
    "Количество кластеров:",
    K
)

print(
    "Размеры кластеров:",
    [
        len(cluster)
        for cluster in clusters
    ]
)

print(
    "Score:",
    round(score, 4)
)

print("=" * 70)
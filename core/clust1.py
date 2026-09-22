import math
import re
import random
import time
from collections import defaultdict, Counter
import matplotlib.pyplot as plt

# ==========================================
# 1. ОЧИСТКА И ПРОСТАЯ ЛЕММАТИЗАЦИЯ
# ==========================================

STOP_WORDS = {
    "и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как", "а", "то", "все", "она",
    "так", "его", "но", "да", "ты", "к", "у", "же", "вы", "за", "бы", "по", "только", "ее",
    "мне", "было", "вот", "от", "меня", "еще", "нет", "о", "из", "ему", "теперь", "когда",
    "даже", "ну", "вдруг", "ли", "если", "уже", "или", "ни", "быть", "был", "него", "до",
    "вас", "нибудь", "опять", "уж", "вам", "ведь", "там", "потом", "себя", "ничего", "ей",
    "может", "они", "тут", "где", "есть", "надо", "ней", "для", "мы", "тебя", "их", "чем",
    "была", "сам", "чтоб", "без", "будто", "чего", "раз", "тоже", "себе", "под", "будет",
    "ж", "тогда", "кто", "этот", "того", "потому", "этого", "какой", "совсем", "ним", "здесь",
    "этом", "один", "почти", "мой", "тем", "чтобы", "нее", "сейчас", "были", "куда", "зачем",
    "всех", "никогда", "можно", "наш", "друг", "про", "картинка", "где", "быть", "при", "пожалуйста"
}

def simple_stemmer_ru(word: str) -> str:
    """ Упрощенный стемминг/лемматизация для русского языка без внешних зависимостей. """
    word = word.lower()
    if len(word) <= 3:
        return word
    # Удаление типичных окончаний/суффиксов
    patterns = [
        r"(иями|ями|ами|ем|ом|ми|ям|ам|ей|ов|ев|ия|ья|а|я|у|ю|о|е|ы|и|ь)$",
        r"(анного|янного|енного|ованного|ующего|ющего|емого|имого|ского|ского|ого|его|ому|ему|ым|им|ых|их|ая|яя|ое|ее|ые|ие)$",
        r"(вшись|вшись|вшись|вшись|вша|вши|в|вшись|ла|ло|ли|л|ть|ти|ет|ют|ней|ем|им|ите|ете|ат|ят)$"
    ]
    for p in patterns:
        new_word = re.sub(p, "", word)
        if len(new_word) >= 3:
            return new_word
    return word

def preprocess_text(text: str) -> list[str]:
    """ Токенизация, очистка и простейшая лемматизация. """
    text = text.lower()
    words = re.findall(r'[а-яёa-z0-9]+', text)
    processed = []
    for w in words:
        if w in STOP_WORDS or len(w) < 2 or w.isdigit():
            continue
        stem = simple_stemmer_ru(w)
        processed.append(stem)
    return processed

def generate_ngrams(tokens: list[str], text: str, word_n_range=(1, 2), char_n=3) -> list[str]:
    """ Генерация словесных N-грамм и буквенных N-грамм. """
    ngrams = []
    # Словесные N-граммы
    for n in range(word_n_range[0], word_n_range[1] + 1):
        for i in range(len(tokens) - n + 1):
            ngrams.append("w:" + "_".join(tokens[i:i+n]))
    
    # Буквенные N-граммы (из очищенных слов, чтобы уменьшить размерность)
    for word in tokens:
        if len(word) >= char_n:
            for i in range(len(word) - char_n + 1):
                ngrams.append("c:" + word[i:i+char_n])
                
    return ngrams

# ==========================================
# 2. РАЗРЕЖЕННАЯ ВЕКТОРНАЯ МОДЕЛЬ (TF-IDF)
# ==========================================

class SparseVector(dict):
    """ Разреженный вектор (словарь: id_признака -> значение). """
    def norm(self):
        return math.sqrt(sum(v * v for v in self.values()))

    def dot(self, other: 'SparseVector') -> float:
        if len(self) > len(other):
            return other.dot(self)
        return sum(val * other.get(k, 0.0) for k, val in self.items())

def normalize_sparse(vec: SparseVector) -> SparseVector:
    n = vec.norm()
    if n == 0:
        return vec
    return SparseVector({k: v / n for k, v in vec.items()})

class FastTfidfVectorizer:
    def __init__(self, min_df=3, max_features=10000):
        self.min_df = min_df
        self.max_features = max_features
        self.vocab = {}
        self.inv_vocab = {}
        self.idf = {}

    def fit_transform(self, documents_ngrams: list[list[str]]) -> list[SparseVector]:
        df = Counter()
        N = len(documents_ngrams)
        for ngrams in documents_ngrams:
            df.update(set(ngrams))

        # Ограничиваем словарь наиболее частыми признаками с фильтрацией редких
        valid_features = [f for f, count in df.items() if count >= self.min_df]
        valid_features.sort(key=lambda x: df[x], reverse=True)
        valid_features = valid_features[:self.max_features]

        self.vocab = {feat: idx for idx, feat in enumerate(valid_features)}
        self.inv_vocab = {idx: feat for feat, idx in self.vocab.items()}
        self.idf = {idx: math.log((1 + N) / (1 + df[feat])) + 1.0 for feat, idx in self.vocab.items()}

        vectors = []
        for ngrams in documents_ngrams:
            tf = Counter(ngrams)
            vec = SparseVector()
            for ng, count in tf.items():
                if ng in self.vocab:
                    idx = self.vocab[ng]
                    vec[idx] = (count / len(ngrams)) * self.idf[idx]
            vectors.append(normalize_sparse(vec))
        return vectors

# ==========================================
# 3. АЛГОРИТМЫ КЛАСТЕРИЗАЦИИ
# ==========================================

def mini_batch_kmeans(vectors: list[SparseVector], k: int, batch_size: int = 1000, max_iters: int = 20):
    """ Mini-Batch K-Means для быстрого обучения на больших объемах данных. """
    n_docs = len(vectors)
    centroids = [SparseVector(v) for v in random.sample(vectors, k)]
    counts = [0] * k

    for it in range(max_iters):
        batch = random.sample(vectors, batch_size)
        for vec in batch:
            # Поиск ближайшего центроида по косинусному расстоянию (dot-product для нормализованных векторов)
            best_k = max(range(k), key=lambda idx: vec.dot(centroids[idx]))
            counts[best_k] += 1
            eta = 1.0 / counts[best_k]
            
            # Обновление центроида
            cent = centroids[best_k]
            for key, val in vec.items():
                cent[key] = (1 - eta) * cent.get(key, 0.0) + eta * val
            
            centroids[best_k] = SparseVector(cent)

    # Окончательное присвоение кластеров
    labels = []
    for vec in vectors:
        best_k = max(range(k), key=lambda idx: vec.dot(centroids[idx]))
        labels.append(best_k)
        
    return labels, centroids

def leader_clustering(vectors: list[SparseVector], threshold: float = 0.35):
    """ Алгоритм Leader (Canopy) для произвольного определения числа кластеров. """
    centroids = []
    labels = []
    
    for vec in vectors:
        best_idx = -1
        best_sim = -1.0
        
        for c_idx, cent in enumerate(centroids):
            sim = vec.dot(cent)
            if sim > best_sim:
                best_sim = sim
                best_idx = c_idx
                
        if best_sim >= threshold:
            labels.append(best_idx)
        else:
            # Создание нового кластера
            centroids.append(vec)
            labels.append(len(centroids) - 1)
            
    return labels, centroids

# ==========================================
# 4. ВЫВОД ТОП-ПРИЗНАКОВ И ВИЗУАЛИЗАЦИЯ
# ==========================================

def print_top_ngrams(centroids, inv_vocab, top_n=8):
    """ Вывод наикратчайшего топа n-грамм для каждого кластера. """
    for idx, cent in enumerate(centroids):
        sorted_feats = sorted(cent.items(), key=lambda x: x[1], reverse=True)[:top_n]
        readable_feats = []
        for feat_id, score in sorted_feats:
            name = inv_vocab.get(feat_id, "")
            prefix = "[Слово]" if name.startswith("w:") else "[Буква]"
            clean_name = name[2:]
            readable_feats.append(f"{prefix} {clean_name} ({score:.3f})")
        print(f"\n--- Кластер {idx} (Всего элементов / Топ n-грамм) ---")
        print(" | ".join(readable_feats))

def plot_clusters_distribution(labels):
    """ Визуализация распределения количества записей по кластерам. """
    counter = Counter(labels)
    clusters = sorted(counter.keys())
    counts = [counter[c] for c in clusters]

    plt.figure(figsize=(10, 5))
    plt.bar([str(c) for c in clusters], counts, color='skyblue', edgecolor='black')
    plt.xlabel('ID Кластера')
    plt.ylabel('Количество обращений')
    plt.title('Распределение обращений в поддержку по кластерам')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

# ==========================================
# 5. СИНТЕТИЧЕСКИЙ ТЕСТОВЫЙ КОРПУС
# ==========================================

templates = [
    # Тема 0: Оплата
    ("Не проходит оплата картой на сайте. Деньги списались, но статус заказа не изменился.",
     "Проверен транзакционный лог эквайринга.",
     "Оформлен возврат денежных средств клиенту."),
    
    # Тема 1: Авторизация
    ("Не могу войти в личный кабинет. Не приходит смс с кодом подтверждения.",
     "Проверена отправка шлюза сообщений.",
     "Сброшен профиль авторизации, отправлена ссылка на email."),
     
    # Тема 2: Доставка
    ("Задерживается доставка заказа курьером. Прошло уже более трех дней.",
     "Связались с логистической службой.",
     "Переназначили время доставки курьером на завтра."),
     
    # Тема 3: Возврат товара
    ("Пришел поврежденный товар в разбитой упаковке. Хочу вернуть деньги.",
     "Запрошены фотографии повреждений у покупателя.",
     "Составлен акт брака, направлен курьер для забора."),
     
    # Тема 4: Удаление данных
    ("Прошу удалить мои персональные данные и заблокировать аккаунт.",
     "Проведена идентификация пользователя.",
     "Профиль заблокирован, данные удалены из базы.")
]

print("Генерация корпуса из 100,000 записей...")
N_SAMPLES = 100000
raw_dataset = []

for i in range(N_SAMPLES):
    tmpl = templates[i % len(templates)]
    # Небольшая вариативность текста
    col1 = tmpl[0] + f" [ID обращение {i}]"
    col2 = tmpl[1]
    col3 = tmpl[2]
    raw_dataset.append((col1, col2, col3))

# ==========================================
# 6. ВЫПОЛНЕНИЕ И ЗАМЕР ВРЕМЕНИ
# ==========================================

start_time = time.time()

# 1. Объединение 3 столбцов в один текст и предобработка
print("1. Предобработка и генерация N-грамм...")
docs_ngrams = []
for col1, col2, col3 in raw_dataset:
    full_text = f"{col1} {col2} {col3}"
    tokens = preprocess_text(full_text)
    ngrams = generate_ngrams(tokens, full_text, word_n_range=(1, 2), char_n=3)
    docs_ngrams.append(ngrams)

# 2. Построение TF-IDF векторов
print("2. Векторизация TF-IDF...")
vectorizer = FastTfidfVectorizer(min_df=5, max_features=5000)
vectors = vectorizer.fit_transform(docs_ngrams)

# 3. Кластеризация
# Вариант А: Заданное число кластеров (Mini-Batch K-Means, K=5)
print("3. Кластеризация Mini-Batch K-Means (K=5)...")
labels, centroids = mini_batch_kmeans(vectors, k=5, batch_size=2000, max_iters=15)

# Вариант Б: Произвольное число кластеров (Leader Clustering)
# labels, centroids = leader_clustering(vectors[:10000], threshold=0.4) # Для тестов на подвыборке

end_time = time.time()
print(f"\nВремя обработки 100,000 записей: {end_time - start_time:.2f} сек.")

# ==========================================
# 7. РЕЗУЛЬТАТЫ
# ==========================================

# Анализ признаков кластеров
print("\n--- ТОП-ПРИЗНАКИ ПО КЛАСТЕРАМ ---")
print_top_ngrams(centroids, vectorizer.inv_vocab, top_n=5)

# График распределения
plot_clusters_distribution(labels)
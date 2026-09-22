import csv
import math
import re
import random
import time
from collections import defaultdict, Counter
import matplotlib.pyplot as plt

# ==========================================
# 1. ОЧИСТКА И ЛЕММАТИЗАЦИЯ
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
    word = word.lower()
    if len(word) <= 3:
        return word
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
    text = text.lower()
    words = re.findall(r'[а-яёa-z0-9]+', text)
    processed = []
    for w in words:
        if w in STOP_WORDS or len(w) < 2 or w.isdigit():
            continue
        stem = simple_stemmer_ru(w)
        processed.append(stem)
    return processed

def generate_ngrams(tokens: list[str], word_n_range=(1, 2), char_n=3) -> list[str]:
    ngrams = []
    # Словесные N-граммы
    for n in range(word_n_range[0], word_n_range[1] + 1):
        for i in range(len(tokens) - n + 1):
            ngrams.append("w:" + "_".join(tokens[i:i+n]))
    
    # Буквенные N-граммы
    for word in tokens:
        if len(word) >= char_n:
            for i in range(len(word) - char_n + 1):
                ngrams.append("c:" + word[i:i+char_n])
                
    return ngrams

# ==========================================
# 2. РАЗРЕЖЕННАЯ ВЕКТОРНАЯ МОДЕЛЬ (TF-IDF)
# ==========================================

class SparseVector(dict):
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

def mini_batch_kmeans(vectors: list[SparseVector], k: int, batch_size: int = 2000, max_iters: int = 20):
    """ Mini-Batch K-Means для фиксированного количества кластеров K. """
    centroids = [SparseVector(v) for v in random.sample(vectors, k)]
    counts = [0] * k

    for it in range(max_iters):
        batch = random.sample(vectors, min(batch_size, len(vectors)))
        for vec in batch:
            best_k = max(range(k), key=lambda idx: vec.dot(centroids[idx]))
            counts[best_k] += 1
            eta = 1.0 / counts[best_k]
            
            cent = centroids[best_k]
            for key, val in vec.items():
                cent[key] = (1 - eta) * cent.get(key, 0.0) + eta * val
            centroids[best_k] = SparseVector(cent)

    labels = [max(range(k), key=lambda idx: vec.dot(centroids[idx])) for vec in vectors]
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
            centroids.append(SparseVector(vec))
            labels.append(len(centroids) - 1)
            
    return labels, centroids

# ==========================================
# 4. ЗАГРУЗКА ДАННЫХ ИЗ CSV
# ==========================================

def load_csv_data(filepath: str, has_header: bool = True, encoding: str = 'utf-8', delimiter: str = ',') -> list[tuple]:
    """ Загрузка 3 столбцов из CSV файла. """
    dataset = []
    with open(filepath, mode='r', encoding=encoding, errors='ignore') as f:
        reader = csv.reader(f, delimiter=delimiter)
        if has_header:
            next(reader, None)  # пропуск заголовков
            
        for row in reader:
            if len(row) >= 3:
                # Берём первые 3 столбца
                c1, c2, c3 = row[0].strip(), row[1].strip(), row[2].strip()
                dataset.append((c1, c2, c3))
            elif len(row) > 0:
                # Если в какой-то строке меньше 3 столбцов, дозаполняем пустыми строками
                c1 = row[0].strip() if len(row) > 0 else ""
                c2 = row[1].strip() if len(row) > 1 else ""
                c3 = row[2].strip() if len(row) > 2 else ""
                dataset.append((c1, c2, c3))
    return dataset

# ==========================================
# 5. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ И ВИЗУАЛИЗАЦИЯ
# ==========================================

def save_clustered_csv(output_filepath: str, original_data: list[tuple], labels: list[int]):
    """ Сохраняет исходные 3 столбца с добавленным 4-м столбцом cluster_id. """
    with open(output_filepath, mode='w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["col1", "col2", "col3", "cluster_id"])
        for row, label in zip(original_data, labels):
            writer.writerow([row[0], row[1], row[2], label])
    print(f"Размеченные данные сохранены в файл: {output_filepath}")

def print_top_ngrams(centroids, inv_vocab, top_n=8):
    for idx, cent in enumerate(centroids):
        sorted_feats = sorted(cent.items(), key=lambda x: x[1], reverse=True)[:top_n]
        readable_feats = []
        for feat_id, score in sorted_feats:
            name = inv_vocab.get(feat_id, "")
            prefix = "[Слово]" if name.startswith("w:") else "[Буква]"
            clean_name = name[2:]
            readable_feats.append(f"{prefix} {clean_name} ({score:.3f})")
        print(f"\n--- Кластер {idx} ---")
        print(" | ".join(readable_feats))

def plot_clusters_distribution(labels):
    counter = Counter(labels)
    clusters = sorted(counter.keys())
    counts = [counter[c] for c in clusters]

    plt.figure(figsize=(10, 5))
    plt.bar([str(c) for c in clusters], counts, color='skyblue', edgecolor='black')
    plt.xlabel('ID Кластера')
    plt.ylabel('Количество записей')
    plt.title('Распределение обращений по кластерам')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

# ==========================================
# 6. ОСНОВНОЙ ЗАПУСК
# ==========================================

if __name__ == "__main__":
    # Укажите путь к вашему CSV-файлу
    CSV_PATH = "data.csv"
    OUTPUT_CSV_PATH = "data_clustered.csv"
    
    print(f"1. Загрузка данных из {CSV_PATH}...")
    # Измените delimiter на ';' если ваш файл использует точку с запятой в качестве разделителя
    raw_dataset = load_csv_data(CSV_PATH, has_header=True, encoding='utf-8', delimiter=',')
    print(f"Загружено записей: {len(raw_dataset)}")

    start_time = time.time()

    print("2. Предобработка текстов и генерация N-грамм...")
    docs_ngrams = []
    for col1, col2, col3 in raw_dataset:
        full_text = f"{col1} {col2} {col3}"
        tokens = preprocess_text(full_text)
        ngrams = generate_ngrams(tokens, word_n_range=(1, 2), char_n=3)
        docs_ngrams.append(ngrams)

    print("3. Векторизация TF-IDF...")
    vectorizer = FastTfidfVectorizer(min_df=3, max_features=5000)
    vectors = vectorizer.fit_transform(docs_ngrams)

    # Выбор режима кластеризации:
    
    # ВАРИАНТ А: С заданным числом кластеров K (например, K=7)
    K = 10
    print(f"4. Кластеризация Mini-Batch K-Means (K={K})...")
    labels, centroids = mini_batch_kmeans(vectors, k=K, batch_size=2000, max_iters=20)

    # ВАРИАНТ Б: С автоматическим определением числа кластеров (раскомментируйте, если нужно)
    # print("4. Кластеризация Leader (произвольное число кластеров)...")
    # labels, centroids = leader_clustering(vectors, threshold=0.35)

    print(f"Обработка завершена за {time.time() - start_time:.2f} сек.")

    # Вывод топ-признаков
    print("\n--- КЛЮЧЕВЫЕ ТЕМЫ КЛАСТЕРОВ ---")
    print_top_ngrams(centroids, vectorizer.inv_vocab, top_n=6)

    # Сохранение результатов в новый CSV
    save_clustered_csv(OUTPUT_CSV_PATH, raw_dataset, labels)

    # Построение графика
    plot_clusters_distribution(labels)
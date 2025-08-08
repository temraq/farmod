import json
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import re
from tqdm import tqdm

def clean_text(text):
    """Очистка медицинского текста"""
    text = re.sub(r'\s+', ' ', text)  # Удаление лишних пробелов
    text = re.sub(r'\[\d+\]', '', text)  # Удаление цитат [1]
    return text.strip()

def chunk_text(text, chunk_size=256, overlap=32):
    """Разбиение текста на перекрывающиеся чанки"""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = end - overlap
    return chunks

def create_faiss_index(dataset_name="pubmed_qa", index_path="data/processed/passages.faiss"):
    dataset = load_dataset(dataset_name, "pqa_labeled")
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    
    passages = []
    for split in ['train', 'validation']:
        for item in tqdm(dataset[split], desc=f"Processing {split}"):
            context = clean_text(" ".join(item['context']))
            passages.extend(chunk_text(context))
    
    # Генерация эмбеддингов
    embeddings = model.encode(passages, show_progress_bar=True, batch_size=64)
    
    # Создание FAISS индекса
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)  # Inner Product ≈ косинусное сходство
    faiss.normalize_L2(embeddings)  # Нормализация для косинусного сходства
    index.add(embeddings)
    
    # Сохранение
    faiss.write_index(index, index_path)
    
    # Сохранение метаданных
    with open("data/processed/metadata.jsonl", "w") as f:
        for passage in passages:
            f.write(json.dumps({"text": passage}) + "\n")
    
    print(f"Index created with {len(passages)} passages")

if __name__ == "__main__":
    create_faiss_index()
import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import RagRetriever, RagTokenizer, RagSequenceForGeneration

class PubMedRetriever:
    def __init__(self, index_path="data/processed/passages.faiss"):
        self.index = faiss.read_index(index_path)
        self.model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        
        # Загрузка метаданных
        self.passages = []
        with open("data/processed/metadata.jsonl", "r") as f:
            for line in f:
                self.passages.append(json.loads(line)['text'])
    
    def retrieve(self, query, top_k=5):
        query_embed = self.model.encode([query])
        faiss.normalize_L2(query_embed)
        
        distances, indices = self.index.search(query_embed, top_k)
        return [self.passages[i] for i in indices[0]]

class QARAGSystem:
    def __init__(self, adapter_path="models/adapters/pubmed_lora"):
        self.retriever = PubMedRetriever()
        self.tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")
        
        # Инициализация генератора с кастомным ретривером
        self.generator = RagSequenceForGeneration.from_pretrained(
            "facebook/rag-sequence-nq",
            retriever=None,  # Используем свой ретривер
            index_name="custom",
            passages_path="data/processed",
            index_path="data/processed/passages.faiss"
        )
        
        # Загрузка адаптеров
        self.generator.model = self.generator.model.from_pretrained(
            "meta-llama/Llama-2-7b-chat-hf",
            load_in_4bit=True,
            device_map="auto"
        )
        self.generator.model = PeftModel.from_pretrained(
            self.generator.model, 
            adapter_path
        )
    
    def generate_answer(self, question, use_rag=True, n_docs=3):
        if use_rag:
            # Ручной режим RAG
            context = " ".join(self.retriever.retrieve(question, top_k=n_docs))
            input_text = f"Ответь на вопрос используя контекст:\nКонтекст: {context}\nВопрос: {question}\nОтвет:"
        else:
            input_text = f"Ответь на медицинский вопрос:\nВопрос: {question}\nОтвет:"
        
        inputs = self.tokenizer(input_text, return_tensors="pt").to("cuda")
        outputs = self.generator.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=200,
            num_return_sequences=1
        )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

# Пример использования
# system = QARAGSystem()
# print(system.generate_answer("What is the first-line treatment for hypertension?"))
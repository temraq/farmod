from datasets import load_dataset
from evaluate import load
from src.retriever import QARAGSystem
import numpy as np

# Инициализация системы
system = QARAGSystem()

def evaluate_ragas(dataset, n_samples=50):
    """Оценка с помощью RAGAS метрик"""
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevance
    
    sample = dataset.shuffle().select(range(n_samples))
    
    questions = [item["question"] for item in sample]
    ground_truths = [item["long_answer"] for item in sample]
    
    # Генерация ответов
    answers = []
    for q in questions:
        answers.append(system.generate_answer(q, use_rag=True))
    
    # Создание датасета для оценки
    eval_dataset = {
        "question": questions,
        "answer": answers,
        "ground_truth": ground_truths
    }
    
    # Расчет метрик
    result = evaluate(
        eval_dataset,
        metrics=[faithfulness, answer_relevance]
    )
    
    return result

if __name__ == "__main__":
    # Загрузка данных
    dataset = load_dataset("pubmed_qa", "pqa_labeled", split="validation")
    
    # Оценка
    rag_results = evaluate_ragas(dataset)
    print("RAGAS Metrics:", rag_results)
    
    # Perplexity (требуется реализация)
    # perplexity = calculate_perplexity(model, tokenizer, dataset)
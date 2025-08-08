# PubMed QA Assistant with Llama-2 and RAG

Проект медицинского ассистента на основе Llama-2-7b с дообучением (QLoRA) и RAG для ответов на вопросы из PubMed.

## Особенности
- 4-битная квантизация модели с bitsandbytes
- Дообучение с LoRA/QLoRA на медицинских данных
- Интеграция FAISS для эффективного поиска
- FastAPI для REST API и Gradio для веб-интерфейса
- Оценка качества с RAGAS метриками

## Требования
- Python 3.10+
- NVIDIA GPU (рекомендуется) или CPU
- Доступ к моделям Llama-2 через Hugging Face

## Установка
1. Клонировать репозиторий:
```bash
git clone https://github.com/yourusername/pubmed-qa-rag.git
cd pubmed-qa-rag
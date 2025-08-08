# Используем официальный образ Python
FROM python:3.10-slim-bullseye

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Установка CUDA (опционально, для GPU-ускорения)
# ENV CUDA_HOME /usr/local/cuda
# RUN curl -L https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin > /etc/apt/preferences.d/cuda-repository-pin-600 && \
#     apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub && \
#     add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /" && \
#     apt-get update && \
#     apt-get install -y cuda-toolkit-11-8

# Рабочая директория
WORKDIR /app

# Копируем зависимости
COPY requirements.txt .

# Устанавливаем Python-зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем исходный код
COPY . .

# Скачиваем модели (требуется предварительная авторизация HF)
# Для этого нужно передать токен при сборке: 
# docker build --build-arg HF_TOKEN=your_token -t pubmed-qa .
ARG HF_TOKEN
RUN huggingface-cli login --token $HF_TOKEN && \
    python -c "from transformers import AutoModel; AutoModel.from_pretrained('meta-llama/Llama-2-7b-chat-hf')" && \
    huggingface-cli logout

# Открываем порты для FastAPI и Gradio
EXPOSE 8000 7860

# Команда запуска по умолчанию
CMD ["python", "src/inference_server.py"]
FROM python:3.10-slim

# Установка системных зависимостей
RUN apt-get update && \
    apt-get install -y \
    git \
    ffmpeg \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Загружаем NLTK данные
RUN python -c "import nltk; nltk.download('stopwords')"

COPY . .

RUN mkdir -p /app/data
RUN mkdir -p /app/chroma_db
RUN mkdir -p /app/vector_cache


CMD ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]

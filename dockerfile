FROM python:3.9.18-slim-bookworm 

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

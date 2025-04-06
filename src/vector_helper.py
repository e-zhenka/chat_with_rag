from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple
import pickle
import os
from nltk.corpus import stopwords
import nltk


class VectorHelper:
    """
    Класс для поиска чанков с помощью TF-IDF
    """
    def __init__(self, cache_dir: str = "vector_cache"):
        # Загружаем стоп-слова для обоих языков
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        
        stop_words = list(set(
            stopwords.words('russian') + 
            stopwords.words('english')
        ))
        
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words=stop_words,
            ngram_range=(1, 2),
            # Добавляем параметры для лучшей обработки обоих языков
            token_pattern=r'(?u)\b\w+\b',  # Поддержка Unicode для русских букв
            max_features=10000  # Ограничиваем словарь для производительности
        )
        self.cache_dir = cache_dir
        self.vectors = None
        self.documents = None
        
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
    
    def preprocess_text(self, text: str) -> str:
        """Предобработка текста для обоих языков"""
        # Приводим к нижнему регистру
        text = text.lower()
        # Удаляем специальные символы, оставляя буквы и цифры
        text = ' '.join(word for word in text.split() 
                       if any(c.isalnum() for c in word))
        return text
    
    def fit_documents(self, documents: List[str]) -> None:
        """Векторизация документов с предобработкой"""
        cache_path = os.path.join(self.cache_dir, "tfidf_cache.pkl")
        
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)
                self.vectorizer = cached_data['vectorizer']
                self.vectors = cached_data['vectors']
                self.documents = cached_data['documents']
        else:
            # Предобработка документов
            processed_docs = [self.preprocess_text(doc) for doc in documents]
            self.documents = documents  # Сохраняем оригинальные документы
            self.vectors = self.vectorizer.fit_transform(processed_docs)
            
            with open(cache_path, 'wb') as f:
                pickle.dump({
                    'vectorizer': self.vectorizer,
                    'vectors': self.vectors,
                    'documents': documents
                }, f)
    
    def find_similar(self, query: str, top_k: int = 6) -> List[Tuple[int, float, str]]:
        """Поиск похожих документов с возвратом оригинального текста"""
        if self.vectors is None:
            raise ValueError("Необходимо сначала выполнить fit_documents")
        
        # Предобработка запроса
        processed_query = self.preprocess_text(query)
        query_vector = self.vectorizer.transform([processed_query])
        
        # Вычисляем косинусную близость
        similarities = cosine_similarity(query_vector, self.vectors).flatten()
        
        # Получаем top_k наиболее похожих документов
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        # Возвращаем тройки (индекс, значение близости, оригинальный текст)
        return [(idx, similarities[idx], self.documents[idx]) 
                for idx in top_indices]

    def get_vector_similarity(self, query: str, document: str) -> float:
        """Вычисление косинусной близости между запросом и документом"""
        processed_query = self.preprocess_text(query)
        processed_doc = self.preprocess_text(document)
        
        # Векторизуем оба текста
        vectors = self.vectorizer.transform([processed_query, processed_doc])
        
        # Вычисляем косинусную близость
        similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
        return similarity 
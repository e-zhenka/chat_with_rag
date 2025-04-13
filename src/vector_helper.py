from rank_bm25 import BM25Okapi
from typing import List, Tuple
import pickle
import os
from nltk.corpus import stopwords
import nltk


class VectorHelper:
    """
    Класс для поиска чанков с помощью BM25
    """
    def __init__(self, cache_dir: str = "vector_cache"):
        # Загружаем стоп-слова для обоих языков
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        
        self.stop_words = list(set(
            stopwords.words('russian') + 
            stopwords.words('english')
        ))
        
        self.cache_dir = cache_dir
        self.bm25 = None
        self.documents = None
        self.tokenized_corpus = None
        
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
    
    def preprocess_text(self, text: str) -> List[str]:
        """Предобработка текста для обоих языков"""
        # Приводим к нижнему регистру
        text = text.lower()
        # Токенизация и удаление стоп-слов
        tokens = [word for word in text.split() 
                 if any(c.isalnum() for c in word) and word not in self.stop_words]
        return tokens
    
    def fit_documents(self, documents: List[str]) -> None:
        """Индексация документов с помощью BM25"""
        cache_path = os.path.join(self.cache_dir, "bm25_cache.pkl")
        
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)
                self.bm25 = cached_data['bm25']
                self.documents = cached_data['documents']
                self.tokenized_corpus = cached_data['tokenized_corpus']
        else:
            # Предобработка и токенизация документов
            self.documents = documents
            self.tokenized_corpus = [self.preprocess_text(doc) for doc in documents]
            self.bm25 = BM25Okapi(self.tokenized_corpus)
            
            with open(cache_path, 'wb') as f:
                pickle.dump({
                    'bm25': self.bm25,
                    'documents': documents,
                    'tokenized_corpus': self.tokenized_corpus
                }, f)
    
    def find_similar(self, query: str, top_k: int = 6) -> List[Tuple[int, float, str]]:
        """Поиск похожих документов с возвратом оригинального текста"""
        if self.bm25 is None:
            raise ValueError("Необходимо сначала выполнить fit_documents")
        
        # Предобработка запроса
        tokenized_query = self.preprocess_text(query)
        
        # Получаем BM25 scores
        scores = self.bm25.get_scores(tokenized_query)
        
        # Получаем top_k наиболее похожих документов
        top_indices = scores.argsort()[-top_k:][::-1]
        
        # Возвращаем тройки (индекс, значение близости, оригинальный текст)
        return [(idx, scores[idx], self.documents[idx]) 
                for idx in top_indices]

    def get_vector_similarity(self, query: str, document: str) -> float:
        """Вычисление близости между запросом и документом с помощью BM25"""
        tokenized_query = self.preprocess_text(query)
        tokenized_doc = self.preprocess_text(document)
        
        # Создаем временный BM25 объект для одного документа
        temp_bm25 = BM25Okapi([tokenized_doc])
        
        # Вычисляем score
        similarity = temp_bm25.get_scores(tokenized_query)[0]
        
        # Нормализуем score в диапазон [0, 1]
        max_possible_score = len(tokenized_query)  # Максимально возможный score
        normalized_similarity = similarity / max_possible_score if max_possible_score > 0 else 0
        
        return normalized_similarity

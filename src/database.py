import chromadb
from chromadb.utils import embedding_functions
import uuid
import os
from vector_helper import VectorHelper
from typing import List, Dict
from custom_embeddings import ONNXEmbedder

class HybridDB:
    def __init__(self, data_dir: str = "data"):
        self.embedding_func = ONNXEmbedder().encode

class HybridDB:
    def __init__(self, data_dir: str = "data"):
        self.chroma_client = chromadb.PersistentClient(path="chroma_db")
        self.embedding_func = ONNXEmbedder().encode
        self.vector_helper = VectorHelper()
        self.data_dir = data_dir
        self.collection = self._init_collection()
        self._load_documents()
    
    def _init_collection(self):
        return self.chroma_client.get_or_create_collection(
            name="knowledge_base",
            embedding_function=self.embedding_func
        )
    
    def _load_documents(self):
        # Сначала соберем все документы
        all_documents = []
        all_metadatas = []
        
        files = [os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) if f.endswith('.txt')]
        
        for file_path in files:
            doc_type = os.path.basename(file_path).split('.')[0]
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                chunks = [chunk.strip() for chunk in text.split('\n\n') if chunk.strip()]
                all_documents.extend(chunks)
                all_metadatas.extend([{"type": doc_type} for _ in chunks])
        
        # Если есть документы и коллекция пуста, загружаем их в ChromaDB
        if all_documents and self.collection.count() == 0:
            ids = [str(uuid.uuid4()) for _ in all_documents]
            self.collection.add(
                documents=all_documents,
                metadatas=all_metadatas,
                ids=ids
            )
        
        # В любом случае обучаем vector_helper на всех документах
        if all_documents:  # Добавляем эту проверку
            self.vector_helper.fit_documents(all_documents)
        else:
            print("Предупреждение: Не найдено документов для загрузки")

    def query(self, query_text: str, doc_type: str = None, n_results: int = 5) -> Dict:
        """
        Комбинированный поиск:
        - ChromaDB: поиск по конкретной категории (n_results документов)
        - TF-IDF: поиск по всем документам (n_results документов)
        
        Returns:
            Dict с двумя списками результатов:
            {
                'chroma_results': [...],
                'tfidf_results': [...]
            }
        """
        try:
            # Получаем результаты из ChromaDB
            chroma_results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results,
                where={"type": doc_type} if doc_type else None
            )
            
            # Получаем TF-IDF результаты
            try:
                tfidf_results = self.vector_helper.find_similar(query_text, top_k=n_results)
            except ValueError as e:
                print(f"Ошибка TF-IDF поиска: {str(e)}")
                tfidf_results = []
            
            # Форматируем результаты ChromaDB
            formatted_chroma = []
            if chroma_results['documents'][0]:
                max_semantic = max(chroma_results['distances'][0]) if chroma_results['distances'][0] else 1.0
                for doc, distance in zip(chroma_results['documents'][0], chroma_results['distances'][0]):
                    formatted_chroma.append({
                        'document': doc,
                        'score': distance / max_semantic,
                        'source': 'chroma_db'
                    })
            
            # Форматируем результаты TF-IDF
            formatted_tfidf = []
            if tfidf_results:
                max_tfidf = max(score for _, score, _ in tfidf_results)
                for _, score, doc in tfidf_results:
                    formatted_tfidf.append({
                        'document': doc,
                        'score': score / max_tfidf if max_tfidf > 0 else 0,
                        'source': 'tfidf'
                    })
            
            return {
                'chroma_results': formatted_chroma,
                'tfidf_results': formatted_tfidf
            }
            
        except Exception as e:
            print(f"Ошибка при выполнении поиска: {str(e)}")
            return {'chroma_results': [], 'tfidf_results': []}

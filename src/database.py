import chromadb
import uuid
import os
from vector_helper import VectorHelper
from typing import Dict
from config import Settings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from custom_embeddings import ONNXEmbedder

settings = Settings.from_yaml("config.yaml")


class HybridDB:
    """
    Класс для гибридного поиска чанков с использованием ChromaDB и BM25
    """
    def __init__(self, data_dir: str = "data"):
        self.chroma_client = chromadb.PersistentClient(path="chroma_db")
        self.embedding_func = ONNXEmbedder().encode
        self.vector_helper = VectorHelper()
        self.data_dir = data_dir
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ".", " ", ""]
        )

        self.collection = self._init_collection()
        self._load_documents()

    def _init_collection(self) -> chromadb.Collection:
        """
        Функция для инициализации базы данных Chroma DB.
        :return: Collection - Новая коллекция
        """
        return self.chroma_client.get_or_create_collection(
            name="knowledge_base",
            embedding_function=self.embedding_func
        )

    def _load_documents(self) -> None:
        """
        Функция для загрузки чанков в Chroma DB
        :return: None
        """
        # Сначала соберем все документы
        all_documents = []
        all_metadatas = []

        files = [os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) if f.endswith('.txt')]

        for file_path in files:
            doc_type = os.path.basename(file_path).split('.')[0]

            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()

                if doc_type in settings.finance_documents:
                    chunks = self.text_splitter.split_text(text)
                else:
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
        Функция для получения контекста для RAG. Контекст получается с помощью комбинированного поиска.

        Комбинированный поиск:
        - ChromaDB: поиск по конкретной категории (n_results документов)
        - BM25: поиск по всем документам (n_results документов)

        :param query_text: str  - запрос к базе данных
        :param doc_type: str    - документ, из которого будет браться информация
        :param n_results: int   - количество документов которое будет возвращено из каждого типа поиска

        Returns:
            Dict с двумя списками результатов:
            {
                'chroma_results': [...],
                'bm25_results': [...]
            }
        """
        try:
            # Получаем результаты из ChromaDB
            chroma_results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results,
                where={"type": doc_type} if doc_type else None
            )

            # Получаем BM25 результаты
            try:
                bm25_results = self.vector_helper.find_similar(query_text, top_k=n_results)
            except ValueError as e:
                print(f"Ошибка BM25 поиска: {str(e)}")
                bm25_results = []

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

            # Форматируем результаты BM25
            formatted_bm25 = []
            if bm25_results:
                max_bm25 = max(score for _, score, _ in bm25_results)
                for _, score, doc in bm25_results:
                    formatted_bm25.append({
                        'document': doc,
                        'score': score / max_bm25 if max_bm25 > 0 else 0,
                        'source': 'bm25'
                    })

            return {
                'chroma_results': formatted_chroma,
                'bm25_results': formatted_bm25
            }

        except Exception as e:
            print(f"Ошибка при выполнении поиска: {str(e)}")
            return {'chroma_results': [], 'bm25_results': []}

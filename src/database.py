import chromadb
import uuid
import os
from vector_helper import VectorHelper
from typing import Dict
from config import Settings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from custom_embeddings import ONNXEmbedder
import re
import pickle

settings = Settings.from_yaml("config.yaml")


class HybridDB:
    """
    Класс для гибридного поиска чанков
    """
    def __init__(self, data_dir: str = "data"):
        self.chroma_client = chromadb.PersistentClient(path="chroma_db")
        self.embedding_func = ONNXEmbedder().encode
        self.vector_helper = VectorHelper()
        self.data_dir = data_dir
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50,
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

        files = [os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) if f.endswith('.txt')]

        print('Индексирование документов')
        add_documents = False
        if self.collection.count() == 0:
            add_documents = True

        all_documents = []
        for file_path in files:
            print(file_path)
            doc_type = os.path.basename(file_path).split('.')[0]

            with open(file_path, 'r', encoding='utf-8') as file:
                all_metadatas = []

                text = file.read()
                pages = []

                if doc_type in settings.finance_documents:
                    chunks = self.text_splitter.split_text(text)
                    current_page = 1
                    if doc_type in settings.finance_documents:
                        for current_chunk in chunks:
                            page_pattern = re.compile(r'Page_ (\d+)')
                            page = re.findall(page_pattern, current_chunk)
                            if page:
                                current_page = int(page[0]) + 1
                            pages.append(current_page)

                else:
                    chunks = [chunk.strip() for chunk in text.split('\n\n') if chunk.strip()]
                all_documents.extend(chunks)

                if add_documents:
                    all_metadatas.extend([{"type": doc_type} for _ in chunks])

                    ids = [str(uuid.uuid4()) for _ in chunks]
                    self.collection.add(
                        documents=chunks,
                        metadatas=all_metadatas,
                        ids=ids
                    )
                    if len(pages) > 0:
                        with open(f'data/{doc_type}_pages.pickle', 'wb') as f:
                            pickle.dump({id_: page for id_, page in zip(ids, pages)}, f)

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
        - TF-IDF: поиск по всем документам (n_results документов)

        :param query_text: str  - запрос к базе данных
        :param doc_type: str    - документ, из которого будет браться информация
        :param n_results: int   - количество документов которое будет возвращено из каждого типа поиска

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

            pages = []
            if doc_type in settings.finance_documents:
                with open(f'data/{doc_type}_pages.pickle', 'rb') as f:
                    all_pages = pickle.load(f)
                for id_ in chroma_results['ids'][0]:
                    pages.append(all_pages[id_])

            # Форматируем результаты ChromaDB
            formatted_chroma = []
            pages = set(pages)
            for page in pages:
                formatted_chroma.append({
                    'document': self.get_page_text(f'data/{doc_type}.txt', page),
                    'score': 1,
                    'source': 'chroma_db'
                })

            if chroma_results['documents'][0] and not formatted_chroma:
                max_semantic = max(chroma_results['distances'][0]) if chroma_results['distances'][0] else 1.0
                for doc, distance in zip(chroma_results['documents'][0], chroma_results['distances'][0]):
                    formatted_chroma.append({
                        'document': doc,
                        'score': distance / max_semantic,
                        'source': 'chroma_db'
                    })

            # Получаем TF-IDF результаты
            try:
                tfidf_results = self.vector_helper.find_similar(query_text, top_k=n_results)
            except ValueError as e:
                print(f"Ошибка TF-IDF поиска: {str(e)}")
                tfidf_results = []

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

    @staticmethod
    def get_page_text(file_path, page_number):
        """
        Возвращает текст с указанной страницы из текстового файла.

        :param file_path: путь к текстовому файлу
        :param page_number: номер страницы (начинается с 1)
        """
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        # Разделяем содержимое по меткам страниц
        pages = content.split('Page_ ')

        # Возвращаем текст страницы (удаляем лишние пробелы и переносы строк)
        page_text = pages[page_number-1].strip()
        # Убираем номер следующей страницы, если он есть
        if page_number < len(pages) - 1:
            page_text = page_text.split('Page_ ')[0].strip()

        return page_text

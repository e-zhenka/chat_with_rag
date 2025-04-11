from openai import OpenAI
from typing import Dict, List
from langchain_openai import ChatOpenAI
import os
import json
from pydantic import BaseModel
from enum import Enum
from config import Settings

settings = Settings.from_yaml("config.yaml")


class QueryType(str, Enum):
    """
    Класс, описывающий все типы документов для structured output
    """
    bulldog = 'bulldog'
    red_mad_robot = 'red_mad_robot'
    world_class = 'world_class'
    michelin = 'michelin'
    telegram_bots = 'telegram_bots'
    bridges_pipes = 'bridges_pipes'
    labour = 'labour'
    payments_taxes = 'payments_taxes'
    education = 'education'
    gosuslugi = 'gosuslugi'
    consumer_basket = 'consumer_basket'
    cosmetics = 'cosmetics'
    brave_bison = 'brave_bison'
    rectifier_technologies = 'rectifier_technologies'
    starvest_plc = 'starvest_plc'
    chat = 'chat'


class CarDescription(BaseModel):
    """
    Класс для описания структуры ответа LLM

    query_type - Тип документа, по теме которого задан вопрос
    search_query - Перефразированный вопрос для лучшего поиска в базе данных
    """
    query_type: QueryType
    search_query: str


class LLMHelper:
    """
    Класс для работы с LLM
    """
    def __init__(self, base_url: str, model: str = settings.model_name, api_key: str = None):
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
        )

        self.model = model
        self.json_schema = CarDescription.model_json_schema()
        self.llm_model = ChatOpenAI(
            base_url=base_url,
            model=model,
            openai_api_key=api_key or os.getenv("OPENAI_API_KEY")
        )

    def analyze_query(self, query: str, previous_messages: List[str] = None) -> Dict:
        """
        Функция для анализа запрос и возвращения тип документа и перефразированного вопроса

        :param query: str   - запрос пользователя.
        :param previous_messages: List[str] - история сообщений
        :return: dict   - ответ LLM, содержащий поля:
            query_type - тип запроса (документ в котором есть ответ)
            search_query - перефразированный запрос для поиска
        """
        context = ""
        if previous_messages:
            context = "Предыдущие сообщения:\n" + "\n".join(previous_messages) + "\n\n"

        prompt = f"""
        {context}
        Проанализируй текущий вопрос пользователя и определи:
        1. К какой категории относится вопрос
        2. Сформулируй поисковый запрос, который наилучшим образом описывает суть вопроса

        Категории:
        bulldog - вопросы про французского бульдога
        red_mad_robot - вопросы про компанию red_mad_robots
        world_class - вопросы про фитнес-клубы World Class и тренировки
        michelin - вопросы про Michelin, Мишлен
        telegram - вопросы про Telegram
        bridges_pipes - вопросы про мосты, трубы, трубопроводы и их конструкции
        labour - вопросы про трудовой кодекс и трудовые отношения
        payments_taxes - вопросы про платежи, налоги, финансы и тарифы
        education - вопросы про высшее образование
        gosuslugi - вопросы про платформу Госуслуги и защите от мошенников
        consumer_basket - вопросы про потребительскую корзину 
        cosmetics - вопросы про уход за кожей лица
        brave_bison - вопросы про маркетинговую компанию Brave Bison
        rectifier_technologies - вопросы про промышленную компанию Rectifier Technologies
        starvest_plc - вопросы про инвестиционную компанию Starvest Plc 
        chat - если вопрос не относится ни к одной из категорий

        Текущий вопрос: {query}

        Верни ТОЛЬКО JSON в формате:
        {{"type": "название_категории", "search_query": "перефразированный поисковый запрос"}}
        """

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            extra_body={"guided_json": self.json_schema},
            temperature=0,
            max_tokens=1024,
        )
        res = json.loads(completion.choices[0].message.content)
        return res

    def rerank_context(self, query: str, search_query: str, contexts: List[str], top_k: int = 3) -> List[str]:
        """
        Функция для переранжирования контекста с помощью LLM

        :param query: str - Оригинальный запрос пользователя
        :param search_query: str - Переформулированный поисковый запрос
        :param contexts: List[str] - Список контекстов для ранжирования
        :param top_k: int - Количество лучших контекстов для возврата
        :return: List[str] - Отсортированный список наиболее релевантных контекстов
        """
        if not contexts:
            return []

        # Создаем строку с контекстами без использования chr(10)
        contexts_str = ""
        for i, text in enumerate(contexts):
            contexts_str += f"Контекст {i+1}:\n{text}\n\n"

        prompt = f"""
        Оцени релевантность каждого контекста для ответа на вопрос пользователя.
        Верни JSON-массив с оценками в формате:
        [
            {{"index": номер_контекста, "score": оценка_от_0_до_10, "explanation": "краткое_объяснение"}}
        ]
        
        Оценка должна учитывать:
        1. Насколько прямо контекст отвечает на оригинальный вопрос (8-10 баллов)
        2. Насколько контекст соответствует переформулированному поисковому запросу (6-8 баллов)
        3. Содержит ли частичный ответ на вопрос или поисковый запрос (4-6 баллов)
        4. Содержит только косвенную информацию (2-4 балла)
        5. Не относится к вопросу или поисковому запросу (0-1 балл)

        Оригинальный вопрос пользователя: {query}
        Переформулированный поисковый запрос: {search_query}

        Контексты для оценки:
        {contexts_str}
        
        При оценке:
        - Если контекст хорошо отвечает и на оригинальный вопрос, и на поисковый запрос - ставь максимальный балл
        - Если контекст отвечает только на одну из формулировок - используй среднюю оценку
        - Учитывай, что поисковый запрос может содержать дополнительные ключевые слова или синонимы
        """

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=1024,
        )
        
        try:
            rankings = json.loads(completion.choices[0].message.content)
            # Сортируем по оценке в убывающем порядке
            sorted_rankings = sorted(rankings, key=lambda x: x['score'], reverse=True)
            # Возвращаем top_k лучших контекстов
            return [contexts[r['index']-1] for r in sorted_rankings[:top_k]]
        except Exception as e:
            print(f"Ошибка при ранжировании контекста: {str(e)}")
            return contexts[:top_k]  # В случае ошибки возвращаем первые top_k контекстов

    def generate_answer(self, query: str, search_query: str, context: List[str]) -> str:
        """
        Функция для генерации ответа пользователю, используя контекст

        :param query: str - Запрос пользователя
        :param search_query: str - Перефразированный запрос
        :param context: List[str] - Контекст, полученный из БД
        :return: str - Ответ для пользователя
        """
        # Сначала переранжируем контекст с учетом обоих запросов
        reranked_context = self.rerank_context(query, search_query, context)
        
        # Используем только отранжированный контекст для генерации ответа
        context_str = "\n\n".join([f"Контекст {i+1}:\n{text}" for i, text in enumerate(reranked_context)])
        
        prompt = f"""
        Используй предоставленный контекст, чтобы ответить на вопрос пользователя.
        Контекст уже отсортирован по релевантности - первые контексты наиболее важны.
        Если ответа нет в контексте, скажи, что не знаешь ответа.
        
        Оригинальный вопрос: {query}
        Поисковый запрос: {search_query}
        
        Контекст: {context_str}
        
        Ответ:
        """
        
        return self.llm_model.invoke(prompt).content
    
    def generate_chat_answer(self, query: str) -> str:
        """Генерация ответа для общих вопросов без контекста"""
        prompt = f"""
        Ты дружелюбный помощник. Ответь на вопрос пользователя естественным образом.
        Вопрос: {query}
        Ответ:
        """
        return self.llm_model.invoke(prompt).content

from openai import OpenAI
from typing import Dict, List
from langchain_openai import ChatOpenAI
import os
import json
from pydantic import BaseModel
from enum import Enum
from config import Settings

settings = Settings.from_yaml("config.yaml")

class RerankResult(BaseModel):
    reasoning: str 
    relevance_score: float

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

    
    def rerank_context(self, search_query: str, contexts: List[str]) -> List[Dict]:  # Исправлено! Теперь принимает 2 аргумента (плюс self)
        """
        Ранжирует контексты по релевантности вопросу
        Возвращает список словарей с оценками и объяснениями
        """
        rerank_schema = RerankResult.model_json_schema()
        
        results = []
        for i, context in enumerate(contexts):
            prompt = f"""
            Оцени релевантность данного контекста вопросу пользователя.
            Вопрос: {search_query}  # Используем search_query вместо query
            Контекст: {context}
            
            Верни JSON с:
            1. reasoning - краткое объяснение оценки
            2. relevance_score - оценка релевантности от 0 до 1
            """
            
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                extra_body={"guided_json": rerank_schema},
                temperature=0,
            )
            
            res = json.loads(completion.choices[0].message.content)
            results.append({
                "index": i+1,
                "score": res['relevance_score'],
                "explanation": res['reasoning'],
                "context": context
            })
        
        return sorted(results, key=lambda x: x['score'], reverse=True)
    

    def generate_answer(self, query: str, search_query: str, context: List[str]) -> str:
        """
        Функция для генерации ответа пользователю, используя контекст
        """
        # Получаем результаты реранкинга
        reranking_results = self.rerank_context(search_query, context)
        
        # Извлекаем отсортированные контексты
        sorted_contexts = [res['context'] for res in reranking_results]
        
        # Формируем строку контекста
        context_str = "\n\n".join([f"Контекст {i+1}:\n{text}" for i, text in enumerate(sorted_contexts)])
        
        prompt = f"""
        Используй предоставленный контекст, чтобы ответить на вопрос пользователя.
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

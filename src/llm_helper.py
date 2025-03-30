from langchain_openai import ChatOpenAI
from typing import Dict, List
import os

class LLMHelper:
    def __init__(self, base_url: str, model: str = "llama-3-8b-instruct-8k", api_key: str = None):
        self.model = ChatOpenAI(
            base_url=base_url,
            model=model,
            openai_api_key=api_key or os.getenv("OPENAI_API_KEY")
        )
    
    def analyze_query(self, query: str, previous_messages: List[str] = None) -> Dict:
        """Анализирует запрос и возвращает тип документа и перефразированный вопрос"""
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
        telegram_bots - вопросы про Telegram, ботов и их разработку
        bridges_pipes - вопросы про мосты, трубы, трубопроводы и их конструкции
        labour - вопросы про трудовой кодекс и трудовые отношения
        payments_taxes - вопросы про платежи, налоги, финансы и тарифы
        education - вопросы про высшее образование
        gosuslugi - вопросы про платформу Госуслуги и защите от мошенников
        consumer_basket - вопросы про потребительскую корзину 
        cosmetics - вопросы про уход за кожей лица
        chat - если вопрос не относится ни к одной из категорий

        Текущий вопрос: {query}

        Верни ТОЛЬКО JSON в формате:
        {{"type": "название_категории", "search_query": "перефразированный поисковый запрос"}}
        """
        
        try:
            response = self.model.invoke(prompt)
            content = response.content
            
            # Извлекаем JSON из ответа
            start = content.find('{')
            end = content.rfind('}') + 1
            
            if start != -1 and end != 0:
                json_str = content[start:end]
                try:
                    result = eval(json_str)
                    if isinstance(result, dict) and "type" in result and "search_query" in result:
                        valid_types = {
                            "bulldog", "red_mad_robot", "world_class", "michelin",
                            "telegram_bots", "bridges_pipes", "labour", "payments_taxes",
                            "education", "gosuslugi", "consumer_basket", "cosmetics", "chat"
                        }
                        if result["type"] in valid_types:
                            return result
                except:
                    pass
            
            return {"type": "chat", "search_query": query}
            
        except Exception as e:
            print(f"Error in analyze_query: {str(e)}")
            return {"type": "chat", "search_query": query}
    
    def generate_answer(self, query: str, search_query: str, context: List[str]) -> str:
        context_str = "\n\n".join([f"Контекст {i+1}:\n{text}" for i, text in enumerate(context)])
        
        prompt = f"""
        Используй предоставленный контекст, чтобы ответить на вопрос пользователя.
        Если ответа нет в контексте, скажи, что не знаешь ответа.
        
        Оригинальный вопрос: {query}
        Поисковый запрос: {search_query}
        
        Контекст: {context_str}
        
        Ответ:
        """
        
        return self.model.invoke(prompt).content
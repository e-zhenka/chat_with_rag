from typing import TypedDict
from langgraph.graph import StateGraph
from llm_helper import LLMHelper
from config import Settings
from database import HybridDB
from moex import get_stock_info
import requests

settings = Settings.from_yaml("config.yaml")


class State(TypedDict):
    input: str
    output: str
    category: str
    search_query: str
    image_path: str
    previous_messages: list
    chroma_results: list
    tfidf_results: list
    reranking_context: list


class Agent:
    """
    Класс, реализующий работу ai-агента
    Реализованы такие функции, как
        - RAG на основании данных файлов
        - Простой чат с пользователем
        - Получение погоды
        - Получение котировок акций мосбиржи
    """
    def __init__(self, api_key):
        self.graph = StateGraph(State)
        self.llm = LLMHelper(settings.url, api_key=api_key)
        self.db = HybridDB()

        # Добавление узлов
        self.graph.add_node("router", self.router)
        self.graph.add_node("rag", self.rag)
        self.graph.add_node("weather", self.weather)
        self.graph.add_node("chat", self.chat)
        self.graph.add_node("stocks", self.stocks)

        # Определение переходов
        self.graph.add_conditional_edges("router", self.condition)

        # Установка начального узла
        self.graph.set_entry_point("router")

        # Компиляция графа
        self.compiled_graph = self.graph.compile()

    def router(self, state: State) -> State:
        """
        Функция для получения категории текста и поискового запроса (роутер)
        :param state: State - Текущее состояние
        :return: State
        """
        router_result = self.llm.analyze_query(
            query=state['input'],
            previous_messages=state['previous_messages']
        )

        state['search_query'] = router_result['search_query']
        state['category'] = router_result['query_type']

        return state

    @staticmethod
    def get_pretty_wind(wind_speed):
        """
        Получаем классификацию ветра по скорости
        :param wind_speed: float - скорость ветра
        :return: str
        """
        if wind_speed < 1.5:
            description = "Штиль"
        elif wind_speed < 5:
            description = "Слабый"
        elif wind_speed < 10:
            description = "Умеренный"
        elif wind_speed < 20:
            description = "Сильный"
        else:
            description = "Шторм"
        return f"{description} - {wind_speed} м/с"

    def weather(self, state: State) -> State:
        """
        Функция для выдачи пользователю текущей погоды
        :param state: State - Текущее состояние
        :return: State - Состояние с ответом пользователю
        """
        city = self.llm.get_city(state['search_query'])

        current_weather = []

        if city == 'None':
            current_weather.append("В запросе отсутствует город, для которого надо выдать погоду.")
        else:
            api_key = settings.weather_api_key
            url = f"{settings.weather_url}?q={city}&appid={api_key}&units=metric"
            response = requests.get(url)
            data = response.json()

            if data["cod"] == 200:
                current_weather.append(f'Погода: {settings.clouds[data["weather"][0]["main"]]} ')
                current_weather.append(f'Текущая температура: {int(data["main"]["temp_min"])}-{int(data["main"]["temp_max"])} C')
                current_weather.append(f'Ощущается как: {int(data["main"]["feels_like"])} C')
                current_weather.append(f'Давление: {data["main"]["pressure"]} hPa')
                current_weather.append(f'Ветер: {self.get_pretty_wind(data["wind"]["speed"])}')
            else:
                current_weather.append("Город из запроса не найден. Пожалуйста, укажите настоящий город")

        output = self.llm.generate_answer(
            query=state["input"],
            search_query=state["search_query"],
            context=current_weather,
        )
        state["output"] = output

        return state

    def stocks(self, state: State) -> State:
        """
        Функция для выдачи пользователю текущей цены акции + отрисовка графика
        :param state: State - Текущее состояние
        :return: State - Состояние с ответом пользователю
        """
        stock = self.llm.get_stock(state['search_query'])

        stock_info = []

        if stock == 'none':
            stock_info.append("В запросе отсутствует город, для которого надо выдать погоду.")
        else:
            info = get_stock_info(stock)
            stock_info += info
            state['image_path'] = settings.img_path

        output = self.llm.generate_answer(
            query=state["input"],
            search_query=state["search_query"],
            context=stock_info,
        )
        state["output"] = output

        return state

    def chat(self, state: State) -> State:
        """
        Функция для простого общения с пользователем.
        :param state: State - Текущее состояние
        :return: State - Состояние с ответом пользователю
        """
        state["output"] = self.llm.generate_chat_answer(state["input"])
        return state

    def rag(self, state: State) -> State:
        """
        Функция для получения информации, используя имеющиеся документы.
        :param state: State - Текущее состояние
        :return: State - Состояние с ответом пользователю
        """
        n_results = 4

        if state["category"] in settings.finance_documents:
            # Если запрос по финансовой теме, то увеличиваем контекст
            n_results = 6

        context = self.db.query(
            query_text=state["search_query"],
            doc_type=state["category"],
            n_results=n_results,
        )
        state["chroma_results"] = context["chroma_results"]
        state["tfidf_results"] = context["tfidf_results"]

        all_contexts = set()

        for r in context['chroma_results'] + context['tfidf_results']:
            all_contexts.add(r['document'])

        reranking_results = self.llm.rerank_context(
            state["search_query"],
            list(all_contexts)
        )
        sorted_contexts = [res['context'] for res in reranking_results]

        state['reranking_context'] = reranking_results

        answer = self.llm.generate_answer(
            query=state['input'],
            search_query=state['search_query'],
            context=sorted_contexts
        )
        state['output'] = answer

        return state

    @staticmethod
    def condition(state: State) -> str:
        if state['category'] in ["chat", "weather", "stocks"]:
            return state['category']
        return "rag"

    def __call__(self, msg_input, previous_messages=None) -> State:
        initial_state = State(
            input=msg_input,
            output="",
            category="",
            search_query='',
            image_path='',
            chroma_results=[],
            previous_messages=previous_messages if previous_messages is not None else [],
            tfidf_results=[],
            reranking_context=[]
        )
        return self.compiled_graph.invoke(initial_state)
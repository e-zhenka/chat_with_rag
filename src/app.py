import streamlit as st
from llm_helper import LLMHelper
from graph import Agent
from database_manager import DatabaseManager
import uuid
import time
from config import Settings

settings = Settings.from_yaml("config.yaml")


def get_session_id():
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    return st.session_state.session_id


def main():
    st.title("Chat with RAG")

    # Инициализация менеджера базы данных
    if 'db_manager' not in st.session_state:
        st.session_state.db_manager = DatabaseManager()

    # Получаем ID сессии
    session_id = get_session_id()

    # Проверка и запрос API ключа
    if "api_key" not in st.session_state:
        st.warning("🔑 Требуется API ключ для доступа к LLM")
        api_key = st.text_input("Введите ваш OpenAI API ключ:", type="password")
        if st.button("Подтвердить ключ"):
            if api_key:
                try:
                    # Пробуем создать LLM с введенным ключом
                    test_llm = LLMHelper(
                        base_url=settings.url,
                        model=settings.model_name,
                        api_key=api_key
                    )
                    # Если успешно, сохраняем ключ
                    st.session_state.api_key = api_key
                    st.success("✅ Ключ API верный! Пожалуйста, подождите загрузки чата...")
                    time.sleep(2)  # Даем время для отображения сообщения
                    st.rerun()
                except Exception as e:
                    st.error("❌ Неверный ключ API. Пожалуйста, проверьте и попробуйте снова.")
            else:
                st.error("Пожалуйста, введите API ключ")
        return

    # Боковая панель для истории чата
    with st.sidebar:
        # Кнопка очистки истории в верхней части
        if st.button("🗑️ Очистить историю", key="clear_history"):
            st.session_state.db_manager.clear_chat_history(session_id)
            if 'last_results' in st.session_state:
                del st.session_state.last_results
            st.success("История чата очищена!")
            # pass

        st.markdown("### История чата")
        history = st.session_state.db_manager.load_chat_history(session_id)
        for msg in reversed(history):
            if msg["role"] == "user":
                st.info(f"👤 Вы: {msg['content']}")
            else:
                st.success(f"🤖 Бот: {msg['content']}")
                if msg["doc_type"]:
                    st.caption(f"📑 Категория: {msg['doc_type']}")

    # Инициализация LLM с ключом
    if "agent" not in st.session_state:
        st.session_state.agent = Agent(
            api_key=st.session_state.api_key
        )

    query = st.chat_input("Введите ваш вопрос:")

    if query:
        try:
            # Получаем последние сообщения пользователя
            previous_messages = st.session_state.db_manager.get_last_user_messages(session_id, limit=3)
            # previous_messages = []

            # Сохраняем текущий вопрос пользователя
            st.session_state.db_manager.save_message(session_id, "user", query)

            result = st.session_state.agent(query, previous_messages)
            # Показываем ответ
            if result['image_path'] != '':
                st.image(result['image_path'], caption="График")

            st.markdown(f"**Ответ:**\n\n{result['output']}")

            # Показываем информацию о классификации запроса
            with st.sidebar:
                st.markdown("### Детали обработки запроса")
                st.markdown(f"**Оригинальный запрос:**\n{query}")
                st.markdown(f"**Категория:**\n{result['category']}")
                st.markdown(f"**Переформулированный запрос:**\n{result['search_query']}")

                # Сохраняем ответ бота
                st.session_state.db_manager.save_message(
                    session_id, "assistant", result['output'], result['category']
                )

                # Показываем контекст в боковой панели, если он был использован
                if result['chroma_results'] or result['tfidf_results']:
                    with st.sidebar:
                        st.markdown("### Использованный контекст")

                        st.markdown("#### Результаты ChromaDB:")
                        for i, context in enumerate(result['chroma_results'], 1):
                            with st.expander(f"Документ {i}"):
                                st.markdown(context['document'])

                        st.markdown("#### Результаты TF-IDF:")
                        for i, context in enumerate(result['tfidf_results'], 1):
                            with st.expander(f"Документ {i}"):
                                st.markdown(context['document'])

        except Exception as e:
            st.error(f"Произошла ошибка: {str(e)}")
            st.write("Debug - full error:", e)


if __name__ == "__main__":
    main()

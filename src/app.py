import streamlit as st
from llm_helper import LLMHelper
from graph import Agent
from database_manager import DatabaseManager
import uuid
import time
from config import Settings
from sound import AudioProcessor
from streamlit_mic_recorder import mic_recorder

settings = Settings.from_yaml("config.yaml")

# Инициализация модели для распознавания голоса
audio_processor = AudioProcessor(model_name="base")
audio_processor.load_model() 

if 'audio_state' not in st.session_state:
    st.session_state.audio_state = 'inactive'


def get_session_id():
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    return st.session_state.session_id


def main():
    col1, col2 = st.columns([0.15, 0.85])
    with col1:
        st.image("pic/robot.png", width=80)  # Используем относительный путь к картинке
    with col2:
        st.title("чат-бот Джуни")

    # Инициализация менеджера базы данных
    if 'db_manager' not in st.session_state:
        st.session_state.db_manager = DatabaseManager()

    if 'audio_processor' not in st.session_state:
        st.session_state.audio_processor = audio_processor

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
            # Очищаем все связанные с чатом состояния
            keys_to_clear = [
                'last_results',
                'chat_history',
                'voice_query',
                'voice_query_active',
                'recorder'  # очищаем состояние рекордера
            ]
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            st.success("История чата очищена!")

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

    # Голосовой ввод
    st.markdown("### Голосовой ввод")
    audio = mic_recorder(
        start_prompt="🎤",
        stop_prompt="⏹️",
        key="recorder"
    )

    # Обработка аудио
    if audio and 'bytes' in audio:
        try:
            query_from_audio = st.session_state.audio_processor.transcribe_audio(audio['bytes'])
            if query_from_audio:
                st.session_state.voice_query = query_from_audio
                st.session_state.voice_query_active = True  # Флаг актуальности голосового ввода
        except Exception as e:
            st.error(f"Ошибка распознавания голоса: {str(e)}")

    # Используем chat_input для удобства ввода
    text_query = st.chat_input("Введите ваш вопрос:")
    
    # Определяем, какой запрос использовать
    if text_query:  # Если есть текстовый ввод, отключаем голосовой
        st.session_state.voice_query_active = False
        query = text_query
    elif hasattr(st.session_state, 'voice_query_active') and st.session_state.voice_query_active:
        query = st.session_state.voice_query
        st.session_state.voice_query_active = False  # Отключаем голосовой ввод после использования
    else:
        query = None
    
    if query:
        try:
            # Получаем последние сообщения пользователя
            previous_messages = st.session_state.db_manager.get_last_user_messages(session_id, limit=3)

            # Сохраняем текущий вопрос пользователя
            st.session_state.db_manager.save_message(session_id, "user", query)

            # Получаем ответ от агента
            result = st.session_state.agent(query, previous_messages)

            # Показываем ответ
            if result['image_path'] != '':
                st.image(result['image_path'], caption="График")

            st.markdown(f"**Ответ:**\n\n{result['output']}")

            # Показываем информацию о классификации запроса
            with st.sidebar:
                st.markdown("### 1️⃣ Анализ запроса")
                st.info(f"**Оригинальный вопрос:**\n{query}")
                st.success(f"**Переформулированный запрос:**\n{result['search_query']}")
                st.caption(f"**Категория:** {result['category']}")

                # Сохраняем ответ бота
                st.session_state.db_manager.save_message(
                    session_id, "assistant", result['output'], result['category']
                )

                if result['chroma_results'] or result['tfidf_results']:
                    
                    with st.sidebar:
                        st.markdown("### 2️⃣ Найденные чанки")
                        
                        if result['chroma_results']:
                            st.markdown("#### ChromaDB:")
                            for i, r in enumerate(result['chroma_results'], 1):
                                with st.expander(f"Чанк {i} (score: {r['score']:.3f})"):
                                    st.markdown(r['document'])
                        
                        if result['tfidf_results']:
                            st.markdown("#### TF-IDF:")
                            for i, r in enumerate(result['tfidf_results'], 1):
                                with st.expander(f"Чанк {i} (score: {r['score']:.3f})"):
                                    st.markdown(r['document'])

                    with st.sidebar:
                        st.markdown("### 3️⃣ Результаты реранкинга")
                        if result['reranking_context']:
                            for rank in result['reranking_context']:
                                with st.expander(f"💯 Оценка: {rank['score']:.2f} | Контекст {rank['index']}"):
                                    st.info(f"**Причина оценки:**\n{rank['explanation']}")
                                    st.success(f"**Контекст:**\n{rank['context']}")
                        else:
                            st.warning("⚠️ Реранкинг не вернул результатов!")

        except Exception as e:
            st.error(f"Произошла ошибка: {str(e)}")
            st.write("Debug - full error:", e)


if __name__ == "__main__":
    main()

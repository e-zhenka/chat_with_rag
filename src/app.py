import streamlit as st
from database import HybridDB
from llm_helper import LLMHelper
from database_manager import DatabaseManager
import uuid
import time
from config import Settings
from sound import AudioProcessor
from streamlit_mic_recorder import mic_recorder

settings = Settings.from_yaml("config.yaml")

if 'audio_state' not in st.session_state:
    st.session_state.audio_state = 'inactive'

def get_session_id():
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    return st.session_state.session_id


def main():
    st.title("Chat with RAG")

    # Инициализация менеджера базы данных
    if 'db_manager' not in st.session_state:
        st.session_state.db_manager = DatabaseManager()

    if 'audio_processor' not in st.session_state:
        st.session_state.audio_processor = AudioProcessor(model_name="base")

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
    if "llm" not in st.session_state:
        st.session_state.llm = LLMHelper(
            base_url=settings.url,
            model=settings.model_name,
            api_key=st.session_state.api_key
        )

    if "db" not in st.session_state:
        st.session_state.db = HybridDB()

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
            previous_messages = st.session_state.db_manager.get_last_user_messages(session_id, limit=3)
            st.session_state.db_manager.save_message(session_id, "user", query)
            analysis = st.session_state.llm.analyze_query(query, previous_messages)

            with st.sidebar:
                st.markdown("### 1️⃣ Анализ запроса")
                st.info(f"**Оригинальный вопрос:**\n{query}")
                st.success(f"**Переформулированный запрос:**\n{analysis['search_query']}")
                st.caption(f"**Категория:** {analysis['query_type']}")

            if analysis["query_type"] == "chat":
                answer = st.session_state.llm.generate_chat_answer(query)
                st.markdown(f"**Ответ:**\n\n{answer}")
                st.session_state.db_manager.save_message(
                    session_id, "assistant", answer, analysis["query_type"]
                )
            else:
                n_results = 3
                if analysis["query_type"] in settings.finance_documents:
                    n_results = 6

                results = st.session_state.db.query(
                    query_text=analysis["search_query"],
                    doc_type=analysis["query_type"],
                    n_results=n_results,
                )

                if results['chroma_results'] or results['tfidf_results']:
                    all_contexts = []
                    
                    with st.sidebar:
                        st.markdown("### 2️⃣ Найденные чанки")
                        
                        if results['chroma_results']:
                            st.markdown("#### ChromaDB:")
                            for i, r in enumerate(results['chroma_results'], 1):
                                with st.expander(f"Чанк {i} (score: {r['score']:.3f})"):
                                    st.markdown(r['document'])
                                all_contexts.append(r['document'])
                        
                        if results['tfidf_results']:
                            st.markdown("#### TF-IDF:")
                            for i, r in enumerate(results['tfidf_results'], 1):
                                if r['document'] not in all_contexts:
                                    with st.expander(f"Чанк {i} (score: {r['score']:.3f})"):
                                        st.markdown(r['document'])
                                    all_contexts.append(r['document'])

                    # Реранкинг контекстов
                    reranking_results = st.session_state.llm.rerank_context(
                        analysis["search_query"], 
                        all_contexts
                    )
                    sorted_contexts = [res['context'] for res in reranking_results]

                    # Генерация ответа
                    answer = st.session_state.llm.generate_answer(
                        query,
                        analysis["search_query"],
                        sorted_contexts  # Передаем отсортированные контексты
                    )

                    with st.sidebar:
                        st.markdown("### 3️⃣ Результаты реранкинга")
                        if reranking_results:
                            for rank in reranking_results:
                                with st.expander(f"💯 Оценка: {rank['score']:.2f} | Контекст {rank['index']}"):
                                    st.info(f"**Причина оценки:**\n{rank['explanation']}")
                                    st.success(f"**Контекст:**\n{rank['context']}")
                        else:
                            st.warning("⚠️ Реранкинг не вернул результатов!")

                    st.markdown("### Ответ:")
                    st.markdown(answer)

                else:
                    answer = "К сожалению, не нашел релевантной информации в базе данных."
                    st.markdown(f"**Ответ:**\n\n{answer}")

                st.session_state.db_manager.save_message(
                    session_id, "assistant", answer, analysis["query_type"]
                )

        except Exception as e:
            st.error(f"Произошла ошибка: {str(e)}")
            st.write("Debug - full error:", e)

if __name__ == "__main__":
    main()

import streamlit as st
from database import HybridDB
from llm_helper import LLMHelper
from database_manager import DatabaseManager
import os
import uuid
import time

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
                        base_url="https://llama3gpu.neuraldeep.tech/v1",
                        model="llama-3-8b-instruct-8k",
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
            st.rerun()
            
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
            base_url="https://llama3gpu.neuraldeep.tech/v1",
            model="llama-3-8b-instruct-8k",
            api_key=st.session_state.api_key
        )
    
    if "db" not in st.session_state:
        st.session_state.db = HybridDB()
    
    # Интерфейс вопроса
    query = st.text_input("Введите ваш вопрос:")
    
    if query:
        try:
            # Получаем последние сообщения пользователя
            previous_messages = st.session_state.db_manager.get_last_user_messages(session_id, limit=3)
            
            # Сохраняем текущий вопрос пользователя
            st.session_state.db_manager.save_message(session_id, "user", query)
            
            # Анализируем запрос и получаем тип документа и поисковый запрос
            analysis = st.session_state.llm.analyze_query(query, previous_messages)
            
            # Показываем информацию о классификации запроса
            with st.sidebar:
                st.markdown("### Детали обработки запроса")
                st.markdown(f"**Оригинальный запрос:**\n{query}")
                st.markdown(f"**Категория:**\n{analysis['type']}")
                st.markdown(f"**Переформулированный запрос:**\n{analysis['search_query']}")
            
            if analysis["type"] == "chat":
                answer = st.session_state.llm.generate_answer(
                    query,
                    analysis["search_query"],
                    ["Это общий разговор без специфического контекста"]
                )
                st.markdown(f"**Ответ:**\n\n{answer}")
            else:
                # Используем перефразированный запрос для поиска
                results = st.session_state.db.query(
                    query_text=analysis["search_query"],
                    doc_type=analysis["type"],
                    n_results=5
                )
                
                st.session_state.last_results = results
                
                if results['chroma_results'] or results['tfidf_results']:
                    all_contexts = []
                    for r in results['chroma_results']:
                        all_contexts.append(r['document'])
                    for r in results['tfidf_results']:
                        if r['document'] not in all_contexts:
                            all_contexts.append(r['document'])
                    
                    answer = st.session_state.llm.generate_answer(
                        query,
                        analysis["search_query"],
                        all_contexts
                    )
                else:
                    answer = "К сожалению, не нашел релевантной информации в базе данных."
                
                # Сохраняем ответ бота
                st.session_state.db_manager.save_message(
                    session_id, "assistant", answer, analysis["type"]
                )
                
                # Показываем ответ
                st.markdown(f"**Ответ:**\n\n{answer}")
                
                # Показываем контекст в боковой панели, если он был использован
                if hasattr(st.session_state, 'last_results') and (results['chroma_results'] or results['tfidf_results']):
                    with st.sidebar:
                        st.markdown("### Использованный контекст")
                        
                        st.markdown("#### Результаты ChromaDB:")
                        for i, result in enumerate(results['chroma_results'], 1):
                            with st.expander(f"Документ {i} (score: {result['score']:.3f})"):
                                st.markdown(result['document'])
                        
                        st.markdown("#### Результаты TF-IDF:")
                        for i, result in enumerate(results['tfidf_results'], 1):
                            with st.expander(f"Документ {i} (score: {result['score']:.3f})"):
                                st.markdown(result['document'])
                
        except Exception as e:
            st.error(f"Произошла ошибка: {str(e)}")
            st.write("Debug - full error:", e)

if __name__ == "__main__":
    main()

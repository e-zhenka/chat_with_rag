import psycopg2
from psycopg2.extras import DictCursor
from config import config

class DatabaseManager:
    def __init__(self):
        self.init_db()

    def get_db_connection(self):
        return psycopg2.connect(
            dbname=config.db.database,
            user=config.db.user,
            password=config.db.password,
            host=config.db.host,
            port=config.db.port
        )

    def init_db(self):
        retries = 5
        for attempt in range(retries):
            try:
                with self.get_db_connection() as conn:
                    with conn.cursor() as cur:
                        # Создаем таблицу для сообщений
                        cur.execute("""
                            CREATE TABLE IF NOT EXISTS chat_messages (
                                id SERIAL PRIMARY KEY,
                                session_id VARCHAR(100),
                                role VARCHAR(50),
                                content TEXT,
                                doc_type VARCHAR(50),
                                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                            )
                        """)
                        conn.commit()
                return
            except psycopg2.Error as e:
                if attempt == retries - 1:
                    raise Exception(f"Не удалось инициализировать базу данных после {retries} попыток: {str(e)}")
                import time
                time.sleep(1)

    def save_message(self, session_id: str, role: str, content: str, doc_type: str = None) -> None:
        with self.get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO chat_messages (session_id, role, content, doc_type) VALUES (%s, %s, %s, %s)",
                    (session_id, role, content, doc_type)
                )
                conn.commit()

    def load_chat_history(self, session_id: str, limit: int = 10) -> list:
        try:
            with self.get_db_connection() as conn:
                with conn.cursor(cursor_factory=DictCursor) as cur:
                    cur.execute(
                        """
                        SELECT role, content, doc_type 
                        FROM chat_messages 
                        WHERE session_id = %s 
                        ORDER BY timestamp DESC 
                        LIMIT %s
                        """,
                        (session_id, limit)
                    )
                    return [
                        {
                            "role": row["role"], 
                            "content": row["content"],
                            "doc_type": row["doc_type"]
                        } for row in cur.fetchall()
                    ]
        except psycopg2.Error as e:
            print(f"Ошибка при загрузке истории чата: {str(e)}")
            return []

    def clear_chat_history(self, session_id: str) -> None:
        with self.get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM chat_messages WHERE session_id = %s", (session_id,))
                conn.commit()

    def get_last_user_messages(self, session_id: str, limit: int = 3) -> list:
        """Получение последних сообщений пользователя"""
        try:
            with self.get_db_connection() as conn:
                with conn.cursor(cursor_factory=DictCursor) as cur:
                    cur.execute(
                        """
                        SELECT content 
                        FROM chat_messages 
                        WHERE session_id = %s AND role = 'user'
                        ORDER BY timestamp DESC 
                        LIMIT %s
                        """,
                        (session_id, limit)
                    )
                    return [row["content"] for row in cur.fetchall()]
        except psycopg2.Error as e:
            print(f"Ошибка при загрузке сообщений пользователя: {str(e)}")
            return []
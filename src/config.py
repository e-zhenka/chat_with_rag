from dataclasses import dataclass
from os import getenv

@dataclass
class DatabaseConfig:
    host: str = getenv('POSTGRES_HOST', 'postgres')
    port: int = int(getenv('POSTGRES_PORT', 5432))
    database: str = getenv('POSTGRES_DB', 'chatbot')
    user: str = getenv('POSTGRES_USER', 'postgres')
    password: str = getenv('POSTGRES_PASSWORD', 'postgres')

class Config:
    db = DatabaseConfig()

config = Config()
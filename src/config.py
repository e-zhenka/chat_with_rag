from dataclasses import dataclass
from os import getenv
import yaml
import os
from typing import Dict, Union
from pydantic import BaseModel, Field


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


class Settings(BaseModel):
    model_name: str = Field(
        default="",
    )
    url: str = Field(
        default="",
    )
    weather_api_key: str = Field(
        default="",
    )
    weather_url: str = Field(
        default="",
    )
    finance_documents: list[str] = Field(
        default=[]
    )
    clouds: dict[str, str] = Field(
        default=dict()
    )

    @classmethod
    def from_yaml(cls, path: str) -> "Settings":
        data = cls()._load_yaml(path)
        return cls(**data)

    def _load_yaml(self, path: str) -> Dict[str, Union[str, bool, int, float]]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File {path} does not exist")
        try:
            with open(path, 'r', encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if data is None:
                raise ValueError(f"Failed to load YAML from {path}")
            return data
        except Exception as e:
            raise ValueError(f"Failed to parse YAML: {e}")

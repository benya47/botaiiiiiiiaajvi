import os
import logging
from dotenv import load_dotenv
from pydantic import BaseSettings, Field

class Settings(BaseSettings):
    """Конфигурация приложения"""
    BYBIT_API_KEY: str = Field(..., env="BYBIT_API_KEY")
    BYBIT_SECRET: str = Field(..., env="BYBIT_API_SECRET")
    BYBIT_TESTNET: bool = Field(True, env="BYBIT_TESTNET")
    MODEL_DIR: str = Field("models", env="MODEL_DIR")
    DATA_DIR: str = Field("data", env="DATA_DIR")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

def initialize_environment():
    """Инициализация окружения"""
    load_dotenv()
    logging.info("Проверка конфигурации окружения")
    
    required_vars = ["BYBIT_API_KEY", "BYBIT_API_SECRET"]
    missing = [var for var in required_vars if not os.getenv(var)]
    
    if missing:
        raise EnvironmentError(f"Отсутствуют переменные окружения: {missing}")

def get_settings() -> Settings:
    """Получение конфигурации"""
    return Settings()
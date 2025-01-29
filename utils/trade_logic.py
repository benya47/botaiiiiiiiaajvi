import asyncio
import logging
from typing import Dict
from threading import Lock
from utils.api import BybitAPI
from utils.websocket_handler import get_market_data

class TradingSession:
    """Управление торговой сессией"""
    
    _sessions: Dict[str, 'TradingSession'] = {}
    _lock = Lock()
    
    def __init__(self, config: dict):
        self.id = config.get("session_id")
        self.strategy = config.get("strategy", "default")
        self.symbol = config["symbol"]
        self.active = False
        self.api = BybitAPI()
        
        with self._lock:
            self._sessions[self.id] = self
    
    def start(self):
        """Запуск сессии"""
        self.active = True
        asyncio.create_task(self._run_strategy())
    
    async def _run_strategy(self):
        """Основной цикл стратегии"""
        while self.active:
            try:
                data = get_market_data(self.symbol)
                decision = await self._analyze_data(data)
                await self._execute_decision(decision)
                await asyncio.sleep(60)
            except Exception as e:
                logging.error(f"Ошибка стратегии: {e}")
    
    async def _analyze_data(self, data: dict) -> str:
        """Анализ рыночных данных"""
        # Здесь должна быть логика нейросети
        return "hold"
    
    async def _execute_decision(self, decision: str):
        """Выполнение торгового решения"""
        if decision == "buy":
            self.api.create_order(self.symbol, "market", "buy", 0.01)
        elif decision == "sell":
            self.api.create_order(self.symbol, "market", "sell", 0.01)
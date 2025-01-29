import ccxt
import logging
from functools import lru_cache
from typing import Dict, List
from utils.config import get_settings

settings = get_settings()
logging.basicConfig(level=logging.INFO)

class BybitAPI:
    """Клиент для работы с API Bybit"""
    
    def __init__(self):
        self.exchange = ccxt.bybit({
            'apiKey': settings.BYBIT_API_KEY,
            'secret': settings.BYBIT_SECRET,
            'enableRateLimit': True,
            'options': {'testnet': settings.BYBIT_TESTNET}
        })
    
    @lru_cache(maxsize=100)
    def fetch_ohlcv(self, symbol: str, timeframe: str = '1m', limit: int = 1000) -> List[list]:
        """Получение исторических данных с кэшированием"""
        try:
            return self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        except ccxt.NetworkError as e:
            logging.error(f"Ошибка сети: {e}")
            return []
    
    def get_balance(self) -> Dict[str, float]:
        """Получение баланса аккаунта"""
        try:
            balance = self.exchange.fetch_balance()
            return balance["total"]
        except Exception as e:
            logging.error(f"Ошибка получения баланса: {e}")
            return {}
    
    def create_order(self, symbol: str, order_type: str, side: str, amount: float) -> Dict:
        """Создание ордера с проверкой баланса"""
        balance = self.get_balance()
        if balance.get("USDT", 0) < amount:
            raise ValueError("Недостаточный баланс USDT")
        
        return self.exchange.create_order(
            symbol=symbol,
            type=order_type,
            side=side,
            amount=amount
        )

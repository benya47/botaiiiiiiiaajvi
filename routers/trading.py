from fastapi import APIRouter, HTTPException
from utils.api import BybitAPI
from utils.trade_logic import TradingSession
import logging

router = APIRouter(prefix="/trading", tags=["Trading"])
api = BybitAPI()

@router.post("/order")
async def create_order(symbol: str, amount: float, order_type: str = "market"):
    """Создание торгового ордера"""
    try:
        result = api.create_order(symbol, order_type, "buy", amount)
        return {"status": "success", "data": result}
    except Exception as e:
        logging.error(f"Ошибка создания ордера: {e}")
        raise HTTPException(500, str(e))

@router.post("/start-strategy")
async def start_strategy(strategy_config: dict):
    """Запуск торговой стратегии"""
    try:
        session = TradingSession(strategy_config)
        session.start()
        return {"status": "started", "session_id": session.id}
    except Exception as e:
        raise HTTPException(500, f"Ошибка запуска стратегии: {e}")

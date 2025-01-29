import logging
from fastapi import APIRouter, HTTPException
from utils.api import BybitAPI

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# Инициализация роутера с префиксом
router = APIRouter(prefix="/dashboard", tags=["Dashboard"])
bybit = BybitAPI()

@router.get("")  # Обрабатывает /dashboard
async def get_main_dashboard():
    """Возвращает список монет с данными"""
    try:
        logging.info("Запрос данных для таблицы монет")
        tickers = bybit.fetch_tickers()
        
        formatted_data = [
            {
                "symbol": symbol,
                "last": data.get("last", "Недоступно"),
                "change": data.get("percentage", 0) / 100 if data.get("percentage") else 0,
                "baseVolume": data.get("baseVolume", "Недоступно"),
                "forecast": "N/A",
            }
            for symbol, data in tickers.items()
        ]
        
        return {"market_data": formatted_data}
    
    except Exception as e:
        logging.error(f"Ошибка: {e}")
        raise HTTPException(500, "Internal Server Error")

@router.get("/{symbol}")
async def get_symbol_data(symbol: str):
    """Возвращает данные для конкретной торговой пары"""
    try:
        logging.info(f"Запрос данных для {symbol}")
        ticker = bybit.fetch_ticker(symbol)
        
        return {
            "symbol": symbol,
            "last": ticker.get("last", "N/A"),
            "high": ticker.get("high", "N/A"),
            "low": ticker.get("low", "N/A"),
            "volume": ticker.get("baseVolume", "N/A")
        }
        
    except Exception as e:
        logging.error(f"Ошибка для {symbol}: {e}")
        raise HTTPException(500, "Internal Server Error")

@router.get("/kline/{symbol}")
async def get_kline_data(
    symbol: str,
    timeframe: str = "1h",
    limit: int = 100
):
    """Возвращает исторические данные свечей"""
    try:
        logging.info(f"Запрос Kline для {symbol} ({timeframe})")
        ohlcv = bybit.fetch_ohlcv(symbol, timeframe, limit=limit)
        
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "data": ohlcv
        }
        
    except Exception as e:
        logging.error(f"Ошибка Kline для {symbol}: {e}")
        raise HTTPException(500, "Internal Server Error")
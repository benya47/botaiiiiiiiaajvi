import logging
import json
from fastapi import APIRouter, HTTPException
from utils.neural_network import get_signals
from utils.api import BybitAPI
from utils.neural_network import load_model_with_metadata

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

router = APIRouter()
bybit = BybitAPI()

@router.get("/get-signals")
async def get_signals_for_pair(pair: str):
    """
    Возвращает данные OHLCV и сигналы нейросети для выбранной пары.
    """
    bybit_api = BybitAPI()
    try:
        if not pair:
            raise HTTPException(status_code=400, detail="Торговая пара не указана")

        logging.info(f"Запрос OHLCV для пары: {pair}")
        ohlcv = bybit_api.fetch_ohlcv(pair, timeframe="1h", limit=100)
        if not ohlcv or len(ohlcv) == 0:
            raise HTTPException(status_code=404, detail=f"Данные OHLCV для пары {pair} не найдены")

        # Форматирование данных OHLCV
        formatted_ohlcv = [
            {
                "time": candle[0] // 1000,  # Преобразуем время в секунды
                "open": candle[1],
                "high": candle[2],
                "low": candle[3],
                "close": candle[4],
                "volume": candle[5],
            }
            for candle in ohlcv
        ]

        # Генерация сигналов на основе закрывающих цен
        closing_prices = [candle[4] for candle in ohlcv]
        signals = get_signals(closing_prices)

        # Привязка сигналов ко времени свечей
        for signal in signals:
            signal_time_index = signal["time"]
            if 0 <= signal_time_index < len(formatted_ohlcv):
                signal["time"] = formatted_ohlcv[signal_time_index]["time"]
            else:
                signal["time"] = None

        signals = [s for s in signals if s["time"] is not None]

        logging.info(f"Данные и сигналы для {pair} успешно сгенерированы")
        return {"ohlcv": formatted_ohlcv, "signals": signals}
    except Exception as e:
        logging.error(f"Ошибка обработки данных для пары {pair}: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка обработки данных для {pair}: {e}")

@router.get("/get-model-info")
async def get_model_info():
    """
    Возвращает информацию о модели для анализа.
    """
    try:
        with open("config/active_model.json", "r") as f:
            active_model = json.load(f).get("active_model")
        if not active_model:
            raise HTTPException(status_code=404, detail="Активная модель не установлена")

        _, metadata = load_model_with_metadata(active_model)
        if not metadata or "coins" not in metadata:
            raise HTTPException(status_code=404, detail="Метаданные для модели не найдены или неполные")

        return {"name": active_model, **metadata}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Файл конфигурации активной модели не найден")
    except Exception as e:
        logging.error(f"Ошибка загрузки информации о модели: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка загрузки информации о модели: {e}")

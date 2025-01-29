import json
import asyncio
import time
import numpy as np
import ta
import logging
import pandas as pd
from threading import Thread
import websocket
from fastapi import WebSocket, WebSocketDisconnect
import threading
from threading import Lock
from utils.api import BybitAPI

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

# Потокобезопасные структуры данных
historical_data_lock = Lock()
realtime_data_lock = Lock()
ws_connection_lock = Lock()
historical_data = {}
realtime_data = {}
last_update_time = 0

class WebSocketManager:
    def __init__(self):
        self.active_connections = []
        self.lock = Lock()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        with self.lock:
            self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        with self.lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        with self.lock:
            for connection in self.active_connections.copy():
                try:
                    await connection.send_json(message)
                except (WebSocketDisconnect, RuntimeError):
                    self.disconnect(connection)

manager = WebSocketManager()

def validate_data_columns(df: pd.DataFrame, required_columns: list) -> None:
    """Проверяет наличие обязательных колонок в DataFrame."""
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Отсутствуют обязательные колонки: {missing}")

def prepare_data_with_features(raw_data: pd.DataFrame) -> np.ndarray:
    """Расчет технических индикаторов: MA, RSI, MACD."""
    df = raw_data.copy()
    try:
        validate_data_columns(df, ['close', 'volume'])
        
        # Скользящие средние
        df['ma_short'] = df['close'].rolling(window=5).mean()
        df['ma_long'] = df['close'].rolling(window=50).mean()
        df['ma_ratio'] = df['ma_short'] / df['ma_long']
        
        # RSI
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        
        # MACD
        macd = ta.trend.MACD(df['close'])
        df['macd_diff'] = macd.macd_diff()

        return df[['ma_short', 'ma_long', 'ma_ratio', 'rsi', 'macd_diff']].fillna(0).values
    except Exception as e:
        logging.error(f"Ошибка подготовки признаков: {e}")
        return np.array([])

def prepare_data_with_indicators(raw_data: pd.DataFrame) -> np.ndarray:
    """Расчет рыночных индикаторов: ATR, Bollinger Bands, Stochastic."""
    df = raw_data.copy()
    try:
        validate_data_columns(df, ['high', 'low', 'close', 'volume'])
        
        # ATR
        df['atr'] = ta.volatility.AverageTrueRange(
            high=df['high'], low=df['low'], close=df['close'], window=14
        ).average_true_range()
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_width'] = df['bb_upper'] - df['bb_lower']
        
        # Stochastic
        stoch = ta.momentum.StochasticOscillator(
            high=df['high'], low=df['low'], close=df['close'],
            window=14, smooth_window=3
        )
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()

        return df[['atr', 'bb_upper', 'bb_lower', 'bb_width', 'stoch_k', 'stoch_d']].fillna(0).values
    except Exception as e:
        logging.error(f"Ошибка подготовки индикаторов: {e}")
        return np.array([])

def process_candle_data(symbol: str, kline: dict) -> dict:
    """Обработка и форматирование данных свечи."""
    return {
        "time": int(kline["start"]),
        "open": float(kline["open"]),
        "high": float(kline["high"]),
        "low": float(kline["low"]),
        "close": float(kline["close"]),
        "volume": float(kline["volume"]),
    }

def initialize_symbol_data(symbol: str) -> None:
    """Инициализация исторических данных для нового символа."""
    with historical_data_lock:
        if symbol not in historical_data:
            logging.info(f"Инициализация данных для {symbol}")
            historical_data[symbol] = {"candles": [], "signals": []}
            
            try:
                bybit_api = BybitAPI()
                ohlcv = bybit_api.fetch_ohlcv(symbol, timeframe="1m", limit=100)
                if ohlcv:
                    df = pd.DataFrame(ohlcv, columns=["time", "open", "high", "low", "close", "volume"])
                    features = prepare_data_with_features(df)
                    indicators = prepare_data_with_indicators(df)
                    
                    # Объединение данных
                    df = pd.concat([
                        df,
                        pd.DataFrame(features, columns=["ma_short", "ma_long", "ma_ratio", "rsi", "macd_diff"]),
                        pd.DataFrame(indicators, columns=["atr", "bb_upper", "bb_lower", "bb_width", "stoch_k", "stoch_d"])
                    ], axis=1)
                    
                    historical_data[symbol]["candles"] = df.to_dict("records")
                    logging.info(f"Загружено {len(df)} исторических свечей для {symbol}")
            except Exception as e:
                logging.error(f"Ошибка инициализации данных для {symbol}: {e}")

def update_historical_data(symbol: str, new_candle: dict) -> None:
    """Обновление исторических данных с потокобезопасным доступом."""
    with historical_data_lock:
        candles = historical_data[symbol]["candles"]
        
        # Обновление или добавление свечи
        if candles and candles[-1]["time"] == new_candle["time"]:
            candles[-1] = new_candle
        else:
            candles.append(new_candle)
        
        # Ограничение размера истории
        if len(candles) > 2000:
            historical_data[symbol]["candles"] = candles[-2000:]

def calculate_latest_indicators(symbol: str) -> None:
    """Пересчет индикаторов для последних 100 свечей."""
    with historical_data_lock:
        candles = historical_data[symbol]["candles"]
        if len(candles) >= 100:
            df = pd.DataFrame(candles[-100:])
            features = prepare_data_with_features(df)
            indicators = prepare_data_with_indicators(df)
            
            # Обновление последней свечи
            latest_index = len(candles) - 1
            candles[latest_index].update({
                **dict(zip(["ma_short", "ma_long", "ma_ratio", "rsi", "macd_diff"], features[-1])),
                **dict(zip(["atr", "bb_upper", "bb_lower", "bb_width", "stoch_k", "stoch_d"], indicators[-1]))
            })

def on_message(ws, message: str) -> None:
    """Обработчик входящих сообщений WebSocket."""
    try:
        data = json.loads(message)
        if "topic" not in data or "data" not in data:
            return

        topic = data["topic"]
        symbol = topic.split(".")[-1]
        klines = data["data"]

        # Инициализация данных при первом упоминании символа
        initialize_symbol_data(symbol)

        # Обработка каждой свечи
        for kline in klines:
            new_candle = process_candle_data(symbol, kline)
            update_historical_data(symbol, new_candle)
            calculate_latest_indicators(symbol)

    except Exception as e:
        logging.error(f"Ошибка обработки сообщения: {e}")

def on_error(ws, error: Exception) -> None:
    """Обработчик ошибок WebSocket."""
    logging.error(f"WebSocket error: {error}")

def on_close(ws, close_status_code: int, close_msg: str) -> None:
    """Обработчик закрытия соединения."""
    logging.info(f"Соединение закрыто. Код: {close_status_code}, Причина: {close_msg}")

def on_open(ws) -> None:
    """Обработчик открытия соединения."""
    def run():
        # Подписка на основные пары по умолчанию
        subscriptions = [
            {"op": "subscribe", "args": ["kline.1.BTCUSDT"]},
            {"op": "subscribe", "args": ["kline.1.ETHUSDT"]}
        ]
        
        for sub in subscriptions:
            ws.send(json.dumps(sub))
            logging.info(f"Отправлена подписка: {sub}")

        # Пинг-механизм для поддержания соединения
        while True:
            time.sleep(20)
            try:
                ws.send(json.dumps({"op": "ping"}))
            except Exception as e:
                logging.error(f"Ошибка отправки ping: {e}")
                break

    Thread(target=run).start()

def start_websocket() -> None:
    """Запуск WebSocket-соединения с механизмом переподключения."""
    def run_ws():
        retry_delay = 5
        while True:
            try:
                with ws_connection_lock:
                    ws = websocket.WebSocketApp(
                        "wss://stream.bybit.com/v5/public/spot",
                        on_open=on_open,
                        on_message=on_message,
                        on_error=on_error,
                        on_close=on_close
                    )
                    logging.info("Подключение к WebSocket...")
                    ws.run_forever()
            except Exception as e:
                logging.error(f"Ошибка соединения: {e}")
                time.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, 60)  # Экспоненциальная задержка

    Thread(target=run_ws, daemon=True).start()

async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Получаем сообщения от клиента
            data = await websocket.receive_json()
            
            # Обработка торговых команд
            if data.action == "place_order":
                result = await execute_trade(data)
                await websocket.send_json(result)
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)

        
def subscribe_to_symbol(symbol: str) -> None:
    """Отправка запроса на подписку для конкретного символа."""
    with ws_connection_lock:
        if not ws or not ws.sock or not ws.sock.connected:
            logging.error("WebSocket соединение не активно")
            return

        formatted_symbol = symbol.replace("/", "").upper()
        payload = {
            "op": "subscribe",
            "args": [f"spot.kline.1.{formatted_symbol}"]
        }
        try:
            ws.send(json.dumps(payload))
            logging.info(f"Отправлена подписка для {symbol}")
        except Exception as e:
            logging.error(f"Ошибка подписки на {symbol}: {e}")

async def get_realtime_data(symbol: str) -> dict:
    """Получение текущих данных с потокобезопасным доступом."""
    with realtime_data_lock:
        return realtime_data.get(symbol, {}).copy()
    

def get_market_data(symbol: str) -> dict:
    """Получение рыночных данных для указанного символа"""
    with historical_data_lock:
        return historical_data.get(symbol, {}).copy()    

if __name__ == "__main__":
    start_websocket()
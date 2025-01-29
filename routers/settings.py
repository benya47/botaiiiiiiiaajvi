from fastapi import APIRouter, HTTPException, UploadFile
from utils.neural_network import train_model, retrain_model, load_model, preprocess_data,  split_sequences
import numpy as np
import os
import time
from datetime import datetime
from utils.websocket_handler import prepare_data_with_features, prepare_data_with_indicators  # ✅ Теперь импортируем из websocket_handler.py

import ta  # Для расчёта технических индикаторов

import json
import logging
import pandas as pd
from utils.api import BybitAPI
import asyncio  # Добавлено для поддержки функций sleep и create_task
from utils.neural_network import load_model_with_metadata









router = APIRouter()
bybit = BybitAPI()
bybit_api = BybitAPI()

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

@router.post("/update-data")
async def update_data(data: dict = None):
    """
    Обновляет данные для выбранных монет начиная с указанного времени.
    """
    try:
        # Убедимся, что файл и данные корректны
        file_path = os.path.join(DATA_DIR, "selected_coins.json")
        if data is None:
            data = {}

        # Извлекаем список монет из запроса
        selected_coins = data.get("selected_coins", [])
        if not selected_coins:
            # Загружаем монеты из файла, если они не переданы в запросе
            if os.path.exists(file_path):
                logging.info("[UPDATE DATA] Загрузка списка монет из файла selected_coins.json")
                with open(file_path, "r") as file:
                    selected_coins = json.load(file)

        # Проверка списка монет
        if not selected_coins:
            logging.error("[UPDATE DATA] Список монет пуст.")
            raise HTTPException(status_code=400, detail="Список монет пуст.")

        # Параметры обновления
        timeframe = data.get("timeframe", "1m")
        limit = int(data.get("limit", 1000))
        since_str = data.get("since", "2023-01-01 00:00:00")
        since = int(datetime.strptime(since_str, "%Y-%m-%d %H:%M:%S").timestamp() * 1000)

        logging.info(f"[UPDATE DATA] Параметры: timeframe={timeframe}, limit={limit}, since={since}")

        
        # Обновление данных через функцию
        update_data(selected_coins, timeframe, limit)
        


        return {"status": "success", "message": "Данные успешно обновлены"}
    except Exception as e:
        logging.error(f"[UPDATE DATA] Ошибка обновления данных: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@router.post("/retrain")
async def retrain_model_endpoint(data: dict):
    """
    Эндпоинт для переобучения модели.
    """
    try:
        raw_data = np.array(data["X"])
        labels = np.array(data["y"])
        processed_data = preprocess_data(raw_data)
        retrain_model(processed_data, labels)
        return {"status": "retraining_completed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка переобучения модели: {e}")

@router.post("/load-model")
async def load_model_endpoint(file: UploadFile):
    """
    Загружает пользовательскую модель.
    """
    try:
        if not file.filename.endswith(".h5"):
            raise HTTPException(status_code=400, detail="Неправильный формат файла. Требуется .h5")
        
        file_location = f"models/{file.filename}"
        with open(file_location, "wb") as f:
            f.write(await file.read())
        
        logging.info(f"Модель загружена: {file.filename}")
        return {"status": "success", "message": f"Модель {file.filename} успешно загружена."}
    except Exception as e:
        logging.error(f"Ошибка загрузки модели: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка загрузки модели: {e}")
    
@router.post("/select-coins")
async def select_coins(coins: list):
    """
    Сохраняет выбранные монеты для анализа.
    """
    file_path = os.path.join(DATA_DIR, "selected_coins.json")
    try:
        if not coins:
            raise HTTPException(status_code=400, detail="Список монет пуст")

        logging.info(f"Сохранение списка монет: {coins}")
        with open(file_path, "w") as file:
            json.dump(coins, file)

        logging.info(f"Список монет успешно сохранён в {file_path}")
        return {"status": "success", "message": "Монеты сохранены"}
    except Exception as e:
        logging.error(f"Ошибка сохранения монет в {file_path}: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка сохранения монет: {e}")

@router.get("/selected-coins")
async def get_selected_coins():
    """
    Возвращает список выбранных монет.
    """
    file_path = os.path.join(DATA_DIR, "selected_coins.json")
    if not os.path.exists(file_path):
        logging.warning(f"Файл {file_path} отсутствует, создаю новый.")
        with open(file_path, "w") as file:
            json.dump([], file)

    try:
        # Проверяем, пуст ли файл
        if os.path.getsize(file_path) == 0:
            logging.warning(f"Файл {file_path} пуст. Записываю [] по умолчанию.")
            with open(file_path, "w") as file:
                json.dump([], file)

        with open(file_path, "r") as file:
            coins = json.load(file)
        return {"status": "success", "coins": coins}
    except json.JSONDecodeError as e:
        logging.error(f"Файл {file_path} повреждён: {e}")
        # Исправляем, записывая пустой список
        with open(file_path, "w") as file:
            json.dump([], file)
        return {"status": "error", "message": f"Файл {file_path} был повреждён и восстановлен."}
    except Exception as e:
        logging.error(f"Ошибка чтения файла {file_path}: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка чтения монет: {e}")

async def periodic_update():
    """
    Периодическое обновление данных.
    """
    file_path = os.path.join(DATA_DIR, "selected_coins.json")
    if not os.path.exists(file_path):
        logging.warning(f"Файл {file_path} отсутствует, создаю новый.")
        with open(file_path, "w") as file:
            json.dump([], file)

    while True:
        try:
            # Проверяем, пуст ли файл
            if os.path.getsize(file_path) == 0:
                logging.warning(f"Файл {file_path} пуст. Записываю [] по умолчанию.")
                with open(file_path, "w") as file:
                    json.dump([], file)

            with open(file_path, "r") as file:
                coins = json.load(file)
            if coins:
                update_data(coins)
            else:
                logging.info("Список монет пуст, обновление данных не требуется.")
            await asyncio.sleep(3600)
        except json.JSONDecodeError as e:
            logging.error(f"Файл {file_path} повреждён: {e}")
            # Исправляем, записывая пустой список
            with open(file_path, "w") as file:
                json.dump([], file)
        except Exception as e:
            logging.error(f"Ошибка периодического обновления: {e}")





def update_data(coins, timeframe="1m", limit=100):
    """
    Обновляет данные с расчётом индикаторов и сохраняет их в CSV.
    Добавлены:
    1. Удаление старых данных перед записью.
    2. Обработка запросов частями, если limit > 1000.
    """
    results = {}
    for coin in coins:
        try:
            # Путь к файлу данных
            file_path = os.path.join(DATA_DIR, f"{coin.replace('/', '_')}.csv")

            # Удаление старого файла
            if os.path.exists(file_path):
                logging.info(f"Удаление старого файла данных: {file_path}")
                os.remove(file_path)

            # Загрузка данных частями
            remaining_limit = limit
            all_data = []

            while remaining_limit > 0:
                current_limit = min(1000, remaining_limit)
                logging.info(f"[FETCH OHLCV] Запрос {current_limit} строк для {coin} с таймфреймом {timeframe}")
                ohlcv = bybit_api.fetch_ohlcv(coin, timeframe, limit=current_limit)

                if not ohlcv or len(ohlcv) == 0:
                    logging.error(f"[FETCH OHLCV] Пустой результат для {coin}.")
                    break

                all_data.extend(ohlcv)
                remaining_limit -= current_limit

            if not all_data:
                logging.error(f"Не удалось загрузить данные для {coin}. Пропускаем.")
                results[coin] = None
                continue

            # Создание DataFrame
            df = pd.DataFrame(all_data, columns=["timestamp", "open", "high", "low", "close", "volume"])

            # Расчёт индикаторов
            if len(df) < 50:
                logging.warning(f"Недостаточно данных для расчёта индикаторов для {coin}.")
                results[coin] = None
                continue

            # Обработка индикаторов
            df_features = pd.DataFrame(prepare_data_with_features(df), columns=["ma_short", "ma_long", "ma_ratio", "rsi", "macd_diff"])
            df_indicators = pd.DataFrame(prepare_data_with_indicators(df), columns=["atr", "bb_upper", "bb_lower", "bb_width", "stoch_k", "stoch_d"])

            # Объединение данных
            df_combined = pd.concat([df, df_features, df_indicators], axis=1)

            # Удаление NaN
            df_combined.dropna(inplace=True)

            # Сохранение данных
            df_combined.to_csv(file_path, index=False)
            logging.info(f"[UPDATE DATA] Данные для {coin} успешно сохранены в файл {file_path}")

            # Сохранение результата
            results[coin] = df_combined
        except Exception as e:
            logging.error(f"[UPDATE DATA] Ошибка обновления данных для {coin}: {e}")
            results[coin] = None

    return results



@router.get("/model-info")
async def get_model_info():
    """
    Возвращает информацию о текущей активной модели.
    """
    try:
        with open("config/active_model.json", "r") as f:
            active_model = json.load(f).get("active_model")
        if not active_model:
            raise HTTPException(status_code=404, detail="Активная модель не установлена")

        _, metadata = load_model_with_metadata(active_model)
        if not metadata:
            raise HTTPException(status_code=404, detail="Метаданные для модели не найдены")

        return {"status": "success", "name": active_model, **metadata}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Файл конфигурации активной модели не найден")
    except Exception as e:
        logging.error(f"Ошибка загрузки информации о модели: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка загрузки информации о модели: {e}")


@router.post("/update-model-info")
async def update_model_info():
    try:
        model_name = "trained_model"
        _, metadata = load_model_with_metadata(model_name)
        if not metadata:
            raise HTTPException(status_code=404, detail="Метаданные для модели не найдены")
        return {"status": "success", "message": f"Информация о модели {model_name} обновлена", **metadata}
    except Exception as e:
        logging.error(f"Ошибка обновления информации о модели: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/training-params")
async def get_training_params():
    """
    Возвращает параметры обучения активной модели.
    """
    try:
        with open("config/active_model.json", "r") as f:
            active_model = json.load(f).get("active_model")
        if not active_model:
            raise HTTPException(status_code=404, detail="Активная модель не установлена")

        _, metadata = load_model_with_metadata(active_model)
        if not metadata:
            raise HTTPException(status_code=404, detail="Метаданные для модели не найдены")

        return metadata
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Файл конфигурации активной модели не найден")
    except Exception as e:
        logging.error(f"Ошибка получения параметров обучения: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка получения параметров обучения: {e}")

@router.get("/available-models")
async def get_available_models():
    try:
        models_dir = "models"
        models = [f.replace(".h5", "") for f in os.listdir(models_dir) if f.endswith(".h5")]
        return models
    except Exception as e:
        logging.error(f"Ошибка получения списка моделей: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/set-active-model")
async def set_active_model(data: dict):
    try:
        selected_model = data.get("model")
        if not selected_model:
            raise HTTPException(status_code=400, detail="Не указана модель")
        metadata_file = f"models/{selected_model}_metadata.json"
        if not os.path.exists(metadata_file):
            raise HTTPException(status_code=404, detail=f"Модель {selected_model} не найдена")
        with open("config/active_model.json", "w") as f:
            json.dump({"active_model": selected_model}, f)
        return {"status": "success", "message": f"Активная модель установлена: {selected_model}"}
    except Exception as e:
        logging.error(f"Ошибка установки активной модели: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models")
async def get_available_models():
    """
    Возвращает список доступных моделей с описанием.
    """
    models = {
        "DQN": "Обучение с подкреплением для автоматической торговли.",
        "Transformer": "Современная модель для анализа длинных последовательностей.",
        "CNN_LSTM": "Комбинация сверточных и рекуррентных слоев.",
        "Autoencoder": "Обнаружение аномалий в данных.",
        "Ensemble": "Ансамбль моделей для повышения точности.",
        "Portfolio": "Модель управления портфелем активов.",
    }
    return models














@router.post("/train")
async def train_model_endpoint(data: dict):
    """
    Эндпоинт для обучения одной или нескольких моделей.
    """
    try:
        logging.info(f"Получены данные для обучения: {data}")
        
        # Чтение параметров из запроса

        model_types = data.get("model_types", ["LSTM"])
        strategy = data.get("strategy", "default")
        timeframe = data.get("timeframe", "1m")
        limit = int(data.get("limit", 1000))
        sequence_length = int(data.get("sequence_length", 10))
        epochs = int(data.get("epochs", 10))
        batch_size = int(data.get("batch_size", 32))
        

        # Загрузка selected_coins из файла
        file_path = os.path.join(DATA_DIR, "selected_coins.json")
        if not os.path.exists(file_path):
            logging.error(f"Файл {file_path} не найден.")
            raise HTTPException(status_code=404, detail="Файл selected_coins.json не найден.")

        with open(file_path, "r") as file:
            selected_coins = json.load(file)

        logging.info(f"Загруженные монеты: {selected_coins}")

        if not isinstance(selected_coins, list) or not all(isinstance(coin, str) for coin in selected_coins):
            logging.error(f"Некорректный формат selected_coins: {selected_coins}")
            raise HTTPException(status_code=400, detail="Некорректный формат selected_coins. Ожидался список строк.")

        # Логирование перед формированием metadata
        logging.info(f"Параметры перед формированием metadata: model_types={model_types}, strategy={strategy}, timeframe={timeframe}, limit={limit}, sequence_length={sequence_length}, epochs={epochs}, batch_size={batch_size}, selected_coins={selected_coins}")


        # Выбор модели и вызов функции обучения
        for model_type in model_types:
            logging.info(f"Обработка model_type={model_type}, strategy={strategy}, selected_coins={selected_coins}")
            model_name = f"model_{strategy}_{model_type}_tf{timeframe}_lim{limit}_seq{sequence_length}_ep{epochs}_bs{batch_size}_{'_'.join(selected_coins)}"
            metadata = {
                "model_type": model_type,
                "strategy": strategy,
                "timeframe": timeframe,
                "limit": limit,
                "sequence_length": sequence_length,
                "epochs": epochs,
                "batch_size": batch_size,
                "coins": selected_coins,
            }
            logging.info(f"Сформированы model_name: {model_name}, metadata: {metadata}")
            train_model(
                model_name=model_name,
                sequence_length=sequence_length,
                epochs=epochs,
                batch_size=batch_size,
                model_type=model_type,
                metadata=metadata,
                use_tuner=data.get("use_tuner", False)  # Передача параметра Keras Tuner

            )

        logging.info("Обучение моделей завершено успешно")
        return {"status": "success", "message": f"Модели {', '.join(model_types)} успешно обучены и сохранены."}
    except Exception as e:
        logging.error(f"Ошибка обучения моделей: {e}")
        raise HTTPException(status_code=500, detail=str(e))





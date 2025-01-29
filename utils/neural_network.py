from datetime import datetime
from utils.api import BybitAPI
from keras_tuner import RandomSearch
from threading import Thread
import pandas as pd

import time
import websocket
from utils.websocket_handler import historical_data  # ✅ Импортируем `historical_data`
from utils.websocket_handler import prepare_data_with_features, prepare_data_with_indicators  # ✅ Теперь импортируем из websocket_handler.py




import tensorflow as tf
import numpy as np
import logging
import json
import os

import ta  # Для расчёта технических индикаторов
import asyncio  # Добавлено для поддержки функций sleep и create_task
import threading


DATA_DIR = "data"




@tf.keras.utils.register_keras_serializable()
def mse(y_true, y_pred):
    """
    Кастомная функция для вычисления среднеквадратичной ошибки.
    """
    return tf.reduce_mean(tf.square(y_true - y_pred))

# Создание модели нейросети
def create_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, return_sequences=True, input_shape=input_shape),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss=mse)  # Используем зарегистрированную функцию mse
    return model

def split_sequences(data, seq_length):
    """
    Разделяет данные на последовательности заданной длины.
    
    :param data: np.array, форма (n_samples, n_features)
    :param seq_length: int, длина последовательности
    :return: np.array, форма (n_sequences, seq_length, n_features)
    """
    sequences = []
    for i in range(len(data) - seq_length + 1):
        sequences.append(data[i:i + seq_length])
    return np.array(sequences)


from keras_tuner import RandomSearch

def train_model(
    model_name="trained_model.h5",
    sequence_length=10,
    epochs=10,
    batch_size=32,
    model_type="LSTM",
    metadata=None,
    use_tuner=False,  # Новый параметр для активации Keras Tuner
    max_trials=10  # Максимальное количество комбинаций гиперпараметров для Keras Tuner
):
    """
    Обучение модели с использованием данных и опциональной настройки гиперпараметров.
    """
    logging.info(f"[TRAIN MODEL] Начало вызова с параметрами: model_name={model_name}, metadata={metadata}, use_tuner={use_tuner}")

    try:
        # Загрузка данных
        file_path = os.path.join(DATA_DIR, "selected_coins.json")
        if not os.path.exists(file_path):
            logging.error(f"Файл {file_path} не найден.")
            raise FileNotFoundError(f"Файл {file_path} не найден.")

        with open(file_path, "r") as file:
            selected_coins = json.load(file)

        if not selected_coins:
            logging.error("Список монет пуст.")
            raise ValueError("Список монет пуст. Обучение невозможно.")

        # Подготовка данных
        data, labels = [], []
        for coin in selected_coins:
            coin_file = os.path.join(DATA_DIR, f"{coin.replace('/', '_')}.csv")
            if not os.path.exists(coin_file):
                logging.warning(f"Файл данных для {coin} не найден.")
                continue

            df = pd.read_csv(coin_file)
            if len(df) < sequence_length:
                logging.warning(f"Недостаточно данных для {coin}: {len(df)} строк.")
                continue

            # Используем все столбцы, кроме 'timestamp' (если он есть)
            features = df.drop(columns=["timestamp"], errors="ignore").values
            target = (df["close"].shift(-1) > df["close"]).astype(int).values[:-1]

            # Разделение на последовательности
            processed_data = split_sequences(features, sequence_length)
            processed_labels = target[sequence_length - 1:]

            data.append(processed_data)
            labels.append(processed_labels)

        # Объединение данных
        data = np.concatenate(data, axis=0)
        labels = np.concatenate(labels, axis=0)
        logging.info(f"Размеры данных: data={data.shape}, labels={labels.shape}")

        # Убедимся, что длины данных и меток совпадают
        min_length = min(len(data), len(labels))
        data = data[:min_length]
        labels = labels[:min_length]

        logging.info(f"После обрезки: data={data.shape}, labels={labels.shape}")

        if len(data) < batch_size:
            logging.error("Недостаточно данных для обучения.")
            raise ValueError("Недостаточно данных для обучения.")

        # Если включен Keras Tuner
        if use_tuner:
            logging.info("[KERAS TUNER] Настройка гиперпараметров включена.")

            # Функция для построения модели
            def build_model(hp):
                model = tf.keras.Sequential()
                for i in range(hp.Int("num_layers", 1, 3)):
                    model.add(tf.keras.layers.LSTM(
                        units=hp.Int(f"units_{i}", min_value=32, max_value=128, step=32),
                        return_sequences=(i < hp.Int("num_layers", 1, 3) - 1),
                        input_shape=(sequence_length, data.shape[2])
                    ))
                model.add(tf.keras.layers.Dense(1, activation='linear'))
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(
                        learning_rate=hp.Choice("learning_rate", [1e-2, 1e-3, 1e-4])
                    ),
                    loss="mse",
                    metrics=["mae"]
                )
                return model

            # Настройка гиперпараметров
            tuner = RandomSearch(
                build_model,
                objective="val_loss",
                max_trials=max_trials,
                executions_per_trial=2,
                directory="tuner_results",
                project_name="crypto_tuning"
            )
            tuner.search(data, labels, epochs=epochs, batch_size=batch_size, validation_split=0.1)
            best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
            logging.info(f"Лучшие гиперпараметры: {best_hps.values}")

            # Обучение модели с лучшими гиперпараметрами
            model = tuner.hypermodel.build(best_hps)
            model.fit(data, labels, epochs=epochs, batch_size=batch_size, validation_split=0.1)

        else:
            # Обычное обучение без настройки гиперпараметров
            input_shape = (sequence_length, data.shape[2])
            if model_type == "LSTM":
                model = create_lstm_model(input_shape)
            logging.info(f"Начало обучения модели: {model_name}")
            model.fit(data, labels, epochs=epochs, batch_size=batch_size, validation_split=0.1)

        # Сохранение модели
        save_model(model_name, model, metadata=metadata)
        logging.info(f"Модель {model_name} успешно обучена и сохранена.")

    except Exception as e:
        logging.error(f"Ошибка обучения модели: {e}")
        raise



def save_model(model_name, model, metadata=None):
    """
    Сохранение модели и метаданных.
    """

        # Заменяем символы '/' на '_'
    model_name = model_name.replace("/", "_")


    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, f"{model_name}.h5")
    metadata_path = os.path.join(model_dir, f"{model_name}_metadata.json")

    model.save(model_path)
    logging.info(f"Модель сохранена в {model_path}")

    if metadata:
        with open(metadata_path, "w") as f:
            json.dump(metadata, f)
        logging.info(f"Метаданные модели сохранены в {metadata_path}")



# Прогнозирование
def predict(data, model=None):
    if not model:
        model = load_model("trained_model")
    predictions = model.predict(data)
    return predictions


# Загрузка модели
def load_model(name):
    """
    Загружает модель с учётом кастомных объектов.
    """
    return tf.keras.models.load_model(
        f"models/{name}.h5",
        custom_objects={"mse": mse}
    )


# Предобработка данных
def preprocess_data(raw_data):
    # Пример простой нормализации данных
    normalized_data = (raw_data - np.mean(raw_data, axis=0)) / np.std(raw_data, axis=0)
    return normalized_data




# Переобучение модели на новых данных
def retrain_model(new_data, new_labels):
    """
    Дообучение модели на новых данных.
    """
    try:
        model = load_model("trained_model")
        model.fit(new_data, new_labels, epochs=5, batch_size=32, validation_split=0.1)
        save_model("trained_model", model)
        logging.info("Модель успешно дообучена.")
    except Exception as e:
        logging.error(f"Ошибка в retrain_model: {e}")




def load_model_with_metadata(name):
    """
    Загружает модель и её метаданные.
    """
    model_path = f"models/{name}.h5"
    metadata_path = f"models/{name}_metadata.json"
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={"mse": mse}  # Указываем зарегистрированную функцию
    )

    metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as meta_file:
            metadata = json.load(meta_file)
        logging.info(f"Метаданные загружены: {metadata}")
    else:
        logging.warning(f"Метаданные для модели {name} не найдены.")

    return model, metadata

def get_signals(prices):
    signals = []
    for i, price in enumerate(prices):
        if price > max(prices[:i+1]) * 0.95:
            signals.append({"time": i, "position": "aboveBar", "color": "green", "shape": "arrowUp", "text": "Buy"})
        elif price < min(prices[:i+1]) * 1.05:
            signals.append({"time": i, "position": "belowBar", "color": "red", "shape": "arrowDown", "text": "Sell"})
    logging.info(f"Сгенерировано {len(signals)} сигналов. Максимальный индекс: {max([s['time'] for s in signals], default=0)}")
    return signals


def create_lstm_model(input_shape):
    """
    LSTM модель для анализа временных рядов.
    """

    from tensorflow.keras.layers import Input
    
    model = tf.keras.Sequential([
        Input(shape=input_shape),  # Используем Input для задания входной формы
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def create_dqn_model(input_shape, action_space):
    """
    Deep Q-Network для автоматической торговли.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=input_shape),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(action_space, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


def create_transformer_model(input_shape, num_heads=4, ff_dim=32):
    """
    Transformer-Based Model для временных рядов.
    """
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=input_shape[-1])(inputs, inputs)
    x = tf.keras.layers.Dense(ff_dim, activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    outputs = tf.keras.layers.Dense(1, activation="linear")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def create_cnn_lstm_model(input_shape):
    """
    Гибридная модель CNN + LSTM для торговли.
    """
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu')(inputs)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    x = tf.keras.layers.LSTM(64, return_sequences=True)(x)
    x = tf.keras.layers.LSTM(32)(x)
    outputs = tf.keras.layers.Dense(1, activation='linear')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def create_autoencoder(input_shape):
    """
    Автокодировщик для обнаружения аномалий.
    """
    inputs = tf.keras.Input(shape=input_shape)
    encoded = tf.keras.layers.Dense(64, activation='relu')(inputs)
    decoded = tf.keras.layers.Dense(input_shape[0], activation='sigmoid')(encoded)
    model = tf.keras.Model(inputs, decoded)
    model.compile(optimizer='adam', loss='mse')
    return model


def create_portfolio_model(input_shape, num_assets):
    """
    Модель управления портфелем.
    """
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.LSTM(64, return_sequences=True)(inputs)
    x = tf.keras.layers.LSTM(32)(x)
    outputs = tf.keras.layers.Dense(num_assets, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


import pandas as pd




def build_model(hp):
    model = tf.keras.Sequential()
    
    # Добавляем LSTM-слои с переменным числом нейронов
    for i in range(hp.Int('num_layers', 1, 3)):  # Количество слоев от 1 до 3
        model.add(tf.keras.layers.LSTM(
            units=hp.Int(f'units_{i}', min_value=32, max_value=128, step=32),
            return_sequences=(i < hp.Int('num_layers', 1, 3) - 1),
            input_shape=(10, 5)  # Пример: длина последовательности 10, 5 признаков
        ))
    
    # Выходной слой
    model.add(tf.keras.layers.Dense(1, activation='linear'))
    
    # Компиляция
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])
        ),
        loss='mse',
        metrics=['mae']
    )
    return model


def tune_hyperparameters(x_train, y_train, x_val, y_val):
    tuner = RandomSearch(
        build_model,
        objective='val_loss',
        max_trials=10,  # Число комбинаций гиперпараметров
        executions_per_trial=2,  # Запуск каждой комбинации дважды
        directory='tuner_results',
        project_name='crypto_tuning'
    )
    
    tuner.search(x_train, y_train, epochs=10, validation_data=(x_val, y_val))
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    
    return tuner, best_hps




def process_stream_data_and_retrain(symbol, sequence_length=10):
    """
    Обрабатывает данные из historical_data и запускает дообучение модели.
    """
    try:
        # Проверяем и инициализируем historical_data
        if symbol not in historical_data:
            logging.warning(f"{symbol} отсутствует в historical_data. Инициализация...")
            historical_data[symbol] = {"candles": [], "signals": []}

        if len(historical_data[symbol]["candles"]) >= sequence_length:
            # Конвертируем данные в DataFrame
            df = pd.DataFrame(historical_data[symbol]["candles"])

            # Используем все признаки и индикаторы
            features = df[[
                "open", "high", "low", "close", "volume",
                "ma_short", "ma_long", "ma_ratio", "rsi", "macd_diff",
                "atr", "bb_upper", "bb_lower", "bb_width", "stoch_k", "stoch_d"
            ]].values
            labels = (df["close"].shift(-1) > df["close"]).astype(int).values[:-1]

            # Формируем последовательности
            data = split_sequences(features, sequence_length)
            labels = labels[sequence_length - 1:]

            # Убедимся, что данные и метки совпадают по длине
            min_length = min(len(data), len(labels))
            data, labels = data[:min_length], labels[:min_length]

            # Дообучение модели
            retrain_model(data, labels)
            logging.info(f"Модель успешно дообучена для {symbol}.")
        else:
            logging.warning(f"Недостаточно данных для {symbol}. Доступно: {len(historical_data[symbol]['candles'])}.")
    except Exception as e:
        logging.error(f"Ошибка в process_stream_data_and_retrain для {symbol}: {e}")


def preprocess_data_for_training(df, sequence_length=10):
    """
    Предобработка данных из DataFrame для обучения модели.
    """
    # Используем все столбцы, включая дополнительные признаки и индикаторы
    features = df[[
        "open", "high", "low", "close", "volume",
        "ma_short", "ma_long", "ma_ratio", "rsi", "macd_diff",
        "atr", "bb_upper", "bb_lower", "bb_width", "stoch_k", "stoch_d"
    ]].values
    labels = (df["close"].shift(-1) > df["close"]).astype(int).values[:-1]

    # Формирование последовательностей
    data = split_sequences(features, sequence_length)
    labels = labels[sequence_length - 1:]

    # Убедимся, что длины совпадают
    min_length = min(len(data), len(labels))
    return data[:min_length], labels[:min_length]



import logging
import asyncio
import json
import os
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from utils.websocket_handler import start_websocket
from routers.dashboard import router as dashboard_router
from routers.analysis import router as analysis_router
from routers.trading import router as trading_router
from routers.settings import router as settings_router

from routers.settings import periodic_update
from utils.websocket_handler import websocket_endpoint

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

app = FastAPI()



# Подключение статических файлов
app.mount("/static", StaticFiles(directory="static"), name="static")

# Подключение маршрутов
app.include_router(dashboard_router, prefix="/dashboard", tags=["Dashboard"])
app.include_router(analysis_router, prefix="/analysis", tags=["Analysis"])
app.include_router(trading_router, prefix="/trading", tags=["Trading"])
app.include_router(settings_router, prefix="/settings", tags=["Settings"])


@app.on_event("startup")
async def startup_event():
    """
    Событие при старте приложения.
    Запускает WebSocket и периодическое обновление данных.
    """
    logging.info("Запуск WebSocket и автообновления данных при старте приложения")
    try:
        file_path = "config/active_model.json"
        if not os.path.exists(file_path):
            logging.warning(f"Файл {file_path} отсутствует, создаю с дефолтным содержимым.")
            with open(file_path, "w") as f:
                json.dump({"active_model": None}, f)

        with open(file_path, "r") as f:
            data = f.read().strip()
            if not data:
                logging.warning(f"Файл {file_path} пуст, записываю дефолтное содержимое.")
                with open(file_path, "w") as f:
                    json.dump({"active_model": None}, f)
                active_model = None
            else:
                active_model = json.loads(data).get("active_model")

        if active_model:
            logging.info(f"Активная модель: {active_model}")
        else:
            logging.warning("Активная модель не установлена. Установите её через настройки.")
    except json.JSONDecodeError:
        logging.error(f"Ошибка чтения JSON в {file_path}. Восстанавливаю файл.")
        with open(file_path, "w") as f:
            json.dump({"active_model": None}, f)
    except Exception as e:
        logging.error(f"Неизвестная ошибка при обработке {file_path}: {e}")

    start_websocket()  # Запуск WebSocket
    asyncio.create_task(periodic_update())  # Асинхронное обновление данных

@app.get("/", response_class=FileResponse)
async def read_root():
    """
    Возвращает главную страницу.
    """
    logging.info("Загрузка главной страницы")
    return FileResponse("templates/index.html")

@app.get("/dashboard.html", response_class=FileResponse)
async def dashboard_page():
    """
    Возвращает страницу дашборда.
    """
    logging.info("Загрузка страницы дашборда")
    return FileResponse("templates/dashboard.html")

@app.get("/trading.html", response_class=FileResponse)
async def trading_page():
    """
    Возвращает страницу торговли.
    """
    logging.info("Загрузка страницы торговли")
    return FileResponse("templates/trading.html")

@app.get("/settings.html", response_class=FileResponse)
async def settings_page():
    """
    Возвращает страницу настроек.
    """
    logging.info("Загрузка страницы настроек")
    return FileResponse("templates/settings.html")

@app.websocket("/ws/analysis")
async def analysis_websocket_endpoint(websocket: WebSocket):
    await websocket_endpoint(websocket)

@app.get("/analysis.html", response_class=FileResponse)
async def analysis_page():
    """
    Возвращает страницу анализа.
    """
    logging.info("Загрузка страницы анализа")
    return FileResponse("templates/analysis.html")

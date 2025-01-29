import asyncio
from utils.config import initialize_exchange, connect_to_websocket
from utils.api import  fetch_all_markets, fetch_ticker_data
from utils.plotting import plot_price_data

def main():
    print("Инициализация клиента Bybit...")
    exchange = initialize_exchange()

    print("Загрузка доступных торговых пар...")
    markets = fetch_all_markets()
    if not markets:
        print("Не удалось загрузить торговые пары!")
        return

    print("Доступные пары:", markets.keys())

    # Дополнительная логика работы с рынками
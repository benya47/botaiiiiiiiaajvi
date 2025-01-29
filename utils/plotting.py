import matplotlib.pyplot as plt

def plot_price_data(historical_data):
    # Строим график на основе исторических данных
    df = pd.DataFrame(historical_data)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    
    plt.plot(df['timestamp'], df['close'])
    plt.title('Price Data')
    plt.xlabel('Timestamp')
    plt.ylabel('Close Price')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

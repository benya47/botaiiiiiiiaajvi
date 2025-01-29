import pandas as pd
import ta

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет технические индикаторы к данным"""
    df = df.copy()
    
    # Moving Averages
    df['ma_short'] = df['close'].rolling(5).mean()
    df['ma_long'] = df['close'].rolling(50).mean()
    df['ma_ratio'] = df['ma_short'] / df['ma_long']
    
    # RSI
    df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
    
    # MACD
    macd = ta.trend.MACD(df['close'])
    df['macd_diff'] = macd.macd_diff()
    
    return df

def add_advanced_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет продвинутые индикаторы"""
    df = df.copy()
    
    # ATR
    df['atr'] = ta.volatility.AverageTrueRange(
        high=df['high'], low=df['low'], close=df['close']
    ).average_true_range()
    
    # Bollinger Bands
    bb = ta.volatility.BollingerBands(df['close'])
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()
    
    return df
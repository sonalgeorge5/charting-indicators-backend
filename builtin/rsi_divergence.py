"""
Example: RSI Divergence Detection
"""

def calculate(ohlcv, params):
    """
    RSI with divergence detection
    """
    period = params.get('period', 14)
    overbought = params.get('overbought', 70)
    oversold = params.get('oversold', 30)
    
    # Calculate RSI
    rsi = ta.rsi(ohlcv['close'], period)
    
    # Simple divergence detection (bullish/bearish)
    close = ohlcv['close']
    
    # Bullish divergence: Price makes lower low, RSI makes higher low
    bullish_div = pd.Series(False, index=close.index)
    bearish_div = pd.Series(False, index=close.index)
    
    lookback = 5
    for i in range(lookback, len(close)):
        # Check for lower low in price
        if close.iloc[i] < close.iloc[i-lookback:i].min():
            # Check if RSI made higher low
            if rsi.iloc[i] > rsi.iloc[i-lookback:i].min():
                bullish_div.iloc[i] = True
        
        # Check for higher high in price
        if close.iloc[i] > close.iloc[i-lookback:i].max():
            # Check if RSI made lower high
            if rsi.iloc[i] < rsi.iloc[i-lookback:i].max():
                bearish_div.iloc[i] = True
    
    return {
        'rsi': rsi,
        'overbought_line': pd.Series(overbought, index=close.index),
        'oversold_line': pd.Series(oversold, index=close.index),
        'signals': {
            'bullish_divergence': bullish_div,
            'bearish_divergence': bearish_div,
            'overbought': rsi > overbought,
            'oversold': rsi < oversold
        }
    }


META = {
    'name': 'RSI Divergence',
    'description': 'RSI with bullish/bearish divergence detection',
    'overlay': False,
    'params': {
        'period': {'type': 'int', 'default': 14, 'min': 2, 'max': 50},
        'overbought': {'type': 'int', 'default': 70, 'min': 50, 'max': 90},
        'oversold': {'type': 'int', 'default': 30, 'min': 10, 'max': 50}
    }
}

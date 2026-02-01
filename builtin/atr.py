"""
ATR - Average True Range
Category: Volatility
"""

def calculate(ohlcv, params):
    """
    ATR - Measures market volatility
    """
    period = params.get('period', 14)
    
    high = ohlcv['high']
    low = ohlcv['low']
    close = ohlcv['close']
    
    # Calculate ATR
    atr = ta.atr(high, low, close, period)
    
    return {'atr': atr}


META = {
    'name': 'ATR',
    'description': 'Average True Range - Volatility indicator',
    'overlay': False,
    'category': 'Volatility',
    'params': {
        'period': {'type': 'int', 'default': 14, 'min': 1, 'max': 100, 'label': 'Period'}
    }
}

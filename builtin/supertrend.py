"""
SuperTrend Indicator
Category: Trend
"""

def calculate(ohlcv, params):
    """
    SuperTrend - ATR-based trend following indicator
    """
    period = params.get('period', 10)
    multiplier = params.get('multiplier', 3.0)
    
    high = ohlcv['high']
    low = ohlcv['low']
    close = ohlcv['close']
    
    # Calculate using built-in function
    st = ta.supertrend(high, low, close, period, multiplier)
    
    return {
        'supertrend': st['supertrend'],
        'direction': st['direction'],
        'signals': {
            'buy': st['direction'] == 1,
            'sell': st['direction'] == -1
        }
    }


META = {
    'name': 'SuperTrend',
    'description': 'ATR-based trend-following indicator with buy/sell signals',
    'overlay': True,
    'category': 'Trend',
    'params': {
        'period': {'type': 'int', 'default': 10, 'min': 1, 'max': 50, 'label': 'ATR Period'},
        'multiplier': {'type': 'float', 'default': 3.0, 'min': 0.5, 'max': 10.0, 'step': 0.1, 'label': 'Multiplier'}
    }
}

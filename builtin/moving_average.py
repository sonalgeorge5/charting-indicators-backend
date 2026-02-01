"""
Moving Average (Multiple Types)
Category: Trend
"""

def calculate(ohlcv, params):
    """
    Configurable Moving Average with multiple types
    
    Types: SMA, EMA, WMA, VWMA, SMMA
    """
    ma_type = params.get('type', 'SMA').upper()
    period = params.get('period', 20)
    source = params.get('source', 'close')
    
    data = ohlcv[source]
    
    if ma_type == 'SMA':
        result = ta.sma(data, period)
    elif ma_type == 'EMA':
        result = ta.ema(data, period)
    elif ma_type == 'WMA':
        result = ta.wma(data, period)
    elif ma_type == 'VWMA':
        # Volume Weighted
        tp = (ohlcv['high'] + ohlcv['low'] + ohlcv['close']) / 3
        result = (tp * ohlcv['volume']).rolling(period).sum() / ohlcv['volume'].rolling(period).sum()
    elif ma_type == 'SMMA':
        # Smoothed MA
        result = data.ewm(alpha=1/period, adjust=False).mean()
    else:
        result = ta.sma(data, period)  # Default to SMA
    
    return {'ma': result}


META = {
    'name': 'Moving Average',
    'description': 'Versatile moving average supporting SMA, EMA, WMA, VWMA, SMMA',
    'overlay': True,
    'category': 'Trend',
    'params': {
        'type': {
            'type': 'select',
            'default': 'SMA',
            'options': ['SMA', 'EMA', 'WMA', 'VWMA', 'SMMA'],
            'label': 'MA Type'
        },
        'period': {'type': 'int', 'default': 20, 'min': 1, 'max': 200, 'label': 'Period'},
        'source': {
            'type': 'select',
            'default': 'close',
            'options': ['open', 'high', 'low', 'close', 'hl2', 'hlc3', 'ohlc4'],
            'label': 'Source'
        }
    }
}

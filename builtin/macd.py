"""
MACD - Moving Average Convergence Divergence
Category: Momentum
"""

def calculate(ohlcv, params):
    """
    MACD - Trend-following momentum indicator
    """
    fast = params.get('fast', 12)
    slow = params.get('slow', 26)
    signal = params.get('signal', 9)
    source = params.get('source', 'close')
    
    data = ohlcv[source]
    
    # Calculate MACD
    macd_result = ta.macd(data, fast, slow, signal)
    
    # Zero line
    zero_line = pd.Series(0, index=data.index)
    
    return {
        'macd': macd_result['macd'],
        'signal': macd_result['signal'],
        'histogram': macd_result['histogram'],
        'zero': zero_line
    }


META = {
    'name': 'MACD',
    'description': 'Moving Average Convergence Divergence',
    'overlay': False,
    'category': 'Momentum',
    'params': {
        'fast': {'type': 'int', 'default': 12, 'min': 2, 'max': 50, 'label': 'Fast Period'},
        'slow': {'type': 'int', 'default': 26, 'min': 2, 'max': 100, 'label': 'Slow Period'},
        'signal': {'type': 'int', 'default': 9, 'min': 2, 'max': 50, 'label': 'Signal Period'},
        'source': {
            'type': 'select',
            'default': 'close',
            'options': ['open', 'high', 'low', 'close'],
            'label': 'Source'
        }
    }
}

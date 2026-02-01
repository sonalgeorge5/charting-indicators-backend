"""
RSI - Relative Strength Index
Category: Momentum
"""

def calculate(ohlcv, params):
    """
    RSI - Measures momentum and overbought/oversold conditions
    """
    period = params.get('period', 14)
    overbought = params.get('overbought', 70)
    oversold = params.get('oversold', 30)
    source = params.get('source', 'close')
    
    data = ohlcv[source]
    
    # Calculate RSI
    rsi = ta.rsi(data, period)
    
    # Create overbought/oversold levels as horizontal lines
    ob_line = pd.Series(overbought, index=data.index)
    os_line = pd.Series(oversold, index=data.index)
    midline = pd.Series(50, index=data.index)
    
    return {
        'rsi': rsi,
        'overbought': ob_line,
        'oversold': os_line,
        'midline': midline,
        'signals': {
            'overbought': rsi > overbought,
            'oversold': rsi < oversold
        }
    }


META = {
    'name': 'RSI',
    'description': 'Relative Strength Index - Momentum oscillator',
    'overlay': False,
    'category': 'Momentum',
    'params': {
        'period': {'type': 'int', 'default': 14, 'min': 2, 'max': 100, 'label': 'Period'},
        'overbought': {'type': 'int', 'default': 70, 'min': 50, 'max': 90, 'label': 'Overbought'},
        'oversold': {'type': 'int', 'default': 30, 'min': 10, 'max': 50, 'label': 'Oversold'},
        'source': {
            'type': 'select',
            'default': 'close',
            'options': ['open', 'high', 'low', 'close'],
            'label': 'Source'
        }
    }
}

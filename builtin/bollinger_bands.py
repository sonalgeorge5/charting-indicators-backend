"""
Bollinger Bands
Category: Trend / Volatility
"""

def calculate(ohlcv, params):
    """
    Bollinger Bands - Moving average with standard deviation bands
    """
    period = params.get('period', 20)
    std_dev = params.get('std_dev', 2.0)
    source = params.get('source', 'close')
    
    data = ohlcv[source]
    
    # Calculate using built-in function
    bb = ta.bollinger_bands(data, period, std_dev)
    
    return {
        'upper': bb['upper'],
        'middle': bb['middle'],
        'lower': bb['lower']
    }


META = {
    'name': 'Bollinger Bands',
    'description': 'Volatility bands using standard deviation',
    'overlay': True,
    'category': 'Trend',
    'params': {
        'period': {'type': 'int', 'default': 20, 'min': 2, 'max': 200, 'label': 'Period'},
        'std_dev': {'type': 'float', 'default': 2.0, 'min': 0.5, 'max': 5.0, 'step': 0.1, 'label': 'Std Dev'},
        'source': {
            'type': 'select',
            'default': 'close',
            'options': ['open', 'high', 'low', 'close', 'hl2', 'hlc3'],
            'label': 'Source'
        }
    }
}

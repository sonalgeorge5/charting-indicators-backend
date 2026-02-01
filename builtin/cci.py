"""
CCI - Commodity Channel Index
Category: Momentum
"""

def calculate(ohlcv, params):
    """
    CCI - Identifies cyclical trends
    """
    period = params.get('period', 20)
    overbought = params.get('overbought', 100)
    oversold = params.get('oversold', -100)
    
    high = ohlcv['high']
    low = ohlcv['low']
    close = ohlcv['close']
    
    # Calculate CCI
    cci = ta.cci(high, low, close, period)
    
    # Levels
    ob_line = pd.Series(overbought, index=close.index)
    os_line = pd.Series(oversold, index=close.index)
    zero_line = pd.Series(0, index=close.index)
    
    return {
        'cci': cci,
        'overbought': ob_line,
        'oversold': os_line,
        'zero': zero_line,
        'signals': {
            'overbought': cci > overbought,
            'oversold': cci < oversold
        }
    }


META = {
    'name': 'CCI',
    'description': 'Commodity Channel Index',
    'overlay': False,
    'category': 'Momentum',
    'params': {
        'period': {'type': 'int', 'default': 20, 'min': 2, 'max': 200, 'label': 'Period'},
        'overbought': {'type': 'int', 'default': 100, 'min': 50, 'max': 300, 'label': 'Overbought'},
        'oversold': {'type': 'int', 'default': -100, 'min': -300, 'max': -50, 'label': 'Oversold'}
    }
}

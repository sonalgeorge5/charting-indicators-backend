"""
Williams %R
Category: Momentum
"""

def calculate(ohlcv, params):
    """
    Williams %R - Momentum indicator showing overbought/oversold
    """
    period = params.get('period', 14)
    overbought = params.get('overbought', -20)
    oversold = params.get('oversold', -80)
    
    high = ohlcv['high']
    low = ohlcv['low']
    close = ohlcv['close']
    
    # Calculate Williams %R
    wr = ta.williams_r(high, low, close, period)
    
    # Levels
    ob_line = pd.Series(overbought, index=close.index)
    os_line = pd.Series(oversold, index=close.index)
    midline = pd.Series(-50, index=close.index)
    
    return {
        'williams_r': wr,
        'overbought': ob_line,
        'oversold': os_line,
        'midline': midline,
        'signals': {
            'overbought': wr > overbought,
            'oversold': wr < oversold
        }
    }


META = {
    'name': 'Williams %R',
    'description': 'Williams Percent Range - Momentum oscillator',
    'overlay': False,
    'category': 'Momentum',
    'params': {
        'period': {'type': 'int', 'default': 14, 'min': 2, 'max': 100, 'label': 'Period'},
        'overbought': {'type': 'int', 'default': -20, 'min': -50, 'max': 0, 'label': 'Overbought'},
        'oversold': {'type': 'int', 'default': -80, 'min': -100, 'max': -50, 'label': 'Oversold'}
    }
}

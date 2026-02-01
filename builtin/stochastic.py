"""
Stochastic Oscillator
Category: Momentum
"""

def calculate(ohlcv, params):
    """
    Stochastic Oscillator - Momentum indicator comparing close to price range
    """
    k_period = params.get('k_period', 14)
    d_period = params.get('d_period', 3)
    smooth_k = params.get('smooth_k', 3)
    overbought = params.get('overbought', 80)
    oversold = params.get('oversold', 20)
    
    high = ohlcv['high']
    low = ohlcv['low']
    close = ohlcv['close']
    
    # Calculate Stochastic
    stoch = ta.stochastic(high, low, close, k_period, d_period)
    
    # Smooth %K if requested
    k_line = stoch['k'] if smooth_k == 1 else ta.sma(stoch['k'], smooth_k)
    
    # Levels
    ob_line = pd.Series(overbought, index=close.index)
    os_line = pd.Series(oversold, index=close.index)
    midline = pd.Series(50, index=close.index)
    
    return {
        'k': k_line,
        'd': stoch['d'],
        'overbought': ob_line,
        'oversold': os_line,
        'midline': midline,
        'signals': {
            'overbought': k_line > overbought,
            'oversold': k_line < oversold
        }
    }


META = {
    'name': 'Stochastic',
    'description': 'Stochastic Oscillator - %K and %D lines',
    'overlay': False,
    'category': 'Momentum',
    'params': {
        'k_period': {'type': 'int', 'default': 14, 'min': 1, 'max': 100, 'label': '%K Period'},
        'd_period': {'type': 'int', 'default': 3, 'min': 1, 'max': 50, 'label': '%D Period'},
        'smooth_k': {'type': 'int', 'default': 3, 'min': 1, 'max': 20, 'label': 'Smooth %K'},
        'overbought': {'type': 'int', 'default': 80, 'min': 50, 'max': 100, 'label': 'Overbought'},
        'oversold': {'type': 'int', 'default': 20, 'min': 0, 'max': 50, 'label': 'Oversold'}
    }
}

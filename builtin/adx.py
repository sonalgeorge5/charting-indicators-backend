"""
ADX - Average Directional Index
Category: Trend
"""

def calculate(ohlcv, params):
    """
    ADX - Measures trend strength
    """
    period = params.get('period', 14)
    show_di = params.get('show_di', True)
    
    high = ohlcv['high']
    low = ohlcv['low']
    close = ohlcv['close']
    
    # Calculate ADX
    adx_result = ta.adx(high, low, close, period)
    
    result = {'adx': adx_result['adx']}
    
    if show_di:
        result['plus_di'] = adx_result['plus_di']
        result['minus_di'] = adx_result['minus_di']
    
    # Add threshold line
    result['threshold'] = pd.Series(25, index=close.index)
    
    return result


META = {
    'name': 'ADX',
    'description': 'Average Directional Index - Trend strength indicator',
    'overlay': False,
    'category': 'Trend',
    'params': {
        'period': {'type': 'int', 'default': 14, 'min': 2, 'max': 100, 'label': 'Period'},
        'show_di': {'type': 'bool', 'default': True, 'label': 'Show +DI/-DI'}
    }
}

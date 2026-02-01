"""
OBV - On Balance Volume
Category: Volume
"""

def calculate(ohlcv, params):
    """
    OBV - Cumulative volume indicator
    """
    show_ma = params.get('show_ma', False)
    ma_period = params.get('ma_period', 20)
    
    close = ohlcv['close']
    volume = ohlcv['volume']
    
    # Calculate OBV
    obv = ta.obv(close, volume)
    
    result = {'obv': obv}
    
    if show_ma:
        obv_ma = ta.sma(obv, ma_period)
        result['obv_ma'] = obv_ma
    
    return result


META = {
    'name': 'OBV',
    'description': 'On Balance Volume - Momentum indicator using volume',
    'overlay': False,
    'category': 'Volume',
    'params': {
        'show_ma': {'type': 'bool', 'default': False, 'label': 'Show MA'},
        'ma_period': {'type': 'int', 'default': 20, 'min': 1, 'max': 100, 'label': 'MA Period'}
    }
}

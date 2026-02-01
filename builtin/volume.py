"""
Volume Indicator
Category: Volume
"""

def calculate(ohlcv, params):
    """
    Volume with optional moving average
    """
    show_ma = params.get('show_ma', True)
    ma_period = params.get('ma_period', 20)
    
    volume = ohlcv['volume']
    
    result = {'volume': volume}
    
    if show_ma:
        volume_ma = ta.sma(volume, ma_period)
        result['volume_ma'] = volume_ma
    
    return result


META = {
    'name': 'Volume',
    'description': 'Trading volume with optional moving average',
    'overlay': False,
    'category': 'Volume',
    'params': {
        'show_ma': {'type': 'bool', 'default': True, 'label': 'Show MA'},
        'ma_period': {'type': 'int', 'default': 20, 'min': 1, 'max': 200, 'label': 'MA Period'}
    }
}

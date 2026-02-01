"""
Ichimoku Cloud
Category: Trend
"""

def calculate(ohlcv, params):
    """
    Ichimoku Cloud - Complete trend-following system
    """
    tenkan = params.get('tenkan', 9)
    kijun = params.get('kijun', 26)
    senkou_b = params.get('senkou_b', 52)
    
    high = ohlcv['high']
    low = ohlcv['low']
    close = ohlcv['close']
    
    # Calculate Ichimoku
    ichi = ta.ichimoku(high, low, close, tenkan, kijun, senkou_b)
    
    return {
        'tenkan_sen': ichi['tenkan'],
        'kijun_sen': ichi['kijun'],
        'senkou_span_a': ichi['senkou_a'],
        'senkou_span_b': ichi['senkou_b'],
        'chikou_span': ichi['chikou']
    }


META = {
    'name': 'Ichimoku Cloud',
    'description': 'Ichimoku Kinko Hyo - Complete trend system',
    'overlay': True,
    'category': 'Trend',
    'params': {
        'tenkan': {'type': 'int', 'default': 9, 'min': 1, 'max': 50, 'label': 'Tenkan Period'},
        'kijun': {'type': 'int', 'default': 26, 'min': 1, 'max': 100, 'label': 'Kijun Period'},
        'senkou_b': {'type': 'int', 'default': 52, 'min': 1, 'max': 200, 'label': 'Senkou B Period'}
    }
}

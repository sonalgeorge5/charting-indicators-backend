"""
MFI - Money Flow Index
Category: Volume
"""

def calculate(ohlcv, params):
    """
    MFI - Volume-weighted RSI
    """
    period = params.get('period', 14)
    overbought = params.get('overbought', 80)
    oversold = params.get('oversold', 20)
    
    high = ohlcv['high']
    low = ohlcv['low']
    close = ohlcv['close']
    volume = ohlcv['volume']
    
    # Calculate MFI
    mfi = ta.mfi(high, low, close, volume, period)
    
    # Levels
    ob_line = pd.Series(overbought, index=close.index)
    os_line = pd.Series(oversold, index=close.index)
    midline = pd.Series(50, index=close.index)
    
    return {
        'mfi': mfi,
        'overbought': ob_line,
        'oversold': os_line,
        'midline': midline,
        'signals': {
            'overbought': mfi > overbought,
            'oversold': mfi < oversold
        }
    }


META = {
    'name': 'MFI',
    'description': 'Money Flow Index - Volume-weighted momentum',
    'overlay': False,
    'category': 'Volume',
    'params': {
        'period': {'type': 'int', 'default': 14, 'min': 2, 'max': 100, 'label': 'Period'},
        'overbought': {'type': 'int', 'default': 80, 'min': 50, 'max': 100, 'label': 'Overbought'},
        'oversold': {'type': 'int', 'default': 20, 'min': 0, 'max': 50, 'label': 'Oversold'}
    }
}

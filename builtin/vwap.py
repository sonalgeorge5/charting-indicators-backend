"""
VWAP - Volume Weighted Average Price
Category: Volume
"""

def calculate(ohlcv, params):
    """
    VWAP - Volume-weighted average of price
    """
    reset_period = params.get('reset_period', 'daily')
    
    high = ohlcv['high']
    low = ohlcv['low']
    close = ohlcv['close']
    volume = ohlcv['volume']
    
    # Calculate VWAP
    vwap = ta.vwap(high, low, close, volume)
    
    return {'vwap': vwap}


META = {
    'name': 'VWAP',
    'description': 'Volume Weighted Average Price',
    'overlay': True,
    'category': 'Volume',
    'params': {
        'reset_period': {
            'type': 'select',
            'default': 'daily',
            'options': ['daily', 'weekly', 'monthly', 'session'],
            'label': 'Reset Period'
        }
    }
}

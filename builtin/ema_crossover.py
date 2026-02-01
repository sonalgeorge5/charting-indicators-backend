"""
Example: EMA Crossover Strategy
Place this in ~/CryptoChartPro/scripts/ema_crossover.py
"""

def calculate(ohlcv, params):
    """
    EMA Crossover with buy/sell signals
    
    Args:
        ohlcv: DataFrame with OHLCV data
        params: User parameters
    """
    fast_period = params.get('fast', 9)
    slow_period = params.get('slow', 21)
    
    # Calculate EMAs using built-in ta functions
    fast_ema = ta.ema(ohlcv['close'], fast_period)
    slow_ema = ta.ema(ohlcv['close'], slow_period)
    
    # Detect crossovers
    cross_up = (fast_ema > slow_ema) & (fast_ema.shift(1) <= slow_ema.shift(1))
    cross_down = (fast_ema < slow_ema) & (fast_ema.shift(1) >= slow_ema.shift(1))
    
    return {
        'fast_ema': fast_ema,
        'slow_ema': slow_ema,
        'signals': {
            'buy': cross_up,
            'sell': cross_down
        }
    }


META = {
    'name': 'EMA Crossover',
    'description': 'Fast/Slow EMA crossover with entry signals',
    'overlay': True,
    'params': {
        'fast': {'type': 'int', 'default': 9, 'min': 1, 'max': 100, 'label': 'Fast EMA'},
        'slow': {'type': 'int', 'default': 21, 'min': 1, 'max': 200, 'label': 'Slow EMA'}
    }
}

"""
Pivot Points
Category: Market Structure
"""

def calculate(ohlcv, params):
    """
    Pivot Points - Support and resistance levels
    """
    pivot_type = params.get('type', 'Standard')
    
    high = ohlcv['high']
    low = ohlcv['low']
    close = ohlcv['close']
    
    # Use previous period's data for pivot calculation
    prev_high = high.iloc[-1] if len(high) > 0 else 0
    prev_low = low.iloc[-1] if len(low) > 0 else 0
    prev_close = close.iloc[-1] if len(close) > 0 else 0
    
    # Calculate pivot points
    pivots = ta.pivot_points(prev_high, prev_low, prev_close)
    
    # Create series for each level
    index = close.index
    
    return {
        'pivot': pd.Series(pivots['pivot'], index=index),
        'r1': pd.Series(pivots['r1'], index=index),
        'r2': pd.Series(pivots['r2'], index=index),
        'r3': pd.Series(pivots['r3'], index=index),
        's1': pd.Series(pivots['s1'], index=index),
        's2': pd.Series(pivots['s2'], index=index),
        's3': pd.Series(pivots['s3'], index=index)
    }


META = {
    'name': 'Pivot Points',
    'description': 'Support and resistance pivot levels',
    'overlay': True,
    'category': 'Market Structure',
    'params': {
        'type': {
            'type': 'select',
            'default': 'Standard',
            'options': ['Standard', 'Fibonacci', 'Woodie', 'Camarilla'],
            'label': 'Type'
        }
    }
}

# 🐍 CryptoChart Pro - Python Indicator Backend

**Production-ready Python backend for technical indicators**

This is a FastAPI server that executes technical indicators written in pure Python (NOT Pine Script), designed to integrate with the CryptoChart Pro frontend.

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start Server
```bash
python server.py
```

Server runs on: `http://localhost:8765`

### 2.5 Optional: Enable Telegram EMA 55/100/200 Alerts
Set these environment variables before starting `server.py`:

```bash
export ALERTS_ENABLED=true
export TELEGRAM_BOT_TOKEN=your_bot_token
export TELEGRAM_CHAT_ID=your_chat_id
export ALERT_TIMEFRAME=30m
export ALERT_HTF_TIMEFRAME=4h
export ALERT_DIRECTION=both
```

Optional tuning:

```bash
export ALERT_SCAN_SECONDS=180
export ALERT_COOLDOWN_SECONDS=21600
export ALERT_RETEST_TOLERANCE_PCT=0.001
export ALERT_MIN_ADX_LONG=30
export ALERT_MIN_ADX_SHORT=25
export ALERT_MIN_EMA_SPREAD_PCT=0.01

# Universe mode:
# top_marketcap_binance = auto Top-N market cap coins listed on Binance (default)
# static = use ALERT_SYMBOLS list directly
export ALERT_UNIVERSE_MODE=top_marketcap_binance
export ALERT_TOP_N=100
export ALERT_EXCLUDE_STABLECOINS=true
export ALERT_QUOTE_ASSET=USDT
export ALERT_UNIVERSE_REFRESH_SECONDS=21600

# Only for static mode:
export ALERT_SYMBOLS=BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT
```

### 3. Test API
Visit: `http://localhost:8765/docs` for interactive API documentation

---

## 📦 What's Included

### ✅ 18 Built-in Indicators

#### Trend (6)
- Moving Average (SMA/EMA/WMA/VWMA/SMMA)
- Bollinger Bands
- SuperTrend
- Ichimoku Cloud
- ADX (Average Directional Index)
- Pivot Points

#### Momentum (5)
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Stochastic Oscillator
- CCI (Commodity Channel Index)
- Williams %R

#### Volume (4)
- Volume (with MA)
- VWAP (Volume Weighted Average Price)
- OBV (On Balance Volume)
- MFI (Money Flow Index)

#### Volatility (1)
- ATR (Average True Range)

#### Strategy Examples (2)
- EMA Crossover
- RSI Divergence

---

## 🔌 API Endpoints

### `GET /`
Health check

**Response:**
```json
{
  "status": "running",
  "service": "CryptoChart Pro Indicator Server"
}
```

---

### `GET /alerts/status`
Returns scanner/telegram worker status and runtime config.

### `POST /alerts/scan-now`
Runs one immediate scan cycle across configured symbols.

### `POST /alerts/test-telegram`
Sends a test Telegram message using configured bot/chat credentials.

---

### `GET /scripts`
List all available indicators

**Response:**
```json
[
  {
    "name": "RSI",
    "path": "/path/to/rsi.py",
    "description": "Relative Strength Index",
    "params": {
      "period": {"type": "int", "default": 14, "min": 2, "max": 100}
    },
    "overlay": false,
    "category": "Momentum"
  }
]
```

---

### `POST /execute`
Execute an indicator

**Request:**
```json
{
  "script_name": "rsi",
  "ohlcv": {
    "time": [1234567890, 1234567900, ...],
    "open": [100.0, 101.5, ...],
    "high": [102.0, 103.0, ...],
    "low": [99.5, 100.0, ...],
    "close": [101.0, 102.5, ...],
    "volume": [1000.0, 1200.0, ...]
  },
  "params": {
    "period": 14,
    "overbought": 70,
    "oversold": 30
  }
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "rsi": [null, null, ..., 67.32],
    "overbought": [70, 70, ...],
    "oversold": [30, 30, ...]
  },
  "signals": {
    "overbought": [false, false, ...],
    "oversold": [false, false, ...]
  },
  "overlay": false
}
```

---

### `POST /validate`
Validate Python indicator code

**Request:**
```json
{
  "code": "def calculate(ohlcv, params):\n    return {'value': ohlcv['close']}"
}
```

**Response:**
```json
{
  "valid": true
}
```

---

### `GET /templates`
Get starter code templates

**Response:**
```json
{
  "basic": "...",
  "crossover": "...",
  "momentum": "..."
}
```

---

## 🧠 Built-in TA Functions

All indicators have access to the `ta` helper object:

### Moving Averages
```python
ta.sma(data, period)      # Simple MA
ta.ema(data, period)      # Exponential MA
ta.wma(data, period)      # Weighted MA
```

### Oscillators
```python
ta.rsi(data, period)
ta.macd(data, fast, slow, signal)
ta.stochastic(high, low, close, k_period, d_period)
ta.cci(high, low, close, period)
ta.williams_r(high, low, close, period)
```

### Trend
```python
ta.bollinger_bands(data, period, std_dev)
ta.supertrend(high, low, close, period, multiplier)
ta.ichimoku(high, low, close, tenkan, kijun, senkou_b)
ta.adx(high, low, close, period)
```

### Volume
```python
ta.vwap(high, low, close, volume)
ta.obv(close, volume)
ta.mfi(high, low, close, volume, period)
```

### Volatility
```python
ta.atr(high, low, close, period)
```

### Support/Resistance
```python
ta.pivot_points(high, low, close)
```

---

## 📝 Creating Custom Indicators

### 1. Create a Python file
`~/CryptoChartPro/scripts/my_indicator.py`

### 2. Write the indicator
```python
"""
My Custom Indicator
Category: Trend
"""

def calculate(ohlcv, params):
    """
    Main calculation function
    
    Args:
        ohlcv: DataFrame with [time, open, high, low, close, volume]
        params: Dict of user parameters
    
    Returns:
        Dict of indicator values
    """
    period = params.get('period', 20)
    close = ohlcv['close']
    
    # Calculate using built-in TA functions
    ma = ta.sma(close, period)
    upper = ma * 1.1
    lower = ma * 0.9
    
    return {
        'ma': ma,
        'upper': upper,
        'lower': lower
    }


# Metadata (defines UI settings)
META = {
    'name': 'My Custom Indicator',
    'description': 'Moving average with bands',
    'overlay': True,  # True = on chart, False = separate pane
    'category': 'Trend',
    'params': {
        'period': {
            'type': 'int',
            'default': 20,
            'min': 1,
            'max': 200,
            'label': 'Period'
        }
    }
}
```

### 3. Use it
```bash
curl -X POST http://localhost:8765/execute \
  -H "Content-Type: application/json" \
  -d '{
    "script_name": "my_indicator",
    "ohlcv": { ... },
    "params": { "period": 20 }
  }'
```

---

## 🎨 Indicator Structure

### Mandatory Components

1. **`calculate()` function**
   - Accepts `ohlcv` DataFrame and `params` dict
   - Returns dict of series or single series

2. **`META` dictionary** (optional but recommended)
   - `name`: Display name
   - `description`: Short description
   - `overlay`: True/False (on chart or separate pane)
   - `category`: Trend/Momentum/Volume/Volatility
   - `params`: Parameter definitions

### Parameter Types
```python
'params': {
    'period': {
        'type': 'int',      # int, float, bool, select
        'default': 14,
        'min': 1,
        'max': 200,
        'label': 'Period'
    },
    'ma_type': {
        'type': 'select',
        'default': 'SMA',
        'options': ['SMA', 'EMA', 'WMA'],
        'label': 'MA Type'
    },
    'show_bands': {
        'type': 'bool',
        'default': True,
        'label': 'Show Bands'
    }
}
```

---

## 🔧 Technical Details

### Dependencies
```txt
fastapi>=0.104.0
pandas>=2.0.0
numpy>=1.24.0
uvicorn>=0.24.0
pydantic>=2.0.0
```

### Performance
- **Vectorized operations** using Pandas/NumPy
- Supports **100,000+ bars** efficiently
- Real-time capable (<100ms per indicator)

### Data Format
- OHLCV data as Pandas DataFrame
- Timestamps in Unix epoch (milliseconds)
- Float64 for prices, Int64 for volume

### Error Handling
- Graceful handling of NaN values
- Validation of input data
- Detailed error messages

---

## 📚 Documentation

- **Full Guide**: `INDICATORS_GUIDE.md`
- **Quick Reference**: `QUICK_REFERENCE.md`
- **Project Overview**: `../PROJECT_OVERVIEW.md`
- **Master Prompt**: `../MASTER_AI_PROMPT.md`

---

## 🧪 Testing

### Manual Test
```python
import pandas as pd

# Sample OHLCV data
ohlcv = pd.DataFrame({
    'time': [1, 2, 3, 4, 5],
    'open': [100, 101, 102, 103, 104],
    'high': [102, 103, 104, 105, 106],
    'low': [99, 100, 101, 102, 103],
    'close': [101, 102, 103, 104, 105],
    'volume': [1000, 1100, 1200, 1300, 1400]
})

# Import TA functions
from server import ta

# Calculate RSI
rsi = ta.rsi(ohlcv['close'], 14)
print(rsi)
```

### API Test
```bash
# List indicators
curl http://localhost:8765/scripts

# Execute RSI
curl -X POST http://localhost:8765/execute \
  -H "Content-Type: application/json" \
  -d @test_request.json
```

---

## 🚀 Deployment

### Development
```bash
python server.py
# or
uvicorn server:app --reload --port 8765
```

### Production
```bash
uvicorn server:app --host 0.0.0.0 --port 8765 --workers 4
```

### Docker
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8765"]
```

---

## ⚠️ Important Notes

### ❌ What NOT to do
- **Don't use Pine Script** - This is pure Python
- **Don't use loops** - Use vectorized operations
- **Don't modify OHLCV data** - Treat as read-only

### ✅ What TO do
- **Use Pandas/NumPy** for calculations
- **Handle NaN values** gracefully
- **Validate inputs** in params
- **Document your code** with docstrings

---

## 🐛 Troubleshooting

### Issue: Import errors
```bash
# Solution: Install dependencies
pip install -r requirements.txt
```

### Issue: Port already in use
```bash
# Solution: Change port
uvicorn server:app --port 8766
```

### Issue: Indicator not found
```bash
# Solution: Check script name and location
# User scripts go in: ~/CryptoChartPro/scripts/
# Built-in scripts are in: builtin/
```

---

## 📄 License

MIT License - See main project for details

---

## 🤝 Contributing

1. Create indicator in `builtin/your_indicator.py`
2. Follow existing structure (see templates)
3. Test with sample data
4. Update documentation

---

## 📞 Support

- **Documentation**: See `.md` files in this directory
- **API Docs**: `http://localhost:8765/docs`
- **GitHub Issues**: (add your repo link)

---

**Last Updated**: 2026-01-21  
**Version**: 1.0.0  
**Status**: Production Ready ✅
# Redeploy trigger Thu Feb 12 01:34:37 AM IST 2026

# 📊 Quick Reference: All Available Indicators

## 🎯 TREND INDICATORS (6)

| Indicator | File | Overlay | Key Params |
|-----------|------|---------|------------|
| **Moving Average** | `moving_average.py` | ✅ Yes | `type` (SMA/EMA/WMA/VWMA/SMMA), `period`, `source` |
| **Bollinger Bands** | `bollinger_bands.py` | ✅ Yes | `period`, `std_dev`, `source` |
| **SuperTrend** | `supertrend.py` | ✅ Yes | `period`, `multiplier` |
| **Ichimoku Cloud** | `ichimoku.py` | ✅ Yes | `tenkan`, `kijun`, `senkou_b` |
| **ADX** | `adx.py` | ❌ No | `period`, `show_di` |
| **Pivot Points** | `pivot_points.py` | ✅ Yes | `type` (Standard/Fibonacci/Woodie/Camarilla) |

---

## 📈 MOMENTUM INDICATORS (6)

| Indicator | File | Overlay | Key Params |
|-----------|------|---------|------------|
| **RSI** | `rsi.py` | ❌ No | `period`, `overbought`, `oversold`, `source` |
| **MACD** | `macd.py` | ❌ No | `fast`, `slow`, `signal`, `source` |
| **Stochastic** | `stochastic.py` | ❌ No | `k_period`, `d_period`, `smooth_k`, `overbought`, `oversold` |
| **CCI** | `cci.py` | ❌ No | `period`, `overbought`, `oversold` |
| **Williams %R** | `williams_r.py` | ❌ No | `period`, `overbought`, `oversold` |

---

## 💧 VOLUME INDICATORS (4)

| Indicator | File | Overlay | Key Params |
|-----------|------|---------|------------|
| **Volume** | `volume.py` | ❌ No | `show_ma`, `ma_period` |
| **VWAP** | `vwap.py` | ✅ Yes | `reset_period` |
| **OBV** | `obv.py` | ❌ No | `show_ma`, `ma_period` |
| **MFI** | `mfi.py` | ❌ No | `period`, `overbought`, `oversold` |

---

## 📊 VOLATILITY INDICATORS (1)

| Indicator | File | Overlay | Key Params |
|-----------|------|---------|------------|
| **ATR** | `atr.py` | ❌ No | `period` |

---

## 🎯 STRATEGY EXAMPLES (2)

| Strategy | File | Overlay | Description |
|----------|------|---------|-------------|
| **EMA Crossover** | `ema_crossover.py` | ✅ Yes | Fast/slow EMA with buy/sell signals |
| **RSI Divergence** | `rsi_divergence.py` | ❌ No | Bullish/bearish divergence detection |

---

## 🔧 Usage Examples

### 1. RSI (Basic Oscillator)
```python
# API Request
{
  "script_name": "rsi",
  "ohlcv": { ... },
  "params": {
    "period": 14,
    "overbought": 70,
    "oversold": 30
  }
}

# Response
{
  "success": true,
  "data": {
    "rsi": [NaN, NaN, ..., 67.32],
    "overbought": [70, 70, ...],
    "oversold": [30, 30, ...]
  },
  "signals": {
    "overbought": [false, false, ..., false],
    "oversold": [false, false, ..., false]
  },
  "overlay": false
}
```

---

### 2. Moving Average (Overlay Indicator)
```python
# API Request
{
  "script_name": "moving_average",
  "ohlcv": { ... },
  "params": {
    "type": "EMA",
    "period": 20,
    "source": "close"
  }
}

# Response
{
  "success": true,
  "data": {
    "ma": [NaN, ..., 52.34]
  },
  "overlay": true
}
```

---

### 3. MACD (Multiple Lines)
```python
# API Request
{
  "script_name": "macd",
  "ohlcv": { ... },
  "params": {
    "fast": 12,
    "slow": 26,
    "signal": 9
  }
}

# Response
{
  "success": true,
  "data": {
    "macd": [...],
    "signal": [...],
    "histogram": [...],
    "zero": [0, 0, ...]
  },
  "overlay": false
}
```

---

### 4. SuperTrend (With Signals)
```python
# API Request
{
  "script_name": "supertrend",
  "ohlcv": { ... },
  "params": {
    "period": 10,
    "multiplier": 3.0
  }
}

# Response
{
  "success": true,
  "data": {
    "supertrend": [...],
    "direction": [1, 1, -1, -1, ...]
  },
  "signals": {
    "buy": [false, false, true, false, ...],
    "sell": [false, false, false, true, ...]
  },
  "overlay": true
}
```

---

## 🎨 Color Coding Guide

### Overlay Indicators (on price chart)
- 🔵 **Moving Averages**: Blue, Green, Purple
- 🟠 **Bollinger Bands**: Orange (upper/lower), Blue (middle)
- 🔴 **SuperTrend**: Green (bullish), Red (bearish)
- 🟣 **Ichimoku**: Multi-color (Tenkan: red, Kijun: blue, Cloud: green/red)
- 💛 **VWAP**: Yellow/Gold

### Separate Pane Indicators
- 📊 **RSI**: Blue line, red/green zones
- 📈 **MACD**: Blue (MACD), Orange (Signal), Gray histogram
- 📉 **Stochastic**: Blue (%K), Red (%D)
- 🌊 **Volume**: Green (up), Red (down)

---

## ⚙️ Default Parameters

| Indicator | Default Period | Default Levels |
|-----------|----------------|----------------|
| RSI | 14 | 70 (OB), 30 (OS) |
| MACD | 12, 26, 9 | 0 (zero line) |
| Stochastic | 14, 3 | 80 (OB), 20 (OS) |
| CCI | 20 | 100 (OB), -100 (OS) |
| Williams %R | 14 | -20 (OB), -80 (OS) |
| MFI | 14 | 80 (OB), 20 (OS) |
| ATR | 14 | - |
| Bollinger | 20, 2σ | - |
| SuperTrend | 10, 3x | - |
| ADX | 14 | 25 (threshold) |

---

## 🔍 Finding the Right Indicator

### For Trend Direction
- ✅ Moving Average
- ✅ SuperTrend
- ✅ ADX
- ✅ Ichimoku Cloud

### For Entry/Exit Signals
- ✅ RSI
- ✅ MACD
- ✅ Stochastic
- ✅ EMA Crossover

### For Volatility
- ✅ ATR
- ✅ Bollinger Bands

### For Volume Analysis
- ✅ Volume
- ✅ VWAP
- ✅ OBV
- ✅ MFI

### For Support/Resistance
- ✅ Pivot Points
- ✅ Bollinger Bands

---

## 🚀 Performance Tips

1. **Use EMA over SMA** for faster response
2. **Combine trend + momentum** for confirmation
3. **Add volume** for validation
4. **Use multiple timeframes** for context
5. **Limit to 5-7 indicators** for clarity

---

## 📝 Common Combinations

### Scalping (1m - 5m)
- EMA(9/21)
- RSI(14)
- Volume

### Day Trading (15m - 1h)
- Moving Averages (20/50/200)
- MACD(12,26,9)
- SuperTrend
- Volume

### Swing Trading (4h - 1D)
- Bollinger Bands(20,2)
- RSI(14)
- ADX(14)
- Pivot Points

### Position Trading (1D - 1W)
- Moving Averages (50/100/200)
- Ichimoku Cloud
- VWAP
- Volume

---

**Last Updated**: 2026-01-21
**Total Indicators**: 18 (16 built-in + 2 strategies)

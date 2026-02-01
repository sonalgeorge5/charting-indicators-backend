# TradingView-Style Indicators System - Python Implementation

**Complete indicator system using Python (NOT Pine Script)**

This implementation provides a TradingView-accurate indicator system built entirely in Python, designed to be integrated with CryptoChart Pro's charting engine.

## 📋 Overview

All indicators follow the same Python structure:
- Accept OHLCV data as Pandas DataFrames
- Return calculated series (lines, bands, histograms)
- Independent, reusable, and modular
- Can overlay on price chart OR appear in separate panes
- Support TradingView-exact settings panels

---

## 🎯 Indicator Categories

### 1. TREND INDICATORS

**Purpose**: Define direction, support, resistance, and channels

**Available Indicators**:
- ✅ **Moving Average** (SMA, EMA, WMA, VWMA, SMMA)
- ✅ **Bollinger Bands**
- ✅ **SuperTrend**
- ✅ **Ichimoku Cloud**
- ✅ **ADX** (Average Directional Index)
- ⬜ Parabolic SAR
- ⬜ Donchian Channels
- ⬜ Keltner Channels

**Behavior**:
- Overlay on main price chart
- Update in real-time
- Anchored to price movements

---

### 2. MOMENTUM INDICATORS

**Purpose**: Measure speed and strength of price movements

**Available Indicators**:
- ✅ **RSI** (Relative Strength Index)
- ✅ **MACD** (Moving Average Convergence Divergence)
- ✅ **Stochastic Oscillator**
- ✅ **CCI** (Commodity Channel Index)
- ✅ **Williams %R**
- ⬜ Stochastic RSI
- ⬜ ROC (Rate of Change)
- ⬜ Momentum
- ⬜ Awesome Oscillator

**Behavior**:
- Appear in separate indicator pane below chart
- Include overbought/oversold levels
- Generate signals (buy/sell)

---

### 3. VOLUME INDICATORS

**Purpose**: Analyze trading volume and money flow

**Available Indicators**:
- ✅ **Volume** (with MA overlay)
- ✅ **VWAP** (Volume Weighted Average Price)
- ✅ **OBV** (On Balance Volume)
- ✅ **MFI** (Money Flow Index)
- ⬜ Accumulation/Distribution
- ⬜ Chaikin Money Flow

**Behavior**:
- VWAP overlays on price chart
- Volume/OBV/MFI shown in separate panes
- Volume bars color-coded (green = up, red = down)

---

### 4. VOLATILITY INDICATORS

**Purpose**: Measure market volatility and risk

**Available Indicators**:
- ✅ **ATR** (Average True Range)
- ✅ **Bollinger Bands** (also classified as Trend)
- ⬜ Standard Deviation
- ⬜ Choppiness Index

**Behavior**:
- ATR shown in separate pane
- Bollinger Bands overlay on price
- Bands expand/contract with volatility

---

### 5. MARKET STRUCTURE

**Purpose**: Identify key levels and patterns

**Available Indicators**:
- ✅ **Pivot Points** (Standard, Fibonacci, Woodie, Camarilla)
- ⬜ ZigZag

**Behavior**:
- Horizontal support/resistance lines
- Updated per period (daily, weekly, etc.)
- Overlay on price chart

---

## 📐 Indicator Structure (Python)

Each indicator follows this exact format:

```python
"""
Indicator Name
Category: Trend/Momentum/Volume/Volatility/Market Structure
"""

def calculate(ohlcv, params):
    """
    Main calculation function
    
    Args:
        ohlcv: DataFrame with columns [time, open, high, low, close, volume]
        params: Dict of user-configurable parameters
    
    Returns:
        Dict of indicator values (series, signals, etc.)
    """
    # 1. Extract parameters
    period = params.get('period', 14)
    
    # 2. Get data
    close = ohlcv['close']
    
    # 3. Calculate using built-in TA functions
    result = ta.sma(close, period)
    
    # 4. Return data
    return {'sma': result}


# Metadata (defines UI settings)
META = {
    'name': 'Indicator Name',
    'description': 'Brief description',
    'overlay': True,  # True = on chart, False = separate pane
    'category': 'Trend',
    'params': {
        'period': {
            'type': 'int',
            'default': 14,
            'min': 1,
            'max': 200,
            'label': 'Period'
        }
    }
}
```

---

## 🎛️ Settings Panel Structure

Each indicator exposes a settings UI with **3 tabs** (TradingView-exact):

### Tab 1: Inputs
- Numeric fields (period, length, multiplier)
- Dropdowns (price source, MA type)
- Checkboxes (enable/disable features)

**Example**:
```python
'params': {
    'period': {'type': 'int', 'default': 14, 'min': 1, 'max': 200},
    'source': {'type': 'select', 'options': ['close', 'open', 'high', 'low']},
    'show_bands': {'type': 'bool', 'default': True}
}
```

### Tab 2: Style
- Line color
- Line thickness
- Line style (solid, dashed, dotted)
- Histogram/area fill toggles
- Background fill between bands
- Precision & value visibility

### Tab 3: Visibility
- Timeframe-based visibility (1m, 5m, 1h, 1D, etc.)
- Chart-type visibility (candles, bars, line)

---

## 📊 Chart Legend Behavior

Each active indicator appears in the chart legend:

```
┌─────────────────────────────────┐
│ 📈 RSI(14)            67.32     │  ← Live value
│    👁️ 👁️⚙️ 🗑️                    │  ← Show/Hide, Settings, Delete
│                                 │
│ 📉 MACD(12,26,9)                │
│    👁️ ⚙️ 🗑️                      │
└─────────────────────────────────┘
```

**Controls**:
- 👁️ Eye icon = Show/Hide
- ⚙️ Gear icon = Open settings
- 🗑️ Trash icon = Remove indicator

---

## 🔧 Built-in TA Functions

All indicators have access to `ta` helper object with these functions:

### Moving Averages
- `ta.sma(data, period)` - Simple MA
- `ta.ema(data, period)` - Exponential MA
- `ta.wma(data, period)` - Weighted MA

### Oscillators
- `ta.rsi(data, period)` - RSI
- `ta.macd(data, fast, slow, signal)` - MACD
- `ta.stochastic(high, low, close, k_period, d_period)` - Stochastic
- `ta.cci(high, low, close, period)` - CCI
- `ta.williams_r(high, low, close, period)` - Williams %R

### Trend
- `ta.bollinger_bands(data, period, std_dev)` - Bollinger Bands
- `ta.supertrend(high, low, close, period, multiplier)` - SuperTrend
- `ta.ichimoku(high, low, close, tenkan, kijun, senkou_b)` - Ichimoku
- `ta.adx(high, low, close, period)` - ADX

### Volume
- `ta.vwap(high, low, close, volume)` - VWAP
- `ta.obv(close, volume)` - On Balance Volume
- `ta.mfi(high, low, close, volume, period)` - Money Flow Index

### Volatility
- `ta.atr(high, low, close, period)` - Average True Range

### Support/Resistance
- `ta.pivot_points(high, low, close)` - Pivot Points

---

## 🎨 Indicator Popup (UI Specification)

### How It Opens
1. User clicks **"Indicators (fx)"** button in top toolbar
2. Modal popup appears centered over chart
3. Semi-transparent overlay behind it

### Layout Structure
```
┌─────────────────────────────────────────────────┐
│ Indicators, Metrics & Strategies            [×] │
│ 🔍 [Search indicators...]                       │
├──────────────┬──────────────────────────────────┤
│              │                                  │
│ 📁 Favorites │  ⭐ EMA Crossover               │
│ 📁 Built-ins │  ⭐ RSI Divergence              │
│   ├ Trend    │  ─────────────────────────────  │
│   ├ Momentum │  📈 Moving Average              │
│   ├ Volume   │  📈 Bollinger Bands             │
│   └ Volatil. │  📈 SuperTrend                  │
│ 📁 Community │  📊 RSI                         │
│ 📁 My Scripts│  📊 MACD                        │
│              │  📊 Stochastic                  │
└──────────────┴──────────────────────────────────┘
```

### Categories Panel (Left Sidebar)
- **Favorites** ⭐ - Starred indicators
- **Built-ins** 📦:
  - Trend
  - Momentum
  - Volume
  - Volatility
  - Market Structure
- **Community Scripts** 👥
- **My Scripts** 📝 (user-created)

### Indicator List (Main Area)
- Each indicator shown as clickable row
- Hover shows description
- Star icon to add to favorites
- Click to instantly add to chart

### Search Behavior
- Real-time filtering
- Matches indicator name, description, author
- Results update without closing popup
- Multi-indicator addition (popup stays open)

---

## 🚀 Performance Requirements

✅ **Vectorized Operations**
- Use Pandas/NumPy vectorized functions
- Avoid loops where possible

✅ **Scalability**
- Support 100,000+ bars
- Multiple indicators without lag

✅ **Real-time Updates**
- Incremental calculation for new bars
- Efficient memory usage

✅ **Bar Replay Support**
- Recompute candle-by-candle
- Maintain indicator state

---

## 📦 Python Dependencies

```txt
fastapi>=0.104.0
pandas>=2.0.0
numpy>=1.24.0
uvicorn>=0.24.0
pydantic>=2.0.0
```

**Optional (for advanced TA)**:
```txt
ta-lib  # Technical Analysis Library (C extension)
```

---

## 🔌 API Endpoints

### GET /scripts
List all available indicators

**Response**:
```json
[
  {
    "name": "RSI",
    "path": "/path/to/rsi.py",
    "description": "Relative Strength Index",
    "params": {...},
    "overlay": false,
    "category": "Momentum"
  }
]
```

### POST /execute
Execute an indicator

**Request**:
```json
{
  "script_name": "rsi",
  "ohlcv": {
    "time": [1234567890, ...],
    "open": [100, 101, ...],
    "high": [102, 103, ...],
    "low": [99, 100, ...],
    "close": [101, 102, ...],
    "volume": [1000, 1100, ...]
  },
  "params": {
    "period": 14,
    "overbought": 70,
    "oversold": 30
  }
}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "rsi": [null, null, ..., 67.32],
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

## 📝 Creating Custom Indicators

Users can create custom indicators by placing `.py` files in:
```
~/CryptoChartPro/scripts/my_indicator.py
```

**Template**:
```python
"""
My Custom Indicator
"""

def calculate(ohlcv, params):
    period = params.get('period', 20)
    close = ohlcv['close']
    
    # Your calculation
    result = ta.sma(close, period)
    
    return {'my_line': result}

META = {
    'name': 'My Custom Indicator',
    'description': 'Does something cool',
    'overlay': True,
    'category': 'Custom',
    'params': {
        'period': {'type': 'int', 'default': 20, 'min': 1, 'max': 200}
    }
}
```

---

## ✅ Implementation Checklist

### Phase 1: Core Indicators (✅ Complete)
- [x] Moving Average
- [x] Bollinger Bands
- [x] SuperTrend
- [x] RSI
- [x] MACD
- [x] Stochastic
- [x] ATR
- [x] VWAP
- [x] Volume
- [x] OBV
- [x] Ichimoku
- [x] CCI
- [x] Williams %R
- [x] ADX
- [x] MFI
- [x] Pivot Points

### Phase 2: Advanced Indicators (Pending)
- [ ] Stochastic RSI
- [ ] Parabolic SAR
- [ ] Donchian Channels
- [ ] Keltner Channels
- [ ] ROC (Rate of Change)
- [ ] Momentum
- [ ] Awesome Oscillator
- [ ] Accumulation/Distribution
- [ ] Chaikin Money Flow
- [ ] Standard Deviation
- [ ] Choppiness Index
- [ ] ZigZag

### Phase 3: Fibonacci Tools (Pending)
- [ ] Fibonacci Retracement
- [ ] Fibonacci Extension
- [ ] Fibonacci Fan
- [ ] Fibonacci Time Zones

### Phase 4: Pattern Recognition (Pending)
- [ ] ABCD Pattern
- [ ] Head & Shoulders
- [ ] Elliott Wave Tools

---

## 🎯 Next Steps

1. ✅ **Implement built-in indicators** (16/30 complete)
2. ⬜ **Create indicator popup UI** (React component)
3. ⬜ **Build settings panel** (3-tab interface)
4. ⬜ **Integrate with chart legend**
5. ⬜ **Add bar replay support**
6. ⬜ **Performance optimize** (vectorization)

---

## 📚 Additional Resources

- **Python Backend**: `python-backend/server.py`
- **Built-in Indicators**: `python-backend/builtin/*.py`
- **TA Functions**: `python-backend/server.py` (TechnicalAnalysis class)
- **Frontend Integration**: `src/indicators/index.ts`

---

**Last Updated**: 2026-01-21
**Version**: 1.0.0
**Status**: In Development 🚧

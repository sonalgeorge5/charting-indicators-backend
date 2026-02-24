"""
CryptoChart Pro - Python Indicator Server
FastAPI server for executing custom Python indicators
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
import json
import os
import importlib.util
import traceback
from pathlib import Path
from datetime import datetime
from alert_worker import AlertConfig, EmaRetestAlertWorker
from vision_cache import (
    SUPPORTED_INTERVALS,
    SUPPORTED_SYMBOLS,
    VisionBackfillService,
    VisionCandleCache,
    VisionConfig,
)

app = FastAPI(
    title="CryptoChart Pro Indicator Server",
    description="Python backend for custom indicator execution",
    version="1.0.0"
)

# CORS for frontend access - allow all origins for public API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=False,  # Must be False when using wildcard origins
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Custom exception handler to ensure CORS headers on error responses
from fastapi import Request
from fastapi.responses import JSONResponse

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Ensure CORS headers are included in error responses"""
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "*",
            "Access-Control-Allow-Headers": "*",
        }
    )


# User scripts directory
SCRIPTS_DIR = Path.home() / "CryptoChartPro" / "scripts"
SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

# Built-in indicators directory
BUILTIN_DIR = Path(__file__).parent / "builtin"
ALERT_CONFIG = AlertConfig.from_env()
ALERT_WORKER = EmaRetestAlertWorker(ALERT_CONFIG)
VISION_CONFIG = VisionConfig.from_env()
VISION_CACHE = VisionCandleCache(VISION_CONFIG)
VISION_BACKFILL = VisionBackfillService(VISION_CACHE, VISION_CONFIG)


class OHLCVData(BaseModel):
    time: List[int]
    open: List[float]
    high: List[float]
    low: List[float]
    close: List[float]
    volume: List[float]


class IndicatorRequest(BaseModel):
    script_name: str
    ohlcv: OHLCVData
    params: Optional[Dict[str, Any]] = {}
    code: Optional[str] = None  # Allow executing raw code directly


class SaveRequest(BaseModel):
    name: str
    code: str
    overwrite: bool = True


class IndicatorResponse(BaseModel):
    success: bool
    data: Optional[Dict[str, List[Optional[float]]]] = None
    signals: Optional[Dict[str, List[bool]]] = None
    overlay: bool = False
    error: Optional[str] = None


class ScriptInfo(BaseModel):
    id: str  # Filename stem for API calls/IDs
    name: str
    path: str
    description: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
    overlay: bool = False


# ============================================
# BUILT-IN TECHNICAL ANALYSIS FUNCTIONS
# ============================================

class TechnicalAnalysis:
    """Built-in TA functions available to all scripts via `ta` object"""
    
    @staticmethod
    def sma(data: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average"""
        return data.rolling(window=period).mean()
    
    @staticmethod
    def ema(data: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average"""
        return data.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def wma(data: pd.Series, period: int) -> pd.Series:
        weights = np.arange(1, period + 1)
        return data.rolling(window=period).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    @staticmethod
    def rsi(data: pd.Series, period: int = 14) -> pd.Series:
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        fast_ema = data.ewm(span=fast, adjust=False).mean()
        slow_ema = data.ewm(span=slow, adjust=False).mean()
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return {'macd': macd_line, 'signal': signal_line, 'histogram': histogram}
    
    @staticmethod
    def bollinger_bands(data: pd.Series, period: int = 20, std_dev: float = 2.0) -> Dict[str, pd.Series]:
        middle = data.rolling(window=period).mean()
        std = data.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return {'upper': upper, 'middle': middle, 'lower': lower}


# Global TA instance for scripts
ta = TechnicalAnalysis()


def ohlcv_to_dataframe(ohlcv: OHLCVData) -> pd.DataFrame:
    """Convert OHLCV data to pandas DataFrame"""
    return pd.DataFrame({
        'time': ohlcv.time,
        'open': ohlcv.open,
        'high': ohlcv.high,
        'low': ohlcv.low,
        'close': ohlcv.close,
        'volume': ohlcv.volume
    })


def execute_script(module, df: pd.DataFrame, params: Dict[str, Any]) -> IndicatorResponse:
    """Execute a Python indicator script safely"""
    try:
        # Inject the `ta` helper into the module's namespace
        module.ta = ta
        module.pd = pd
        module.np = np
        
        # Get metadata (indicator dict as per User requirement)
        # Fallback to META for backward compatibility
        meta = getattr(module, 'indicator', getattr(module, 'META', {}))
        overlay = meta.get('overlay', False)
        
        # Determine inputs
        inputs = meta.get('inputs', {}).copy()
        inputs.update(params)  # Override defaults with params
        
        # Call the calculate function
        if not hasattr(module, 'calculate'):
            raise ValueError("Script must have a 'calculate(data, inputs)' function")
        
        # Calculate
        result = module.calculate(df, inputs)
        
        # Process result
        data = {}
        signals = {}
        
        def _to_float_or_none(x: Any) -> Optional[float]:
            if pd.isna(x):
                return None
            try:
                return float(x)
            except Exception:
                return None

        if isinstance(result, dict):
            for key, value in result.items():
                if key == 'signals' and isinstance(value, dict):
                    # Handle signals if returned separately
                    for sig_name, sig_data in value.items():
                        if isinstance(sig_data, pd.Series):
                            signals[sig_name] = sig_data.fillna(False).astype(bool).tolist()
                elif isinstance(value, pd.Series):
                    data[key] = [_to_float_or_none(x) for x in value.tolist()]
                elif isinstance(value, (list, np.ndarray)):
                    data[key] = [_to_float_or_none(x) for x in value]
                elif isinstance(value, (int, float)):
                    data[key] = [float(value)] * len(df)
        elif isinstance(result, pd.Series):
            data['value'] = [_to_float_or_none(x) for x in result.tolist()]
        elif isinstance(result, pd.DataFrame):
            for col in result.columns:
                data[col] = [_to_float_or_none(x) for x in result[col].tolist()]
        
        return IndicatorResponse(
            success=True,
            data=data,
            signals=signals if signals else None,
            overlay=overlay
        )
        
    except Exception as e:
        return IndicatorResponse(
            success=False,
            error=f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        )


@app.get("/")
async def root():
    return {
        "status": "running",
        "service": "CryptoChart Pro Indicator Server",
        "alerts": ALERT_WORKER.status(),
        "vision": VISION_CACHE.status(),
    }


@app.on_event("startup")
async def startup_event():
    await ALERT_WORKER.start()
    await VISION_BACKFILL.start()


@app.on_event("shutdown")
async def shutdown_event():
    await ALERT_WORKER.stop()
    await VISION_BACKFILL.stop()


@app.get("/vision/status")
async def vision_status():
    return {
        "ok": True,
        "config": {
            "db_path": VISION_CONFIG.db_path,
            "recent_days": VISION_CONFIG.recent_days,
            "oldest_backfill_month": VISION_CONFIG.oldest_backfill_month,
            "worker_interval_seconds": VISION_CONFIG.worker_interval_seconds,
            "bootstrap_days_on_demand": VISION_CONFIG.bootstrap_days_on_demand,
        },
        "status": VISION_CACHE.status(),
        "supported_symbols": sorted(SUPPORTED_SYMBOLS),
        "supported_intervals": sorted(SUPPORTED_INTERVALS.keys(), key=lambda x: SUPPORTED_INTERVALS[x]),
    }


@app.post("/vision/bootstrap")
async def vision_bootstrap(symbol: Optional[str] = None, days: Optional[int] = None):
    symbols = [symbol.upper()] if symbol else sorted(SUPPORTED_SYMBOLS)
    added: Dict[str, int] = {}
    for s in symbols:
        if s not in SUPPORTED_SYMBOLS:
            continue
        count = await VISION_BACKFILL.ensure_recent(s, days=days)
        if count == 0:
            # Fallback short seed from Binance 1s klines for immediate chart usability.
            count = VISION_CACHE.seed_recent_from_binance_klines(s, seconds=3600)
        added[s] = count
    return {"ok": True, "added": added, "status": VISION_CACHE.status()}


@app.get("/candles")
async def get_candles(
    symbol: str = Query(..., description="Symbol, e.g. BTCUSDT"),
    interval: str = Query("1s", description="Interval, e.g. 1s, 5s, 1m"),
    from_ts: Optional[int] = Query(None, alias="from", description="From unix seconds"),
    to_ts: Optional[int] = Query(None, alias="to", description="To unix seconds"),
    limit: Optional[int] = Query(2000, description="Maximum number of bars to return"),
):
    sym = symbol.upper().strip()
    tf = interval.strip()
    if sym not in SUPPORTED_SYMBOLS:
        raise HTTPException(status_code=400, detail=f"Unsupported symbol: {sym}")
    if tf not in SUPPORTED_INTERVALS:
        raise HTTPException(status_code=400, detail=f"Unsupported interval: {tf}")

    now_sec = int(datetime.utcnow().timestamp())
    to_s = to_ts if to_ts is not None else now_sec
    from_s = from_ts if from_ts is not None else max(0, to_s - SUPPORTED_INTERVALS[tf] * max(1000, (limit or 2000)))
    if from_s > to_s:
        from_s, to_s = to_s, from_s

    rows = VISION_CACHE.get_candles(
        symbol=sym,
        interval=tf,
        from_ts=from_s,
        to_ts=to_s,
        limit=max(1, min(10000, limit or 2000)),
    )
    if not rows:
        # On-demand bootstrap for first request on fresh deployment.
        await VISION_BACKFILL.ensure_recent(sym, days=VISION_CONFIG.bootstrap_days_on_demand)
        rows = VISION_CACHE.get_candles(
            symbol=sym,
            interval=tf,
            from_ts=from_s,
            to_ts=to_s,
            limit=max(1, min(10000, limit or 2000)),
        )

    return {
        "ok": True,
        "symbol": sym,
        "interval": tf,
        "from": from_s,
        "to": to_s,
        "count": len(rows),
        "candles": rows,
    }


@app.get("/scripts", response_model=List[ScriptInfo])
async def list_scripts():
    scripts = []
    
    # Helper to process file
    def process_file(f: Path, is_builtin=False):
        try:
            spec = importlib.util.spec_from_file_location("indicator", f)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            meta = getattr(module, 'indicator', getattr(module, 'META', {}))
            scripts.append(ScriptInfo(
                id=f.stem,
                name=meta.get('name', f.stem),
                path=str(f),
                description=meta.get('description', ''),
                params=meta.get('inputs', meta.get('params', {})),
                overlay=meta.get('overlay', False)
            ))
        except Exception as e:
            # Skip invalid scripts
            pass

    # User scripts
    if SCRIPTS_DIR.exists():
        for f in SCRIPTS_DIR.glob("*.py"):
            process_file(f)
    
    # Built-in scripts
    if BUILTIN_DIR.exists():
        for f in BUILTIN_DIR.glob("*.py"):
            if f.stem != "__init__":
                process_file(f, True)
    
    return scripts


@app.post("/execute", response_model=IndicatorResponse)
async def execute_indicator(request: IndicatorRequest):
    df = ohlcv_to_dataframe(request.ohlcv)
    
    if request.code:
        # Execute raw code
        try:
            # Create temporary module
            spec = importlib.util.spec_from_loader("indicator", loader=None)
            module = importlib.util.module_from_spec(spec)
            exec(request.code, module.__dict__)
            return execute_script(module, df, request.params or {})
        except Exception as e:
            return IndicatorResponse(success=False, error=str(e))
    else:
        # Execute from file
        script_path = None
        user_script = SCRIPTS_DIR / f"{request.script_name}.py"
        if user_script.exists():
            script_path = user_script
        elif BUILTIN_DIR.exists():
            builtin_script = BUILTIN_DIR / f"{request.script_name}.py"
            if builtin_script.exists():
                script_path = builtin_script
                
        if not script_path or not script_path.exists():
            raise HTTPException(status_code=404, detail=f"Script not found: {request.script_name}")
            
        spec = importlib.util.spec_from_file_location("indicator", script_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return execute_script(module, df, request.params or {})


@app.post("/save")
async def save_script(request: SaveRequest):
    """Save a custom indicator script"""
    try:
        # Sanitize name
        filename = "".join(x for x in request.name if x.isalnum() or x in (' ', '_', '-')).strip()
        filename = filename.replace(' ', '_')
        if not filename:
            raise HTTPException(400, "Invalid script name")
            
        file_path = SCRIPTS_DIR / f"{filename}.py"
        
        with open(file_path, 'w') as f:
            f.write(request.code)
            
        return {"success": True, "path": str(file_path), "name": filename}
    except Exception as e:
        raise HTTPException(500, str(e))


class ValidateRequest(BaseModel):
    code: str


@app.post("/validate")
async def validate_script(request: ValidateRequest):
    """Validate Python code syntax"""
    try:
        # Try to compile the code to check for syntax errors
        compile(request.code, '<string>', 'exec')
        
        # Check if code has required structure
        temp_globals = {}
        exec(request.code, temp_globals)
        
        has_indicator = 'indicator' in temp_globals
        has_calculate = 'calculate' in temp_globals and callable(temp_globals.get('calculate'))
        
        if not has_calculate:
            return {
                "valid": False, 
                "error": "Script must have a 'calculate(data, inputs)' function"
            }
        
        return {
            "valid": True,
            "has_metadata": has_indicator,
            "message": "Code is valid"
        }
    except SyntaxError as e:
        return {
            "valid": False,
            "error": f"Syntax error at line {e.lineno}: {e.msg}"
        }
    except Exception as e:
        return {
            "valid": False,
            "error": str(e)
        }



@app.get("/templates")
async def get_templates():
    return {
        "basic": '''# === Indicator Metadata ===
indicator = {
    "name": "My Moving Average",
    "overlay": True,
    "inputs": {
        "length": 14,
        "source": "close"
    }
}

# === Indicator Logic ===
def calculate(data, inputs):
    length = inputs["length"]
    source = data["close"]

    sma = source.rolling(length).mean()
    return {
        "plot": sma
    }
''',
        "complex": '''# === Indicator Metadata ===
indicator = {
    "name": "RSI Strategy",
    "overlay": False,
    "inputs": {
        "period": 14,
        "overbought": 70,
        "oversold": 30
    }
}

def calculate(data, inputs):
    period = inputs["period"]
    close = data["close"]
    
    # Calculate RSI
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return {
        "rsi": rsi,
        "overbought": inputs["overbought"],
        "oversold": inputs["oversold"]
    }
'''
    }


@app.get("/alerts/status")
async def alerts_status():
    return ALERT_WORKER.status()


@app.post("/alerts/scan-now")
async def alerts_scan_now():
    queued = await ALERT_WORKER.trigger_scan_async()
    return {
        "ok": True,
        "queued": queued,
        "status": ALERT_WORKER.status(),
    }


@app.post("/alerts/test-telegram")
async def alerts_test_telegram():
    ok = await ALERT_WORKER.send_test_async()
    return {"ok": ok, "status": ALERT_WORKER.status()}

if __name__ == "__main__":
    import uvicorn
    import os
    
    port = int(os.environ.get("PORT", 8765))
    
    print(f"CryptoChart Pro Indicator Server")
    print(f"User scripts directory: {SCRIPTS_DIR}")
    print(f"Starting server on port {port}")
    
    uvicorn.run(app, host="0.0.0.0", port=port)

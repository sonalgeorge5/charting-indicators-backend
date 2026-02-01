"""
CryptoChart Pro - Python Indicator Server
FastAPI server for executing custom Python indicators
"""

from fastapi import FastAPI, HTTPException
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

app = FastAPI(
    title="CryptoChart Pro Indicator Server",
    description="Python backend for custom indicator execution",
    version="1.0.0"
)

# CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173", 
        "http://localhost:3000", 
        "https://charting-platform.onrender.com",
        "*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# User scripts directory
SCRIPTS_DIR = Path.home() / "CryptoChartPro" / "scripts"
SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

# Built-in indicators directory
BUILTIN_DIR = Path(__file__).parent / "builtin"


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
    data: Optional[Dict[str, List[float]]] = None
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
        
        if isinstance(result, dict):
            for key, value in result.items():
                if key == 'signals' and isinstance(value, dict):
                    # Handle signals if returned separately
                    for sig_name, sig_data in value.items():
                        if isinstance(sig_data, pd.Series):
                            signals[sig_name] = sig_data.fillna(False).astype(bool).tolist()
                elif isinstance(value, pd.Series):
                    data[key] = value.where(pd.notnull(value), None).tolist()
                elif isinstance(value, (list, np.ndarray)):
                    data[key] = [float(x) if not np.isnan(x) else None for x in value]
                elif isinstance(value, (int, float)):
                    data[key] = [value] * len(df)
        elif isinstance(result, pd.Series):
            data['value'] = result.where(pd.notnull(result), None).tolist()
        elif isinstance(result, pd.DataFrame):
            for col in result.columns:
                data[col] = result[col].where(pd.notnull(result[col]), None).tolist()
        
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
    return {"status": "running", "service": "CryptoChart Pro Indicator Server"}


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

if __name__ == "__main__":
    import uvicorn
    import os
    
    port = int(os.environ.get("PORT", 8765))
    
    print(f"CryptoChart Pro Indicator Server")
    print(f"User scripts directory: {SCRIPTS_DIR}")
    print(f"Starting server on port {port}")
    
    uvicorn.run(app, host="0.0.0.0", port=port)


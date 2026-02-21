import asyncio
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError


BINANCE_SPOT_API = "https://api.binance.com/api/v3"
BINANCE_FUTURES_API = "https://fapi.binance.com/fapi/v1"
BYBIT_API = "https://api.bybit.com/v5/market"


def _as_bool(value: str, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _interval_to_ms(interval: str) -> int:
    if interval.endswith("m"):
        return int(interval[:-1]) * 60_000
    if interval.endswith("h"):
        return int(interval[:-1]) * 3_600_000
    if interval.endswith("d"):
        return int(interval[:-1]) * 86_400_000
    if interval.endswith("w"):
        return int(interval[:-1]) * 604_800_000
    return 60_000


def _ema_series(values: List[float], period: int) -> List[float]:
    out = [float("nan")] * len(values)
    if period <= 0 or len(values) < period:
        return out
    mult = 2.0 / (period + 1)
    seed = sum(values[:period]) / period
    out[period - 1] = seed
    ema = seed
    for i in range(period, len(values)):
        ema = (values[i] - ema) * mult + ema
        out[i] = ema
    return out


def _adx_series(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> List[float]:
    n = len(closes)
    out = [float("nan")] * n
    if n < (period * 2 + 1):
        return out

    tr = [0.0] * n
    plus_dm = [0.0] * n
    minus_dm = [0.0] * n

    for i in range(1, n):
        up_move = highs[i] - highs[i - 1]
        down_move = lows[i - 1] - lows[i]
        plus_dm[i] = up_move if (up_move > down_move and up_move > 0) else 0.0
        minus_dm[i] = down_move if (down_move > up_move and down_move > 0) else 0.0
        tr[i] = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )

    atr = sum(tr[1:period + 1])
    sm_plus = sum(plus_dm[1:period + 1])
    sm_minus = sum(minus_dm[1:period + 1])
    dx_vals: List[float] = [float("nan")] * n

    for i in range(period, n):
        if i > period:
            atr = atr - (atr / period) + tr[i]
            sm_plus = sm_plus - (sm_plus / period) + plus_dm[i]
            sm_minus = sm_minus - (sm_minus / period) + minus_dm[i]

        if atr <= 0:
            continue
        plus_di = 100.0 * (sm_plus / atr)
        minus_di = 100.0 * (sm_minus / atr)
        denom = plus_di + minus_di
        dx_vals[i] = 0.0 if denom == 0 else (100.0 * abs(plus_di - minus_di) / denom)

    start = period
    end = period * 2
    seed_vals = [v for v in dx_vals[start:end] if v == v]
    if len(seed_vals) < period:
        return out

    adx = sum(seed_vals) / period
    out[end - 1] = adx
    for i in range(end, n):
        dx = dx_vals[i]
        if dx != dx:
            continue
        adx = ((adx * (period - 1)) + dx) / period
        out[i] = adx

    return out


@dataclass
class AlertConfig:
    enabled: bool
    telegram_bot_token: str
    telegram_chat_id: str
    binance_api_key: str
    symbols: List[str]
    timeframe: str
    htf_timeframe: str
    direction: str
    scan_seconds: int
    universe_mode: str
    top_n: int
    universe_refresh_seconds: int
    quote_asset: str
    exclude_stablecoins: bool
    cooldown_seconds: int
    retest_tolerance_pct: float
    min_ema_spread_pct: float
    min_adx_long: float
    min_adx_short: float
    state_path: Path

    @staticmethod
    def from_env() -> "AlertConfig":
        default_symbols = "BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT,ADAUSDT"
        raw_symbols = os.getenv("ALERT_SYMBOLS", default_symbols)
        symbols = [s.strip().upper() for s in raw_symbols.split(",") if s.strip()]
        state_dir = Path.home() / "CryptoChartPro"
        state_dir.mkdir(parents=True, exist_ok=True)
        universe_mode = os.getenv("ALERT_UNIVERSE_MODE", "top_marketcap_binance").strip().lower()
        scan_seconds = max(20, int(os.getenv("ALERT_SCAN_SECONDS", "60")))
        if universe_mode == "top_marketcap_binance" and scan_seconds < 180:
            scan_seconds = 180
        return AlertConfig(
            enabled=_as_bool(os.getenv("ALERTS_ENABLED", "false"), False),
            telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN", "").strip(),
            telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID", "").strip(),
            binance_api_key=os.getenv("BINANCE_API_KEY", "").strip(),
            symbols=symbols,
            timeframe=os.getenv("ALERT_TIMEFRAME", "30m").strip(),
            htf_timeframe=os.getenv("ALERT_HTF_TIMEFRAME", "4h").strip(),
            direction=os.getenv("ALERT_DIRECTION", "both").strip().lower(),
            scan_seconds=scan_seconds,
            universe_mode=universe_mode,
            top_n=max(1, int(os.getenv("ALERT_TOP_N", "100"))),
            universe_refresh_seconds=max(300, int(os.getenv("ALERT_UNIVERSE_REFRESH_SECONDS", "21600"))),
            quote_asset=os.getenv("ALERT_QUOTE_ASSET", "USDT").strip().upper(),
            exclude_stablecoins=_as_bool(os.getenv("ALERT_EXCLUDE_STABLECOINS", "true"), True),
            cooldown_seconds=max(60, int(os.getenv("ALERT_COOLDOWN_SECONDS", "21600"))),
            retest_tolerance_pct=float(os.getenv("ALERT_RETEST_TOLERANCE_PCT", "0.001")),
            min_ema_spread_pct=float(os.getenv("ALERT_MIN_EMA_SPREAD_PCT", "0.01")),
            min_adx_long=float(os.getenv("ALERT_MIN_ADX_LONG", "30")),
            min_adx_short=float(os.getenv("ALERT_MIN_ADX_SHORT", "25")),
            state_path=state_dir / "ema_alert_state.json",
        )

    def is_ready(self) -> bool:
        has_symbols = bool(self.symbols) or self.universe_mode == "top_marketcap_binance"
        return bool(self.telegram_bot_token and self.telegram_chat_id and has_symbols)


class EmaRetestAlertWorker:
    def __init__(self, cfg: AlertConfig):
        self.cfg = cfg
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._last_error: Optional[str] = None
        self._last_scan_at: Optional[int] = None
        self._effective_symbols: List[str] = list(cfg.symbols)
        self._last_universe_refresh: Optional[int] = None
        self._binance_spot_cache: Dict[str, Any] = {"symbols": set(), "updated_at": 0}
        self._state = self._load_state()

    def _load_state(self) -> Dict[str, Any]:
        if self.cfg.state_path.exists():
            try:
                return json.loads(self.cfg.state_path.read_text())
            except Exception:
                return {"sent": {}}
        return {"sent": {}}

    def _save_state(self) -> None:
        try:
            self.cfg.state_path.write_text(json.dumps(self._state))
        except Exception:
            pass

    def _request_json(self, url: str, extra_headers: Optional[Dict[str, str]] = None) -> Any:
        headers = {}
        if self.cfg.binance_api_key:
            headers["X-MBX-APIKEY"] = self.cfg.binance_api_key
        if extra_headers:
            headers.update(extra_headers)
        req = Request(url, headers=headers, method="GET")
        with urlopen(req, timeout=12) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _post_form(self, url: str, payload: Dict[str, Any]) -> Any:
        data = urlencode(payload).encode("utf-8")
        req = Request(url, data=data, method="POST")
        with urlopen(req, timeout=12) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _normalize_bybit_rows(self, rows: List[List[Any]]) -> List[List[Any]]:
        # Bybit row format:
        # [startTime, open, high, low, close, volume, turnover]
        norm: List[List[Any]] = []
        for r in rows:
            try:
                t = int(r[0])
                o = str(r[1])
                h = str(r[2])
                l = str(r[3])
                c = str(r[4])
                v = str(r[5])
                close_time = t + 1
                norm.append([t, o, h, l, c, v, close_time])
            except Exception:
                continue
        norm.sort(key=lambda x: int(x[0]))
        return norm

    def _to_bybit_interval(self, interval: str) -> Optional[str]:
        map_tf = {
            "1m": "1",
            "3m": "3",
            "5m": "5",
            "15m": "15",
            "30m": "30",
            "1h": "60",
            "2h": "120",
            "4h": "240",
            "1d": "D",
            "1w": "W",
        }
        return map_tf.get(interval)

    def _fetch_klines(self, symbol: str, interval: str, limit: int) -> List[List[Any]]:
        # 1) Binance spot
        try:
            q = urlencode({"symbol": symbol, "interval": interval, "limit": limit})
            url = f"{BINANCE_SPOT_API}/klines?{q}"
            rows = self._request_json(url)
            if isinstance(rows, list) and rows:
                return rows
        except HTTPError:
            pass
        except Exception:
            pass

        # 2) Binance futures
        try:
            q = urlencode({"symbol": symbol, "interval": interval, "limit": limit})
            url = f"{BINANCE_FUTURES_API}/klines?{q}"
            rows = self._request_json(url)
            if isinstance(rows, list) and rows:
                return rows
        except HTTPError:
            pass
        except Exception:
            pass

        # 3) Bybit linear
        try:
            bybit_interval = self._to_bybit_interval(interval)
            if bybit_interval:
                q = urlencode({"category": "linear", "symbol": symbol, "interval": bybit_interval, "limit": limit})
                url = f"{BYBIT_API}/kline?{q}"
                body = self._request_json(url)
                rows = body.get("result", {}).get("list", []) if isinstance(body, dict) else []
                if isinstance(rows, list) and rows:
                    return self._normalize_bybit_rows(rows)
        except Exception:
            pass

        return []

    def _fetch_binance_listed_symbols(self) -> set:
        now = int(time.time())
        cached = self._binance_spot_cache
        if cached["symbols"] and now - int(cached["updated_at"]) < 3600:
            return cached["symbols"]

        out = set()

        # Spot listing lookup (can be region-blocked in some Render regions).
        try:
            body = self._request_json(f"{BINANCE_SPOT_API}/exchangeInfo")
            for s in body.get("symbols", []):
                if s.get("status") != "TRADING":
                    continue
                if s.get("quoteAsset") != self.cfg.quote_asset:
                    continue
                if not s.get("isSpotTradingAllowed", True):
                    continue
                sym = str(s.get("symbol", "")).upper()
                if sym:
                    out.add(sym)
        except Exception:
            pass

        # Futures listing fallback.
        try:
            body = self._request_json(f"{BINANCE_FUTURES_API}/exchangeInfo")
            for s in body.get("symbols", []):
                if s.get("status") != "TRADING":
                    continue
                if s.get("quoteAsset") != self.cfg.quote_asset:
                    continue
                sym = str(s.get("symbol", "")).upper()
                if sym:
                    out.add(sym)
        except Exception:
            pass

        self._binance_spot_cache = {"symbols": out, "updated_at": now}
        return out

    def _is_binance_pair_supported(self, pair: str, interval: str) -> bool:
        # Probe Binance spot then futures directly when exchangeInfo is blocked.
        try:
            q = urlencode({"symbol": pair, "interval": interval, "limit": 1})
            rows = self._request_json(f"{BINANCE_SPOT_API}/klines?{q}")
            if isinstance(rows, list) and rows:
                return True
        except Exception:
            pass
        try:
            q = urlencode({"symbol": pair, "interval": interval, "limit": 1})
            rows = self._request_json(f"{BINANCE_FUTURES_API}/klines?{q}")
            if isinstance(rows, list) and rows:
                return True
        except Exception:
            pass
        return False

    def _looks_like_stablecoin(self, symbol: str, name: str) -> bool:
        sym = symbol.strip().lower()
        nm = name.strip().lower()
        stable_syms = {
            "usdt", "usdc", "busd", "dai", "tusd", "fdusd", "usde", "usdd",
            "usdp", "frax", "lusd", "susd", "gusd", "pyusd", "rlusd",
            "eurt", "eurs", "eurc",
        }
        if sym in stable_syms:
            return True
        stable_name_keys = [
            "stablecoin", "usd coin", "tether", "trueusd", "first digital usd",
            "pax dollar", "paypal usd", "digital usd",
        ]
        if any(k in nm for k in stable_name_keys):
            return True
        return False

    def _fetch_top_marketcap_binance_symbols(self) -> List[str]:
        listed = self._fetch_binance_listed_symbols()
        use_pair_probe = len(listed) == 0

        selected: List[str] = []
        seen_base = set()
        per_page = 250
        # Pull extra pages so we can still fill top N after stablecoin + listing filters.
        for page in range(1, 5):
            q = urlencode({
                "vs_currency": "usd",
                "order": "market_cap_desc",
                "per_page": per_page,
                "page": page,
                "sparkline": "false",
            })
            url = f"https://api.coingecko.com/api/v3/coins/markets?{q}"
            rows = self._request_json(url, extra_headers={"accept": "application/json", "user-agent": "charting-alert-worker/1.0"})
            if not isinstance(rows, list) or not rows:
                break
            for coin in rows:
                base = str(coin.get("symbol", "")).upper()
                name = str(coin.get("name", ""))
                if not base or base in seen_base:
                    continue
                if self.cfg.exclude_stablecoins and self._looks_like_stablecoin(base, name):
                    continue
                pair = f"{base}{self.cfg.quote_asset}"
                if not use_pair_probe:
                    if pair not in listed:
                        continue
                else:
                    if not self._is_binance_pair_supported(pair, self.cfg.timeframe):
                        continue
                selected.append(pair)
                seen_base.add(base)
                if len(selected) >= self.cfg.top_n:
                    return selected
        return selected

    def _refresh_symbol_universe(self, force: bool = False) -> None:
        if self.cfg.universe_mode != "top_marketcap_binance":
            self._effective_symbols = list(self.cfg.symbols)
            return

        now = int(time.time())
        if not force and self._last_universe_refresh and (now - self._last_universe_refresh) < self.cfg.universe_refresh_seconds:
            return

        try:
            dynamic_symbols = self._fetch_top_marketcap_binance_symbols()
            if dynamic_symbols:
                self._effective_symbols = dynamic_symbols
                self._last_universe_refresh = now
            elif not self._effective_symbols:
                self._effective_symbols = list(self.cfg.symbols)
        except Exception as exc:
            self._last_error = f"universe: {exc}"
            if not self._effective_symbols:
                self._effective_symbols = list(self.cfg.symbols)

    def _closed_index(self, rows: List[List[Any]], interval: str) -> int:
        if not rows:
            return -1
        now_ms = int(time.time() * 1000)
        interval_ms = _interval_to_ms(interval)
        idx = len(rows) - 1
        open_ms = int(rows[idx][0])
        close_ms = open_ms + interval_ms - 1
        if close_ms > now_ms:
            idx -= 1
        return idx

    def _htf_bias(self, symbol: str) -> int:
        rows = self._fetch_klines(symbol, self.cfg.htf_timeframe, 260)
        idx = self._closed_index(rows, self.cfg.htf_timeframe)
        if idx < 220:
            return 0
        closes = [float(r[4]) for r in rows[:idx + 1]]
        ema200 = _ema_series(closes, 200)[idx]
        close = closes[idx]
        if ema200 != ema200:
            return 0
        if close > ema200:
            return 1
        if close < ema200:
            return -1
        return 0

    def _signal_for_symbol(self, symbol: str) -> Optional[Dict[str, Any]]:
        rows = self._fetch_klines(symbol, self.cfg.timeframe, 420)
        idx = self._closed_index(rows, self.cfg.timeframe)
        if idx < 220:
            return None

        scoped = rows[:idx + 1]
        highs = [float(r[2]) for r in scoped]
        lows = [float(r[3]) for r in scoped]
        closes = [float(r[4]) for r in scoped]

        ema55 = _ema_series(closes, 55)
        ema100 = _ema_series(closes, 100)
        ema200 = _ema_series(closes, 200)
        adx14 = _adx_series(highs, lows, closes, 14)

        i = len(scoped) - 1
        if i < 2:
            return None
        e55 = ema55[i]
        e100 = ema100[i]
        e200 = ema200[i]
        adx = adx14[i]
        if any(v != v for v in [e55, e100, e200, adx]):
            return None

        current_close = closes[i]
        prev_close = closes[i - 1]
        current_high = highs[i]
        current_low = lows[i]
        spread_pct = abs(e55 - e200) / current_close if current_close else 0.0
        if spread_pct < self.cfg.min_ema_spread_pct:
            return None

        bullish_aligned = e55 > e100 > e200
        bearish_aligned = e55 < e100 < e200
        upper_touch = e100 * (1.0 + self.cfg.retest_tolerance_pct)
        lower_touch = e100 * (1.0 - self.cfg.retest_tolerance_pct)

        signal: Optional[str] = None
        if bullish_aligned:
            if adx > self.cfg.min_adx_long:
                from_above = prev_close >= e100
                retest = current_low <= upper_touch
                close_back = current_close >= e100
                if from_above and retest and close_back:
                    signal = "long"
        if bearish_aligned and signal is None:
            if adx > self.cfg.min_adx_short:
                from_below = prev_close <= e100
                retest = current_high >= lower_touch
                close_back = current_close <= e100
                if from_below and retest and close_back:
                    signal = "short"

        if signal is None:
            return None

        if self.cfg.direction == "long" and signal != "long":
            return None
        if self.cfg.direction == "short" and signal != "short":
            return None

        htf = self._htf_bias(symbol)
        if signal == "long" and htf == -1:
            return None
        if signal == "short" and htf == 1:
            return None

        candle_open_sec = int(scoped[i][0] / 1000)
        return {
            "symbol": symbol,
            "signal": signal,
            "timeframe": self.cfg.timeframe,
            "htf_timeframe": self.cfg.htf_timeframe,
            "candle_open_sec": candle_open_sec,
            "price": current_close,
            "ema55": e55,
            "ema100": e100,
            "ema200": e200,
            "adx14": adx,
            "htf_bias": htf,
        }

    def _should_send(self, alert: Dict[str, Any]) -> bool:
        key = f"{alert['symbol']}:{alert['signal']}"
        prev = self._state.get("sent", {}).get(key)
        if not prev:
            return True
        last_bar = int(prev.get("candle_open_sec", 0))
        last_sent = int(prev.get("sent_at", 0))
        if alert["candle_open_sec"] <= last_bar:
            return False
        if int(time.time()) - last_sent < self.cfg.cooldown_seconds:
            return False
        return True

    def _mark_sent(self, alert: Dict[str, Any]) -> None:
        key = f"{alert['symbol']}:{alert['signal']}"
        self._state.setdefault("sent", {})[key] = {
            "candle_open_sec": alert["candle_open_sec"],
            "sent_at": int(time.time()),
        }
        self._save_state()

    def _telegram_text(self, alert: Dict[str, Any]) -> str:
        side = "LONG" if alert["signal"] == "long" else "SHORT"
        htf_txt = "Bullish" if alert["htf_bias"] == 1 else "Bearish" if alert["htf_bias"] == -1 else "Neutral"
        candle_dt = datetime.fromtimestamp(alert["candle_open_sec"], tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        return (
            f"EMA 55/100/200 Multi-Retest Alert\n"
            f"Symbol: {alert['symbol']}\n"
            f"Signal: {side}\n"
            f"TF: {alert['timeframe']} | HTF: {alert['htf_timeframe']} ({htf_txt})\n"
            f"Price: {alert['price']:.6f}\n"
            f"EMA55: {alert['ema55']:.6f}\n"
            f"EMA100: {alert['ema100']:.6f}\n"
            f"EMA200: {alert['ema200']:.6f}\n"
            f"ADX14: {alert['adx14']:.2f}\n"
            f"Candle Open: {candle_dt}"
        )

    def _send_telegram(self, text: str) -> bool:
        if not self.cfg.telegram_bot_token or not self.cfg.telegram_chat_id:
            return False
        url = f"https://api.telegram.org/bot{self.cfg.telegram_bot_token}/sendMessage"
        body = {
            "chat_id": self.cfg.telegram_chat_id,
            "text": text,
            "disable_web_page_preview": "true",
        }
        try:
            resp = self._post_form(url, body)
            return bool(resp.get("ok"))
        except Exception:
            return False

    def scan_once(self) -> List[Dict[str, Any]]:
        self._last_error = None
        self._refresh_symbol_universe()
        sent: List[Dict[str, Any]] = []
        symbols = self._effective_symbols if self._effective_symbols else self.cfg.symbols
        for symbol in symbols:
            try:
                alert = self._signal_for_symbol(symbol)
                if not alert:
                    continue
                if not self._should_send(alert):
                    continue
                if self._send_telegram(self._telegram_text(alert)):
                    self._mark_sent(alert)
                    sent.append(alert)
            except Exception as exc:
                self._last_error = f"{symbol}: {exc}"
        self._last_scan_at = int(time.time())
        return sent

    async def scan_once_async(self) -> List[Dict[str, Any]]:
        return await asyncio.to_thread(self.scan_once)

    async def send_test_async(self) -> bool:
        self._refresh_symbol_universe(force=True)
        symbols = self._effective_symbols if self._effective_symbols else self.cfg.symbols
        text = (
            "Test message from EMA 55/100/200 alert worker.\n"
            f"Symbols: {', '.join(symbols[:20])}{' ...' if len(symbols) > 20 else ''}\n"
            f"TF: {self.cfg.timeframe}, HTF: {self.cfg.htf_timeframe}"
        )
        return await asyncio.to_thread(self._send_telegram, text)

    async def _loop(self) -> None:
        while self._running:
            try:
                await self.scan_once_async()
            except Exception as exc:
                self._last_error = str(exc)
            await asyncio.sleep(self.cfg.scan_seconds)

    async def start(self) -> None:
        if self._running:
            return
        if not (self.cfg.enabled and self.cfg.is_ready()):
            return
        self._running = True
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except Exception:
                pass
            self._task = None

    def status(self) -> Dict[str, Any]:
        symbols = self._effective_symbols if self._effective_symbols else self.cfg.symbols
        return {
            "enabled": self.cfg.enabled,
            "ready": self.cfg.is_ready(),
            "running": self._running,
            "symbols": symbols,
            "symbol_count": len(symbols),
            "configured_symbols": self.cfg.symbols,
            "universe_mode": self.cfg.universe_mode,
            "top_n": self.cfg.top_n,
            "quote_asset": self.cfg.quote_asset,
            "exclude_stablecoins": self.cfg.exclude_stablecoins,
            "universe_refresh_seconds": self.cfg.universe_refresh_seconds,
            "last_universe_refresh": self._last_universe_refresh,
            "timeframe": self.cfg.timeframe,
            "htf_timeframe": self.cfg.htf_timeframe,
            "direction": self.cfg.direction,
            "scan_seconds": self.cfg.scan_seconds,
            "cooldown_seconds": self.cfg.cooldown_seconds,
            "telegram_token_set": bool(self.cfg.telegram_bot_token),
            "telegram_chat_id_set": bool(self.cfg.telegram_chat_id),
            "last_scan_at": self._last_scan_at,
            "last_error": self._last_error,
            "state_path": str(self.cfg.state_path),
        }

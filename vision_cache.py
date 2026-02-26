import asyncio
import csv
import io
import json
import os
import sqlite3
import threading
import time
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

try:
    import psycopg
    from psycopg.rows import dict_row
except Exception:
    psycopg = None
    dict_row = None


SUPPORTED_SYMBOLS = {"BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"}
SUPPORTED_INTERVALS: Dict[str, int] = {
    "1s": 1,
    "5s": 5,
    "15s": 15,
    "30s": 30,
    "45s": 45,
    "1m": 60,
    "2m": 120,
    "3m": 180,
    "5m": 300,
}

VISION_SPOT_DAILY_TRADES = (
    "https://data.binance.vision/data/spot/daily/trades/{symbol}/{symbol}-trades-{day}.zip"
)
VISION_SPOT_MONTHLY_TRADES = (
    "https://data.binance.vision/data/spot/monthly/trades/{symbol}/{symbol}-trades-{month}.zip"
)
BINANCE_SPOT_KLINES_ENDPOINTS = (
    "https://api.binance.com/api/v3/klines",
    "https://data-api.binance.vision/api/v3/klines",
    "https://api1.binance.com/api/v3/klines",
)


@dataclass
class VisionConfig:
    db_path: str
    recent_days: int
    oldest_backfill_month: str
    worker_interval_seconds: int
    bootstrap_days_on_demand: int
    request_timeout_seconds: int
    worker_enabled: bool
    low_memory_mode: bool
    normalize_on_start: bool
    max_zip_download_bytes: int
    klines_seed_seconds: int

    @staticmethod
    def from_env() -> "VisionConfig":
        def _env_bool(key: str, default: bool) -> bool:
            raw = os.getenv(key)
            if raw is None:
                return default
            return raw.strip().lower() in {"1", "true", "yes", "on"}

        db_path = os.getenv("VISION_DB_PATH", "./data/vision_candles.db")
        recent_days = int(os.getenv("VISION_RECENT_DAYS", "1"))
        oldest_backfill_month = os.getenv("VISION_OLDEST_BACKFILL_MONTH", "2017-09")
        worker_interval_seconds = int(os.getenv("VISION_WORKER_INTERVAL_SECONDS", "60"))
        bootstrap_days_on_demand = int(os.getenv("VISION_BOOTSTRAP_DAYS_ON_DEMAND", "1"))
        request_timeout_seconds = int(os.getenv("VISION_REQUEST_TIMEOUT_SECONDS", "20"))
        worker_enabled = _env_bool("VISION_WORKER_ENABLED", False)
        low_memory_mode = _env_bool("VISION_LOW_MEMORY_MODE", True)
        normalize_on_start = _env_bool("VISION_NORMALIZE_ON_START", False)
        max_zip_download_bytes = int(os.getenv("VISION_MAX_ZIP_DOWNLOAD_BYTES", str(25 * 1024 * 1024)))
        klines_seed_seconds = int(os.getenv("VISION_KLINES_SEED_SECONDS", str(6 * 60 * 60)))
        return VisionConfig(
            db_path=db_path,
            recent_days=max(1, recent_days),
            oldest_backfill_month=oldest_backfill_month,
            worker_interval_seconds=max(5, worker_interval_seconds),
            bootstrap_days_on_demand=max(1, bootstrap_days_on_demand),
            request_timeout_seconds=max(5, request_timeout_seconds),
            worker_enabled=worker_enabled,
            low_memory_mode=low_memory_mode,
            normalize_on_start=normalize_on_start,
            max_zip_download_bytes=max(1_000_000, max_zip_download_bytes),
            klines_seed_seconds=max(600, klines_seed_seconds),
        )


def _month_string(dt: date) -> str:
    return dt.strftime("%Y-%m")


def _prev_month(month_str: str) -> str:
    dt = datetime.strptime(f"{month_str}-01", "%Y-%m-%d").date()
    first = dt.replace(day=1)
    prev_last = first - timedelta(days=1)
    return _month_string(prev_last)


def _to_utc_date(ts_seconds: int) -> date:
    return datetime.fromtimestamp(ts_seconds, tz=timezone.utc).date()


def _to_epoch_seconds(dt: datetime) -> int:
    return int(dt.replace(tzinfo=timezone.utc).timestamp())


class VisionCandleCache:
    def __init__(self, cfg: VisionConfig) -> None:
        self.cfg = cfg
        self._lock = threading.Lock()
        raw_database_url = os.getenv("DATABASE_URL", "").strip()
        if raw_database_url.startswith("postgres://"):
            raw_database_url = "postgresql://" + raw_database_url[len("postgres://"):]
        self._database_url = raw_database_url
        self._use_postgres = self._database_url.startswith("postgresql://")
        if self._use_postgres:
            if psycopg is None:
                raise RuntimeError("DATABASE_URL is set but psycopg is not installed")
        else:
            os.makedirs(os.path.dirname(self.cfg.db_path) or ".", exist_ok=True)
        self._init_db()
        if self.cfg.normalize_on_start and not self._use_postgres:
            self._normalize_existing_candle_times()

    @contextmanager
    def _conn(self):
        if self._use_postgres:
            conn = psycopg.connect(
                self._database_url,
                row_factory=dict_row,
                connect_timeout=self.cfg.request_timeout_seconds,
            )
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()
            return

        conn = sqlite3.connect(self.cfg.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._conn() as conn:
            if self._use_postgres:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS candles_1s (
                        symbol TEXT NOT NULL,
                        time BIGINT NOT NULL,
                        open DOUBLE PRECISION NOT NULL,
                        high DOUBLE PRECISION NOT NULL,
                        low DOUBLE PRECISION NOT NULL,
                        close DOUBLE PRECISION NOT NULL,
                        volume DOUBLE PRECISION NOT NULL,
                        trades INTEGER NOT NULL DEFAULT 0,
                        PRIMARY KEY (symbol, time)
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS ingested_files (
                        file_key TEXT PRIMARY KEY,
                        symbol TEXT NOT NULL,
                        source TEXT NOT NULL,
                        processed_at BIGINT NOT NULL
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS symbol_progress (
                        symbol TEXT PRIMARY KEY,
                        recent_bootstrap_done INTEGER NOT NULL DEFAULT 0,
                        backfill_cursor_month TEXT NOT NULL,
                        last_recent_sync_at BIGINT,
                        last_backfill_sync_at BIGINT,
                        last_message TEXT
                    )
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_candles_symbol_time ON candles_1s(symbol, time)"
                )
            else:
                conn.executescript(
                    """
                    PRAGMA journal_mode=WAL;
                    CREATE TABLE IF NOT EXISTS candles_1s (
                        symbol TEXT NOT NULL,
                        time INTEGER NOT NULL,
                        open REAL NOT NULL,
                        high REAL NOT NULL,
                        low REAL NOT NULL,
                        close REAL NOT NULL,
                        volume REAL NOT NULL,
                        trades INTEGER NOT NULL DEFAULT 0,
                        PRIMARY KEY (symbol, time)
                    );

                    CREATE TABLE IF NOT EXISTS ingested_files (
                        file_key TEXT PRIMARY KEY,
                        symbol TEXT NOT NULL,
                        source TEXT NOT NULL,
                        processed_at INTEGER NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS symbol_progress (
                        symbol TEXT PRIMARY KEY,
                        recent_bootstrap_done INTEGER NOT NULL DEFAULT 0,
                        backfill_cursor_month TEXT NOT NULL,
                        last_recent_sync_at INTEGER,
                        last_backfill_sync_at INTEGER,
                        last_message TEXT
                    );

                    CREATE INDEX IF NOT EXISTS idx_candles_symbol_time ON candles_1s(symbol, time);
                    """
                )

            today = datetime.now(timezone.utc).date()
            recent_start = today - timedelta(days=max(1, self.cfg.recent_days) - 1)
            start_backfill_month = _prev_month(_month_string(recent_start))
            for symbol in SUPPORTED_SYMBOLS:
                if self._use_postgres:
                    conn.execute(
                        """
                        INSERT INTO symbol_progress (symbol, backfill_cursor_month)
                        VALUES (%s, %s)
                        ON CONFLICT(symbol) DO NOTHING
                        """,
                        (symbol, start_backfill_month),
                    )
                else:
                    conn.execute(
                        """
                        INSERT INTO symbol_progress (symbol, backfill_cursor_month)
                        VALUES (?, ?)
                        ON CONFLICT(symbol) DO NOTHING
                        """,
                        (symbol, start_backfill_month),
                    )

    def _normalize_existing_candle_times(self) -> None:
        """Repair older datasets that were stored with millisecond timestamps.

        The table is expected to hold 1-second candles. If older buggy ingests wrote
        millisecond-level rows, convert and aggregate them into 1-second bars.
        """
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS c FROM candles_1s WHERE time > 100000000000"
            ).fetchone()
            needs_fix = int(row["c"]) > 0 if row else False
            if not needs_fix:
                return

            conn.executescript(
                """
                DROP TABLE IF EXISTS candles_1s_new;
                CREATE TABLE candles_1s_new (
                    symbol TEXT NOT NULL,
                    time INTEGER NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL,
                    trades INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (symbol, time)
                );
                """
            )

            conn.execute(
                """
                INSERT INTO candles_1s_new (symbol, time, open, high, low, close, volume, trades)
                WITH normalized AS (
                    SELECT
                        symbol,
                        CASE
                            WHEN time > 100000000000 THEN CAST(time / 1000 AS INTEGER)
                            ELSE time
                        END AS sec_time,
                        time AS raw_time,
                        open, high, low, close, volume, trades
                    FROM candles_1s
                ),
                ranked AS (
                    SELECT
                        symbol, sec_time, raw_time, open, high, low, close, volume, trades,
                        ROW_NUMBER() OVER (
                            PARTITION BY symbol, sec_time
                            ORDER BY raw_time ASC
                        ) AS rn_open,
                        ROW_NUMBER() OVER (
                            PARTITION BY symbol, sec_time
                            ORDER BY raw_time DESC
                        ) AS rn_close
                    FROM normalized
                )
                SELECT
                    symbol,
                    sec_time AS time,
                    MAX(CASE WHEN rn_open = 1 THEN open END) AS open,
                    MAX(high) AS high,
                    MIN(low) AS low,
                    MAX(CASE WHEN rn_close = 1 THEN close END) AS close,
                    SUM(volume) AS volume,
                    SUM(trades) AS trades
                FROM ranked
                GROUP BY symbol, sec_time
                """
            )

            conn.executescript(
                """
                DROP TABLE candles_1s;
                ALTER TABLE candles_1s_new RENAME TO candles_1s;
                CREATE INDEX IF NOT EXISTS idx_candles_symbol_time ON candles_1s(symbol, time);
                """
            )

    def _is_ingested(self, file_key: str) -> bool:
        with self._conn() as conn:
            if self._use_postgres:
                row = conn.execute(
                    "SELECT 1 FROM ingested_files WHERE file_key = %s",
                    (file_key,),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT 1 FROM ingested_files WHERE file_key = ?",
                    (file_key,),
                ).fetchone()
            return row is not None

    def _mark_ingested(self, file_key: str, symbol: str, source: str) -> None:
        with self._conn() as conn:
            if self._use_postgres:
                conn.execute(
                    """
                    INSERT INTO ingested_files (file_key, symbol, source, processed_at)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT(file_key) DO UPDATE SET
                        symbol = EXCLUDED.symbol,
                        source = EXCLUDED.source,
                        processed_at = EXCLUDED.processed_at
                    """,
                    (file_key, symbol, source, int(time.time())),
                )
            else:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO ingested_files (file_key, symbol, source, processed_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (file_key, symbol, source, int(time.time())),
                )

    def _set_progress_message(self, symbol: str, message: str) -> None:
        with self._conn() as conn:
            if self._use_postgres:
                conn.execute(
                    "UPDATE symbol_progress SET last_message = %s WHERE symbol = %s",
                    (message, symbol),
                )
            else:
                conn.execute(
                    "UPDATE symbol_progress SET last_message = ? WHERE symbol = ?",
                    (message, symbol),
                )

    def _set_recent_done(self, symbol: str) -> None:
        with self._conn() as conn:
            if self._use_postgres:
                conn.execute(
                    """
                    UPDATE symbol_progress
                    SET recent_bootstrap_done = 1, last_recent_sync_at = %s, last_message = %s
                    WHERE symbol = %s
                    """,
                    (int(time.time()), "recent bootstrap completed", symbol),
                )
            else:
                conn.execute(
                    """
                    UPDATE symbol_progress
                    SET recent_bootstrap_done = 1, last_recent_sync_at = ?, last_message = ?
                    WHERE symbol = ?
                    """,
                    (int(time.time()), "recent bootstrap completed", symbol),
                )

    def _advance_backfill_month(self, symbol: str, next_month: str, message: str) -> None:
        with self._conn() as conn:
            if self._use_postgres:
                conn.execute(
                    """
                    UPDATE symbol_progress
                    SET backfill_cursor_month = %s, last_backfill_sync_at = %s, last_message = %s
                    WHERE symbol = %s
                    """,
                    (next_month, int(time.time()), message, symbol),
                )
            else:
                conn.execute(
                    """
                    UPDATE symbol_progress
                    SET backfill_cursor_month = ?, last_backfill_sync_at = ?, last_message = ?
                    WHERE symbol = ?
                    """,
                    (next_month, int(time.time()), message, symbol),
                )

    def status(self) -> Dict[str, Any]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT symbol, recent_bootstrap_done, backfill_cursor_month,
                       last_recent_sync_at, last_backfill_sync_at, last_message
                FROM symbol_progress
                ORDER BY symbol
                """
            ).fetchall()
            counts = conn.execute(
                """
                SELECT symbol, COUNT(*) AS candle_count, MIN(time) AS min_time, MAX(time) AS max_time
                FROM candles_1s
                GROUP BY symbol
                """
            ).fetchall()

        count_map = {r["symbol"]: dict(r) for r in counts}
        out = []
        for r in rows:
            c = count_map.get(r["symbol"], {})
            out.append(
                {
                    "symbol": r["symbol"],
                    "recent_bootstrap_done": bool(r["recent_bootstrap_done"]),
                    "backfill_cursor_month": r["backfill_cursor_month"],
                    "last_recent_sync_at": r["last_recent_sync_at"],
                    "last_backfill_sync_at": r["last_backfill_sync_at"],
                    "last_message": r["last_message"],
                    "candle_count_1s": c.get("candle_count", 0),
                    "min_time": c.get("min_time"),
                    "max_time": c.get("max_time"),
                }
            )
        return {
            "db_path": self.cfg.db_path,
            "symbols": out,
        }

    def _download_zip(self, url: str) -> Optional[bytes]:
        try:
            with urlopen(url, timeout=self.cfg.request_timeout_seconds) as resp:
                content_length = resp.headers.get("Content-Length")
                if content_length:
                    try:
                        if int(content_length) > self.cfg.max_zip_download_bytes:
                            return None
                    except Exception:
                        pass
                chunks: List[bytes] = []
                total = 0
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > self.cfg.max_zip_download_bytes:
                        return None
                    chunks.append(chunk)
                return b"".join(chunks)
        except HTTPError as exc:
            if exc.code == 404:
                return None
            raise
        except URLError:
            return None

    def _insert_batch(self, symbol: str, rows: Iterable[Tuple[int, float, float, float, float, float, int]]) -> None:
        params = [(symbol, *r) for r in rows]
        if not params:
            return
        with self._conn() as conn:
            if self._use_postgres:
                conn.executemany(
                    """
                    INSERT INTO candles_1s (symbol, time, open, high, low, close, volume, trades)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT(symbol, time) DO UPDATE SET
                        open = EXCLUDED.open,
                        high = EXCLUDED.high,
                        low = EXCLUDED.low,
                        close = EXCLUDED.close,
                        volume = EXCLUDED.volume,
                        trades = EXCLUDED.trades
                    """,
                    params,
                )
            else:
                conn.executemany(
                    """
                    INSERT INTO candles_1s (symbol, time, open, high, low, close, volume, trades)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(symbol, time) DO UPDATE SET
                        open = excluded.open,
                        high = excluded.high,
                        low = excluded.low,
                        close = excluded.close,
                        volume = excluded.volume,
                        trades = excluded.trades
                    """,
                    params,
                )

    def _ingest_trade_zip(self, symbol: str, zip_bytes: bytes, file_key: str, source: str) -> int:
        if self._is_ingested(file_key):
            return 0

        with self._lock:
            if self._is_ingested(file_key):
                return 0

            with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
                names = zf.namelist()
                if not names:
                    return 0
                name = names[0]

                batch: List[Tuple[int, float, float, float, float, float, int]] = []
                inserted = 0
                current_sec: Optional[int] = None
                o = h = l = c = v = 0.0
                trades = 0

                with zf.open(name, "r") as fp:
                    reader = csv.reader(io.TextIOWrapper(fp, encoding="utf-8"))
                    for row in reader:
                        if not row:
                            continue
                        if row[0].lower() in {"id", "agg_trade_id"}:
                            continue
                        if len(row) < 5:
                            continue
                        try:
                            price = float(row[1])
                            qty = float(row[2])
                        except Exception:
                            continue

                        ts_ms: Optional[int] = None
                        for idx in (4, 5):
                            if idx >= len(row):
                                continue
                            try:
                                cand = int(float(row[idx]))
                            except Exception:
                                continue
                            if cand > 1_000_000_000_000:
                                ts_ms = cand
                                break
                        if ts_ms is None:
                            continue

                        sec = ts_ms // 1000
                        if current_sec is None:
                            current_sec = sec
                            o = h = l = c = price
                            v = qty
                            trades = 1
                            continue

                        if sec == current_sec:
                            h = max(h, price)
                            l = min(l, price)
                            c = price
                            v += qty
                            trades += 1
                            continue

                        batch.append((current_sec, o, h, l, c, v, trades))
                        if len(batch) >= 10000:
                            self._insert_batch(symbol, batch)
                            inserted += len(batch)
                            batch.clear()

                        current_sec = sec
                        o = h = l = c = price
                        v = qty
                        trades = 1

                if current_sec is not None:
                    batch.append((current_sec, o, h, l, c, v, trades))
                if batch:
                    self._insert_batch(symbol, batch)
                    inserted += len(batch)

            self._mark_ingested(file_key, symbol, source)
            return inserted

    def ingest_recent_days(self, symbol: str, days: int) -> int:
        if symbol not in SUPPORTED_SYMBOLS:
            return 0
        total = 0
        today = datetime.now(timezone.utc).date()
        for d in range(days):
            day = today - timedelta(days=d)
            day_str = day.strftime("%Y-%m-%d")
            key = f"daily:{symbol}:{day_str}"
            if self._is_ingested(key):
                continue
            url = VISION_SPOT_DAILY_TRADES.format(symbol=symbol, day=day_str)
            blob = self._download_zip(url)
            if not blob:
                continue
            total += self._ingest_trade_zip(symbol, blob, key, "daily")
        if total > 0:
            self._set_progress_message(symbol, f"recent ingest +{total} candles")
        self._set_recent_done(symbol)
        return total

    def backfill_month_step(self, symbol: str) -> int:
        if symbol not in SUPPORTED_SYMBOLS:
            return 0
        with self._conn() as conn:
            if self._use_postgres:
                row = conn.execute(
                    "SELECT backfill_cursor_month FROM symbol_progress WHERE symbol = %s",
                    (symbol,),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT backfill_cursor_month FROM symbol_progress WHERE symbol = ?",
                    (symbol,),
                ).fetchone()
        if not row:
            return 0
        month = row["backfill_cursor_month"]
        if month < self.cfg.oldest_backfill_month:
            self._set_progress_message(symbol, "backfill completed")
            return 0

        key = f"monthly:{symbol}:{month}"
        inserted = 0
        if not self._is_ingested(key):
            url = VISION_SPOT_MONTHLY_TRADES.format(symbol=symbol, month=month)
            blob = self._download_zip(url)
            if blob:
                inserted = self._ingest_trade_zip(symbol, blob, key, "monthly")

        next_month = _prev_month(month)
        self._advance_backfill_month(symbol, next_month, f"backfilled {month} (+{inserted})")
        return inserted

    def _query_1s(
        self,
        symbol: str,
        from_ts: int,
        to_ts: int,
        limit: Optional[int],
    ) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            if self._use_postgres:
                sql = """
                    SELECT time, open, high, low, close, volume, trades
                    FROM candles_1s
                    WHERE symbol = %s AND time >= %s AND time <= %s
                    ORDER BY time ASC
                """
                params: List[Any] = [symbol, from_ts, to_ts]
                if limit is not None and limit > 0:
                    sql += " LIMIT %s"
                    params.append(limit)
                rows = conn.execute(sql, tuple(params)).fetchall()
            else:
                sql = """
                    SELECT time, open, high, low, close, volume, trades
                    FROM candles_1s
                    WHERE symbol = ? AND time >= ? AND time <= ?
                    ORDER BY time ASC
                """
                params = [symbol, from_ts, to_ts]
                if limit is not None and limit > 0:
                    sql += " LIMIT ?"
                    params.append(limit)
                rows = conn.execute(sql, tuple(params)).fetchall()
        return [dict(r) for r in rows]

    def _query_aggregated(
        self,
        symbol: str,
        from_ts: int,
        to_ts: int,
        bucket_seconds: int,
        limit: Optional[int],
    ) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            if self._use_postgres:
                sql = """
                    WITH bucketed AS (
                        SELECT
                            ((time / %s) * %s) AS bucket_time,
                            time, open, high, low, close, volume
                        FROM candles_1s
                        WHERE symbol = %s AND time >= %s AND time <= %s
                    ),
                    ranked AS (
                        SELECT
                            bucket_time, time, open, high, low, close, volume,
                            ROW_NUMBER() OVER (PARTITION BY bucket_time ORDER BY time ASC) AS rn_open,
                            ROW_NUMBER() OVER (PARTITION BY bucket_time ORDER BY time DESC) AS rn_close
                        FROM bucketed
                    )
                    SELECT
                        bucket_time AS time,
                        MAX(CASE WHEN rn_open = 1 THEN open END) AS open,
                        MAX(high) AS high,
                        MIN(low) AS low,
                        MAX(CASE WHEN rn_close = 1 THEN close END) AS close,
                        SUM(volume) AS volume
                    FROM ranked
                    GROUP BY bucket_time
                    ORDER BY bucket_time ASC
                """
                params: List[Any] = [bucket_seconds, bucket_seconds, symbol, from_ts, to_ts]
                if limit is not None and limit > 0:
                    sql += " LIMIT %s"
                    params.append(limit)
                rows = conn.execute(sql, tuple(params)).fetchall()
            else:
                sql = """
                    WITH bucketed AS (
                        SELECT
                            ((time / :bucket) * :bucket) AS bucket_time,
                            time, open, high, low, close, volume
                        FROM candles_1s
                        WHERE symbol = :symbol AND time >= :from_ts AND time <= :to_ts
                    ),
                    ranked AS (
                        SELECT
                            bucket_time, time, open, high, low, close, volume,
                            ROW_NUMBER() OVER (PARTITION BY bucket_time ORDER BY time ASC) AS rn_open,
                            ROW_NUMBER() OVER (PARTITION BY bucket_time ORDER BY time DESC) AS rn_close
                        FROM bucketed
                    )
                    SELECT
                        bucket_time AS time,
                        MAX(CASE WHEN rn_open = 1 THEN open END) AS open,
                        MAX(high) AS high,
                        MIN(low) AS low,
                        MAX(CASE WHEN rn_close = 1 THEN close END) AS close,
                        SUM(volume) AS volume
                    FROM ranked
                    GROUP BY bucket_time
                    ORDER BY bucket_time ASC
                """
                if limit is not None and limit > 0:
                    sql += " LIMIT :limit"
                params = {
                    "symbol": symbol,
                    "from_ts": from_ts,
                    "to_ts": to_ts,
                    "bucket": bucket_seconds,
                    "limit": limit if limit is not None else -1,
                }
                rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def get_candles(
        self,
        symbol: str,
        interval: str,
        from_ts: int,
        to_ts: int,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if symbol not in SUPPORTED_SYMBOLS:
            return []
        if interval not in SUPPORTED_INTERVALS:
            return []
        if to_ts < from_ts:
            return []
        if interval == "1s":
            return self._query_1s(symbol, from_ts, to_ts, limit)
        return self._query_aggregated(symbol, from_ts, to_ts, SUPPORTED_INTERVALS[interval], limit)

    def seed_range_from_binance_klines(
        self,
        symbol: str,
        from_ts: int,
        to_ts: int,
        max_points: int = 12000,
    ) -> int:
        if symbol not in SUPPORTED_SYMBOLS:
            return 0
        if to_ts < from_ts:
            return 0
        max_points = max(1, min(500000, max_points))
        cursor_ms = from_ts * 1000
        end_ms = to_ts * 1000
        total = 0
        batch_rows: List[Tuple[int, float, float, float, float, float, int]] = []
        last_err: Optional[str] = None

        while cursor_ms <= end_ms and total < max_points:
            batch_limit = min(1000, max_points - total)
            req_end_ms = min(end_ms, cursor_ms + (batch_limit - 1) * 1000)
            query = (
                f"symbol={symbol}&interval=1s"
                f"&startTime={cursor_ms}&endTime={req_end_ms}&limit={batch_limit}"
            )
            data: Optional[str] = None
            for base in BINANCE_SPOT_KLINES_ENDPOINTS:
                url = f"{base}?{query}"
                try:
                    with urlopen(url, timeout=self.cfg.request_timeout_seconds) as resp:
                        data = resp.read().decode("utf-8")
                    break
                except HTTPError as exc:
                    last_err = f"{base}:http_{exc.code}"
                except URLError as exc:
                    last_err = f"{base}:url_{type(exc.reason).__name__}"
                except Exception as exc:
                    last_err = f"{base}:{type(exc).__name__}"

            if data is None:
                break

            try:
                rows = json.loads(data)
            except Exception:
                last_err = "klines:invalid_json"
                break
            if isinstance(rows, dict):
                code = rows.get("code")
                last_err = f"klines:api_error_{code}" if code is not None else "klines:api_error"
                break
            if not isinstance(rows, list):
                last_err = "klines:unexpected_payload"
                break
            if not rows:
                if total == 0 and last_err is None:
                    last_err = "klines:empty_window"
                break

            last_open_ms = None
            for k in rows:
                try:
                    sec = int(k[0]) // 1000
                    batch_rows.append(
                        (
                            sec,
                            float(k[1]),
                            float(k[2]),
                            float(k[3]),
                            float(k[4]),
                            float(k[5]),
                            0,
                        )
                    )
                    total += 1
                    last_open_ms = int(k[0])
                    if len(batch_rows) >= 5000:
                        self._insert_batch(symbol, batch_rows)
                        batch_rows.clear()
                except Exception:
                    continue

            if last_open_ms is None:
                break
            next_cursor = last_open_ms + 1000
            if next_cursor <= cursor_ms:
                break
            cursor_ms = next_cursor

        if batch_rows:
            self._insert_batch(symbol, batch_rows)
            batch_rows.clear()
        if total > 0:
            self._set_progress_message(symbol, f"seeded from klines +{total}")
        else:
            self._set_progress_message(symbol, f"klines seed returned 0 ({last_err or 'no_data'})")
        return total

    def seed_recent_from_binance_klines(self, symbol: str, seconds: int = 3600, max_points: Optional[int] = None) -> int:
        if symbol not in SUPPORTED_SYMBOLS:
            return 0
        now_sec = int(time.time())
        start_sec = max(0, now_sec - max(1, seconds))
        points = max_points if max_points is not None else min(500000, max(12000, seconds + 120))
        return self.seed_range_from_binance_klines(symbol, start_sec, now_sec, max_points=points)


class VisionBackfillService:
    def __init__(self, cache: VisionCandleCache, cfg: VisionConfig) -> None:
        self.cache = cache
        self.cfg = cfg
        self._task: Optional[asyncio.Task] = None
        self._stop = asyncio.Event()
        self._symbol_index = 0
        self._symbols = sorted(SUPPORTED_SYMBOLS)

    async def start(self) -> None:
        if not self.cfg.worker_enabled:
            return
        if self._task and not self._task.done():
            return
        self._stop.clear()
        self._task = asyncio.create_task(self._run_loop(), name="vision-backfill-loop")

    async def stop(self) -> None:
        self._stop.set()
        if self._task:
            await self._task
            self._task = None

    async def ensure_recent(self, symbol: str, days: Optional[int] = None) -> int:
        if self.cfg.low_memory_mode:
            return await asyncio.to_thread(
                self.cache.seed_recent_from_binance_klines,
                symbol,
                self.cfg.klines_seed_seconds,
            )
        d = days if days is not None else self.cfg.bootstrap_days_on_demand
        return await asyncio.to_thread(self.cache.ingest_recent_days, symbol, d)

    async def _run_loop(self) -> None:
        while not self._stop.is_set():
            try:
                if self.cfg.low_memory_mode:
                    for symbol in self._symbols:
                        await asyncio.to_thread(
                            self.cache.seed_recent_from_binance_klines,
                            symbol,
                            self.cfg.klines_seed_seconds,
                        )
                    await asyncio.wait_for(self._stop.wait(), timeout=self.cfg.worker_interval_seconds)
                    continue

                # Keep recent data warm for all symbols.
                for symbol in self._symbols:
                    await asyncio.to_thread(self.cache.ingest_recent_days, symbol, self.cfg.recent_days)

                # Backfill one symbol per loop iteration for fairness.
                symbol = self._symbols[self._symbol_index % len(self._symbols)]
                self._symbol_index += 1
                await asyncio.to_thread(self.cache.backfill_month_step, symbol)
            except Exception as exc:
                # Keep worker alive on transient failures.
                for symbol in self._symbols:
                    self.cache._set_progress_message(symbol, f"worker error: {type(exc).__name__}")
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.cfg.worker_interval_seconds)
            except asyncio.TimeoutError:
                continue

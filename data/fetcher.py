import ccxt
import pandas as pd
import os
import time
from datetime import datetime, timezone

from config import DATA_DIR, TIMEFRAME, START_DATE


class DataFetcher:
    def __init__(self, exchange_name='binance'):
        self.exchange = getattr(ccxt, exchange_name)()
        self.exchange.enableRateLimit = True
        os.makedirs(DATA_DIR, exist_ok=True)

    def _cache_path(self, symbol):
        name = symbol.replace('/', '_')
        return os.path.join(DATA_DIR, f'{name}_{TIMEFRAME}.csv')

    def fetch_ohlcv(self, symbol):
        """拉取K线数据，优先读缓存。从START_DATE到当前"""
        cache_file = self._cache_path(symbol)

        if os.path.exists(cache_file):
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            if not df.empty:
                return df

        since = self.exchange.parse8601(f'{START_DATE}T00:00:00Z')
        all_ohlcv = []

        while True:
            ohlcv = self.exchange.fetch_ohlcv(symbol, TIMEFRAME, since=since, limit=1000)
            if not ohlcv:
                break
            all_ohlcv.extend(ohlcv)
            since = ohlcv[-1][0] + 1
            time.sleep(self.exchange.rateLimit / 1000)

        df = pd.DataFrame(
            all_ohlcv,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        df = df[~df.index.duplicated()]
        df.to_csv(cache_file)
        return df

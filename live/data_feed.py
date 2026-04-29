"""
LiveDataFeed — 实时K线数据源
从cache加载历史 + ccxt轮询最新bar + bar闭合检测
"""
import os
import time
import ccxt
import pandas as pd
import numpy as np


class LiveDataFeed:
    def __init__(self, symbols, timeframe='1h', cache_dir='data/cache'):
        self.symbols = list(symbols)
        self.timeframe = timeframe
        self.cache_dir = cache_dir
        self.df_dict = {}
        self._last_closed_time = None
        self._bar_count = 0
        self._all_times = []

        self.exchange = getattr(ccxt, 'binance')()
        self.exchange.enableRateLimit = True

    def _cache_path(self, symbol):
        name = symbol.replace('/', '_')
        return os.path.join(self.cache_dir, f'{name}_{self.timeframe}.csv')

    def load_history(self):
        """从cache CSV加载历史数据"""
        for symbol in self.symbols:
            cache_file = self._cache_path(symbol)
            if os.path.exists(cache_file):
                df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                if not df.empty:
                    # 统一时区：CSV数据是UTC但无时区标记，添加UTC时区
                    if df.index.tz is None:
                        df.index = df.index.tz_localize('UTC')
                    self.df_dict[symbol] = df
                    print(f'  {symbol}: {len(df)} 根K线 [{df.index[0]} ~ {df.index[-1]}]')
                else:
                    raise RuntimeError(f'{symbol} cache file is empty: {cache_file}')
            else:
                raise RuntimeError(f'{symbol} cache not found: {cache_file}. Run main_backtest.py first to download data.')

        self._all_times = sorted(set.union(*[set(df.index) for df in self.df_dict.values()]))
        self._last_closed_time = max(self._all_times)
        self._bar_count = len(self._all_times)
        print(f'  共 {self._bar_count} 个时间点，最新: {self._last_closed_time}')
        return self.df_dict

    def sync_latest(self):
        """拉取最新K线，检测新bar闭合。返回新bar时间戳或None

        在1h K线中，当 exchange 返回的最新 bar 的 timestamp 比上次闭合时间更新，
        且该 bar 不是当前正在形成的 bar（通过比较当前时间判断），则视为新闭合 bar。
        """
        new_closed = None
        now = pd.Timestamp.now(tz='UTC')

        for symbol in self.symbols:
            try:
                ohlcv = self.exchange.fetch_ohlcv(symbol, self.timeframe, limit=5)
            except Exception as e:
                print(f'  [WARN] fetch_ohlcv({symbol}) failed: {e}')
                return None

            if not ohlcv or len(ohlcv) < 2:
                return None

            # 最后两个bar: 倒数第二个是最近闭合的，最后一个是当前未闭合的
            new_bars = []
            for row in ohlcv:
                ts = pd.Timestamp(row[0], unit='ms', tz='UTC')
                if ts > self._last_closed_time:
                    new_bars.append({
                        'timestamp': ts,
                        'open': row[1], 'high': row[2], 'low': row[3],
                        'close': row[4], 'volume': row[5]
                    })

            if not new_bars:
                continue

            # 最后一个 new_bar 可能是未闭合的当前bar，排除它
            last_bar_time = new_bars[-1]['timestamp']
            # 1h bar: 如果当前时间距离 bar 开始不足 55 分钟，认为未闭合
            bar_age_minutes = (now - last_bar_time).total_seconds() / 60
            timeframe_minutes = 60 if self.timeframe == '1h' else int(self.timeframe[:-1]) * {'m': 1, 'h': 60, 'd': 1440}[self.timeframe[-1]]

            closed_bars = new_bars[:-1] if bar_age_minutes < timeframe_minutes * 0.9 else new_bars

            if closed_bars:
                new_rows = pd.DataFrame(closed_bars).set_index('timestamp')
                new_rows = new_rows[~new_rows.index.duplicated()]
                # 只追加不存在于当前df_dict的时间
                existing_idx = self.df_dict[symbol].index
                new_rows = new_rows[~new_rows.index.isin(existing_idx)]
                if not new_rows.empty:
                    self.df_dict[symbol] = pd.concat([self.df_dict[symbol], new_rows])
                    self.df_dict[symbol].sort_index(inplace=True)

            if closed_bars:
                candidate_time = closed_bars[-1]['timestamp']
                if candidate_time > self._last_closed_time:
                    new_closed = candidate_time

        if new_closed is not None:
            self._last_closed_time = new_closed
            self._bar_count += 1
            self._all_times.append(new_closed)

        return new_closed

    def get_df_dict(self):
        return self.df_dict

    @property
    def bar_count(self):
        return self._bar_count

    @property
    def last_time(self):
        return self._last_closed_time

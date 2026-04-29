"""
Funding Rate 数据获取（独立于OHLCV价格数据）
信息源: 永续合约资金费率 — 反映多空拥挤程度
"""
import ccxt
import pandas as pd
import os
import time
from datetime import datetime
from config import DATA_DIR, START_DATE


class FundingFetcher:
    def __init__(self):
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'future'},
        })
        self.cache_dir = os.path.join(DATA_DIR, 'funding')
        os.makedirs(self.cache_dir, exist_ok=True)

    def _cache_path(self, symbol):
        name = symbol.replace('/', '')
        return os.path.join(self.cache_dir, f'{name}_funding.csv')

    def fetch_all(self, symbols, start_date=START_DATE):
        """获取所有币种的funding rate历史"""
        since_ms = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)

        result = {}
        for sym in symbols:
            df = self._fetch_single(sym, since_ms)
            if not df.empty:
                result[sym] = df
                print(f'  {sym}: {len(df)} 条funding [{df.index[0]} ~ {df.index[-1]}]')
            else:
                print(f'  {sym}: 无数据')
        return result

    def _fetch_single(self, symbol, since_ms):
        cache_file = self._cache_path(symbol)
        if os.path.exists(cache_file):
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            if not df.empty:
                return df

        symbol_fmt = symbol.replace('/', '')
        all_data = []

        try:
            end_time = None  # 首次不设endTime，拿最新数据
            while True:
                params = {'symbol': symbol_fmt, 'limit': 1000}
                if end_time is not None:
                    params['endTime'] = end_time

                data = self.exchange.fapiPublicGetFundingRate(params)
                if not data:
                    break

                all_data.extend(data)
                oldest = min(int(d['fundingTime']) for d in data)

                # 如果最老记录已经早于START_DATE，停止
                if oldest <= since_ms:
                    break

                # 往回翻：下页拿比当前页最老记录更早的数据
                end_time = oldest - 1
                time.sleep(0.2)
        except Exception as e:
            print(f'    {symbol}: funding fetch warning: {e}')

        if not all_data:
            return pd.DataFrame()

        df = pd.DataFrame(all_data)
        df['timestamp'] = pd.to_datetime(df['fundingTime'].astype(int), unit='ms')
        df['fundingRate'] = df['fundingRate'].astype(float)
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        df = df[~df.index.duplicated()]
        df = df[['fundingRate']]
        df.to_csv(cache_file)
        return df


def align_funding_to_bars(funding_df, bar_index):
    """
    将funding rate数据对齐到K线时间轴
    使用前向填充：每根K线取最近一次funding rate
    """
    aligned = pd.Series(index=bar_index, dtype=float)
    funding_sorted = funding_df.sort_index()

    for t in bar_index:
        mask = funding_sorted.index <= t
        if mask.any():
            aligned[t] = funding_sorted[mask]['fundingRate'].iloc[-1]
        else:
            aligned[t] = 0.0

    return aligned

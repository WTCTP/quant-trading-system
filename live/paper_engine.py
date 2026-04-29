"""
PaperTradingEngine — 纸盘交易引擎
继承 PortfolioBacktest，复用所有信号/优化/风控逻辑
将 backtest 的历史循环替换为实时 bar 等待
"""
import time
import numpy as np
import pandas as pd
from datetime import datetime

from config import (
    SYMBOLS, FEE_RATE, SLIPPAGE_K, SLIPPAGE_MIN_BPS, SLIPPAGE_MAX_BPS,
    MAX_POSITION_ADV_PCT, MAX_ORDER_ADV_PCT,
    RISK_PORTFOLIO_STOP, RISK_VOL_SHOCK, RISK_CONSECUTIVE_LOSS, RISK_PAUSE_BARS,
    VOL_LOOKBACK, VOL_SCALE_CAP, VOL_FLOOR,
    MIN_WEIGHT_DELTA, MAKER_TIMEOUT_BARS, MAX_RETRIES,
    MAKER_SLIP_BPS, TAKER_SLIP_EXTRA_BPS,
    FUNDING_COST_ENABLED, TRAIN_WINDOW, RETRAIN_EVERY,
)
from portfolio.executor import PortfolioExecutor
from alpha.model import RegimeAlphaModel
from backtest.engine import PortfolioBacktest


class PaperTradingEngine(PortfolioBacktest):
    """纸盘引擎：复用回测逻辑，实时驱动"""

    def initialize(self, df_dict):
        """在历史数据上初始化：创建executor、训练模型、加载funding数据"""
        self.symbols = list(df_dict.keys())
        n = len(self.symbols)

        print(f'初始化纸盘引擎: {n} 个交易对, 初始资金 ${self.initial_capital:,.0f}')
        print(f'历史数据最新: {max(df_dict[s].index[-1] for s in self.symbols)}')

        # 创建执行器
        self.executor = PortfolioExecutor(
            self.symbols, self.initial_capital,
            fee_rate=FEE_RATE,
            slippage_k=SLIPPAGE_K,
            slippage_min_bps=SLIPPAGE_MIN_BPS,
            slippage_max_bps=SLIPPAGE_MAX_BPS,
            max_position_adv_pct=MAX_POSITION_ADV_PCT,
            max_order_adv_pct=MAX_ORDER_ADV_PCT,
            risk_portfolio_stop=RISK_PORTFOLIO_STOP,
            risk_vol_shock=RISK_VOL_SHOCK,
            risk_consecutive_loss=RISK_CONSECUTIVE_LOSS,
            risk_pause_bars=RISK_PAUSE_BARS,
            vol_lookback=VOL_LOOKBACK,
            vol_scale_cap=VOL_SCALE_CAP,
            vol_floor=VOL_FLOOR,
            min_weight_delta=MIN_WEIGHT_DELTA,
            maker_timeout_bars=MAKER_TIMEOUT_BARS,
            max_retries=MAX_RETRIES,
            maker_slip_bps=MAKER_SLIP_BPS,
            taker_slip_extra_bps=TAKER_SLIP_EXTRA_BPS,
        )

        # 初始化模型
        self.models = {s: RegimeAlphaModel() for s in self.symbols}
        self.prev_signal = np.zeros(n)

        # 加载Funding Rate数据
        if self.use_funding:
            from data.funding_fetcher import FundingFetcher, align_funding_to_bars
            try:
                ff = FundingFetcher()
                funding_raw = ff.fetch_all(self.symbols)
                bar_idx = df_dict[self.symbols[0]].index
                self.funding_data = {
                    s: align_funding_to_bars(funding_raw.get(s, pd.DataFrame()), bar_idx)
                    for s in self.symbols
                }
                print(f'  Funding数据已加载')
            except Exception as e:
                print(f'  Warning: funding data unavailable ({e}), disabling')
                self.use_funding = False
                self.funding_data = None

        # 初始ADV
        last_time = max(df_dict[s].index[-1] for s in self.symbols)
        self.executor.update_adv(df_dict, last_time)

        # 在全部历史数据上训练模型
        print('  训练Alpha模型...')
        for s in self.symbols:
            train_df = self._prepare_slice(df_dict[s], df_dict[s].index[-1])
            if len(train_df) >= TRAIN_WINDOW:
                self.models[s].train(train_df)
        print(f'  训练完成 ({len(self.models)} 个模型)')

        print('  纸盘引擎初始化完成，等待实时bar...\n')

    def run_live(self, data_feed):
        """主循环：轮询新bar → process_bar → 输出状态"""
        self._feed = data_feed

        print('纸盘交易开始 (Ctrl+C 停止)')
        print(f'{"─" * 70}')
        print(f'  {"时间":20s} {"资金":>10s} {"Regime":>8s} {"信号":>8s} {"订单":>6s}')
        print(f'  {"─" * 70}')

        while True:
            try:
                new_time = data_feed.sync_latest()
            except Exception as e:
                print(f'  [ERROR] sync_latest: {e}')
                time.sleep(30)
                continue

            if new_time:
                df_dict = data_feed.get_df_dict()
                self.process_bar(data_feed.bar_count, new_time, df_dict)
                self._print_live_status(new_time, df_dict)

            time.sleep(30)

    def _print_live_status(self, time, df_dict):
        """输出当前状态到控制台"""
        prices = self.executor.get_prices(df_dict, time)
        total_value = self.executor.portfolio_value(prices)
        pending = len(self.executor.pending_orders)

        time_str = str(time)[:19] if hasattr(time, '__str__') else str(time)

        print(f'  {time_str:20s} ${total_value:>9,.0f} {self.current_regime:>8s} '
              f'{self.prev_signal_max if hasattr(self, "prev_signal_max") else "-":>8s} '
              f'{pending:>5d}')

"""
纸盘交易系统 — Paper Trading
实时行情 + 模拟成交，观察虚拟订单在真实市场中的行为
"""
import sys
import signal

from config import (
    SYMBOLS, INITIAL_CAPITAL, TIMEFRAME,
    BASE_LOW, BASE_MID, BASE_HIGH,
    PORTFOLIO_VOL_TARGET, PORTFOLIO_LEVERAGE,
    USE_FUNDING, USE_CROSS_SECTIONAL,
)
from live.data_feed import LiveDataFeed
from live.paper_engine import PaperTradingEngine
from risk.manager import RiskManager
from log.recorder import TradeLogger


def main():
    print('=' * 60)
    print('  纸盘交易系统 — Paper Trading')
    print('=' * 60)
    print(f'交易对: {", ".join(SYMBOLS)}')
    print(f'K线周期: {TIMEFRAME} | 初始资金: ${INITIAL_CAPITAL:,.0f}')
    print(f'目标波动率: {PORTFOLIO_VOL_TARGET:.0%} | 杠杆: {PORTFOLIO_LEVERAGE}x')
    print(f'Alpha: Breakout-Only (生产配置)')
    print()

    # 1. 加载历史数据
    print('Step 1/3: 加载历史数据')
    feed = LiveDataFeed(SYMBOLS, timeframe=TIMEFRAME)
    df_dict = feed.load_history()

    if not df_dict:
        print('错误: 无历史数据。请先运行 main_backtest.py 下载数据。')
        return

    # 2. 初始化纸盘引擎
    print()
    print('Step 2/3: 初始化纸盘引擎')
    risk_mgr = RiskManager(INITIAL_CAPITAL)
    logger = TradeLogger('trades_paper.csv')
    engine = PaperTradingEngine(
        risk_mgr, logger,
        base_low=BASE_LOW, base_mid=BASE_MID, base_high=BASE_HIGH,
        portfolio_vol_target=PORTFOLIO_VOL_TARGET,
        portfolio_leverage=PORTFOLIO_LEVERAGE,
        use_funding=USE_FUNDING,
        use_cross_sectional=USE_CROSS_SECTIONAL,
    )
    engine.initialize(df_dict)

    # 3. 进入实时循环
    print()
    print('Step 3/3: 进入实时循环')

    def handle_signal(sig, frame):
        print('\n纸盘交易已停止')
        trades_df = engine.get_trades_df()
        if not trades_df.empty:
            trades_df.to_csv('trades_paper.csv', index=False)
            print(f'交易记录已保存到 trades_paper.csv ({len(trades_df)} 笔)')
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    engine.run_live(feed)


if __name__ == '__main__':
    main()

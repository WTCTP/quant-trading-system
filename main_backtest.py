"""
量化交易系统 v2 - 组合优化版回测
架构: Alpha(ML分类) → 收缩 → EWMA协方差 → Σ⁻¹μ优化 → 权重平滑
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd

from config import (
    SYMBOLS, INITIAL_CAPITAL, START_DATE,
    TRAIN_WINDOW, RETRAIN_EVERY, FORWARD_BARS,
    FEE_RATE, SLIPPAGE_BPS, SHRINK_MU,
    MIN_WEIGHT_CHANGE, MIN_TRADE_INTERVAL_HOURS,
    REGIME_LOW_THRESH, REGIME_HIGH_THRESH,
)
from data.fetcher import DataFetcher
from risk.manager import RiskManager
from log.recorder import TradeLogger
from backtest.engine import PortfolioBacktest


def prepare_data():
    fetcher = DataFetcher()
    df_dict = {}
    for symbol in SYMBOLS:
        df = fetcher.fetch_ohlcv(symbol)
        if df.empty:
            continue
        df_dict[symbol] = df
        print(f'  {symbol}: {len(df)} 根K线 [{df.index[0]} ~ {df.index[-1]}]')
    return df_dict


def print_section(title):
    print()
    print('─' * 60)
    print(f'  {title}')
    print('─' * 60)


def main():
    print('=' * 60)
    print('  量化交易系统 v2 — 组合优化回测')
    print('=' * 60)
    print(f'交易对: {", ".join(SYMBOLS)}')
    print(f'数据起点: {START_DATE}')
    print(f'初始资金: {INITIAL_CAPITAL} USDT')
    from config import MIN_HOLD_BARS, EXIT_BUFFER_BARS, SIGNAL_CONFIRM, REGIME_ENTRY_BARS
    from config import SIGNAL_CHANGE_THRESH, COOLING_BARS
    from config import REGIME_LOW_THRESH, REGIME_HIGH_THRESH, VOL_REGIME_EMA_SPAN
    from config import ENTRY_DELAY_BARS, ENTRY_PRICE_CONFIRM
    print(f'架构: 半连续执行 (迟滞层 + 时机优化)')
    print(f'  Regime平滑: EMA{ VOL_REGIME_EMA_SPAN}bar 带[{REGIME_LOW_THRESH}~{REGIME_HIGH_THRESH}] | 确认{REGIME_ENTRY_BARS}bar | 缓冲{EXIT_BUFFER_BARS}bar')
    print(f'  暴露迟滞: 等级变化(base↔mid↔full)才调仓 | 冷却{COOLING_BARS}bar')
    print(f'  入场时机: 升仓延迟{ENTRY_DELAY_BARS}bar + 价格确认(期间至少一次突破)')
    print(f'训练窗口: {TRAIN_WINDOW}根 | 重训间隔: {RETRAIN_EVERY}根')
    print(f'手续费: {FEE_RATE:.2%} | 滑点: {SLIPPAGE_BPS:.2%}')
    from config import DISABLE_TRADING, SIGNAL_INVERT, SIGNAL_CONFIRM_BARS
    from config import USE_SIGNAL_TIER, SIGNAL_Q_CORE, SIGNAL_Q_ATTACK, SIGNAL_CHANGE_FILTER
    from config import SIGNAL_MUST_INCREASE, PRICE_BREAKOUT_BARS, MIN_HOLD_BARS_EXEC
    if DISABLE_TRADING: print(f'🛑 [诊断模式] 禁止交易')
    if SIGNAL_INVERT: print(f'🔄 [诊断模式] 信号反转')
    if SIGNAL_CONFIRM_BARS: print(f'⏳ [诊断模式] 延迟确认: {SIGNAL_CONFIRM_BARS}bar')
    if USE_SIGNAL_TIER: print(f'📊 [分层仓位] Q8核心(≥Q{SIGNAL_Q_CORE:.0%}) weight=1.0 | Q9进攻(≥Q{SIGNAL_Q_ATTACK:.0%})')
    from config import BASE_EXPOSURE, SIGNAL_BOOST_MID, SIGNAL_BOOST_HIGH
    print(f'🔺 [升仓触发] 信号增强 + 最小持仓{MIN_HOLD_BARS_EXEC}bar')
    print(f'📈 [价格确认] 延迟期间至少一次BTC突破{PRICE_BREAKOUT_BARS}bar高点')
    print(f'📊 [暴露管理] base={BASE_EXPOSURE:.0%} | mid(≥{SIGNAL_BOOST_MID})→60% | high(≥{SIGNAL_BOOST_HIGH})→100%')

    # 获取数据
    print_section('Step 1/3: 获取数据')
    df_dict = prepare_data()
    if not df_dict:
        print('错误: 无数据')
        return

    # 初始化模块
    print_section('Step 2/3: 初始化模块')
    risk_mgr = RiskManager(INITIAL_CAPITAL)
    logger = TradeLogger('trades.csv')

    print_section('Step 3/3: Walk-Forward 回测 (每币种独立模型)')
    engine = PortfolioBacktest(risk_mgr, logger)
    results = engine.run(df_dict)

    # 输出
    print_section('一、回测概况')
    for k, v in results.items():
        print(f'  {k}: {v}')

    # 模型系数（分regime）
    coef = engine.get_coefficients()
    if coef:
        print_section('二、Regime模型系数（+ = 做多，- = 做空 | 高波动=不交易）')
        for symbol, regimes in coef.items():
            print(f'  [{symbol}]')
            for regime_name, items in regimes.items():
                print(f'    ── {regime_name} ──')
                for name, val in items[:4]:
                    sign = '+' if val > 0 else ' '
                    bar_len = int(abs(val) * 50)
                    bar = '█' * min(bar_len, 30)
                    print(f'      {name:18s} {sign}{val:+.4f}  {bar}')
            print()

    # Regime分析
    ra = engine.get_regime_analysis(df_dict)
    if ra:
        print_section('三、状态分组收益分析')
        for s in ra:
            if s['bars'] > 0:
                print(f'  {s["label"]:20s} | bars:{s["bars"]:6d} | return:{s["cum_return"]:>8s} | sharpe:{s["sharpe"]:>6}')
            else:
                print(f'  {s["label"]:20s} | 无数据')

    # 信号分层分析 (10档)
    ba = engine.get_signal_bucket_analysis(n_buckets=10)
    if ba:
        print_section('四、信号强度分层（中波内 | 10档 = 弱→强）')
        print(f'  {"档位":6s} {"K线数":>6s} {"信号范围":>16s} {"累计收益":>10s} {"Sharpe":>8s} {"胜率":>8s}')
        print(f'  {"─" * 60}')
        for b in ba:
            if b['bucket'] >= 7:
                marker = ' ★ Q8-Q9' if b['bucket'] >= 8 else ' ◆ Q7'
            else:
                marker = ''
            print(f'  Q{b["bucket"]:<5d} {b["count"]:>6d} {b["signal_range"]:>16s} '
                  f'{b["cum_return"]:>10s} {b["sharpe"]:>8} {b["winrate"]:>8s}{marker}')

    # 时间稳定性
    sa = engine.get_signal_stability_analysis()
    if sa:
        print_section('五、时间稳定性（10档 × 年份 | 仅显示Q8-Q9）')
        years = sorted(set(s['year'] for s in sa))
        print(f'  {"年份":6s} {"档位":6s} {"K线数":>6s} {"累计收益":>10s} {"Sharpe":>8s}')
        print(f'  {"─" * 50}')
        for y in years:
            for s in sa:
                if s['year'] == y and s['bucket'] >= 8:
                    print(f'  {s["year"]:<6d} Q{s["bucket"]:<5d} {s["count"]:>6d} '
                          f'{s["cum_return"]:>10s} {s["sharpe"]:>8}')
            print()

    # 进场偏移测试
    ta = engine.get_entry_timing_analysis(max_shift=5)
    if ta:
        print_section('六、进场偏移测试（信号 → 收益时间窗口）')
        print(f'  {"偏移":6s} {"档位":6s} {"K线数":>6s} {"5bar收益":>10s} {"Sharpe":>8s} {"胜率":>8s}')
        print(f'  {"─" * 55}')
        shifts = sorted(set(t['shift'] for t in ta))
        for shift in shifts:
            for t in ta:
                if t['shift'] == shift and t['bucket'] >= 7:
                    marker = ' ★' if t['bucket'] >= 8 else ' ◆'
                    print(f'  +{t["shift"]}bar  Q{t["bucket"]:<5d} {t["count"]:>6d} '
                          f'{t["cum_return"]:>9.2%} {t["sharpe"]:>8.2f} {t["winrate"]:>7.1%}{marker}')
            print()

    # 近期交易
    td = engine.get_trades_df()
    print_section('七、近期调仓（最后20笔）')
    if not td.empty:
        for _, t in td.tail(20).iterrows():
            side = '买入' if t['delta_value'] > 0 else '卖出'
            print(f'  {str(t["time"])[:19]} {t["symbol"]:10s} {side:4s} '
                  f'¥{abs(t["delta_value"]):8.2f}  权重→{t["weight"]:.2%}  费用{t["fee"]:.4f}')

    print()
    print(f'交易日志已保存到 trades.csv')


if __name__ == '__main__':
    main()

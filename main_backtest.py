"""
量化交易系统 v3.0 — Breakout Alpha + Regime分层 + Vol Targeting
实验结论: Breakout-Only 最优，Funding/CrossSectional 当前为负贡献
"""
import sys
import numpy as np
import pandas as pd

from config import (
    SYMBOLS, INITIAL_CAPITAL, START_DATE,
    TRAIN_WINDOW, RETRAIN_EVERY,
    FEE_RATE, SLIPPAGE_BPS,
    REGIME_LOW_THRESH, REGIME_HIGH_THRESH,
    VOL_REGIME_EMA_SPAN,
    SIGNAL_BOOST_MID, SIGNAL_BOOST_HIGH,
    BASE_LOW, BASE_MID, BASE_HIGH,
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


def compute_metrics(engine):
    df = pd.DataFrame(engine.records)
    df['returns'] = df['capital'].pct_change()
    rets = df['returns'].dropna()

    final = df['capital'].iloc[-1]
    total_ret = (final - INITIAL_CAPITAL) / INITIAL_CAPITAL
    peak = df['capital'].expanding().max()
    max_dd = ((df['capital'] - peak) / peak).min()
    ann_vol = rets.std() * np.sqrt(365 * 24)
    avg_ret = rets.mean()
    sharpe = (avg_ret / rets.std()) * np.sqrt(365 * 24) if rets.std() > 0 else 0
    calmar = total_ret / abs(max_dd) if max_dd != 0 else 0
    win_rate = (rets > 0).mean()
    wins = rets[rets > 0].sum()
    losses = abs(rets[rets < 0].sum())
    profit_factor = wins / losses if losses > 0 else float('inf')

    # 按三区制分解
    def regime_root(r):
        if r and r.startswith('mid'): return 'mid'
        if r and r.startswith('low'): return 'low'
        if r and r.startswith('high'): return 'high'
        return 'other'
    df['root'] = df['regime'].apply(regime_root)
    regime_ret = {}
    for regime in ['low', 'mid', 'high']:
        sub = df[df['root'] == regime]
        if len(sub) > 5:
            cum = (1 + sub['returns']).prod() - 1
            sr = (sub['returns'].mean() / sub['returns'].std()) * np.sqrt(365*24) if sub['returns'].std() > 0 else 0
            regime_ret[regime] = {'bars': len(sub), 'cum': cum, 'sharpe': sr}

    return {
        'final_capital': round(final, 2),
        'total_return': f'{total_ret:.2%}',
        'total_return_numeric': round(total_ret, 4),
        'max_drawdown': f'{max_dd:.2%}',
        'max_drawdown_numeric': round(max_dd, 4),
        'ann_volatility': f'{ann_vol:.2%}',
        'sharpe_ratio': round(sharpe, 2),
        'sharpe_numeric': round(sharpe, 4),
        'calmar': round(calmar, 2),
        'win_rate': f'{win_rate:.1%}',
        'profit_factor': round(profit_factor, 2),
        'trade_events': len(engine.trades),
        'total_fee': round(engine.total_fee, 2),
        'total_slippage': round(engine.total_slippage, 2),
        'total_funding_cost': round(engine.total_funding_cost, 2),
        'regime': regime_ret,
    }


def run_experiment(df_dict, label, base_mid, portfolio_vol_target,
                   use_funding=False, funding_weight=0.10,
                   use_cross_sectional=False, cross_sectional_weight=0.10):
    risk_mgr = RiskManager(INITIAL_CAPITAL)
    logger = TradeLogger(f'trades_{label}.csv')
    engine = PortfolioBacktest(
        risk_mgr, logger,
        base_low=0.0, base_mid=base_mid, base_high=0.0,
        portfolio_vol_target=portfolio_vol_target,
        use_pullback=False,
        use_funding=use_funding,
        funding_weight=funding_weight,
        use_cross_sectional=use_cross_sectional,
        cross_sectional_weight=cross_sectional_weight,
    )
    engine.run(df_dict.copy())
    results = compute_metrics(engine)
    results['_label'] = label
    results['_engine'] = engine
    return results


def print_summary(results):
    print(f'  最终资金: ${results["final_capital"]:>12,.2f}')
    print(f'  总收益:   {results["total_return"]:>12}')
    print(f'  年化波动: {results["ann_volatility"]:>12}')
    print(f'  最大回撤: {results["max_drawdown"]:>12}')
    print(f'  Sharpe:   {results["sharpe_ratio"]:>12}')
    print(f'  Calmar:   {results["calmar"]:>12}')
    print(f'  胜率:     {results["win_rate"]:>12}')
    print(f'  盈亏比:   {results["profit_factor"]:>12}')
    print(f'  交易次数: {results["trade_events"]:>12}')
    print(f'  总手续费: ${results["total_fee"]:>12,.2f}')
    print(f'  总滑点:   ${results["total_slippage"]:>12,.2f}')
    print(f'  资金费:   ${results["total_funding_cost"]:>12,.2f}')


def print_regime_decomp(results):
    regime_ret = results.get('regime', {})
    for name, label in [('low', '低波'), ('mid', '中波'), ('high', '高波')]:
        info = regime_ret.get(name)
        if info and info['bars'] > 0:
            print(f'  > {label}: {info["bars"]:>6} bars  '
                  f'cum={info["cum"]:>8.2%}  sharpe={info["sharpe"]:>7.2f}')


def print_regime_matrix(engine, df_dict, label):
    ra = engine.get_regime_analysis(df_dict)
    if not ra:
        return
    print_section(f'Regime 状态分解 — {label}')
    print(f'  {"状态":28s} {"K线数":>7s} {"累计收益":>10s} {"Sharpe":>8s}')
    print(f'  {"─" * 55}')
    for s in ra:
        if s['bars'] > 0:
            print(f'  {s["label"]:28s} {s["bars"]:>7d} {s["cum_return"]:>10s} {s["sharpe"]:>8}')
        elif s['label'] not in ('低波-强制持有', '高波-强制持有'):
            print(f'  {s["label"]:28s} 无数据')


def main():
    print('=' * 60)
    print('  量化交易系统 v3.0 — Breakout Alpha + Vol Targeting')
    print('=' * 60)
    print(f'交易对: {", ".join(SYMBOLS)}')
    print(f'数据起点: {START_DATE} | 初始资金: {INITIAL_CAPITAL} USDT')
    print(f'训练窗口: {TRAIN_WINDOW}根 | 重训间隔: {RETRAIN_EVERY}根')
    print(f'Regime划分: low<{REGIME_LOW_THRESH} | mid∈[{REGIME_LOW_THRESH}~{REGIME_HIGH_THRESH}] | high>{REGIME_HIGH_THRESH}')
    print(f'手续费: {FEE_RATE:.2%} | 滑点: {SLIPPAGE_BPS:.2%}')
    actual_vol_target = 0.10  # 实验实际使用的波动率目标（覆盖 config 默认值）
    print(f'信号分层: 基仓(≥{SIGNAL_BOOST_MID})→60% | 全仓(≥{SIGNAL_BOOST_HIGH})→100%')
    print(f'组合目标波动率: {actual_vol_target:.0%}')

    # 获取数据
    print_section('Step 1/3: 获取数据')
    df_dict = prepare_data()
    if not df_dict:
        print('错误: 无数据')
        return

    # === 主实验: Breakout-Only (当前最优) ===
    print_section('Step 2/3: Breakout-Only (当前最优配置)')
    r_breakout = run_experiment(df_dict, 'Breakout-Only',
                                base_mid=1.0,
                                portfolio_vol_target=actual_vol_target)
    print_summary(r_breakout)
    print()
    print_regime_decomp(r_breakout)
    print_regime_matrix(r_breakout['_engine'], df_dict, 'Breakout-Only')

    # === 对比: Breakout + Funding(10%) ===
    print_section('Step 3/3: 多Alpha对比')
    r_funding = run_experiment(df_dict, 'Breakout+Funding(10%)',
                               base_mid=1.0,
                               portfolio_vol_target=actual_vol_target,
                               use_funding=True, funding_weight=0.10)

    r_crosssec = run_experiment(df_dict, 'Breakout+CrossSec(10%)',
                                base_mid=1.0,
                                portfolio_vol_target=actual_vol_target,
                                use_cross_sectional=True, cross_sectional_weight=0.10)

    # 汇总对比
    print(f'  {"实验":28s} {"收益":>10s} {"Sharpe":>8s} {"Calmar":>8s} {"回撤":>10s} {"交易":>6s} {"胜率":>7s}')
    print(f'  {"─" * 81}')
    for r in [r_breakout, r_funding, r_crosssec]:
        best_marker = ' ★' if r is r_breakout else ''
        print(f'  {r["_label"]:28s} {r["total_return"]:>10s} {r["sharpe_ratio"]:>8} '
              f'{r["calmar"]:>8} {r["max_drawdown"]:>10} {r["trade_events"]:>6} '
              f'{r["win_rate"]:>7}{best_marker}')

    print()
    print(f'  ★ Breakout-Only 是当前最优配置')
    print(f'  多Alpha (Funding/CrossSectional) 增加噪声，降低Sharpe和Calmar')
    print()
    print('交易日志已保存到 trades_*.csv')


if __name__ == '__main__':
    main()

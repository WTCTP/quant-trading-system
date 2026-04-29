"""
Funding Rate Alpha 实验 — 独立信息源验证
Stage 1: 独立回测（验证alpha是否存在）
Stage 2: 与Breakout组合（验证是否提升Sharpe）
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, r'C:\Users\16079\Desktop\lianghua')
from config import SYMBOLS, INITIAL_CAPITAL, TRAIN_WINDOW, RETRAIN_EVERY
from data.fetcher import DataFetcher
from risk.manager import RiskManager
from log.recorder import TradeLogger
from backtest.engine import PortfolioBacktest


def compute_extended_metrics(records, trades):
    df = pd.DataFrame(records)
    df['returns'] = df['capital'].pct_change()
    rets = df['returns'].dropna()
    if len(rets) < 10:
        return {}
    final = df['capital'].iloc[-1]
    total_ret = (final - INITIAL_CAPITAL) / INITIAL_CAPITAL
    ann_vol = rets.std() * np.sqrt(365 * 24)
    peak = df['capital'].expanding().max()
    max_dd = ((df['capital'] - peak) / peak).min()
    avg_ret = rets.mean()
    sharpe = (avg_ret / rets.std()) * np.sqrt(365 * 24) if rets.std() > 0 else 0
    calmar = total_ret / abs(max_dd) if max_dd != 0 else 0
    win_rate = (rets > 0).mean()

    return {
        'final_capital': round(final, 2),
        'total_return': f'{total_ret:.2%}',
        'total_return_numeric': round(total_ret, 4),
        'ann_volatility': f'{ann_vol:.2%}',
        'max_drawdown': f'{max_dd:.2%}',
        'max_drawdown_numeric': round(max_dd, 4),
        'sharpe': round(sharpe, 2),
        'sharpe_numeric': round(sharpe, 4),
        'calmar': round(calmar, 2),
        'win_rate': f'{win_rate:.1%}',
        'n_trades': len(trades),
        'total_fee': round(sum(t['fee'] for t in trades), 2),
    }


def print_regime_decomp(records):
    df = pd.DataFrame(records)
    df['returns'] = df['capital'].pct_change()
    def root(r):
        if r and 'mid' in str(r): return 'mid'
        if r and 'low' in str(r): return 'low'
        if r and 'high' in str(r): return 'high'
        return 'other'
    df['root'] = df['regime'].apply(root)
    for r in ['low', 'mid', 'high']:
        sub = df[df['root'] == r]
        if len(sub) > 5:
            cum = (1 + sub['returns']).prod() - 1
            sr = (sub['returns'].mean() / sub['returns'].std()) * np.sqrt(365*24) if sub['returns'].std() > 0 else 0
            print(f'    {r}: {len(sub)}b | cum={cum:.2%} | SR={sr:.2f}')


def run_funding_standalone(df_dict):
    """
    Stage 1: Funding-only（验证alpha独立存在）
    在mid regime只用funding信号交易（不用breakout）
    """
    print('\n' + '─'*55)
    print('  Stage 1: Funding-Only（独立验证）')
    print('─'*55)

    # 用特殊配置模拟funding-only：base_mid很小，主要靠funding
    risk = RiskManager(INITIAL_CAPITAL)
    logger = TradeLogger('trades_funding_only.csv')

    # 用funding作为主alpha
    engine = PortfolioBacktest(
        risk, logger,
        base_low=0.0, base_mid=0.30, base_high=0.0,
        portfolio_vol_target=0.08,
        use_pullback=False, use_cross_sectional=False,
        use_funding=True, funding_weight=1.0,  # 100% funding
    )
    results = engine.run(df_dict.copy())
    m = compute_extended_metrics(engine.records, engine.trades)
    m['_label'] = 'Funding-Only'
    m['_records'] = engine.records

    print_regime_decomp(engine.records)
    print(f'  收益: {m["total_return"]} | 回撤: {m["max_drawdown"]} | Sharpe: {m["sharpe"]}')
    print(f'  交易: {m["n_trades"]} | 手续费: ¥{m["total_fee"]}')
    return m


def run_combined(df_dict, funding_wt):
    """
    Stage 2: Breakout + Funding 组合
    """
    label = f'Breakout+Funding({int(funding_wt*100)}%)'
    print('\n' + '─'*55)
    print(f'  Stage 2: {label}')
    print('─'*55)

    risk = RiskManager(INITIAL_CAPITAL)
    logger = TradeLogger(f'trades_{label}.csv')
    engine = PortfolioBacktest(
        risk, logger,
        base_low=0.0, base_mid=0.30, base_high=0.0,
        portfolio_vol_target=0.08,
        use_pullback=False, use_cross_sectional=False,
        use_funding=True, funding_weight=funding_wt,
    )
    results = engine.run(df_dict.copy())
    m = compute_extended_metrics(engine.records, engine.trades)
    m['_label'] = label
    m['_records'] = engine.records

    print_regime_decomp(engine.records)
    print(f'  收益: {m["total_return"]} | 回撤: {m["max_drawdown"]} | Sharpe: {m["sharpe"]}')
    print(f'  交易: {m["n_trades"]} | 手续费: ¥{m["total_fee"]}')
    return m


def main():
    print('=' * 55)
    print('  Funding Rate Alpha — 独立信息源验证')
    print('=' * 55)

    fetcher = DataFetcher()
    df_dict = {}
    for s in SYMBOLS:
        df_dict[s] = fetcher.fetch_ohlcv(s)
        print(f'  {s}: {len(df_dict[s])} 根')

    # === Stage 1: Funding独立验证 ===
    r_fund = run_funding_standalone(df_dict)

    # === Stage 2: 不同权重组合 ===
    r_combos = []
    for wt in [0.10, 0.15, 0.20]:
        r = run_combined(df_dict, wt)
        r_combos.append(r)

    # === 对比 ===
    print()
    print('=' * 55)
    print('  对比总结')
    print('=' * 55)
    print(f'  {"实验":22s} {"收益":>10s} {"Sharpe":>8s} {"Calmar":>8s} {"交易":>6s}')
    print(f'  {"─" * 56}')

    for r in [r_fund] + r_combos:
        print(f'  {r["_label"]:22s} {r["total_return"]:>10} {r["sharpe"]:>8} {r["calmar"]:>8} {r["n_trades"]:>6}')

    # 结论
    print()
    fund_sharpe = r_fund['sharpe_numeric']
    if fund_sharpe > 0.5:
        print(f'  ✅ Funding alpha 独立有效 (Sharpe={fund_sharpe:.2f})')
    elif fund_sharpe > 0:
        print(f'  ⚠ Funding alpha 略有效但弱 (Sharpe={fund_sharpe:.2f})')
    else:
        print(f'  ❌ Funding alpha 无效 (Sharpe={fund_sharpe:.2f})')

    best_combo = max(r_combos, key=lambda r: r['sharpe_numeric'])
    print(f'  最佳组合: {best_combo["_label"]} (Sharpe={best_combo["sharpe"]})')


if __name__ == '__main__':
    main()

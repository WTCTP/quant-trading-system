"""
Cross-Sectional Alpha 实验: Breakout vs Breakout+CrossSectional
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


def compute_metrics(records, trades):
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
    sharpe = (rets.mean() / rets.std()) * np.sqrt(365 * 24) if rets.std() > 0 else 0
    calmar = total_ret / abs(max_dd) if max_dd != 0 else 0
    return {
        'final_capital': round(final, 2),
        'total_return': f'{total_ret:.2%}',
        'total_return_numeric': round(total_ret, 4),
        'ann_volatility': f'{ann_vol:.2%}',
        'max_drawdown': f'{max_dd:.2%}',
        'max_drawdown_numeric': round(max_dd, 4),
        'sharpe': round(sharpe, 2),
        'calmar': round(calmar, 2),
        'n_trades': len(trades),
        'total_fee': round(sum(t['fee'] for t in trades), 2),
    }


def run_experiment(df_dict, label, use_cs, cs_weight=0.30):
    print(f'\n{"─"*55}')
    print(f'  {label}')
    print(f'{"─"*55}')
    risk = RiskManager(INITIAL_CAPITAL)
    logger = TradeLogger(f'trades_{label}.csv')
    engine = PortfolioBacktest(
        risk, logger,
        base_low=0.0, base_mid=0.30, base_high=0.0,
        portfolio_vol_target=0.08,
        use_pullback=False, use_cross_sectional=use_cs,
        cross_sectional_weight=cs_weight,
    )
    results = engine.run(df_dict.copy())
    m = compute_metrics(engine.records, engine.trades)
    m['_label'] = label

    # 按regime分解
    df = pd.DataFrame(engine.records)
    df['returns'] = df['capital'].pct_change()
    def regime_root(r):
        if r and r.startswith('mid'): return 'mid'
        if r and r.startswith('low'): return 'low'
        if r and r.startswith('high'): return 'high'
        return 'other'
    df['root'] = df['regime'].apply(regime_root)

    for regime in ['low', 'mid', 'high']:
        sub = df[df['root'] == regime]
        if len(sub) > 5:
            cum = (1 + sub['returns']).prod() - 1
            sr = (sub['returns'].mean() / sub['returns'].std()) * np.sqrt(365*24) if sub['returns'].std() > 0 else 0
            print(f'    {regime}: {len(sub)} bars | cum={cum:.2%} | SR={sr:.2f}')

    print(f'    收益: {m["total_return"]} | 回撤: {m["max_drawdown"]} | Sharpe: {m["sharpe"]} | Calmar: {m["calmar"]}')
    print(f'    交易: {m["n_trades"]} | 手续费: ¥{m["total_fee"]}')
    return m


def main():
    print('=' * 55)
    print('  Cross-Sectional Alpha: Breakout vs Breakout+CrossSec')
    print('=' * 55)

    fetcher = DataFetcher()
    df_dict = {}
    for s in SYMBOLS:
        df_dict[s] = fetcher.fetch_ohlcv(s)
        print(f'  {s}: {len(df_dict[s])} 根')

    # 实验
    r_bo = run_experiment(df_dict, 'Breakout-Only', use_cs=False)
    r_cs = run_experiment(df_dict, 'Breakout+CrossSec(30%)', use_cs=True, cs_weight=0.30)

    # 对比
    print()
    print('=' * 55)
    print('  对比')
    print('=' * 55)
    print(f'  {"指标":18s} {"Breakout-Only":>14s} {"+CrossSec(30%)":>14s}')
    print(f'  {"─" * 48}')
    for name, key in [('收益', 'total_return'), ('Sharpe', 'sharpe'),
                       ('Calmar', 'calmar'), ('交易', 'n_trades'),
                       ('手续费', 'total_fee')]:
        print(f'  {name:18s} {str(r_bo[key]):>14s} {str(r_cs[key]):>14s}')

    ret_bo = r_bo['total_return_numeric']
    ret_cs = r_cs['total_return_numeric']
    sharpe_bo = r_bo['sharpe']
    sharpe_cs = r_cs['sharpe']

    print()
    print(f'  收益: {ret_bo:.2%} → {ret_cs:.2%} ({ret_cs-ret_bo:+.2%})')
    print(f'  Sharpe: {sharpe_bo} → {sharpe_cs} ({sharpe_cs-sharpe_bo:+.2f})')

    if sharpe_cs > sharpe_bo:
        print('  ✅ Cross-Sectional 提升了 Sharpe！不同信息源起作用了')
    elif ret_cs > ret_bo:
        print('  → 收益上升但Sharpe未升，需关注风险')
    else:
        print('  → Cross-Sectional 未改善，可能权重/频率需要调整')


if __name__ == '__main__':
    main()

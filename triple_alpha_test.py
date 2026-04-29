"""
三 Alpha 融合测试: Breakout + Funding + CrossSectional
扫描不同融合权重，找最优组合
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, r'C:\Users\16079\Desktop\lianghua')
from config import SYMBOLS, INITIAL_CAPITAL
from data.fetcher import DataFetcher
from risk.manager import RiskManager
from log.recorder import TradeLogger
from backtest.engine import PortfolioBacktest


def compute_metrics(engine):
    df = pd.DataFrame(engine.records)
    df['returns'] = df['capital'].pct_change()
    rets = df['returns'].dropna()
    if len(rets) < 10:
        return {}
    final = df['capital'].iloc[-1]
    total_ret = (final - INITIAL_CAPITAL) / INITIAL_CAPITAL
    peak = df['capital'].expanding().max()
    max_dd = ((df['capital'] - peak) / peak).min()
    sharpe = (rets.mean() / rets.std()) * np.sqrt(365 * 24) if rets.std() > 0 else 0
    calmar = total_ret / abs(max_dd) if max_dd != 0 else 0
    ann_vol = rets.std() * np.sqrt(365 * 24)
    win_rate = (rets > 0).mean()
    wins = rets[rets > 0].sum()
    losses = abs(rets[rets < 0].sum())
    profit_factor = wins / losses if losses > 0 else float('inf')

    def regime_root(r):
        if r and r.startswith('mid'): return 'mid'
        if r and r.startswith('low'): return 'low'
        if r and r.startswith('high'): return 'high'
        return 'other'
    df['root'] = df['regime'].apply(regime_root)
    regime_info = {}
    for regime in ['low', 'mid', 'high']:
        sub = df[df['root'] == regime]
        if len(sub) > 5:
            cum = (1 + sub['returns']).prod() - 1
            sr = (sub['returns'].mean() / sub['returns'].std()) * np.sqrt(365 * 24) if sub['returns'].std() > 0 else 0
            regime_info[regime] = {'bars': len(sub), 'cum': cum, 'sharpe': sr}

    return {
        'total_return': f'{total_ret:.2%}',
        'total_return_numeric': round(total_ret, 4),
        'ann_volatility': f'{ann_vol:.2%}',
        'max_drawdown': f'{max_dd:.2%}',
        'max_drawdown_numeric': round(max_dd, 4),
        'sharpe': round(sharpe, 2),
        'sharpe_numeric': round(sharpe, 4),
        'calmar': round(calmar, 2),
        'calmar_numeric': round(calmar, 4),
        'win_rate': f'{win_rate:.1%}',
        'profit_factor': round(profit_factor, 2),
        'n_trades': len(engine.trades),
        'total_fee': round(sum(t['fee'] for t in engine.trades), 2),
        'regime': regime_info,
    }


def run_config(df_dict, label, base_mid, funding_wt, cs_wt):
    print(f'\n{"─"*60}')
    print(f'  {label}')
    print(f'{"─"*60}')

    risk = RiskManager(INITIAL_CAPITAL)
    logger = TradeLogger(f'trades_{label}.csv')
    engine = PortfolioBacktest(
        risk, logger,
        base_low=0.0, base_mid=base_mid, base_high=0.0,
        portfolio_vol_target=0.08,
        use_pullback=False,
        use_funding=(funding_wt > 0),
        funding_weight=funding_wt,
        use_cross_sectional=(cs_wt > 0),
        cross_sectional_weight=cs_wt,
    )
    engine.run(df_dict.copy())
    m = compute_metrics(engine)
    m['_label'] = label

    for reg in ['low', 'mid', 'high']:
        info = m['regime'].get(reg, {})
        if info:
            print(f'    {reg}: {info["bars"]}b | cum={info["cum"]:.2%} | SR={info["sharpe"]:.2f}')
    print(f'    收益={m["total_return"]} | Sharpe={m["sharpe"]} | Calmar={m["calmar"]} | 交易={m["n_trades"]}')
    return m


def main():
    print('=' * 60)
    print('  三Alpha融合: Breakout + Funding + CrossSectional')
    print('=' * 60)

    fetcher = DataFetcher()
    df_dict = {}
    for s in SYMBOLS:
        df_dict[s] = fetcher.fetch_ohlcv(s)
        print(f'  {s}: {len(df_dict[s])} 根')

    results = []

    # Baseline: Breakout-Only
    results.append(run_config(df_dict, 'Breakout-Only', 0.30, 0.0, 0.0))

    # Breakout + Funding (best from sweep: 10%)
    results.append(run_config(df_dict, 'Breakout+Funding(10%)', 0.30, 0.10, 0.0))

    # Breakout + CrossSectional (need to find best weight)
    results.append(run_config(df_dict, 'Breakout+CrossSec(10%)', 0.30, 0.0, 0.10))

    # Triple: Breakout + Funding(10%) + CrossSec(various)
    for csw in [0.05, 0.10, 0.15]:
        label = f'Breakout+F10+CS{int(csw*100)}%'
        results.append(run_config(df_dict, label, 0.30, 0.10, csw))

    print()
    print('=' * 85)
    print('  融合对比结果')
    print('=' * 85)
    h = f'  {"实验":28s} {"收益":>10s} {"Sharpe":>8s} {"Calmar":>8s} {"回撤":>10s} {"胜率":>7s} {"交易":>6s}'
    print(h)
    print(f'  {"─" * 83}')

    best_sharpe = max(results, key=lambda r: r['sharpe_numeric'])
    best_calmar = max(results, key=lambda r: r['calmar_numeric'])
    best_score = max(results, key=lambda r: r['sharpe_numeric'] * r['calmar_numeric'])

    for r in results:
        markers = []
        if r is best_sharpe: markers.append('S')
        if r is best_calmar: markers.append('C')
        if r is best_score: markers.append('★')
        marker = ' ' + '/'.join(markers) if markers else ''
        print(f'  {r["_label"]:28s} {r["total_return"]:>10s} {r["sharpe"]:>8} '
              f'{r["calmar"]:>8} {r["max_drawdown"]:>10} {r["win_rate"]:>7} '
              f'{r["n_trades"]:>6}{marker}')

    print()
    bo = results[0]
    best = best_score
    print(f'  Baseline (Breakout-Only): Sharpe={bo["sharpe"]}, Calmar={bo["calmar"]}')
    print(f'  最优融合: {best["_label"]} (Sharpe={best["sharpe"]}, Calmar={best["calmar"]})')
    ret_diff = best['total_return_numeric'] - bo['total_return_numeric']
    sr_diff = best['sharpe_numeric'] - bo['sharpe_numeric']
    print(f'  收益变化: {bo["total_return_numeric"]:.2%} → {best["total_return_numeric"]:.2%} ({ret_diff:+.2%})')
    print(f'  Sharpe变化: {bo["sharpe"]} → {best["sharpe"]} ({sr_diff:+.2f})')

    if sr_diff > 0.05 and ret_diff > 0:
        print(f'\n  多Alpha融合成功提升系统质量')
    elif sr_diff > 0:
        print(f'\n  Sharpe提升但收益增幅有限')
    else:
        print(f'\n  多Alpha融合未显著优于单一Breakout')
    print()


if __name__ == '__main__':
    main()

"""
Cross-Sectional Alpha 权重扫描
在 Breakout + Funding(15%) 基础上，扫描 CS 权重 0%~30%
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, r'C:\Users\16079\Desktop\lianghua')
from config import SYMBOLS, INITIAL_CAPITAL, TRAIN_WINDOW
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
    ann_vol = rets.std() * np.sqrt(365 * 24)
    peak = df['capital'].expanding().max()
    max_dd = ((df['capital'] - peak) / peak).min()
    sharpe = (rets.mean() / rets.std()) * np.sqrt(365 * 24) if rets.std() > 0 else 0
    calmar = total_ret / abs(max_dd) if max_dd != 0 else 0

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
            regime_info[regime] = {'bars': len(sub), 'cum': cum}

    return {
        'total_return': f'{total_ret:.2%}',
        'total_return_numeric': round(total_ret, 4),
        'ann_volatility': f'{ann_vol:.2%}',
        'max_drawdown': f'{max_dd:.2%}',
        'max_drawdown_numeric': round(max_dd, 4),
        'sharpe': round(sharpe, 2),
        'sharpe_numeric': round(sharpe, 4),
        'calmar': round(calmar, 2),
        'n_trades': len(engine.trades),
        'total_fee': round(sum(t['fee'] for t in engine.trades), 2),
        'regime': regime_info,
    }


def run_sweep(df_dict, cs_weights, base_mid=0.30, funding_wt=0.15):
    results = []
    for csw in cs_weights:
        label = f'CS={int(csw*100)}%'
        print(f'\n{"─"*55}')
        print(f'  {label} (base_mid={base_mid}, funding={int(funding_wt*100)}%)')
        print(f'{"─"*55}')

        risk = RiskManager(INITIAL_CAPITAL)
        logger = TradeLogger(f'trades_cs_sweep_{int(csw*100)}.csv')
        engine = PortfolioBacktest(
            risk, logger,
            base_low=0.0, base_mid=base_mid, base_high=0.0,
            portfolio_vol_target=0.08,
            use_pullback=False,
            use_cross_sectional=(csw > 0),
            cross_sectional_weight=csw,
            use_funding=True,
            funding_weight=funding_wt,
        )
        engine.run(df_dict.copy())
        m = compute_metrics(engine)
        m['_label'] = label
        m['_cs_weight'] = csw
        results.append(m)

        for reg in ['low', 'mid', 'high']:
            info = m['regime'].get(reg, {})
            if info:
                print(f'    {reg}: {info["bars"]}b | cum={info["cum"]:>8.2%}')
        print(f'    收益={m["total_return"]} | Sharpe={m["sharpe"]} | Calmar={m["calmar"]} | 交易={m["n_trades"]}')

    return results


def main():
    print('=' * 55)
    print('  Cross-Sectional 权重扫描')
    print('=' * 55)

    fetcher = DataFetcher()
    df_dict = {}
    for s in SYMBOLS:
        df_dict[s] = fetcher.fetch_ohlcv(s)
        print(f'  {s}: {len(df_dict[s])} 根')

    cs_weights = [0.00, 0.05, 0.10, 0.15, 0.20, 0.30]
    results = run_sweep(df_dict, cs_weights)

    print()
    print('=' * 75)
    print('  权重扫描结果')
    print('=' * 75)
    h = f'  {"CS权重":>8s} {"收益":>10s} {"Sharpe":>8s} {"Calmar":>8s} {"回撤":>10s} {"交易":>6s} {"手续费":>8s}'
    print(h)
    print(f'  {"─" * 73}')

    best = max(results, key=lambda r: r['sharpe_numeric'])
    for r in results:
        marker = ' ★' if r is best else ''
        print(f'  {r["_label"]:>8s} {r["total_return"]:>10s} {r["sharpe"]:>8} '
              f'{r["calmar"]:>8} {r["max_drawdown"]:>10} {r["n_trades"]:>6} '
              f'¥{r["total_fee"]:>7,.2f}{marker}')

    print()
    print(f'  最佳: {best["_label"]} (Sharpe={best["sharpe"]}, Calmar={best["calmar"]})')
    print()


if __name__ == '__main__':
    main()

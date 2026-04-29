"""
Portfolio Layer — Target Vol 扫描
找到最优组合目标波动率
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, r'C:\Users\16079\Desktop\lianghua')
from config import SYMBOLS, INITIAL_CAPITAL, TRAIN_WINDOW, VOL_SCALE_CAP
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
    realized_vol = rets.std() * np.sqrt(365 * 24)

    # 平均暴露: 从交易记录反推
    if engine.trades:
        trade_vals = [abs(t['delta_value']) for t in engine.trades]
        avg_trade_size = np.mean(trade_vals)
        avg_capital = df['capital'].mean()
        avg_exposure_est = avg_trade_size / max(avg_capital, 1)
    else:
        avg_exposure_est = 0

    return {
        'final_capital': round(final, 2),
        'total_return': f'{total_ret:.2%}',
        'total_return_numeric': round(total_ret, 4),
        'realized_vol': f'{realized_vol:.2%}',
        'realized_vol_numeric': round(realized_vol, 4),
        'max_drawdown': f'{max_dd:.2%}',
        'max_drawdown_numeric': round(max_dd, 4),
        'sharpe': round(sharpe, 2),
        'sharpe_numeric': round(sharpe, 4),
        'calmar': round(calmar, 2),
        'calmar_numeric': round(calmar, 4),
        'n_trades': len(engine.trades),
        'total_fee': round(sum(t['fee'] for t in engine.trades), 2),
        'avg_exposure_est': f'{avg_exposure_est:.1%}',
    }


def run_sweep(df_dict, vol_targets, base_mid=1.0):
    """扫描不同目标波动率"""
    results = []

    for vt in vol_targets:
        label = f'Vol={vt:.0%}'
        print(f'\n{"─"*55}')
        print(f'  {label} (BASE_MID={base_mid})')
        print(f'{"─"*55}')

        risk = RiskManager(INITIAL_CAPITAL)
        logger = TradeLogger(f'trades_sweep_{int(vt*100)}.csv')
        engine = PortfolioBacktest(
            risk, logger,
            base_low=0.0, base_mid=base_mid, base_high=0.0,
            portfolio_vol_target=vt,
        )
        engine.run(df_dict.copy())

        m = compute_metrics(engine)
        m['_label'] = label
        m['_target_vol'] = vt
        m['_base_mid'] = base_mid
        results.append(m)

        dd_pct = float(m['max_drawdown'].rstrip('%')) / 100
        print(f'  收益={m["total_return"]} | 已实现波={m["realized_vol"]} | '
              f'Sharpe={m["sharpe"]} | DD={m["max_drawdown"]} | '
              f'Calmar={m["calmar"]} | 平均暴露≈{m["avg_exposure_est"]}')

    return results


def print_comparison(results):
    print()
    print('=' * 85)
    print('  Portfolio Layer — Target Vol 扫描结果')
    print('=' * 85)
    header = (f'  {"Target Vol":>10s} {"收益":>10s} {"已实现波":>9s} '
              f'{"Sharpe":>7s} {"Max DD":>8s} {"Calmar":>7s} '
              f'{"暴露≈":>7s} {"交易":>5s} {"手续费":>8s}')
    print(header)
    print(f'  {"─" * 83}')

    best_sharpe = max(results, key=lambda r: r['sharpe_numeric'])
    best_calmar = max(results, key=lambda r: r['calmar_numeric'])
    best_score = max(results, key=lambda r: r['sharpe_numeric'] * r['calmar_numeric'])

    for r in results:
        marker = ''
        if r is best_score:
            marker = ' ★'
        print(f'  {r["_label"]:>10s} {r["total_return"]:>10s} {r["realized_vol"]:>9s} '
              f'{r["sharpe"]:>7} {r["max_drawdown"]:>8s} {r["calmar"]:>7} '
              f'{r["avg_exposure_est"]:>7s} {r["n_trades"]:>5} ¥{r["total_fee"]:>7,.2f}{marker}')

    print()
    print(f'  ★ 最优综合 (Sharpe×Calmar): {best_score["_label"]} '
          f'(Sharpe={best_score["sharpe"]}, Calmar={best_score["calmar"]})')
    print(f'  Best Sharpe: {best_sharpe["_label"]} ({best_sharpe["sharpe"]})')
    print(f'  Best Calmar: {best_calmar["_label"]} ({best_calmar["calmar"]})')
    print(f'  VOL_SCALE_CAP = {VOL_SCALE_CAP}')


def main():
    print('=' * 85)
    print('  Portfolio Layer — 风险预算 + 杠杆 + 目标波动率')
    print('=' * 85)

    fetcher = DataFetcher()
    df_dict = {}
    for s in SYMBOLS:
        df_dict[s] = fetcher.fetch_ohlcv(s)
        print(f'  {s}: {len(df_dict[s])} 根')

    # 扫描目标波动率
    vol_targets = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20]
    results = run_sweep(df_dict, vol_targets, base_mid=1.0)

    print_comparison(results)


if __name__ == '__main__':
    main()

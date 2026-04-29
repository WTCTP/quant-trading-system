"""
组合杠杆扫描: 测试 1.0x ~ 3.0x 外层杠杆对收益/回撤的影响
原则: 杠杆加在组合层，不是信号层
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


def run_leverage(df_dict, leverage, base_mid=1.0, target_vol=0.05):
    label = f'Lev={leverage:.1f}x'
    print(f'\n{"─"*55}')
    print(f'  {label}')
    print(f'{"─"*55}')

    risk = RiskManager(INITIAL_CAPITAL)
    logger = TradeLogger(f'trades_lev_{int(leverage*10)}.csv')
    engine = PortfolioBacktest(
        risk, logger,
        base_low=0.0, base_mid=base_mid, base_high=0.0,
        portfolio_vol_target=target_vol,
        use_pullback=False,
        portfolio_leverage=leverage,
    )
    engine.run(df_dict.copy())

    df = pd.DataFrame(engine.records)
    df['returns'] = df['capital'].pct_change()
    rets = df['returns'].dropna()
    final = df['capital'].iloc[-1]
    total_ret = (final - INITIAL_CAPITAL) / INITIAL_CAPITAL
    peak = df['capital'].expanding().max()
    max_dd = ((df['capital'] - peak) / peak).min()
    ann_vol = rets.std() * np.sqrt(365 * 24)
    sharpe = (rets.mean() / rets.std()) * np.sqrt(365 * 24) if rets.std() > 0 else 0
    calmar = total_ret / abs(max_dd) if max_dd != 0 else 0

    # 风险事件统计
    risk_events = len(df[df['regime'].str.startswith('risk_', na=False)])
    vol_shocks = len(df[df['regime'] == 'risk_vol_shock'])

    m = {
        '_label': label,
        '_leverage': leverage,
        'total_return': f'{total_ret:.2%}',
        'total_return_numeric': round(total_ret, 4),
        'max_drawdown': f'{max_dd:.2%}',
        'max_drawdown_numeric': round(max_dd, 4),
        'ann_volatility': f'{ann_vol:.2%}',
        'sharpe': round(sharpe, 2),
        'calmar': round(calmar, 2),
        'n_trades': len(engine.trades),
        'total_fee': round(engine.total_fee, 2),
        'total_slippage': round(engine.total_slippage, 2),
        'total_funding_cost': round(engine.total_funding_cost, 2),
        'risk_events': risk_events,
        'vol_shocks': vol_shocks,
    }

    print(f'  收益={m["total_return"]} | DD={m["max_drawdown"]} | '
          f'Sharpe={m["sharpe"]} | Calmar={m["calmar"]} | 交易={m["n_trades"]}')
    print(f'  滑点=¥{m["total_slippage"]} | 资金费=¥{m["total_funding_cost"]} | '
          f'风险事件={risk_events} (熔断={vol_shocks})')
    return m


def main():
    print('=' * 55)
    print('  组合杠杆扫描 — Portfolio Leverage')
    print('=' * 55)

    fetcher = DataFetcher()
    df_dict = {}
    for s in SYMBOLS:
        df_dict[s] = fetcher.fetch_ohlcv(s)
        print(f'  {s}: {len(df_dict[s])} 根')

    leverages = [1.0, 1.5, 2.0, 2.5, 3.0]
    results = []
    for lev in leverages:
        r = run_leverage(df_dict, lev)
        results.append(r)

    print()
    print('=' * 85)
    print('  杠杆扫描结果')
    print('=' * 85)
    h = (f'  {"杠杆":>6s} {"收益":>10s} {"DD":>10s} {"波动":>10s} '
         f'{"Sharpe":>8s} {"Calmar":>8s} {"交易":>6s} {"滑点":>8s} {"风险事件":>8s}')
    print(h)
    print(f'  {"─" * 83}')

    best_calmar = max(results, key=lambda r: r['calmar'])
    best_sharpe = max(results, key=lambda r: r['sharpe'])

    for r in results:
        markers = []
        if r is best_sharpe: markers.append('S')
        if r is best_calmar: markers.append('C')
        marker = ' ' + '/'.join(markers) if markers else ''
        print(f'  {r["_label"]:>6s} {r["total_return"]:>10s} {r["max_drawdown"]:>10} '
              f'{r["ann_volatility"]:>10} {r["sharpe"]:>8} {r["calmar"]:>8} '
              f'{r["n_trades"]:>6} ¥{r["total_slippage"]:>6,.0f} {r["risk_events"]:>8}{marker}')

    print()
    r1 = results[0]
    r2 = results[-1]
    print(f'  基准 (1.0x): 收益={r1["total_return"]}, DD={r1["max_drawdown"]}')
    print(f'  最大 (3.0x): 收益={r2["total_return"]}, DD={r2["max_drawdown"]}')

    if r1['calmar'] >= max(r['calmar'] for r in results):
        print(f'\n  建议: 保持 1.0x（不加杠杆）— 当前Calmar最优')
    else:
        best = best_calmar
        print(f'\n  建议: {best["_label"]} — 综合最优 Sharpe={best["sharpe"]}, Calmar={best["calmar"]}')
    print()


if __name__ == '__main__':
    main()

"""
容量测试: 模拟不同资金规模的执行成本
capital = [10k, 50k, 100k, 500k, 1M]
观察: 收益是否下降 / 滑点是否爆炸 / 是否吃掉流动性
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


def run_capacity(df_dict, capital, base_mid=1.0, target_vol=0.05):
    """以指定资金规模运行回测"""
    label = f'${capital/1000:.0f}k' if capital < 1e6 else f'${capital/1e6:.1f}M'
    label_clean = label.replace('$', '')

    print(f'\n{"─"*55}')
    print(f'  {label}')
    print(f'{"─"*55}')

    risk = RiskManager(capital)
    logger = TradeLogger(f'trades_cap_{label_clean}.csv')
    engine = PortfolioBacktest(
        risk, logger,
        base_low=0.0, base_mid=base_mid, base_high=0.0,
        portfolio_vol_target=target_vol,
        use_pullback=False,
        initial_capital=capital,
    )
    engine.run(df_dict.copy())

    df = pd.DataFrame(engine.records)
    df['returns'] = df['capital'].pct_change()
    rets = df['returns'].dropna()
    final = df['capital'].iloc[-1]
    total_ret = (final - capital) / capital
    peak = df['capital'].expanding().max()
    max_dd = ((df['capital'] - peak) / peak).min()
    ann_vol = rets.std() * np.sqrt(365 * 24)
    sharpe = (rets.mean() / rets.std()) * np.sqrt(365 * 24) if rets.std() > 0 else 0
    calmar = total_ret / abs(max_dd) if max_dd != 0 else 0

    # 滑点占比
    total_cost = engine.total_fee + engine.total_slippage + engine.total_funding_cost
    cost_pct = total_cost / capital

    # 平均滑点 (bps)
    if engine.trades:
        avg_slip_bps = engine.total_slippage / capital * 10000
    else:
        avg_slip_bps = 0

    m = {
        '_label': label,
        '_capital': capital,
        'total_return': f'{total_ret:.2%}',
        'total_return_numeric': round(total_ret, 4),
        'max_drawdown': f'{max_dd:.2%}',
        'sharpe': round(sharpe, 2),
        'calmar': round(calmar, 2),
        'n_trades': len(engine.trades),
        'total_fee': round(engine.total_fee, 2),
        'total_slippage': round(engine.total_slippage, 2),
        'total_funding_cost': round(engine.total_funding_cost, 2),
        'total_cost': round(total_cost, 2),
        'cost_pct': cost_pct,
        'avg_slip_bps': round(avg_slip_bps, 1),
        'vol_shocks': len(df[df['regime'] == 'risk_vol_shock']),
    }

    print(f'  收益={m["total_return"]} | DD={m["max_drawdown"]} | '
          f'Sharpe={m["sharpe"]} | Calmar={m["calmar"]}')
    print(f'  总成本=¥{total_cost:,.0f} ({cost_pct:.2%}) | '
          f'平均滑点≈{avg_slip_bps:.0f}bps | 交易={m["n_trades"]}')
    return m


def main():
    print('=' * 60)
    print('  容量测试 — Capacity Test')
    print('=' * 60)

    fetcher = DataFetcher()
    df_dict = {}
    for s in SYMBOLS:
        df_dict[s] = fetcher.fetch_ohlcv(s)
        print(f'  {s}: {len(df_dict[s])} 根')

    capitals = [10_000, 50_000, 100_000, 500_000, 1_000_000]
    results = []
    for cap in capitals:
        r = run_capacity(df_dict, cap)
        results.append(r)

    print()
    print('=' * 95)
    print('  容量测试结果')
    print('=' * 95)
    h = (f'  {"资金":>8s} {"收益":>10s} {"DD":>10s} {"Sharpe":>8s} {"Calmar":>8s} '
         f'{"总成本":>10s} {"成本占比":>8s} {"平均滑点":>8s} {"交易":>6s}')
    print(h)
    print(f'  {"─" * 93}')

    r10k = results[0]
    for r in results:
        degrade = ''
        if r is not r10k:
            ret_diff = r['total_return_numeric'] - r10k['total_return_numeric']
            if ret_diff < -0.01:
                degrade = f' ↓{abs(ret_diff):.1%}'

        print(f'  {r["_label"]:>8s} {r["total_return"]:>10s} {r["max_drawdown"]:>10} '
              f'{r["sharpe"]:>8} {r["calmar"]:>8} '
              f'¥{r["total_cost"]:>8,.0f} {r["cost_pct"]:>7.2%} {r["avg_slip_bps"]:>6}bps '
              f'{r["n_trades"]:>6}{degrade}')

    print()
    print('─' * 60)
    print('  结论')
    print('─' * 60)

    r_big = results[-1]
    ret_decline = r10k['total_return_numeric'] - r_big['total_return_numeric']
    cost_increase = r_big['cost_pct'] - r10k['cost_pct']

    if ret_decline > 0.05:
        print(f'  ❌ 容量瓶颈: $1M收益下降 {ret_decline:.1%}')
        print(f'  原因: 滑点爆炸 (成本占比 +{cost_increase:.2%})')
    elif ret_decline > 0.02:
        print(f'  ⚠ 轻度容量限制: $1M收益下降 {ret_decline:.1%}')
        print(f'  建议: 分仓执行 / 拆单优化')
    elif r_big['vol_shocks'] > 10:
        print(f'  ⚠ 波动率熔断增多: {r_big["vol_shocks"]}次')
        print(f'  建议: 降低杠杆或增加延迟')
    else:
        print(f'  ✅ 容量充足: $1M收益仅下降 {ret_decline:.1%}')
        print(f'  系统可支持到 $1M 以上资金')

    print(f'\n  基准 ($10k): 收益={r10k["total_return"]}, 成本占比={r10k["cost_pct"]:.2%}')
    print(f'  最大 ($1M):  收益={r_big["total_return"]}, 成本占比={r_big["cost_pct"]:.2%}')
    print()


if __name__ == '__main__':
    main()

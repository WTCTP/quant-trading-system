"""
组合风险预算系统 — Mid-Only vs Mid+Low Beta 对照实验
核心问题: low regime 是否值得占用风险预算？
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, r'C:\Users\16079\Desktop\lianghua')
from config import (
    SYMBOLS, INITIAL_CAPITAL, TRAIN_WINDOW, RETRAIN_EVERY,
    FEE_RATE, SLIPPAGE_BPS,
    REGIME_LOW_THRESH, REGIME_HIGH_THRESH,
)
from data.fetcher import DataFetcher
from risk.manager import RiskManager
from log.recorder import TradeLogger
from backtest.engine import PortfolioBacktest


def compute_metrics(records, trades):
    """计算扩展风险指标"""
    df = pd.DataFrame(records)
    df['returns'] = df['capital'].pct_change()
    rets = df['returns'].dropna()

    if len(rets) < 10:
        return {}

    # 基础指标
    final = df['capital'].iloc[-1]
    total_ret = (final - INITIAL_CAPITAL) / INITIAL_CAPITAL
    ann_vol = rets.std() * np.sqrt(365 * 24)

    # 回撤
    peak = df['capital'].expanding().max()
    dd = (df['capital'] - peak) / peak
    max_dd = dd.min()
    avg_dd = dd[dd < 0].mean()

    # 回撤恢复速度
    dd_events = []
    in_dd = False
    dd_start = 0
    dd_peak_val = INITIAL_CAPITAL
    for i, (_, row) in enumerate(df.iterrows()):
        val = row['capital']
        if val < dd_peak_val and not in_dd:
            in_dd = True
            dd_start = i
        elif val >= dd_peak_val and in_dd:
            recovery_bars = i - dd_start
            dd_events.append(recovery_bars)
            in_dd = False
            dd_peak_val = val
        elif val > dd_peak_val:
            dd_peak_val = val
    avg_recovery = np.mean(dd_events) if dd_events else 0

    # Sharpe & Calmar
    mean_ret = rets.mean()
    sharpe = (mean_ret / rets.std()) * np.sqrt(365 * 24) if rets.std() > 0 else 0
    calmar = total_ret / abs(max_dd) if max_dd != 0 else 0

    # Sortino (只考虑下行波动)
    down_rets = rets[rets < 0]
    down_std = down_rets.std() * np.sqrt(365 * 24) if len(down_rets) > 0 else ann_vol
    sortino = total_ret / down_std if down_std > 0 else 0

    # 盈亏比
    win_rets = rets[rets > 0]
    loss_rets = rets[rets < 0]
    win_rate = len(win_rets) / max(len(rets), 1)
    avg_win = win_rets.mean() if len(win_rets) > 0 else 0
    avg_loss = abs(loss_rets.mean()) if len(loss_rets) > 0 else 0
    profit_factor = (avg_win * len(win_rets)) / (avg_loss * len(loss_rets)) if avg_loss > 0 else float('inf')

    # 风险预算利用率
    risk_per_trade = ann_vol / max(len(trades), 1) * np.sqrt(365 * 24)
    fee_efficiency = total_ret / max(sum(t['fee'] for t in trades), 0.001)

    return {
        'final_capital': round(final, 2),
        'total_return': f'{total_ret:.2%}',
        'total_return_numeric': total_ret,
        'ann_volatility': f'{ann_vol:.2%}',
        'max_drawdown': f'{max_dd:.2%}',
        'max_drawdown_numeric': max_dd,
        'avg_drawdown': f'{avg_dd:.2%}',
        'avg_recovery_bars': round(avg_recovery, 1),
        'sharpe': round(sharpe, 2),
        'calmar': round(calmar, 2),
        'sortino': round(sortino, 2),
        'win_rate': f'{win_rate:.1%}',
        'profit_factor': round(profit_factor, 2),
        'n_trades': len(trades),
        'total_fee': round(sum(t['fee'] for t in trades), 2),
        'fee_efficiency': round(fee_efficiency, 2),
    }


def regime_exposure_analysis(records):
    """分析各regime的实际风险暴露"""
    df = pd.DataFrame(records)
    df['returns'] = df['capital'].pct_change()

    def regime_root(r):
        if r and r.startswith('mid'): return 'mid'
        if r and r.startswith('low'): return 'low'
        if r and r.startswith('high'): return 'high'
        return 'other'

    df['root'] = df['regime'].apply(regime_root)

    results = {}
    for regime in ['low', 'mid', 'high']:
        sub = df[df['root'] == regime]
        if len(sub) < 10:
            results[regime] = {'bars': len(sub), 'vol_pct': 0, 'ret_pct': 0}
            continue
        rets = sub['returns'].dropna()
        if len(rets) == 0:
            results[regime] = {'bars': len(sub), 'vol_pct': 0, 'ret_pct': 0}
            continue

        cum_ret = (1 + rets).prod() - 1
        vol = rets.std() * np.sqrt(365 * 24)

        results[regime] = {
            'bars': len(sub),
            'vol_ann': vol,
            'cum_return': cum_ret,
            'sharpe': (rets.mean() / rets.std()) * np.sqrt(365 * 24) if rets.std() > 0 else 0,
        }

    return results


def run_experiment(df_dict, label, base_low, base_mid, portfolio_vol_target):
    """运行单个实验"""
    risk = RiskManager(INITIAL_CAPITAL)
    logger = TradeLogger(f'trades_{label}.csv')
    engine = PortfolioBacktest(
        risk, logger,
        base_low=base_low, base_mid=base_mid, base_high=0.0,
        portfolio_vol_target=portfolio_vol_target,
    )
    results = engine.run(df_dict.copy())

    metrics = compute_metrics(engine.records, engine.trades)
    metrics['_label'] = label
    metrics['_regime'] = regime_exposure_analysis(engine.records)
    metrics['_records'] = engine.records
    return metrics


def main():
    print('=' * 65)
    print('  组合风险预算实验 — Mid-Only vs Mid+Low Beta')
    print('=' * 65)

    fetcher = DataFetcher()
    df_dict = {}
    for s in SYMBOLS:
        df = fetcher.fetch_ohlcv(s)
        df_dict[s] = df
        print(f'  {s}: {len(df)} 根')
    print(f'  训练窗口: {TRAIN_WINDOW} | 重训: {RETRAIN_EVERY}')

    # === 实验 A: Mid-Only ===
    print()
    print('─' * 65)
    print('  实验A: Mid-Only Alpha（低波=空仓，不给风险预算）')
    print('─' * 65)
    r_a = run_experiment(df_dict, 'Mid-Only',
                          base_low=0.00, base_mid=0.30,
                          portfolio_vol_target=0.08)

    # === 实验 B: Mid+Low Beta ===
    print()
    print('─' * 65)
    print('  实验B: Mid+Low Beta（低波=10%被动beta）')
    print('─' * 65)
    r_b = run_experiment(df_dict, 'Mid+Low',
                          base_low=0.10, base_mid=0.30,
                          portfolio_vol_target=0.08)

    # === 对照表 ===
    print()
    print('=' * 65)
    print('  风险效率对照表')
    print('=' * 65)

    metrics_rows = [
        ('收益', 'total_return', '>10s'),
        ('年化波动', 'ann_volatility', '>10s'),
        ('最大回撤', 'max_drawdown', '>10s'),
        ('平均回撤', 'avg_drawdown', '>10s'),
        ('回撤恢复(bar)', 'avg_recovery_bars', '>10.0f'),
        ('Sharpe', 'sharpe', '>10.2f'),
        ('Calmar', 'calmar', '>10.2f'),
        ('Sortino', 'sortino', '>10.2f'),
        ('胜率', 'win_rate', '>10s'),
        ('盈亏比', 'profit_factor', '>10.2f'),
        ('交易次数', 'n_trades', '>10d'),
        ('手续费', 'total_fee', '>10.2f'),
        ('收益/手续费', 'fee_efficiency', '>10.2f'),
    ]

    print(f'  {"指标":18s} {"Mid-Only":>14s} {"Mid+Low":>14s} {"差异":>10s}')
    print(f'  {"─" * 58}')
    for name, key, fmt in metrics_rows:
        v_a = r_a[key]
        v_b = r_b[key]
        if isinstance(v_a, str):
            print(f'  {name:18s} {v_a:{fmt}} {v_b:{fmt}}')
        elif isinstance(v_a, float):
            diff = v_b - v_a
            print(f'  {name:18s} {v_a:{fmt}} {v_b:{fmt}} {diff:+{fmt}}')
        else:
            print(f'  {name:18s} {v_a:{fmt}} {v_b:{fmt}}')

    # === Regime风险贡献 ===
    print()
    print('─' * 65)
    print('  Regime 风险贡献分解')
    print('─' * 65)
    print(f'  {"Regime":8s} {"实验":12s} {"K线数":>7s} {"年化波动":>10s} {"累计收益":>10s} {"Sharpe":>8s}')
    print(f'  {"─" * 55}')

    for label, r in [('Mid-Only', r_a), ('Mid+Low', r_b)]:
        for regime in ['low', 'mid', 'high']:
            ra = r['_regime'].get(regime, {})
            bars = ra.get('bars', 0)
            vol = ra.get('vol_ann', 0)
            cum = ra.get('cum_return', 0)
            sr = ra.get('sharpe', 0)
            if bars > 0:
                print(f'  {regime:8s} {label:12s} {bars:>7d} {vol:>10.2%} {cum:>10.2%} {sr:>8.2f}')
        print()

    # === 结论 ===
    print('─' * 65)
    print('  结论')
    print('─' * 65)

    sharpe_a = r_a['sharpe']
    sharpe_b = r_b['sharpe']
    calmar_a = r_a['calmar']
    calmar_b = r_b['calmar']
    fee_a = r_a['total_fee']
    fee_b = r_b['total_fee']

    if sharpe_a > sharpe_b:
        print(f'  Mid-Only Sharpe ({sharpe_a}) > Mid+Low ({sharpe_b})')
        print(f'  → 低波 beta 稀释了 alpha 质量')
    else:
        print(f'  Mid+Low Sharpe ({sharpe_b}) > Mid-Only ({sharpe_a})')

    if calmar_a > calmar_b:
        print(f'  Mid-Only Calmar ({calmar_a}) > Mid+Low ({calmar_b})')
        print(f'  → 低波 beta 降低了风险调整收益')

    ret_diff = r_b['total_return_numeric'] - r_a['total_return_numeric']
    if ret_diff < 0:
        print(f'  Mid+Low 收益更低 ({ret_diff:+.2%}) — 低波beta是负贡献')
    else:
        fee_increase = fee_b - fee_a
        net_gain = ret_diff - fee_increase / INITIAL_CAPITAL
        print(f'  Mid+Low 收益更高 ({ret_diff:+.2%})')
        print(f'  但额外手续费 +¥{fee_increase:.0f}，净增收益 ≈ {net_gain:+.2%}')

    print()
    print(f'  🔥 核心判断: ', end='')
    if sharpe_a > sharpe_b and calmar_a > calmar_b:
        print('低波不值得占用风险预算。Mid-Only 是更优结构。')
    elif ret_diff > 0.02:
        print('低波beta带来显著额外收益，值得保留。')
    else:
        print('低波beta收益有限，风险效率下降。建议Mid-Only + 预留未来低波alpha的接口。')

    print()


if __name__ == '__main__':
    main()

"""
Mid Regime 双Alpha实验: Breakout-only vs Breakout+Pullback
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, r'C:\Users\16079\Desktop\lianghua')
from config import (
    SYMBOLS, INITIAL_CAPITAL, TRAIN_WINDOW, RETRAIN_EVERY,
    FEE_RATE, SLIPPAGE_BPS,
)
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
    dd = (df['capital'] - peak) / peak
    max_dd = dd.min()
    avg_ret = rets.mean()
    sharpe = (avg_ret / rets.std()) * np.sqrt(365 * 24) if rets.std() > 0 else 0
    calmar = total_ret / abs(max_dd) if max_dd != 0 else 0

    # Win/loss stats
    wins = rets[rets > 0]
    losses = rets[rets < 0]
    win_rate = len(wins) / max(len(rets), 1)
    profit_factor = (wins.sum() / abs(losses.sum())) if len(losses) > 0 and losses.sum() != 0 else float('inf')

    # Mid-only metrics
    mid_df = df[df['regime'].str.startswith('mid', na=False)]
    mid_rets = mid_df['returns'].dropna()
    mid_cum = (1 + mid_rets).prod() - 1 if len(mid_rets) > 0 else 0
    mid_sharpe = (mid_rets.mean() / mid_rets.std()) * np.sqrt(365 * 24) if len(mid_rets) > 0 and mid_rets.std() > 0 else 0

    return {
        'final_capital': round(final, 2),
        'total_return': f'{total_ret:.2%}',
        'total_return_numeric': round(total_ret, 4),
        'ann_volatility': f'{ann_vol:.2%}',
        'max_drawdown': f'{max_dd:.2%}',
        'max_drawdown_numeric': round(max_dd, 4),
        'sharpe': round(sharpe, 2),
        'calmar': round(calmar, 2),
        'win_rate': f'{win_rate:.1%}',
        'profit_factor': round(profit_factor, 2),
        'n_trades': len(trades),
        'total_fee': round(sum(t['fee'] for t in trades), 2),
        'mid_cum_return': f'{mid_cum:.2%}',
        'mid_sharpe': round(mid_sharpe, 2),
    }


def run(df_dict, label, use_pullback):
    print(f'\n{"─"*55}')
    print(f'  {label}')
    print(f'{"─"*55}')
    risk = RiskManager(INITIAL_CAPITAL)
    logger = TradeLogger(f'trades_{label}.csv')
    engine = PortfolioBacktest(
        risk, logger,
        base_low=0.0, base_mid=0.30, base_high=0.0,
        portfolio_vol_target=0.08,
        use_pullback=use_pullback,
    )
    results = engine.run(df_dict.copy())
    m = compute_extended_metrics(engine.records, engine.trades)
    m['_label'] = label

    # Regime分解
    df = pd.DataFrame(engine.records)
    df['returns'] = df['capital'].pct_change()
    for regime in ['low', 'mid', 'high']:
        mask = df['regime'].str.startswith(regime, na=False)
        sub = df[mask]
        if len(sub) > 5:
            cum = (1 + sub['returns']).prod() - 1
            print(f'    {regime}: {len(sub)} bars | cum={cum:>8.2%}')

    print(f'    收益: {m["total_return"]} | 回撤: {m["max_drawdown"]} | Sharpe: {m["sharpe"]} | Calmar: {m["calmar"]}')
    print(f'    交易: {m["n_trades"]} | 手续费: ¥{m["total_fee"]}')
    return m


def main():
    print('=' * 55)
    print('  Mid Regime 双Alpha: Breakout vs Breakout+Pullback')
    print('=' * 55)

    fetcher = DataFetcher()
    df_dict = {}
    for s in SYMBOLS:
        df = fetcher.fetch_ohlcv(s)
        df_dict[s] = df
        print(f'  {s}: {len(df)} 根')

    # 实验
    r_bo = run(df_dict, 'Breakout-Only', use_pullback=False)
    r_bp = run(df_dict, 'Breakout+Pullback', use_pullback=True)

    # 对比
    print()
    print('=' * 55)
    print('  双Alpha 对比')
    print('=' * 55)
    print(f'  {"指标":18s} {"Breakout":>14s} {"B+O+Pullback":>14s}')
    print(f'  {"─" * 48}')

    keys = [
        ('收益', 'total_return'), ('年化波动', 'ann_volatility'),
        ('最大回撤', 'max_drawdown'), ('Sharpe', 'sharpe'),
        ('Calmar', 'calmar'), ('胜率', 'win_rate'),
        ('盈亏比', 'profit_factor'), ('交易次数', 'n_trades'),
        ('手续费', 'total_fee'), ('Mid累计收益', 'mid_cum_return'),
        ('Mid Sharpe', 'mid_sharpe'),
    ]
    for name, key in keys:
        print(f'  {name:18s} {str(r_bo[key]):>14s} {str(r_bp[key]):>14s}')

    # 结论
    print()
    print('─' * 55)
    sharpe_bo = r_bo['sharpe']
    sharpe_bp = r_bp['sharpe']
    calmar_bo = r_bo['calmar']
    calmar_bp = r_bp['calmar']

    if sharpe_bp > sharpe_bo and calmar_bp > calmar_bo:
        print('  Pullback成功提升系统质量（Sharpe+Calmar双升）')
    elif sharpe_bp > sharpe_bo:
        print(f'  Pullback提升Sharpe ({sharpe_bo}→{sharpe_bp})，但Calmar变化需关注')
    else:
        print('  Pullback未显著改善，需检查信号相关性')

    ret_bo = r_bo['total_return_numeric']
    ret_bp = r_bp['total_return_numeric']
    print(f'  收益变化: {ret_bo:.2%} → {ret_bp:.2%} ({ret_bp-ret_bo:+.2%})')
    print()


if __name__ == '__main__':
    main()

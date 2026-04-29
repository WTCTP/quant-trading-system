"""
低波市场结构识别 — 基准测试
目的: 回答"低波市场是横盘/趋势/假突破？"，决定用哪种策略
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, r'C:\Users\16079\Desktop\lianghua')
from config import SYMBOLS, REGIME_LOW_THRESH, REGIME_HIGH_THRESH, VOL_REGIME_EMA_SPAN
from data.fetcher import DataFetcher
from alpha.features import build_features
from alpha.model import get_regime


def analyze_low_vol_structure(df_dict):
    """分析低波市场的收益结构"""
    btc = df_dict['BTC/USDT']
    eth = df_dict['ETH/USDT']

    # 用BTC的vol_regime给所有bar打标签
    feats = build_features(btc)
    regime_col = feats['vol_regime']
    alpha = 2.0 / (VOL_REGIME_EMA_SPAN + 1)
    ema_vol = regime_col.ewm(alpha=alpha, adjust=False).mean()

    labels = ema_vol.apply(get_regime)
    btc_ret = btc['close'].pct_change().reindex(labels.index)

    low_mask = labels == 'low'
    mid_mask = labels == 'mid'
    high_mask = labels == 'high'

    low_ret = btc_ret[low_mask].dropna()
    mid_ret = btc_ret[mid_mask].dropna()
    high_ret = btc_ret[high_mask].dropna()

    print('=' * 60)
    print('  低波市场结构分析')
    print('=' * 60)
    print(f'  BTC 1h数据: {len(btc)} 根K线')
    print(f'  Regime划分: low<{REGIME_LOW_THRESH} | mid∈[{REGIME_LOW_THRESH}~{REGIME_HIGH_THRESH}] | high>{REGIME_HIGH_THRESH}')
    print(f'  EMA平滑: span={VOL_REGIME_EMA_SPAN}')

    # === 1. 基础统计 ===
    print()
    print('─' * 60)
    print('  一、基础统计')
    print('─' * 60)
    print(f'  {"":20s} {"低波(low)":>12s} {"中波(mid)":>12s} {"高波(high)":>12s}')
    print(f'  {"─" * 56}')

    for label, data in [('低波', low_ret), ('中波', mid_ret), ('高波', high_ret)]:
        n = len(data)
        mean_r = data.mean() * 10000  # bps
        std_r = data.std() * 10000
        sharpe = (mean_r / std_r) * np.sqrt(365 * 24) if std_r > 0 else 0
        winrate = (data > 0).mean()
        skew = data.skew()
        kurt = data.kurtosis()
        print(f'  {label:20s} {n:>6d}根  | μ={mean_r:>6.1f}bps | σ={std_r:>6.1f}bps | SR={sharpe:>5.2f} | 胜率={winrate:>5.1%} | 偏度={skew:>+5.2f} | 峰度={kurt:>5.2f}')

    # === 2. 持有收益（Buy & Hold在各regime的表现）===
    print()
    print('─' * 60)
    print('  二、纯持有BTC在各Regime的累计收益')
    print('─' * 60)

    for label, mask in [('低波', low_mask), ('中波', mid_mask), ('高波', high_mask)]:
        ret_sub = btc_ret[mask].dropna()
        if len(ret_sub) > 0:
            cum = (1 + ret_sub).prod() - 1
            ann_ret = cum * (365 * 24 / len(ret_sub))  # 年化近似
            max_dd = (ret_sub.cumsum().cummax() - ret_sub.cumsum()).max()  # log-return近似
            print(f'  {label}: {len(ret_sub)}根 | 累计={cum:>8.2%} | 年化≈{ann_ret:>8.2%} | 最大回撤(log)≈{max_dd:>8.2%}')

    # === 3. 自相关分析（判断趋势 vs 反转）===
    print()
    print('─' * 60)
    print('  三、自相关分析（趋势 or 反转？）')
    print('─' * 60)
    print(f'  Lag-1自相关: 正值→趋势延续, 负值→均值回归')
    print()

    for label, data in [('低波', low_ret), ('中波', mid_ret), ('高波', high_ret)]:
        if len(data) < 10:
            continue
        ac1 = data.autocorr(lag=1)
        ac2 = data.autocorr(lag=2)
        ac4 = data.autocorr(lag=4)
        ac12 = data.autocorr(lag=12)
        direction = '📈 趋势延续' if ac1 > 0.01 else '📉 均值回归' if ac1 < -0.01 else '➡ 随机游走'
        print(f'  {label}: AC(1)={ac1:+.4f}  AC(2)={ac2:+.4f}  AC(4)={ac4:+.4f}  AC(12)={ac12:+.4f}  → {direction}')

    # === 4. Z-score分析 ===
    print()
    print('─' * 60)
    print('  四、Z-score分布（判断震荡区间）')
    print('─' * 60)
    close = btc['close'].reindex(labels.index)
    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    zscore = (close - ma20) / std20

    for label, mask in [('低波', low_mask), ('中波', mid_mask), ('高波', high_mask)]:
        zs = zscore[mask].dropna()
        if len(zs) < 10:
            continue
        pct_oversold = (zs < -1.5).mean()  # 超卖
        pct_overbought = (zs > 1.5).mean()  # 超买
        pct_range = ((zs > -1) & (zs < 1)).mean()  # 正常区间
        print(f'  {label}: |z|<1={pct_range:.1%} | z<-1.5={pct_oversold:.1%} | z>1.5={pct_overbought:.1%} | mean(z)={zs.mean():+.2f}')

    # === 5. 反转概率分析 ===
    print()
    print('─' * 60)
    print('  五、反转概率（超买超卖后N根内反转的概率）')
    print('─' * 60)

    for label, mask in [('低波', low_mask), ('中波', mid_mask)]:
        zs = zscore[mask].dropna()
        rets = btc_ret[mask].dropna()
        common_idx = zs.index.intersection(rets.index)
        if len(common_idx) < 100:
            continue
        zs = zs[common_idx]
        rets = rets[common_idx]

        for threshold, tag in [(-1.5, '超卖后'), (1.5, '超买后')]:
            if threshold < 0:
                signals = zs < threshold
            else:
                signals = zs > threshold

            n_signals = signals.sum()
            if n_signals < 10:
                continue

            # 反转概率: 接下来N根累计收益方向
            for horizon in [4, 8, 24]:
                fwd_ret = rets.rolling(horizon).apply(lambda x: (1 + x).prod() - 1).shift(-horizon + 1)
                if threshold < 0:
                    reversal_mask = signals & (fwd_ret > 0)
                else:
                    reversal_mask = signals & (fwd_ret < 0)
                rev_pct = reversal_mask.sum() / max(n_signals, 1)
                print(f'  {label} {tag} {horizon}bar反转概率: {rev_pct:.1%}  (n={n_signals})')

    # === 6. 结论与建议 ===
    print()
    print('─' * 60)
    print('  六、策略建议')
    print('─' * 60)

    ac1_low = low_ret.autocorr(lag=1) if len(low_ret) > 10 else 0
    zs_low = zscore[low_mask].dropna()
    reveral_4h = 0
    if len(zs_low) > 100:
        signals = zs_low < -1.5
        rets_low = btc_ret[low_mask].dropna()
        common = zs_low.index.intersection(rets_low.index)
        if len(common) > 100:
            zs_c = zs_low[common]
            rets_c = rets_low[common]
            fwd = rets_c.rolling(4).apply(lambda x: (1 + x).prod() - 1).shift(-3)
            s = zs_c < -1.5
            if s.sum() > 5:
                reveral_4h = (s & (fwd > 0)).sum() / s.sum()

    print(f'  低波Lag-1自相关: {ac1_low:+.4f}')
    print(f'  低波超卖4bar反转概率: {reveral_4h:.1%}')
    print()

    if ac1_low < -0.01:
        print('  → 低波呈现【均值回归】特征 → 适合 MR 策略')
    elif ac1_low > 0.02:
        print('  → 低波呈现【趋势延续】特征 → 适合 Trend 策略')
    else:
        print('  → 低波接近【随机游走】→ 不建议单独建仓')

    if reveral_4h > 0.5:
        print('  → 超卖反转率高 → Z-score MR 值得尝试')
    else:
        print('  → 超卖反转率低 → MR 需要额外确认条件')


def main():
    fetcher = DataFetcher()
    df_dict = {}
    for s in ['BTC/USDT', 'ETH/USDT']:
        df = fetcher.fetch_ohlcv(s)
        df_dict[s] = df
        print(f'  {s}: {len(df)} 根')

    analyze_low_vol_structure(df_dict)


if __name__ == '__main__':
    main()

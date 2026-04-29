"""
Pullback Alpha — 回调入场策略
逻辑: 趋势存在但不过热 → 等价格回撤到支撑位 → 确认反转 → 入场
与 Breakout Alpha 互补: Breakout追突破, Pullback等回调
"""
import numpy as np
import pandas as pd


def build_pullback_features(df):
    """构建回调相关特征"""
    c = df['close']
    h = df['high']
    l = df['low']
    v = df['volume']
    ret = c.pct_change()

    feats = pd.DataFrame(index=df.index)

    # 1. 回调深度: 从N日高点的回撤幅度
    high_24 = h.rolling(24).max()
    high_48 = h.rolling(48).max()
    feats['pullback_1d'] = (high_24 - c) / (high_24 - l.rolling(24).min() + 1e-10)  # 24h回撤
    feats['pullback_2d'] = (high_48 - c) / (high_48 - l.rolling(48).min() + 1e-10)  # 48h回撤

    # 2. MA支撑距离: 正=在MA上方, 负=跌破
    ma20 = c.rolling(20).mean()
    ma50 = c.rolling(50).mean()
    feats['ma20_dist'] = (c - ma20) / (ma20 + 1e-10)
    feats['ma50_dist'] = (c - ma50) / (ma50 + 1e-10)

    # 3. 波动收缩: 回调期间的振幅是否在缩小（理想的回调特征）
    atr6 = (h - l).rolling(6).mean()
    atr24 = (h - l).rolling(24).mean()
    feats['vol_contraction'] = atr6 / (atr24 + 1e-10)

    # 4. 成交量萎缩: 回调时缩量是健康信号
    vol_ma12 = v.rolling(12).mean()
    vol_ma48 = v.rolling(48).mean()
    feats['vol_shrink_12'] = v / (vol_ma12 + 1e-10)
    feats['vol_shrink_48'] = v / (vol_ma48 + 1e-10)

    # 5. 反转确认: K线形态
    feats['reversal_bar'] = ((c > l.shift(1)) & (c > c.shift(1))).astype(float)  # 今日收涨且高于昨日低点
    feats['hammer'] = ((c > (h + l) / 2) & ((h - l) > 2 * abs(c - c.shift(1)))).astype(float)  # 下影线

    # 6. 趋势背景: 长期趋势方向
    feats['trend_20'] = ret.rolling(20).mean() * 100  # 20bar趋势
    feats['trend_50'] = ret.rolling(50).mean() * 100  # 50bar趋势

    return feats


class PullbackSignal:
    """回调入场信号生成器"""

    def __init__(self):
        self.feature_names = None

    def compute_signal(self, df):
        """
        返回 pullback_score (0~1) 和 entry_flag (bool)
        score > 0.5 表示回调买入机会
        """
        feats = build_pullback_features(df)
        self.feature_names = feats.columns.tolist()

        # 取最新一行
        row = feats.iloc[-1]
        if row.isna().any():
            return 0.0, 0.0

        score = 0.0
        reasons = []

        # 条件1: 趋势背景（必须偏多）
        trend_20 = row['trend_20']
        trend_50 = row['trend_50']
        trend_score = 0
        if trend_20 > 0.01 and trend_50 > 0:
            trend_score = 1.0
            reasons.append('趋势↑')
        elif trend_20 > 0 and trend_50 > -0.01:
            trend_score = 0.5
            reasons.append('趋势→')
        else:
            reasons.append('趋势↓-跳过')
            return 0.0, 0.0  # 趋势向下，不找回调

        # 条件2: 回调深度（越大越好，但不能跌破MA支撑）
        pb_1d = row['pullback_1d']
        pb_2d = row['pullback_2d']
        ma20_d = row['ma20_dist']
        ma50_d = row['ma50_dist']

        if pb_1d > 0.3:  # 起码回调了30%
            pb_score = min(pb_1d, 0.8)  # 回调越深越好，但封顶
            if ma20_d > -0.02:  # 未跌破MA20太多
                pb_score *= 1.0
            elif ma50_d > -0.03:  # 在MA50上方
                pb_score *= 0.7
            else:
                pb_score *= 0.3  # 跌破MA50，风险大
            reasons.append(f'回调{pb_1d:.0%}')
        else:
            pb_score = 0
            reasons.append('无回调')

        # 条件3: 波动收缩（回调缩量）
        vc = row['vol_contraction']
        if vc < 0.9:  # ATR缩小
            vc_score = 1.0
            reasons.append('缩波')
        elif vc < 1.1:
            vc_score = 0.6
        else:
            vc_score = 0.2  # 放量回调不好
            reasons.append('放量')

        # 条件4: 成交量确认
        vs12 = row['vol_shrink_12']
        if vs12 < 0.8:  # 缩量
            vol_score = 1.0
            reasons.append('缩量')
        elif vs12 < 1.0:
            vol_score = 0.7
        else:
            vol_score = 0.3
            reasons.append('放量')

        # 条件5: 反转K线确认
        rev = row['reversal_bar']
        hammer = row['hammer']
        if rev > 0 and hammer > 0:
            rev_score = 1.0
            reasons.append('反转+锤子')
        elif rev > 0:
            rev_score = 0.8
            reasons.append('反转')
        elif hammer > 0:
            rev_score = 0.5
            reasons.append('锤子')
        else:
            rev_score = 0.1
            reasons.append('未确认')

        # 加权合成
        score = (
            trend_score * 0.20 +
            pb_score * 0.30 +
            vc_score * 0.20 +
            vol_score * 0.15 +
            rev_score * 0.15
        )

        return score, trend_score

    def get_entry_signal(self, df, alpha_signal=None):
        """
        生成入场信号:
        - positive: 回调买入机会 (0~1)
        - alpha_signal: 现有breakout信号，用于判断趋势方向
        """
        score, trend = self.compute_signal(df)
        if score < 0.40:
            return 0.0

        # 如果breakout信号已经很强（>0.10），说明正在突破中，不抢
        if alpha_signal is not None and abs(alpha_signal) > 0.10:
            return score * 0.3  # 打折，避免和breakout冲突

        return score

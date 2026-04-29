"""
Funding Rate Alpha — 资金费率拥挤信号
信息来源: 永续合约资金费率（完全独立于价格）
逻辑: 极端funding → 多空拥挤 → 反向挤压机会
"""
import numpy as np
import pandas as pd


class FundingAlpha:
    """
    资金费率极端信号:
    - funding_percentile > 95 → 多头拥挤 → 反向做空/减仓
    - funding_percentile < 5  → 空头拥挤 → 反向做多/加仓
    - 中间区域 → 无信号（不交易）
    """

    def __init__(self, lookback_days=90, extreme_pct=95, max_weight=0.10):
        self.lookback_days = lookback_days
        self.extreme_pct = extreme_pct  # 极端阈值 (95 → 只看top 5%和bottom 5%)
        self.max_weight = max_weight   # 单币最大权重分配

    def compute_signals(self, funding_dict, bar_time):
        """
        输入: {symbol: funding_series_up_to_bar_time}
        输出: np.array of signals per coin, range [-1, 1]
        正值=看多(空头拥挤), 负值=看空(多头拥挤), 0=无信号
        """
        n = len(funding_dict)
        signals = np.zeros(n)

        for i, (symbol, series) in enumerate(funding_dict.items()):
            if series is None or len(series) < 30:
                continue

            # 截取到当前时间
            mask = series.index <= bar_time
            hist = series[mask]
            if len(hist) < 30:
                continue

            current = hist.iloc[-1]
            lookback = hist.iloc[-min(len(hist), self.lookback_days * 3):]  # ~270个8h数据点

            if len(lookback) < 20:
                continue

            pct = (lookback < current).mean()

            if pct >= self.extreme_pct / 100:
                # 多头极端拥挤 → 看空
                extremity = min((pct - self.extreme_pct/100) / (1 - self.extreme_pct/100), 1.0)
                signals[i] = -extremity
            elif pct <= (1 - self.extreme_pct / 100):
                # 空头极端拥挤 → 看多
                extremity = min(((1 - self.extreme_pct/100) - pct) / (1 - self.extreme_pct/100), 1.0)
                signals[i] = extremity

        return signals

    def get_weights(self, funding_dict, bar_time, breakout_signal):
        """
        返回 funding alpha 的独立权重
        breakout_signal: 现有breakout信号，用于避免冲突
        """
        signals = self.compute_signals(funding_dict, bar_time)
        weights = np.zeros(len(signals))

        for i in range(len(signals)):
            s = signals[i]
            if abs(s) < 0.01:
                continue

            # 如果和breakout方向一致 → 增强
            # 如果和breakout方向相反 → 减半（冲突时breakout优先）
            if i < len(breakout_signal):
                breakout_dir = 1 if breakout_signal[i] > 0 else -1 if breakout_signal[i] < 0 else 0
                funding_dir = 1 if s > 0 else -1

                if breakout_dir == funding_dir:
                    weights[i] = s * self.max_weight  # 同向，全权重
                elif breakout_dir == 0:
                    weights[i] = s * self.max_weight * 0.7  # breakout无信号，0.7倍
                else:
                    weights[i] = s * self.max_weight * 0.3  # 反向，降到0.3倍
            else:
                weights[i] = s * self.max_weight

        return weights

"""
Cross-Sectional Alpha — 横截面动量轮动
信息来源: 币与币之间的相对强弱（与 time-series breakout 本质不同）
逻辑: 在 mid regime 内，定期选出最强 N 个币，等风险配置
"""
import numpy as np
import pandas as pd


class CrossSectionalAlpha:
    """横截面动量: 买最强、卖最弱（或只做多最强）"""

    def __init__(self, top_n=3, lookback_bars=20, rebalance_bars=24):
        self.top_n = top_n
        self.lookback_bars = lookback_bars
        self.rebalance_bars = rebalance_bars  # 每N根K线重排一次
        self.bars_since_rebalance = {}

    def compute_scores(self, df_dict, time):
        """
        计算所有币种的横截面动量得分
        返回 dict: {symbol: score} 和 排名
        """
        scores = {}
        for symbol, df in df_dict.items():
            score = self._momentum_score(df, time)
            if score is not None:
                scores[symbol] = score

        if len(scores) < self.top_n:
            return scores, {}

        # 排名: 高分 = 强动量
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        ranks = {s: i for i, (s, _) in enumerate(ranked)}
        return scores, ranks

    def get_weights(self, df_dict, time, current_weights, bars_since):
        """
        返回横截面权重 (np.array, len=币种数)
        仅在 rebalance 时更新，其余时间返回 None（保持不动）
        """
        symbols = list(df_dict.keys())
        n = len(symbols)

        # 初始化
        for s in symbols:
            if s not in self.bars_since_rebalance:
                self.bars_since_rebalance[s] = self.rebalance_bars

        # 检查是否需要重排
        should_rebalance = all(
            self.bars_since_rebalance.get(s, 0) >= self.rebalance_bars
            for s in symbols
        )

        if not should_rebalance and current_weights.sum() > 0:
            return None  # 不动

        # 计算得分和排名
        scores, ranks = self.compute_scores(df_dict, time)

        if len(ranks) < self.top_n:
            return np.zeros(n)

        # 选 top N，等权重
        weights = np.zeros(n)
        top_symbols = [s for s, r in ranks.items() if r < self.top_n]
        if not top_symbols:
            return np.zeros(n)

        weight_per = 1.0 / len(top_symbols)
        for i, s in enumerate(symbols):
            if s in top_symbols:
                weights[i] = weight_per

        # 重置计数
        for s in symbols:
            self.bars_since_rebalance[s] = 0

        return weights

    def tick(self, df_dict):
        """每根bar调用，更新计数器"""
        for s in df_dict:
            if s in self.bars_since_rebalance:
                self.bars_since_rebalance[s] += 1
            else:
                self.bars_since_rebalance[s] = 1

    def _momentum_score(self, df, time):
        """计算单个币种的横截面动量得分"""
        mask = df.index <= time
        close = df[mask]['close']
        if len(close) < self.lookback_bars + 1:
            return None

        # 20bar 收益率
        ret_20 = (close.iloc[-1] / close.iloc[-self.lookback_bars - 1]) - 1

        # 波动率调整（Sharpe-like，避免选到剧烈波动的币）
        daily_rets = close.pct_change().iloc[-self.lookback_bars:]
        vol = daily_rets.std() * np.sqrt(365 * 24)  # 年化波动
        if vol > 0:
            score = ret_20 / (vol + 1e-10)
        else:
            score = ret_20

        return score

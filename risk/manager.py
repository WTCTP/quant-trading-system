import numpy as np


class RiskManager:
    """风控：回撤监控 + EWMA协方差计算"""

    def __init__(self, initial_capital):
        self.peak_capital = initial_capital
        self.drawdown = 0.0

    def update(self, current_capital):
        if current_capital > self.peak_capital:
            self.peak_capital = current_capital
        self.drawdown = (current_capital - self.peak_capital) / self.peak_capital
        return self.drawdown

    def check(self):
        if self.drawdown <= -0.20:
            return 'liquidate'
        return 'ok'

import numpy as np

from config import MIN_WEIGHT_CHANGE, FEE_RATE, SLIPPAGE_BPS


class Executor:
    """组合执行器：调仓执行 + 成本计算"""

    def __init__(self, initial_capital):
        self.capital = initial_capital
        self.positions = {}       # {symbol: current_holdings_value}
        self.current_weights = None

    def rebalance(self, target_weights, symbols, prices, cash):
        """
        执行调仓，返回 (trade_logs, total_fee)
        target_weights: np.array, 目标权重
        symbols: list of str
        prices: np.array, 当前价格
        cash: float, 当前现金
        """
        trades = []
        total_fee = 0.0
        total_value = cash + sum(self.positions.values())

        # 检查是否需要调仓
        if self.current_weights is not None:
            diff = np.abs(target_weights - self.current_weights).max()
            if diff < MIN_WEIGHT_CHANGE:
                return trades, total_fee

        # 计算目标持仓值
        target_values = total_value * target_weights

        for i, symbol in enumerate(symbols):
            current_val = self.positions.get(symbol, 0)
            target_val = target_values[i]
            delta_val = target_val - current_val

            if abs(delta_val) < 1e-6:
                continue

            price = prices[i]
            delta_qty = delta_val / price

            # 手续费
            fee = abs(delta_val) * FEE_RATE
            total_fee += fee

            # 滑点
            slip = SLIPPAGE_BPS * abs(delta_val)
            total_fee += slip

            # 更新持仓
            self.positions[symbol] = target_val

            trades.append({
                'symbol': symbol,
                'delta_value': delta_val,
                'delta_qty': delta_qty,
                'price': price,
                'fee': fee,
            })

        # 扣费
        self.capital = total_value - total_fee - cash
        # 更新权重记录
        self.current_weights = target_weights.copy()

        return trades, total_fee

    def get_portfolio_value(self, prices, symbols):
        """按现价估算持仓市值"""
        value = 0
        for i, symbol in enumerate(symbols):
            if symbol in self.positions:
                value += self.positions[symbol]
        return value

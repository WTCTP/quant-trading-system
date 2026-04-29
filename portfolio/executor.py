"""
PortfolioExecutor v2 — 组合执行层 + 订单状态机
负责：仓位管理、订单生命周期、Maker/Taker路由、动态滑点、ADV限制、风控规则
不负责：信号生成、Alpha模型、策略逻辑
"""
from dataclasses import dataclass, field
import numpy as np
import pandas as pd


@dataclass
class OrderTask:
    """单个symbol的订单任务"""
    symbol: str
    side: str                     # 'buy' | 'sell'
    target_qty: float             # 目标数量（币本位）
    filled_qty: float = 0.0
    state: str = 'pending'        # pending/working/partial/filled/cancelled/failed
    limit_price: float = 0.0      # maker限价
    retry_count: int = 0
    bars_pending: int = 0         # 已等待的K线数
    created_time: object = None


class PortfolioExecutor:
    """执行层：接收目标权重 → 过滤+路由 → 分批执行 → 记录成交"""

    def __init__(self, symbols, initial_capital, *,
                 fee_rate, slippage_k, slippage_min_bps, slippage_max_bps,
                 max_position_adv_pct, max_order_adv_pct,
                 risk_portfolio_stop, risk_vol_shock,
                 risk_consecutive_loss, risk_pause_bars,
                 vol_lookback, vol_scale_cap, vol_floor,
                 min_weight_delta=0.05, maker_timeout_bars=2,
                 max_retries=3, maker_slip_bps=0.00002,
                 taker_slip_extra_bps=0.0005):
        self.symbols = list(symbols)
        self.initial_capital = initial_capital
        self.cash = initial_capital

        # 成本参数
        self.fee_rate = fee_rate
        self.slippage_k = slippage_k
        self.slippage_min_bps = slippage_min_bps
        self.slippage_max_bps = slippage_max_bps

        # 执行约束
        self.max_position_adv_pct = max_position_adv_pct
        self.max_order_adv_pct = max_order_adv_pct

        # 风控参数
        self.risk_portfolio_stop = risk_portfolio_stop
        self.risk_vol_shock = risk_vol_shock
        self.risk_consecutive_loss = risk_consecutive_loss
        self.risk_pause_bars = risk_pause_bars

        # Vol targeting 参数
        self.vol_lookback = vol_lookback
        self.vol_scale_cap = vol_scale_cap
        self.vol_floor = vol_floor

        # 执行层状态机参数
        self.min_weight_delta = min_weight_delta
        self.maker_timeout_bars = maker_timeout_bars
        self.max_retries = max_retries
        self.maker_slip_bps = maker_slip_bps
        self.taker_slip_extra_bps = taker_slip_extra_bps

        # 仓位状态
        self.qty = {s: 0.0 for s in self.symbols}
        self.current_weights = np.zeros(len(self.symbols))

        # 成本累计
        self.total_fee = 0.0
        self.total_slippage = 0.0
        self.total_funding_cost = 0.0
        self.trades = []

        # 流动性
        self.adv_data = {}

        # Vol targeting 状态
        self.portfolio_returns = []
        self.vol_scale = 1.0

        # 风控状态
        self.risk_paused = False
        self.risk_pause_bars_remaining = 0
        self.risk_leverage_halved = False
        self.consecutive_losses = 0

        # 订单追踪（跨bar）
        self.pending_orders = {}   # {symbol: OrderTask}

    # ─── 查询 ─────────────────────────────────

    def get_prices(self, df_dict, time):
        return np.array([df_dict[s].loc[time, 'close'] for s in self.symbols])

    def portfolio_value(self, prices):
        pos_val = sum(self.qty[s] * prices[i] for i, s in enumerate(self.symbols))
        return self.cash + pos_val

    @property
    def leverage_multiplier(self):
        return 0.5 if self.risk_leverage_halved else 1.0

    @property
    def has_position(self):
        return sum(self.qty.values()) > 0

    # ─── 流动性 ───────────────────────────────

    def update_adv(self, df_dict, time):
        for s in self.symbols:
            mask = df_dict[s].index <= time
            vol_series = df_dict[s][mask]['volume']
            if len(vol_series) < 24:
                self.adv_data[s] = vol_series.sum() if len(vol_series) > 0 else 1e9
            else:
                self.adv_data[s] = vol_series.iloc[-24:].mean() * 24
            self.adv_data[s] = max(self.adv_data[s], 1e6)

    # ─── 成本 ─────────────────────────────────

    def calc_slippage(self, order_value, symbol, is_taker=False):
        adv = self.adv_data.get(symbol, 1e9)
        slip_rate = self.slippage_k * abs(order_value) / adv
        slip_rate = max(self.slippage_min_bps, min(slip_rate, self.slippage_max_bps))
        if is_taker:
            slip_rate += self.taker_slip_extra_bps
        return slip_rate * abs(order_value)

    # ─── 资金费率 ─────────────────────────────

    def deduct_funding(self, funding_data, time, prices):
        for j, s in enumerate(self.symbols):
            pos_val = self.qty[s] * prices[j]
            if abs(pos_val) < 1:
                continue
            fund_series = funding_data.get(s)
            if fund_series is not None and time in fund_series.index:
                rate = fund_series.loc[time]
                if rate and not np.isnan(rate):
                    cost = pos_val * rate
                    self.cash -= cost
                    self.total_funding_cost += cost

    # ─── Vol Targeting ────────────────────────

    def update_vol_tracking(self, period_return):
        self.portfolio_returns.append(period_return)
        if len(self.portfolio_returns) > self.vol_lookback:
            self.portfolio_returns.pop(0)

    def get_vol_scale(self, portfolio_vol_target):
        if portfolio_vol_target <= 0 or len(self.portfolio_returns) < max(self.vol_lookback // 2, 6):
            return 1.0
        realized_vol = np.std(self.portfolio_returns) * np.sqrt(365 * 24)
        realized_vol = max(realized_vol, self.vol_floor)
        return min(portfolio_vol_target / realized_vol, self.vol_scale_cap)

    # ─── 连续亏损追踪 ─────────────────────────

    def update_consecutive_losses(self, new_value, old_value):
        if not self.has_position:
            return
        bar_pnl = new_value - old_value
        if bar_pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

    # ═══════════════════════════════════════════
    # 订单状态机 (V1)
    # ═══════════════════════════════════════════

    def _prepare_orders(self, weights, prices, total_value):
        """计算delta → 过滤trade buffer → 卖出优先排序 → 生成OrderTask"""
        symbol_index = {s: i for i, s in enumerate(self.symbols)}
        tasks = []

        for s in self.symbols:
            j = symbol_index[s]
            current_val = self.qty[s] * prices[j]
            current_weight = current_val / total_value if total_value > 0 else 0
            target_weight = max(weights[j], 0)

            delta_weight = target_weight - current_weight
            if abs(delta_weight) < self.min_weight_delta:
                # 撤销该symbol的旧订单（目标已变化很小）
                self.pending_orders.pop(s, None)
                continue

            target_val = total_value * target_weight

            # ADV仓位上限
            adv_s = self.adv_data.get(s, 1e9)
            max_allowed = adv_s * self.max_position_adv_pct
            target_val = min(target_val, max_allowed)

            delta_val = target_val - current_val
            if abs(delta_val) < 1.0:
                self.pending_orders.pop(s, None)
                continue

            # ADV单笔上限
            max_order = adv_s * self.max_order_adv_pct
            if abs(delta_val) > max_order:
                delta_val = max_order if delta_val > 0 else -max_order

            side = 'buy' if delta_val > 0 else 'sell'
            target_qty = self.qty[s] + delta_val / prices[j]

            # 检查是否有该symbol的旧订单
            old_order = self.pending_orders.get(s)
            if old_order and old_order.state in ('pending', 'working', 'partial'):
                # 目标变化不大→继续旧订单；变化大→取消旧订单重建
                old_target = old_order.target_qty
                if abs(target_qty - old_target) / max(old_target, 1e-8) < 0.3:
                    continue  # 复用旧订单

            # 创建新订单
            task = OrderTask(
                symbol=s,
                side=side,
                target_qty=target_qty,
                state='pending',
                created_time=time if 'time' in dir() else None,
            )
            tasks.append(task)

        # 排序：卖出优先（释放现金），然后买入
        tasks.sort(key=lambda t: 0 if t.side == 'sell' else 1)
        return tasks

    def _try_maker_fill(self, task, prices, ohlc):
        """尝试maker成交：检查OHLC是否穿过限价单"""
        j = self.symbols.index(task.symbol)
        close = prices[j]
        high = ohlc['high'][j] if 'high' in ohlc else close
        low = ohlc['low'][j] if 'low' in ohlc else close

        if task.side == 'buy':
            # 买单限价 = close下方0.1% (偏有利)
            task.limit_price = close * 0.999
            if low <= task.limit_price:
                return True, task.limit_price
        else:
            # 卖单限价 = close上方0.1%
            task.limit_price = close * 1.001
            if high >= task.limit_price:
                return True, task.limit_price

        return False, close

    def _execute_order_maker(self, task, prices, ohlc):
        """以maker价成交，使用maker滑点"""
        j = self.symbols.index(task.symbol)
        fill_price = task.limit_price

        current_val = self.qty[task.symbol] * prices[j]
        target_qty = task.target_qty
        delta_qty = target_qty - self.qty[task.symbol]
        delta_val = delta_qty * fill_price

        if abs(delta_qty) < 1e-8 or abs(delta_val) < 1.0:
            task.state = 'filled'
            self.pending_orders.pop(task.symbol, None)
            return

        # maker滑点 + 手续费
        slippage = abs(delta_val) * self.maker_slip_bps
        fee = abs(delta_val) * self.fee_rate

        self.qty[task.symbol] = target_qty
        self.cash -= delta_val + fee + slippage
        self.total_fee += fee
        self.total_slippage += slippage

        self.trades.append({
            'time': task.created_time, 'symbol': task.symbol,
            'delta_value': round(delta_val, 2),
            'price': round(fill_price, 2),
            'fee': round(fee, 4),
            'slippage': round(slippage, 4),
            'weight': 0,
            'type': 'maker',
        })
        task.filled_qty = target_qty
        task.state = 'filled'
        self.pending_orders.pop(task.symbol, None)

    def _execute_order_taker(self, task, prices):
        """以taker价成交，使用taker滑点"""
        j = self.symbols.index(task.symbol)
        fill_price = prices[j]

        current_val = self.qty[task.symbol] * prices[j]
        target_qty = task.target_qty
        delta_qty = target_qty - self.qty[task.symbol]
        delta_val = delta_qty * fill_price

        if abs(delta_qty) < 1e-8 or abs(delta_val) < 1.0:
            task.state = 'filled'
            self.pending_orders.pop(task.symbol, None)
            return

        slippage = self.calc_slippage(delta_val, task.symbol, is_taker=True)
        fee = abs(delta_val) * self.fee_rate

        self.qty[task.symbol] = target_qty
        self.cash -= delta_val + fee + slippage
        self.total_fee += fee
        self.total_slippage += slippage

        self.trades.append({
            'time': task.created_time, 'symbol': task.symbol,
            'delta_value': round(delta_val, 2),
            'price': round(fill_price, 2),
            'fee': round(fee, 4),
            'slippage': round(slippage, 4),
            'weight': 0,
            'type': 'taker',
        })
        task.filled_qty = target_qty
        task.state = 'filled'
        self.pending_orders.pop(task.symbol, None)

    def _process_order(self, task, prices, ohlc, time):
        """单订单状态机: pending → working → (filled | taker fallback | cancelled)"""
        if task.state == 'pending':
            task.state = 'working'
            task.bars_pending = 0
            self.pending_orders[task.symbol] = task

        if task.state == 'working':
            task.bars_pending += 1

            # 尝试maker成交
            filled, _ = self._try_maker_fill(task, prices, ohlc)
            if filled:
                self._execute_order_maker(task, prices, ohlc)
                return

            # 超时→转taker
            if task.bars_pending >= self.maker_timeout_bars:
                task.retry_count += 1
                if task.retry_count >= self.max_retries:
                    self._execute_order_taker(task, prices)
                else:
                    # 重置pending让maker重试
                    task.state = 'pending'
                    task.bars_pending = 0
                return

            # 还在等待maker成交
            task.state = 'working'

    def _process_orders(self, tasks, prices, ohlc, time):
        """遍历订单列表，逐个执行"""
        for task in tasks:
            self._process_order(task, prices, ohlc, time)

        # 更新current_weights
        total_value = self.portfolio_value(prices)
        if total_value > 0:
            for j, s in enumerate(self.symbols):
                self.current_weights[j] = self.qty[s] * prices[j] / total_value

    def _process_existing_orders(self, prices, ohlc, time):
        """处理上一bar残留的pending订单"""
        for s, task in list(self.pending_orders.items()):
            if task.state in ('working', 'pending'):
                self._process_order(task, prices, ohlc, time)

    # ═══════════════════════════════════════════
    # 主执行入口
    # ═══════════════════════════════════════════

    def rebalance_to(self, weights, prices, total_value, time, ohlc=None):
        """两阶段执行：
        1. prepare_orders: 计算delta → 过滤 → 排序 → 生成订单
        2. process_orders: 状态机执行 (maker→timeout→taker)
        """
        # 先处理残留订单
        if ohlc is None:
            ohlc = {'high': prices, 'low': prices}
        self._process_existing_orders(prices, ohlc, time)

        # 准备新订单
        tasks = self._prepare_orders(weights, prices, total_value)
        if not tasks:
            return

        self._process_orders(tasks, prices, ohlc, time)

    def force_close(self, prices, time, reason=''):
        # 取消所有pending订单
        self.pending_orders.clear()

        for j, s in enumerate(self.symbols):
            if self.qty[s] > 0:
                val = self.qty[s] * prices[j]
                fee = val * self.fee_rate
                slippage = self.calc_slippage(-val, s, is_taker=True)
                self.cash += val - fee - slippage
                self.total_fee += fee
                self.total_slippage += slippage
                self.trades.append({
                    'time': time, 'symbol': s,
                    'delta_value': round(-val, 2),
                    'price': round(prices[j], 2),
                    'fee': round(fee, 4),
                    'slippage': round(slippage, 4),
                    'weight': 0,
                    'type': 'force',
                })
                self.qty[s] = 0
        self.current_weights = np.zeros(len(self.symbols))

    # ─── 风控规则 ─────────────────────────────

    def check_risk_rules(self, prices, total_value, time,
                         drawdown, portfolio_vol_target):
        """返回 (halted, reason)"""
        if self.risk_paused:
            self.risk_pause_bars_remaining -= 1
            if self.risk_pause_bars_remaining <= 0:
                self.risk_paused = False
                self.risk_leverage_halved = False
                self.consecutive_losses = 0
                self.pending_orders.clear()
            return self.risk_paused, 'paused'

        # 规则1: 组合止损
        if drawdown <= self.risk_portfolio_stop and not self.risk_leverage_halved:
            self.risk_leverage_halved = True
            self.force_close(prices, time, f'组合止损 DD={drawdown:.1%} -> 杠杆减半')

        # 规则2: 波动率熔断
        if len(self.portfolio_returns) >= 6:
            realized_vol = np.std(self.portfolio_returns) * np.sqrt(365 * 24)
            if realized_vol > self.risk_vol_shock * portfolio_vol_target:
                self.force_close(prices, time, f'波动率熔断 vol={realized_vol:.1%}')
                self.risk_paused = True
                self.risk_pause_bars_remaining = self.risk_pause_bars
                return True, 'vol_shock'

        # 规则3: 连续亏损
        if self.consecutive_losses >= self.risk_consecutive_loss:
            self.force_close(prices, time, f'连续亏损 {self.consecutive_losses}笔 -> 暂停')
            self.risk_paused = True
            self.risk_pause_bars_remaining = self.risk_pause_bars
            return True, 'kill_switch'

        return False, 'ok'

    # ─── 分析工具 ─────────────────────────────

    def get_trades_df(self):
        return pd.DataFrame(self.trades)

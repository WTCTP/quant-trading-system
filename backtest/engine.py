import numpy as np
import pandas as pd

from config import (
    SYMBOLS, INITIAL_CAPITAL, TRAIN_WINDOW, RETRAIN_EVERY,
    FEE_RATE, SLIPPAGE_K, SLIPPAGE_MIN_BPS, SLIPPAGE_MAX_BPS,
    MAX_POSITION_ADV_PCT, MAX_ORDER_ADV_PCT, MAX_WEIGHT,
    EXIT_BUFFER_BARS, SIGNAL_CONFIRM,
    SIGNAL_CHANGE_THRESH, MIN_WEIGHT_DELTA, COOLING_BARS, REGIME_ENTRY_BARS,
    VOL_REGIME_EMA_SPAN, ENTRY_DELAY_BARS,
    SMOOTH_WEIGHT_EXEC, GLOBAL_VOL_FILTER, NO_TRADE_ZONE,
    DISABLE_TRADING, SIGNAL_INVERT, SIGNAL_CONFIRM_BARS,
    USE_SIGNAL_TIER, SIGNAL_Q_CORE, SIGNAL_Q_ATTACK, SIGNAL_CHANGE_FILTER,
    SIGNAL_MUST_INCREASE, ENTRY_PRICE_CONFIRM, PRICE_BREAKOUT_BARS, MIN_HOLD_BARS_EXEC,
    BASE_EXPOSURE, SIGNAL_BOOST_MID, SIGNAL_BOOST_HIGH,
    TARGET_VOL, VOL_LOOKBACK, VOL_SCALE_CAP, VOL_FLOOR,
    BASE_LOW, BASE_MID, BASE_HIGH,
    TARGET_VOL_LOW, TARGET_VOL_MID, TARGET_VOL_HIGH,
    PORTFOLIO_VOL_TARGET, PORTFOLIO_LEVERAGE,
    FUNDING_COST_ENABLED,
    RISK_PORTFOLIO_STOP, RISK_VOL_SHOCK, RISK_CONSECUTIVE_LOSS, RISK_PAUSE_BARS,
    USE_FUNDING, USE_CROSS_SECTIONAL,
    MIN_WEIGHT_DELTA, MAKER_TIMEOUT_BARS, MAX_RETRIES,
    MAKER_SLIP_BPS, TAKER_SLIP_EXTRA_BPS,
)
from portfolio.optimizer import (
    shrink_returns, shrink_covariance, compute_ewma_cov,
    optimize_weights,
)
from portfolio.executor import PortfolioExecutor
from alpha.model import RegimeAlphaModel, get_regime
from alpha.pullback import PullbackSignal
from alpha.cross_sectional import CrossSectionalAlpha
from alpha.funding_alpha import FundingAlpha


class PortfolioBacktest:
    """半连续执行：Regime过滤 + 多Alpha(Breakout+Funding+CrossSectional) + 风险预算"""

    def __init__(self, risk_manager, logger,
                 base_low=None, base_mid=None, base_high=None,
                 portfolio_vol_target=None,
                 use_pullback=False, use_cross_sectional=None, use_funding=None,
                 cross_sectional_weight=0.30, funding_weight=0.15,
                 portfolio_leverage=None, initial_capital=None):
        self.risk = risk_manager
        self.logger = logger
        self.initial_capital = initial_capital if initial_capital is not None else INITIAL_CAPITAL
        self.symbols = SYMBOLS
        self.models = {}
        self.records = []

        # 执行器（在 run() 中根据实际 symbols 创建）
        self.executor = None

        # Regime分层底仓
        self.base_low = base_low if base_low is not None else BASE_LOW
        self.base_mid = base_mid if base_mid is not None else BASE_MID
        self.base_high = base_high if base_high is not None else BASE_HIGH
        # Portfolio Layer
        self.portfolio_vol_target = portfolio_vol_target if portfolio_vol_target is not None else PORTFOLIO_VOL_TARGET
        self.portfolio_leverage = portfolio_leverage if portfolio_leverage is not None else PORTFOLIO_LEVERAGE
        # Alpha开关
        self.use_pullback = use_pullback
        self.use_cross_sectional = use_cross_sectional if use_cross_sectional is not None else USE_CROSS_SECTIONAL
        self.use_funding = use_funding if use_funding is not None else USE_FUNDING
        # Pullback (已弃用)
        self.pullback = PullbackSignal() if use_pullback else None
        # Cross-Sectional
        self.cross_sectional_weight = cross_sectional_weight
        self.cs_alpha = CrossSectionalAlpha(top_n=3, lookback_bars=20, rebalance_bars=24) if self.use_cross_sectional else None
        # Funding (默认关闭)
        self.funding_weight = funding_weight
        self.funding_alpha = FundingAlpha(lookback_days=90, extreme_pct=95, max_weight=0.10) if self.use_funding else None
        self.funding_data = None
        # 执行状态
        self.prev_signal = None
        self.bars_since_trade = COOLING_BARS
        self.signal_confirm_counter = 0
        self.signal_history = []
        self.prev_signal_max = 0.0
        self.bars_since_entry = MIN_HOLD_BARS_EXEC
        self.current_exposure_level = 'none'
        self.current_regime = 'low'
        self.regime_entry_counter = 0
        self.ema_vol_regime = 1.0
        self.entry_delay_counter = 0
        self.entry_delay_target = None
        self.entry_delay_breakout = False

    # ─── 委托给 executor 的属性（向后兼容）───

    @property
    def trades(self):
        return self.executor.trades if self.executor else []

    @property
    def total_fee(self):
        return self.executor.total_fee if self.executor else 0.0

    @property
    def total_slippage(self):
        return self.executor.total_slippage if self.executor else 0.0

    @property
    def total_funding_cost(self):
        return self.executor.total_funding_cost if self.executor else 0.0

    # ─── 工具方法 ─────────────────────────────

    def _prepare_slice(self, df, end_time):
        return df[df.index <= end_time]

    def _get_returns_matrix(self, df_dict, end_time, window=120):
        ret_data = {}
        for s in self.symbols:
            df = df_dict[s]
            mask = df.index <= end_time
            ret_data[s] = df[mask]['close'].pct_change().iloc[-window:]
        returns_df = pd.DataFrame(ret_data).dropna()
        return returns_df

    def _get_current_regime(self, df_dict, time):
        from alpha.features import build_features
        first_sym = self.symbols[0]
        feats = build_features(df_dict[first_sym])
        if time not in feats.index:
            return 'mid'
        raw_vol = feats.loc[time, 'vol_regime']
        alpha = 2.0 / (VOL_REGIME_EMA_SPAN + 1)
        self.ema_vol_regime = alpha * raw_vol + (1 - alpha) * self.ema_vol_regime
        return get_regime(self.ema_vol_regime)

    def _get_btc_vol_regime(self, df_dict, time):
        from alpha.features import build_features
        btc_sym = 'BTC/USDT'
        if btc_sym not in df_dict:
            return None
        feats = build_features(df_dict[btc_sym])
        if time not in feats.index:
            return None
        return feats.loc[time, 'vol_regime']

    def _get_ohlc(self, df_dict, time):
        """获取当前bar各symbol的high/low价格数组"""
        highs = []
        lows = []
        for s in self.symbols:
            df = df_dict[s]
            if time in df.index:
                highs.append(df.loc[time, 'high'])
                lows.append(df.loc[time, 'low'])
            else:
                close = df.loc[time, 'close'] if time in df.index else 0
                highs.append(close)
                lows.append(close)
        return {'high': np.array(highs), 'low': np.array(lows)}

    def _check_price_breakout(self, df_dict, time):
        """价格确认: BTC收盘价突破近期N根K线最高价"""
        btc_sym = 'BTC/USDT'
        if btc_sym not in df_dict:
            return True
        df = df_dict[btc_sym]
        mask = df.index <= time
        slice_df = df[mask]
        if len(slice_df) < PRICE_BREAKOUT_BARS + 1:
            return True
        recent_high = slice_df['high'].iloc[-(PRICE_BREAKOUT_BARS+1):-1].max()
        current_close = slice_df['close'].iloc[-1]
        return current_close > recent_high

    def _get_signal_percentile(self, signal_val):
        """扩展窗口：当前信号在过去所有信号中的分位数 (0~1)"""
        if len(self.signal_history) < 50:
            return 1.0  # 样本不足，允许交易
        arr = np.array(self.signal_history)
        return (arr < signal_val).mean()

    def _reset_mid_state(self):
        """退出中波时重置执行状态"""
        self.regime_entry_counter = 0
        self.entry_delay_counter = 0
        self.entry_delay_target = None
        self.entry_delay_breakout = False
        self.current_exposure_level = 'none'

    def _handle_low_regime(self, df_dict, time, prices, total_value, vol_scale, force=False):
        """低波轻仓：固定底仓 + alpha引导权重 + 简化执行（无信号分层）"""
        mu_raw = self._predict_all(df_dict, time)

        should_trade = force or (self.executor.current_weights.sum() == 0 or
                        self.bars_since_trade >= 24)

        if DISABLE_TRADING:
            should_trade = False

        if should_trade:
            ret_df = self._get_returns_matrix(df_dict, time)
            cov = compute_ewma_cov(ret_df) if len(ret_df) > 30 else np.eye(len(self.symbols))
            mu = shrink_returns(mu_raw)
            cov_reg = shrink_covariance(cov)
            raw_weights = optimize_weights(mu, cov_reg)

            effective_leverage = self.portfolio_leverage * self.executor.leverage_multiplier
            target_exposure = self.base_low * vol_scale * effective_leverage

            smoothed_w = SMOOTH_WEIGHT_EXEC * self.executor.current_weights + (1 - SMOOTH_WEIGHT_EXEC) * raw_weights
            smoothed_w *= target_exposure

            ohlc = self._get_ohlc(df_dict, time)
            self.executor.rebalance_to(smoothed_w, prices, total_value, time, ohlc)
            self.prev_signal = mu_raw.copy()
            self.current_exposure_level = 'base'
            self.bars_since_trade = 0
            self.bars_since_entry = 0

    def _handle_mid_regime(self, df_dict, time, prices, total_value, vol_scale):
        """中波完整逻辑：信号分层 + 迟滞层 + 入场时机优化"""
        # 进入确认
        if self.executor.current_weights.sum() == 0 and REGIME_ENTRY_BARS > 0:
            self.regime_entry_counter += 1
            if self.regime_entry_counter < REGIME_ENTRY_BARS:
                return 'mid_entry_wait', float('nan')
        self.regime_entry_counter = REGIME_ENTRY_BARS

        mu_raw = self._predict_all(df_dict, time)
        signal_max = float(abs(mu_raw).max())

        # 延迟确认
        if SIGNAL_CONFIRM_BARS > 0:
            if self.signal_confirm_counter < SIGNAL_CONFIRM_BARS:
                self.signal_confirm_counter += 1
                return 'mid_confirming', signal_max
        self.signal_confirm_counter = SIGNAL_CONFIRM_BARS

        # 诊断禁止交易
        if DISABLE_TRADING:
            self.signal_history.append(signal_max)
            return 'mid_disabled', signal_max

        # No-Trade Zone: 信号太弱且当前空仓 → 不开仓
        if signal_max < NO_TRADE_ZONE and self.executor.current_weights.sum() == 0:
            self.signal_history.append(signal_max)
            return 'mid_no_trade', signal_max

        # 1. 信号分层 → 目标暴露等级
        signal_increasing = signal_max > self.prev_signal_max
        self.prev_signal_max = signal_max
        breakout_ok = self._check_price_breakout(df_dict, time) if ENTRY_PRICE_CONFIRM else True

        if signal_max >= SIGNAL_BOOST_HIGH:
            if SIGNAL_MUST_INCREASE and not signal_increasing:
                target_exposure = self.base_mid * 0.60
                target_level = 'mid'
                exp_label = 'mid_base_hi'
            elif ENTRY_PRICE_CONFIRM and not breakout_ok:
                target_exposure = self.base_mid * 0.60
                target_level = 'mid'
                exp_label = 'mid_base_hi'
            else:
                target_exposure = self.base_mid * 1.00
                target_level = 'full'
                exp_label = 'mid_full'
        elif signal_max >= SIGNAL_BOOST_MID:
            target_exposure = self.base_mid * 0.60
            target_level = 'mid'
            exp_label = 'mid_mid'
        else:
            target_exposure = self.base_mid * 0.30
            target_level = 'base'
            exp_label = 'mid_base'

        # 2. 非对称迟滞：升仓延迟、降仓立即
        level_order = {'none': 0, 'base': 1, 'mid': 2, 'full': 3}
        scaling_up = level_order.get(target_level, 0) > level_order.get(self.current_exposure_level, 0)
        scaling_down = level_order.get(target_level, 0) < level_order.get(self.current_exposure_level, 0)
        level_changed = scaling_up or scaling_down
        cooled_down = self.bars_since_trade >= COOLING_BARS
        first_entry = (self.executor.current_weights.sum() == 0)

        should_execute = False
        execute_level = target_level
        execute_exposure = target_exposure

        # 延迟确认状态机
        in_delay = (self.entry_delay_target is not None)

        if in_delay:
            delay_target = self.entry_delay_target
            if level_order.get(target_level, 0) < level_order.get(delay_target, 0):
                self.entry_delay_target = None
                self.entry_delay_counter = 0
                self.entry_delay_breakout = False
                regime_tag = 'mid_hold'
            else:
                if breakout_ok:
                    self.entry_delay_breakout = True
                delay_ok = True
                if ENTRY_PRICE_CONFIRM and not self.entry_delay_breakout:
                    delay_ok = False

                if delay_ok:
                    self.entry_delay_counter += 1
                    if self.entry_delay_counter >= ENTRY_DELAY_BARS:
                        should_execute = True
                        execute_level = delay_target
                        execute_exposure = {'base': self.base_mid, 'mid': 0.6, 'full': 1.0}[delay_target]
                        exp_label = {'base': 'mid_base', 'mid': 'mid_mid', 'full': 'mid_full'}[delay_target]
                        self.entry_delay_target = None
                        self.entry_delay_counter = 0
                        self.entry_delay_breakout = False
                    else:
                        regime_tag = 'mid_entry_delay'
                else:
                    regime_tag = 'mid_entry_delay'

        elif (level_changed and cooled_down) or first_entry:
            if (scaling_up or first_entry) and ENTRY_DELAY_BARS > 0:
                self.entry_delay_target = target_level
                self.entry_delay_counter = 0
                self.entry_delay_breakout = False
                regime_tag = 'mid_entry_delay'
            else:
                should_execute = True
        else:
            regime_tag = 'mid_hold'

        if should_execute:
            ret_df = self._get_returns_matrix(df_dict, time)
            cov = compute_ewma_cov(ret_df) if len(ret_df) > 30 else np.eye(len(self.symbols))
            mu = shrink_returns(mu_raw)
            cov_reg = shrink_covariance(cov)

            # === 多Alpha融合: Breakout + Funding + CrossSectional ===
            raw_weights = optimize_weights(mu, cov_reg)

            # Funding Alpha: 独立信息源，极端拥挤信号
            if self.use_funding and self.funding_data is not None:
                fund_weights = self.funding_alpha.get_weights(
                    self.funding_data, time, mu_raw
                )
                if fund_weights.sum() > 1e-6:
                    raw_weights = (1 - self.funding_weight) * raw_weights + \
                                  self.funding_weight * fund_weights

            # Cross-Sectional Alpha
            if self.use_cross_sectional and self.cs_alpha is not None:
                cs_weights = self.cs_alpha.get_weights(
                    {s: self._prepare_slice(df_dict[s], time) for s in self.symbols},
                    time, self.executor.current_weights, self.bars_since_trade
                )
                if cs_weights is not None and cs_weights.sum() > 0:
                    raw_weights = (1 - self.cross_sectional_weight) * raw_weights + \
                                  self.cross_sectional_weight * cs_weights

            smoothed_w = SMOOTH_WEIGHT_EXEC * self.executor.current_weights + (1 - SMOOTH_WEIGHT_EXEC) * raw_weights

            effective_leverage = self.portfolio_leverage * self.executor.leverage_multiplier
            smoothed_w *= execute_exposure * vol_scale * effective_leverage

            ohlc = self._get_ohlc(df_dict, time)
            self.executor.rebalance_to(smoothed_w, prices, total_value, time, ohlc)
            self.prev_signal = mu_raw.copy()
            self.current_exposure_level = execute_level
            self.bars_since_trade = 0
            self.bars_since_entry = 0
            regime_tag = exp_label

        self.signal_history.append(signal_max)
        return regime_tag, signal_max

    def process_bar(self, i, time, df_dict):
        """处理单根K线 — 供 backtest 和 paper trading 共用"""
        n = len(self.symbols)

        prices = self.executor.get_prices(df_dict, time)
        total_value = self.executor.portfolio_value(prices)

        # === 更新ADV (每24根K线更新一次) ===
        if i % 24 == 0:
            self.executor.update_adv(df_dict, time)

        # === 资金费率成本 ===
        if FUNDING_COST_ENABLED and self.funding_data is not None:
            self.executor.deduct_funding(self.funding_data, time, prices)

        # === 风控规则检查 ===
        risk_halted, risk_reason = self.executor.check_risk_rules(
            prices, total_value, time,
            self.risk.drawdown, self.portfolio_vol_target)
        if risk_halted:
            self._reset_mid_state()
            self.prev_signal = np.zeros(len(self.symbols))
            self.records.append({
                'time': time, 'capital': self.executor.portfolio_value(prices),
                'regime': f'risk_{risk_reason}', 'signal_max': float('nan')
            })
            return

        # === Vol Targeting ===
        prev_total = self.records[-1]['capital'] if self.records else total_value
        period_return = (total_value / prev_total) - 1 if prev_total > 0 else 0
        self.executor.update_vol_tracking(period_return)
        vol_scale = self.executor.get_vol_scale(self.portfolio_vol_target)

        # 简单回撤清仓
        self.risk.update(total_value)
        if self.risk.check() == 'liquidate':
            self.executor.force_close(prices, time, '风控清仓')
            self._reset_mid_state()
            self.prev_signal = np.zeros(n)
            self.records.append({
                'time': time, 'capital': self.executor.portfolio_value(prices),
                'regime': 'LIQ', 'signal_max': float('nan')
            })
            return

        # 全局波动率过滤
        global_vol = self._get_btc_vol_regime(df_dict, time)
        if global_vol and global_vol > GLOBAL_VOL_FILTER:
            self.executor.force_close(prices, time, '全局高波')
            self._reset_mid_state()
            self.prev_signal = np.zeros(n)
            self.records.append({
                'time': time, 'capital': self.executor.portfolio_value(prices),
                'regime': 'global_high', 'signal_max': float('nan')
            })
            return

        # 重训
        if i % RETRAIN_EVERY == 0 or i == TRAIN_WINDOW:
            for s in self.symbols:
                train_df = self._prepare_slice(df_dict[s], time)
                self.models[s].train(train_df)

        # 横截面alpha计时器
        if self.use_cross_sectional:
            self.cs_alpha.tick(df_dict)

        self.current_regime = self._get_current_regime(df_dict, time)
        self.bars_since_trade += 1

        # === 高波 → 清仓 ===
        if self.current_regime == 'high':
            self._reset_mid_state()
            self.bars_since_entry += 1
            if self.executor.has_position and self.bars_since_entry < MIN_HOLD_BARS_EXEC:
                self.records.append({
                    'time': time, 'capital': total_value,
                    'regime': 'high_forced_hold', 'signal_max': float('nan')
                })
                return
            self.executor.force_close(prices, time, '高波空仓')
            self.records.append({
                'time': time, 'capital': self.executor.portfolio_value(prices),
                'regime': 'high', 'signal_max': float('nan')
            })
            return

        # === 低波 → 轻仓试探 ===
        if self.current_regime == 'low':
            self._reset_mid_state()
            self.bars_since_entry += 1

            if self.current_exposure_level not in ('none', 'base'):
                self._handle_low_regime(df_dict, time, prices, total_value, vol_scale, force=True)
            elif self.bars_since_trade >= 24:
                self._handle_low_regime(df_dict, time, prices, total_value, vol_scale)

            regime_tag = 'low'
            signal_max = float('nan')
        else:
            # === 中波 → 完整alpha逻辑 ===
            self.bars_since_entry = 0
            regime_tag, signal_max = self._handle_mid_regime(
                df_dict, time, prices, total_value, vol_scale)

        # 记录
        new_value = self.executor.portfolio_value(prices)
        self.records.append({
            'time': time,
            'capital': new_value,
            'regime': regime_tag,
            'signal_max': signal_max,
        })

        # 连续亏损追踪
        self.executor.update_consecutive_losses(new_value, total_value)

    def run(self, df_dict):
        self.symbols = list(df_dict.keys())
        n = len(self.symbols)

        # 创建执行器
        self.executor = PortfolioExecutor(
            self.symbols, self.initial_capital,
            fee_rate=FEE_RATE,
            slippage_k=SLIPPAGE_K,
            slippage_min_bps=SLIPPAGE_MIN_BPS,
            slippage_max_bps=SLIPPAGE_MAX_BPS,
            max_position_adv_pct=MAX_POSITION_ADV_PCT,
            max_order_adv_pct=MAX_ORDER_ADV_PCT,
            risk_portfolio_stop=RISK_PORTFOLIO_STOP,
            risk_vol_shock=RISK_VOL_SHOCK,
            risk_consecutive_loss=RISK_CONSECUTIVE_LOSS,
            risk_pause_bars=RISK_PAUSE_BARS,
            vol_lookback=VOL_LOOKBACK,
            vol_scale_cap=VOL_SCALE_CAP,
            vol_floor=VOL_FLOOR,
            min_weight_delta=MIN_WEIGHT_DELTA,
            maker_timeout_bars=MAKER_TIMEOUT_BARS,
            max_retries=MAX_RETRIES,
            maker_slip_bps=MAKER_SLIP_BPS,
            taker_slip_extra_bps=TAKER_SLIP_EXTRA_BPS,
        )

        self.models = {s: RegimeAlphaModel() for s in self.symbols}
        self.prev_signal = np.zeros(n)

        # 加载Funding Rate数据
        if self.use_funding:
            from data.funding_fetcher import FundingFetcher, align_funding_to_bars
            try:
                ff = FundingFetcher()
                funding_raw = ff.fetch_all(self.symbols)
                bar_idx = df_dict[self.symbols[0]].index
                self.funding_data = {
                    s: align_funding_to_bars(funding_raw.get(s, pd.DataFrame()), bar_idx)
                    for s in self.symbols
                }
            except Exception as e:
                print(f'  Warning: funding data unavailable ({e}), disabling funding alpha')
                self.use_funding = False
                self.funding_data = None

        all_times = sorted(set.union(*[set(df.index) for df in df_dict.values()]))

        for i, time in enumerate(all_times):
            if i < TRAIN_WINDOW:
                continue
            self.process_bar(i, time, df_dict)

        return self._summary()

    def _compute_pullback_scores(self, df_dict, time, mu_raw):
        """计算各币种的回调入场评分 (0~1)，用于调制breakout权重"""
        scores = np.zeros(len(self.symbols))
        for i, s in enumerate(self.symbols):
            data = self._prepare_slice(df_dict[s], time)
            if len(data) < 60:
                continue
            alpha_signal = mu_raw[i] if i < len(mu_raw) else 0
            scores[i] = self.pullback.get_entry_signal(data, alpha_signal)
        # 归一化到 [0, 1]
        smax = scores.max()
        if smax > 0:
            scores = scores / smax
        return scores

    def _predict_all(self, df_dict, time):
        mu_raw = []
        for s in self.symbols:
            data = self._prepare_slice(df_dict[s], time)
            mu = self.models[s].predict(data)
            mu_raw.append(mu[0] if len(mu) > 0 else 0)
        result = np.array(mu_raw)
        if SIGNAL_INVERT:
            result = -result
        return result

    def _summary(self):
        df = pd.DataFrame(self.records)
        if df.empty:
            return {'error': 'no records'}
        df['returns'] = df['capital'].pct_change()
        final_capital = df['capital'].iloc[-1]
        total_return = (final_capital - self.initial_capital) / self.initial_capital
        peak = df['capital'].expanding().max()
        drawdowns = (df['capital'] - peak) / peak
        max_dd = drawdowns.min()
        avg_ret = df['returns'].mean()
        std_ret = df['returns'].std()
        sharpe = (avg_ret / std_ret) * np.sqrt(365 * 24) if std_ret and std_ret > 0 else 0

        return {
            'initial_capital': self.initial_capital,
            'final_capital': round(final_capital, 2),
            'total_return': f'{total_return:.2%}',
            'total_return_numeric': round(total_return, 4),
            'max_drawdown': f'{max_dd:.2%}',
            'max_drawdown_numeric': round(max_dd, 4),
            'sharpe_ratio': round(sharpe, 2),
            'sharpe_ratio_numeric': round(sharpe, 2),
            'total_bars': len(df),
            'trade_events': len(self.trades),
            'total_fee': round(self.total_fee, 2),
            'total_slippage': round(self.total_slippage, 2),
            'total_funding_cost': round(self.total_funding_cost, 2),
        }

    def get_trades_df(self):
        return pd.DataFrame(self.trades)

    def get_coefficients(self):
        result = {}
        for s in self.symbols:
            coef = self.models[s].get_coefficients()
            if coef:
                result[s] = coef
        return result

    def get_signal_bucket_analysis(self, n_buckets=10):
        """按信号强度分桶，统计每档收益/Sharpe/胜率"""
        df = pd.DataFrame(self.records)
        if df.empty:
            return {}
        df['returns'] = df['capital'].pct_change()

        mid_df = df[df['signal_max'].notna()].copy()
        if len(mid_df) < 50:
            return {}

        try:
            mid_df['bucket'] = pd.qcut(mid_df['signal_max'], n_buckets, labels=False, duplicates='drop')
        except ValueError:
            return {}

        results = []
        for b in sorted(mid_df['bucket'].unique()):
            subset = mid_df[mid_df['bucket'] == b]
            n = len(subset)
            cum = (1 + subset['returns']).prod() - 1
            avg_r = subset['returns'].mean()
            std_r = subset['returns'].std()
            sharpe = (avg_r / std_r) * np.sqrt(365 * 24) if std_r and std_r > 0 else 0
            winrate = (subset['returns'] > 0).mean()

            lo = subset['signal_max'].min()
            hi = subset['signal_max'].max()

            results.append({
                'bucket': int(b),
                'count': n,
                'signal_range': f'{lo:.3f}~{hi:.3f}',
                'cum_return': f'{cum:.2%}',
                'sharpe': round(sharpe, 2),
                'winrate': f'{winrate:.1%}',
            })

        return results

    def get_signal_stability_analysis(self):
        """时间稳定性：按年份拆分，看高分位信号是否跨时间稳定"""
        df = pd.DataFrame(self.records)
        if df.empty:
            return {}
        df['returns'] = df['capital'].pct_change()
        df['year'] = pd.to_datetime(df['time']).dt.year

        mid_df = df[df['signal_max'].notna()].copy()
        if len(mid_df) < 50:
            return {}

        try:
            mid_df['bucket'] = pd.qcut(mid_df['signal_max'], 10, labels=False, duplicates='drop')
        except ValueError:
            return {}

        results = []
        for year in sorted(mid_df['year'].unique()):
            year_df = mid_df[mid_df['year'] == year]
            for b in sorted(year_df['bucket'].unique()):
                subset = year_df[year_df['bucket'] == b]
                if len(subset) < 10:
                    continue
                cum = (1 + subset['returns']).prod() - 1
                avg_r = subset['returns'].mean()
                std_r = subset['returns'].std()
                sharpe = (avg_r / std_r) * np.sqrt(365 * 24) if std_r and std_r > 0 else 0
                results.append({
                    'year': int(year),
                    'bucket': int(b),
                    'count': len(subset),
                    'cum_return': f'{cum:.2%}',
                    'sharpe': round(sharpe, 2),
                })

        return results

    def get_regime_analysis(self, df_dict):
        df = pd.DataFrame(self.records)
        if df.empty:
            return {}
        df['returns'] = df['capital'].pct_change()

        def stats(subset, label):
            if len(subset) < 10:
                return {'label': label, 'bars': 0}
            cum = (1 + subset['returns']).prod() - 1
            sharpe = (subset['returns'].mean() / subset['returns'].std()) * np.sqrt(365 * 24) \
                if subset['returns'].std() > 0 else 0
            return {
                'label': label,
                'bars': len(subset),
                'cum_return': f'{cum:.2%}',
                'sharpe': round(sharpe, 2),
            }

        return [
            stats(df[df['regime'] == 'mid_full'], '中波-全仓(信号≥0.15)'),
            stats(df[df['regime'] == 'mid_mid'], '中波-中仓(0.10~0.15)'),
            stats(df[df['regime'] == 'mid_base'], '中波-基仓'),
            stats(df[df['regime'] == 'mid_base_hi'], '中波-降级(强信号未确认)'),
            stats(df[df['regime'] == 'mid_hold'], '中波-持有'),
            stats(df[df['regime'] == 'mid_entry_wait'], '中波-等待确认(防闪烁)'),
            stats(df[df['regime'] == 'mid_entry_delay'], '中波-升仓延迟(入场时机)'),
            stats(df[df['regime'] == 'low'], '低波-轻仓试探'),
            stats(df[df['regime'] == 'low_forced_hold'], '低波-强制持有'),
            stats(df[df['regime'] == 'high'], '高波-空仓'),
            stats(df[df['regime'] == 'high_forced_hold'], '高波-强制持有'),
        ]

    def get_entry_timing_analysis(self, max_shift=5):
        """进场偏移测试：信号在哪个时间窗口兑现收益？
        对每个中波bar计算 forward_N 收益（延迟shift根K线入场）
        按信号分桶聚合，找到最佳入场偏移
        """
        df = pd.DataFrame(self.records)
        if df.empty:
            return {}
        df['returns'] = df['capital'].pct_change()

        mid_df = df[df['signal_max'].notna()].copy()
        if len(mid_df) < 100:
            return {}

        # 计算forward returns（从每根bar往后看）
        ret_arr = df['returns'].values
        results = []

        for shift in range(max_shift):
            # 对每个bar，模拟"延迟shift根后入场，持有5根"的收益
            fwd_returns = []
            signal_vals = []
            for idx in mid_df.index:
                start = df.index.get_loc(idx) + shift
                end = start + 5
                if end < len(ret_arr):
                    fwd_ret = (1 + ret_arr[start:end]).prod() - 1
                    fwd_returns.append(fwd_ret)
                    signal_vals.append(mid_df.loc[idx, 'signal_max'])

            if len(fwd_returns) < 50:
                continue

            fwd_arr = np.array(fwd_returns)
            sig_arr = np.array(signal_vals)

            # 按信号分桶
            try:
                buckets = pd.qcut(sig_arr, 10, labels=False, duplicates='drop')
            except ValueError:
                continue

            for b in sorted(set(buckets)):
                mask = buckets == b
                n = mask.sum()
                if n < 10:
                    continue
                cum = (1 + fwd_arr[mask]).prod() - 1
                avg_r = fwd_arr[mask].mean()
                std_r = fwd_arr[mask].std()
                sharpe = (avg_r / std_r) * np.sqrt(365 * 24) if std_r and std_r > 0 else 0
                winrate = (fwd_arr[mask] > 0).mean()
                results.append({
                    'shift': shift,
                    'bucket': int(b),
                    'count': n,
                    'cum_return': cum,
                    'sharpe': sharpe,
                    'winrate': winrate,
                })

        return results

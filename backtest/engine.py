import numpy as np
import pandas as pd

from config import (
    SYMBOLS, INITIAL_CAPITAL, TRAIN_WINDOW, RETRAIN_EVERY,
    FEE_RATE, SLIPPAGE_BPS, MAX_WEIGHT,
    EXIT_BUFFER_BARS, SIGNAL_CONFIRM,
    SIGNAL_CHANGE_THRESH, MIN_WEIGHT_DELTA, COOLING_BARS, REGIME_ENTRY_BARS,
    VOL_REGIME_EMA_SPAN, ENTRY_DELAY_BARS,
    SMOOTH_WEIGHT_EXEC, GLOBAL_VOL_FILTER,
    DISABLE_TRADING, SIGNAL_INVERT, SIGNAL_CONFIRM_BARS,
    USE_SIGNAL_TIER, SIGNAL_Q_CORE, SIGNAL_Q_ATTACK, SIGNAL_CHANGE_FILTER,
    SIGNAL_MUST_INCREASE, ENTRY_PRICE_CONFIRM, PRICE_BREAKOUT_BARS, MIN_HOLD_BARS_EXEC,
    BASE_EXPOSURE, SIGNAL_BOOST_MID, SIGNAL_BOOST_HIGH,
)
from portfolio.optimizer import (
    shrink_returns, shrink_covariance, compute_ewma_cov,
    optimize_weights,
)
from alpha.model import RegimeAlphaModel, get_regime


class PortfolioBacktest:
    """半连续执行：Regime过滤 + Band Trading + No-Trade Zone"""

    def __init__(self, risk_manager, logger):
        self.risk = risk_manager
        self.logger = logger
        self.cash = INITIAL_CAPITAL
        self.qty = {s: 0.0 for s in SYMBOLS}
        self.records = []
        self.trades = []
        self.total_fee = 0.0
        self.models = {s: RegimeAlphaModel() for s in SYMBOLS}
        # 执行状态
        self.current_weights = np.zeros(len(SYMBOLS))
        self.prev_signal = np.zeros(len(SYMBOLS))
        self.bars_since_trade = COOLING_BARS  # 初始允许交易
        self.exit_buffer = 0
        self.was_in_mid = False
        self.signal_confirm_counter = 0       # 延迟确认计数器
        self.signal_history = []              # 扩展窗口信号记录（用于分位数计算）
        self.prev_signal_max = 0.0            # 上根K线的signal_max（信号变化过滤）
        self.bars_since_entry = MIN_HOLD_BARS_EXEC  # 初始允许退出
        self.current_exposure_level = 'none'  # 暴露等级迟滞: 'none' | 'base' | 'mid' | 'full'
        self.regime_entry_counter = 0         # 中波进入确认计数器
        self.ema_vol_regime = 1.0             # vol_regime的EMA平滑值（防闪烁）
        self.entry_delay_counter = 0          # 升仓延迟计数器
        self.entry_delay_target = None        # 升仓延迟的目标等级
        self.entry_delay_breakout = False     # 延迟期间是否已出现价格突破

    def _get_prices(self, df_dict, time):
        return np.array([df_dict[s].loc[time, 'close'] for s in SYMBOLS])

    def _portfolio_value(self, prices):
        pos_val = sum(self.qty[s] * prices[i] for i, s in enumerate(SYMBOLS))
        return self.cash + pos_val

    def _prepare_slice(self, df, end_time):
        return df[df.index <= end_time]

    def _get_returns_matrix(self, df_dict, end_time, window=120):
        ret_data = {}
        for s in SYMBOLS:
            df = df_dict[s]
            mask = df.index <= end_time
            ret_data[s] = df[mask]['close'].pct_change().iloc[-window:]
        returns_df = pd.DataFrame(ret_data).dropna()
        return returns_df

    def _get_current_regime(self, df_dict, time):
        from alpha.features import build_features
        first_sym = SYMBOLS[0]
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

    def run(self, df_dict):
        all_times = sorted(set.union(*[set(df.index) for df in df_dict.values()]))

        for i, time in enumerate(all_times):
            if i < TRAIN_WINDOW:
                continue

            prices = self._get_prices(df_dict, time)
            total_value = self._portfolio_value(prices)

            # 风控
            self.risk.update(total_value)
            if self.risk.check() == 'liquidate':
                self._force_close(prices, time, '风控清仓')
                self.records.append({
                    'time': time, 'capital': self._portfolio_value(prices),
                    'regime': 'LIQ', 'signal_max': float('nan')
                })
                continue

            # 全局波动率过滤：用BTC vol作为市场总开关
            global_vol = self._get_btc_vol_regime(df_dict, time)
            if global_vol and global_vol > GLOBAL_VOL_FILTER:
                self._force_close(prices, time, '全局高波')
                self.records.append({
                    'time': time, 'capital': self._portfolio_value(prices),
                    'regime': 'global_high', 'signal_max': float('nan')
                })
                continue

            # 重训
            if i % RETRAIN_EVERY == 0 or i == TRAIN_WINDOW:
                for s in SYMBOLS:
                    train_df = self._prepare_slice(df_dict[s], time)
                    self.models[s].train(train_df)

            current_regime = self._get_current_regime(df_dict, time)
            in_mid = (current_regime == 'mid')
            self.bars_since_trade += 1

            # === 离开中波 → 清仓（受最小持仓约束） ===
            if not in_mid:
                self.regime_entry_counter = 0  # 重置进入确认
                self.entry_delay_counter = 0   # 重置升仓延迟
                self.entry_delay_target = None
                self.entry_delay_breakout = False
                self.bars_since_entry += 1
                if self.current_weights.sum() > 0 and self.bars_since_entry < MIN_HOLD_BARS_EXEC:
                    # 最小持仓期内，不退出（风控和全局过滤已在上方处理）
                    self.records.append({
                        'time': time, 'capital': total_value,
                        'regime': 'forced_hold', 'signal_max': float('nan')
                    })
                    continue
                if self.was_in_mid:
                    self.exit_buffer += 1
                    if self.exit_buffer >= EXIT_BUFFER_BARS:
                        self._force_close(prices, time, f'退出中波→{current_regime}')
                        self.was_in_mid = False
                        self.exit_buffer = 0
                else:
                    self._force_close(prices, time, f'{current_regime}')
                    self.exit_buffer = 0
            else:
                self.exit_buffer = 0
                self.was_in_mid = True

                # === 中波进入确认：需连续N根中波K线才允许建仓 ===
                if self.current_weights.sum() == 0 and REGIME_ENTRY_BARS > 0:
                    self.regime_entry_counter += 1
                    if self.regime_entry_counter < REGIME_ENTRY_BARS:
                        self.records.append({
                            'time': time, 'capital': total_value,
                            'regime': 'mid_entry_wait', 'signal_max': float('nan')
                        })
                        continue
                self.regime_entry_counter = REGIME_ENTRY_BARS

                # === 中波内：暴露管理 + 迟滞层 ===
                mu_raw = self._predict_all(df_dict, time)
                signal_max = float(abs(mu_raw).max())

                # 延迟确认
                if SIGNAL_CONFIRM_BARS > 0:
                    if self.signal_confirm_counter < SIGNAL_CONFIRM_BARS:
                        self.signal_confirm_counter += 1
                        self.records.append({
                            'time': time, 'capital': total_value,
                            'regime': 'mid_confirming', 'signal_max': signal_max
                        })
                        continue
                self.signal_confirm_counter = SIGNAL_CONFIRM_BARS

                # 禁止交易诊断
                if DISABLE_TRADING:
                    self.records.append({
                        'time': time, 'capital': total_value,
                        'regime': 'mid_disabled', 'signal_max': signal_max
                    })
                    self.signal_history.append(signal_max)
                    continue

                # 1. 计算目标暴露等级
                signal_increasing = signal_max > self.prev_signal_max
                self.prev_signal_max = signal_max
                breakout_ok = self._check_price_breakout(df_dict, time) if ENTRY_PRICE_CONFIRM else True

                if signal_max >= SIGNAL_BOOST_HIGH:
                    if SIGNAL_MUST_INCREASE and not signal_increasing:
                        target_exposure = 0.6
                        target_level = 'mid'
                        exp_label = 'mid_base_hi'
                    elif ENTRY_PRICE_CONFIRM and not breakout_ok:
                        target_exposure = 0.6
                        target_level = 'mid'
                        exp_label = 'mid_base_hi'
                    else:
                        target_exposure = 1.0
                        target_level = 'full'
                        exp_label = 'mid_full'
                elif signal_max >= SIGNAL_BOOST_MID:
                    target_exposure = 0.6
                    target_level = 'mid'
                    exp_label = 'mid_mid'
                else:
                    target_exposure = BASE_EXPOSURE
                    target_level = 'base'
                    exp_label = 'mid_base'

                # 2. 非对称迟滞：升仓延迟确认（等N根K线），降仓立即执行
                level_order = {'none': 0, 'base': 1, 'mid': 2, 'full': 3}
                scaling_up = level_order.get(target_level, 0) > level_order.get(self.current_exposure_level, 0)
                scaling_down = level_order.get(target_level, 0) < level_order.get(self.current_exposure_level, 0)
                level_changed = scaling_up or scaling_down
                cooled_down = self.bars_since_trade >= COOLING_BARS
                first_entry = (self.current_weights.sum() == 0)

                should_execute = False
                execute_level = target_level
                execute_exposure = target_exposure

                # ── 延迟确认状态机 ──
                in_delay = (self.entry_delay_target is not None)

                if in_delay:
                    delay_target = self.entry_delay_target
                    # 条件恶化（当前目标 < 延迟目标）→ 取消升仓
                    if level_order.get(target_level, 0) < level_order.get(delay_target, 0):
                        self.entry_delay_target = None
                        self.entry_delay_counter = 0
                        self.entry_delay_breakout = False
                        regime_tag = 'mid_hold'
                    else:
                        # 验证确认条件：价格突破只需在延迟期间至少出现一次
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
                                execute_exposure = {'base': BASE_EXPOSURE, 'mid': 0.6, 'full': 1.0}[delay_target]
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
                        # 升仓/首入 → 启动延迟确认
                        self.entry_delay_target = target_level
                        self.entry_delay_counter = 0
                        self.entry_delay_breakout = False
                        regime_tag = 'mid_entry_delay'
                    else:
                        # 降仓 或 无延迟模式 → 立即执行
                        should_execute = True

                else:
                    regime_tag = 'mid_hold'

                if should_execute:
                    # 优化 + 平滑 + 缩放
                    ret_df = self._get_returns_matrix(df_dict, time)
                    cov = compute_ewma_cov(ret_df) if len(ret_df) > 30 else np.eye(len(SYMBOLS))
                    mu = shrink_returns(mu_raw)
                    cov_reg = shrink_covariance(cov)
                    raw_weights = optimize_weights(mu, cov_reg)
                    smoothed_w = SMOOTH_WEIGHT_EXEC * self.current_weights + (1 - SMOOTH_WEIGHT_EXEC) * raw_weights
                    smoothed_w *= execute_exposure

                    self._rebalance_to(smoothed_w, prices, total_value, time)
                    self.current_weights = smoothed_w.copy()
                    self.prev_signal = mu_raw.copy()
                    self.current_exposure_level = execute_level
                    self.bars_since_trade = 0
                    self.bars_since_entry = 0
                    regime_tag = exp_label if should_execute else regime_tag

                self.signal_history.append(signal_max)

            self.records.append({
                'time': time,
                'capital': self._portfolio_value(prices),
                'regime': regime_tag if in_mid else current_regime,
                'signal_max': signal_max if in_mid else float('nan'),
            })

        return self._summary()

    def _predict_all(self, df_dict, time):
        mu_raw = []
        for s in SYMBOLS:
            data = self._prepare_slice(df_dict[s], time)
            mu = self.models[s].predict(data)
            mu_raw.append(mu[0] if len(mu) > 0 else 0)
        result = np.array(mu_raw)
        if SIGNAL_INVERT:
            result = -result
        return result

    def _rebalance_to(self, weights, prices, total_value, time):
        fee_this_step = 0
        for j, s in enumerate(SYMBOLS):
            current_val = self.qty[s] * prices[j]
            target_val = total_value * max(weights[j], 0)
            delta_val = target_val - current_val
            if abs(delta_val) < 1e-6:
                continue
            fee = abs(delta_val) * (FEE_RATE + SLIPPAGE_BPS)
            fee_this_step += fee
            self.qty[s] += delta_val / prices[j]
            self.cash -= delta_val + fee
            self.trades.append({
                'time': time, 'symbol': s,
                'delta_value': round(delta_val, 2),
                'price': round(prices[j], 2),
                'fee': round(fee, 4),
                'weight': round(max(weights[j], 0), 4),
            })
        self.total_fee += fee_this_step

    def _force_close(self, prices, time, reason=''):
        for j, s in enumerate(SYMBOLS):
            if self.qty[s] > 0:
                val = self.qty[s] * prices[j]
                fee = val * (FEE_RATE + SLIPPAGE_BPS)
                self.cash += val - fee
                self.total_fee += fee
                self.trades.append({
                    'time': time, 'symbol': s,
                    'delta_value': round(-val, 2),
                    'price': round(prices[j], 2),
                    'fee': round(fee, 4),
                    'weight': 0,
                })
                self.qty[s] = 0
        self.current_weights = np.zeros(len(SYMBOLS))
        self.prev_signal = np.zeros(len(SYMBOLS))
        self.current_exposure_level = 'none'
        self.regime_entry_counter = 0
        self.entry_delay_counter = 0
        self.entry_delay_target = None
        self.entry_delay_breakout = False

    def _summary(self):
        df = pd.DataFrame(self.records)
        if df.empty:
            return {'error': 'no records'}
        df['returns'] = df['capital'].pct_change()
        final_capital = df['capital'].iloc[-1]
        total_return = (final_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL
        peak = df['capital'].expanding().max()
        drawdowns = (df['capital'] - peak) / peak
        max_dd = drawdowns.min()
        avg_ret = df['returns'].mean()
        std_ret = df['returns'].std()
        sharpe = (avg_ret / std_ret) * np.sqrt(365 * 24) if std_ret and std_ret > 0 else 0

        return {
            'initial_capital': INITIAL_CAPITAL,
            'final_capital': round(final_capital, 2),
            'total_return': f'{total_return:.2%}',
            'max_drawdown': f'{max_dd:.2%}',
            'sharpe_ratio': round(sharpe, 2),
            'total_bars': len(df),
            'trade_events': len(self.trades),
            'total_fee': round(self.total_fee, 2),
        }

    def get_trades_df(self):
        return pd.DataFrame(self.trades)

    def get_coefficients(self):
        result = {}
        for s in SYMBOLS:
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
            stats(df[df['regime'] == 'mid_base'], '中波-基仓(30%)'),
            stats(df[df['regime'] == 'mid_base_hi'], '中波-降级(强信号未确认)'),
            stats(df[df['regime'] == 'mid_hold'], '中波-持有'),
            stats(df[df['regime'] == 'mid_entry_wait'], '中波-等待确认(防闪烁)'),
            stats(df[df['regime'] == 'mid_entry_delay'], '中波-升仓延迟(入场时机)'),
            stats(df[df['regime'] == 'forced_hold'], '强制持有(最小持仓)'),
            stats(df[df['regime'] == 'high'], '高波(空仓)'),
            stats(df[df['regime'] == 'low'], '低波(空仓)'),
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

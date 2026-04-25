import numpy as np
import pandas as pd

from config import SHRINK_MU, SHRINK_COV, MAX_WEIGHT


def shrink_returns(mu_raw):
    """收益收缩：mu = shrink * mu_raw，防极端预测"""
    return SHRINK_MU * np.array(mu_raw)


def shrink_covariance(cov):
    """协方差收缩：向单位阵收缩，提高稳定性"""
    n = len(cov)
    return (1 - SHRINK_COV) * cov + SHRINK_COV * np.eye(n)


def compute_ewma_cov(returns, span=60):
    """计算EWMA协方差矩阵"""
    if len(returns) < span:
        return np.cov(returns, rowvar=False)
    ewm_cov = returns.ewm(span=span).cov()
    # 取最后一组
    last_idx = ewm_cov.index.get_level_values(0)[-1]
    cov = ewm_cov.loc[last_idx].values
    return cov


def optimize_weights(mu, cov):
    """解析解：w ∝ Σ⁻¹μ，只做多单资产上限"""
    n = len(mu)
    cov_reg = cov + np.eye(n) * 1e-6
    try:
        inv_cov = np.linalg.pinv(cov_reg)
    except np.linalg.LinAlgError:
        inv_cov = np.eye(n)

    w = inv_cov @ mu
    w = np.clip(w, 0, MAX_WEIGHT)

    total = w.sum()
    if total > 1e-10:
        w = w / total
    else:
        w = np.ones(n) / n

    return w


def smooth_weights(old_w, new_w, alpha=0.9):
    """权重平滑：减少换手"""
    if old_w is None:
        return new_w
    return alpha * np.array(old_w) + (1 - alpha) * new_w

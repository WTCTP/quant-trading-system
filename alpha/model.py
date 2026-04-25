import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from config import FORWARD_BARS, REGIME_LOW_THRESH, REGIME_HIGH_THRESH


def get_regime(vol_ratio_val):
    """根据波动率比值返回状态标签"""
    if vol_ratio_val is None or np.isnan(vol_ratio_val):
        return 'mid'
    if vol_ratio_val < REGIME_LOW_THRESH:
        return 'low'
    if vol_ratio_val > REGIME_HIGH_THRESH:
        return 'high'
    return 'mid'


class AlphaModel:
    """单一逻辑回归模型"""

    def __init__(self):
        self.model = LogisticRegression(C=1.0, solver='liblinear', max_iter=500)
        self.feature_names = None
        self.fitted = False

    def prepare_data(self, df):
        from alpha.features import build_features, build_label
        X = build_features(df)
        y = build_label(df, FORWARD_BARS)
        mask = ~(X.isna().any(axis=1) | y.isna())
        return X[mask], y[mask]

    def train(self, df):
        X, y = self.prepare_data(df)
        if len(X) < 200:
            return False
        self.feature_names = X.columns.tolist()
        self.model.fit(X, y)
        self.fitted = True
        return True

    def predict(self, df):
        if not self.fitted:
            return np.zeros(1)
        from alpha.features import build_features
        X = build_features(df)
        X_latest = X.iloc[[-1]]
        if X_latest.isna().any(axis=1).iloc[0]:
            return np.zeros(1)
        proba = self.model.predict_proba(X_latest)[:, 1]
        return proba - 0.5

    def get_coefficients(self):
        if not self.fitted:
            return {}
        return dict(zip(self.feature_names, self.model.coef_[0]))


class RegimeAlphaModel:
    """Regime Switching：每个币种按波动率状态持有多个子模型"""

    def __init__(self):
        self.models = {
            'low': AlphaModel(),
            'mid': AlphaModel(),
        }

    def train(self, df):
        """按regime拆分数据，分别训练低波动和中等波动模型"""
        from alpha.features import build_features, build_label

        X_all, y_all = self._prepare(df)
        if len(X_all) < 200:
            return False

        # 用vol_regime列分组
        regime_col = X_all['vol_regime']

        for regime_name, model in self.models.items():
            if regime_name == 'low':
                mask = regime_col < REGIME_LOW_THRESH
            else:  # mid
                mask = (regime_col >= REGIME_LOW_THRESH) & (regime_col <= REGIME_HIGH_THRESH)

            X_r = X_all[mask].drop(columns=['vol_regime'])
            y_r = y_all[mask]

            if len(X_r) >= 100:
                model.feature_names = X_r.columns.tolist()
                model.model.fit(X_r, y_r)
                model.fitted = True

        return True

    def predict(self, df):
        """根据当前波动率状态选择模型预测"""
        from alpha.features import build_features
        X = build_features(df)
        X_latest = X.iloc[[-1]]
        if X_latest.isna().any(axis=1).iloc[0]:
            return np.zeros(1)

        vol_r = X_latest['vol_regime'].iloc[0]
        regime = get_regime(vol_r)

        # 高波动不交易
        if regime == 'high':
            return np.zeros(1)

        model = self.models[regime]
        if not model.fitted:
            return np.zeros(1)

        # 预测时去掉vol_regime列（模型训练时已去掉）
        X_pred = X_latest.drop(columns=['vol_regime'])
        proba = model.model.predict_proba(X_pred)[:, 1]
        return proba - 0.5

    def get_coefficients(self):
        result = {}
        for regime, model in self.models.items():
            coef = model.get_coefficients()
            if coef:
                result[f'{regime}_vol'] = sorted(coef.items(), key=lambda x: abs(x[1]), reverse=True)
        return result

    def _prepare(self, df):
        from alpha.features import build_features, build_label
        X = build_features(df)
        y = build_label(df, FORWARD_BARS)
        mask = ~(X.isna().any(axis=1) | y.isna())
        return X[mask], y[mask]

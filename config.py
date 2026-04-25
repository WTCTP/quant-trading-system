# 交易对
SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'LINK/USDT', 'AVAX/USDT', 'ADA/USDT', 'DOGE/USDT']

# K线周期
TIMEFRAME = '1h'

# 初始资金 (USDT)
INITIAL_CAPITAL = 10000

# ---- Alpha 模型参数 ----
# 标签：预测未来N根K线的方向
FORWARD_BARS = 5

# walk-forward 训练窗口 (K线数)
TRAIN_WINDOW = 5000   # 约7个月

# 重新训练频率 (K线数)
RETRAIN_EVERY = 500   # 约20天

# ---- 因子参数（特征计算用）----
EMA_SHORT = 20
EMA_LONG = 50
ZSCORE_PERIOD = 20
MOMENTUM_PERIOD = 10
VOL_PERIOD = 20

# ---- 组合优化参数 ----
# 收益收缩强度
SHRINK_MU = 0.7

# 协方差收缩强度 (向单位阵收缩)
SHRINK_COV = 0.3

# 权重平滑系数
SMOOTH_WEIGHT = 0.9

# 单资产最大权重
MAX_WEIGHT = 0.30

# 最小调仓阈值（权重变化超过此值才执行）
MIN_WEIGHT_CHANGE = 0.05

# 交易频率限制（小时）
MIN_TRADE_INTERVAL_HOURS = 4

# ---- 交易成本 ----
FEE_RATE = 0.0004
SLIPPAGE_BPS = 0.0001

# ---- Regime Switching ----
# 波动率状态划分阈值（vol_regime = vol_20 / vol_100，集中在1.0附近）
REGIME_LOW_THRESH = 0.90
REGIME_HIGH_THRESH = 1.10

# 状态驱动持仓参数
MIN_HOLD_BARS = 6
EXIT_BUFFER_BARS = 6            # 退出中波缓冲
REGIME_ENTRY_BARS = 2           # 进入中波需持续确认
VOL_REGIME_EMA_SPAN = 12        # vol_regime的EMA平滑窗口（核心防闪烁）
SIGNAL_CONFIRM = 0.1

# 半连续执行参数（Band Trading）
SIGNAL_CHANGE_THRESH = 0.1     # 信号变化超过此值才考虑调仓
MIN_WEIGHT_DELTA = 0.05        # 权重变化超过此值才执行
COOLING_BARS = 3               # 调仓冷却K线数（等级变化后冷却）
SMOOTH_WEIGHT_EXEC = 0.8       # 执行层权重平滑
NO_TRADE_ZONE = 0.15           # 信号绝对值<此值不开仓

# 全局波动率过滤（用BTC波动作为市场开关）
GLOBAL_VOL_FILTER = 1.3        # BTC vol_regime > 此值 → 全部空仓

# ---- 风控 ----
MAX_DRAWDOWN = 0.20

# ---- Alpha 诊断开关 ----
DISABLE_TRADING = False        # 实验1: True → 永远不交易，验证alpha是否来自过滤
SIGNAL_INVERT = False          # 实验2: True → 信号取反，验证方向是否反了
SIGNAL_CONFIRM_BARS = 0        # 实验3: 延迟确认K线数，0=不延迟 (建议2-3)

# ---- 分层仓位 (signal tiering) ----
USE_SIGNAL_TIER = False        # True → 按信号分位数分层下注
SIGNAL_Q_CORE = 0.80           # Q8核心仓位 (percentile ≥ 此值 → weight 1.0)
SIGNAL_Q_ATTACK = 0.90         # Q9进攻仓位 (percentile ≥ 此值 → weight 0.8)
SIGNAL_CHANGE_FILTER = 0.02    # 信号变化过滤: |signal_t - signal_t-1| > 此值才允许交易

# ---- 执行层优化 ----
SIGNAL_MUST_INCREASE = True    # 信号必须增强(signal > prev)才交易，过滤静态强信号
ENTRY_PRICE_CONFIRM = True     # 价格确认: BTC收盘价突破近期高点才入场
ENTRY_DELAY_BARS = 2           # 升仓延迟确认（向上调仓需等N根K线确认）
PRICE_BREAKOUT_BARS = 6        # 突破确认的回看K线数
MIN_HOLD_BARS_EXEC = 6         # 最小持仓K线数，防止1h反复进出

# ---- 暴露管理 (exposure-based) ----
BASE_EXPOSURE = 0.30           # 中波内基础仓位（始终持有）
SIGNAL_BOOST_MID = 0.10       # 中等信号阈值 → 暴露×0.6
SIGNAL_BOOST_HIGH = 0.15      # 强信号阈值 → 暴露×1.0

# ---- 数据拉取 ----
DATA_DIR = 'data/cache'
START_DATE = '2024-01-01'

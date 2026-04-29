# 交易对
SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'LINK/USDT', 'AVAX/USDT', 'ADA/USDT', 'DOGE/USDT']

# K线周期
TIMEFRAME = '1h'

# 初始资金 (USDT)
INITIAL_CAPITAL = 10000

# ============================================================
# 生产配置（冻结区 — 不再调参/改信号/加策略）
# ============================================================
ALPHA_MODE = 'breakout_only'         # 只跑 Breakout Alpha（实验验证最优）
PORTFOLIO_VOL_TARGET = 0.05          # 组合目标年化波动率 5%（生产档）
BASE_MID = 1.0                       # 中波底仓乘数
BASE_LOW = 0.0                       # 低波不参与
BASE_HIGH = 0.0                      # 高波空仓
NO_TRADE_ZONE = 0.02                 # |signal| < 此值且空仓 → 不开仓（信号≈0.5±0.03，0.02过滤纯噪声）
USE_FUNDING = False                  # 默认关闭（实验验证负贡献）
USE_CROSS_SECTIONAL = False          # 默认关闭（实验验证负贡献）

# — 杠杆控制（组合层，不调信号层）—
PORTFOLIO_LEVERAGE = 1.0             # 1.0x基准，扫描 1.5x/2.0x

# — 执行层状态机 —
MIN_WEIGHT_DELTA = 0.05              # trade buffer: 权重变化<5%不执行
MAKER_TIMEOUT_BARS = 2               # maker挂单超时K线数→转taker
MAX_RETRIES = 3                      # 单订单最大重试次数
MAKER_SLIP_BPS = 0.00002             # maker成交滑点 0.2bps
TAKER_SLIP_EXTRA_BPS = 0.0005        # taker额外滑点 5bps（在基础滑点之上）

# — 真实成本模型 —
FEE_RATE = 0.0004                    # 手续费 4bps
SLIPPAGE_K = 1.0                     # 滑点系数: slippage = k * |delta| / ADV
SLIPPAGE_MIN_BPS = 0.000005          # 最小滑点 0.05bps（大盘币）
SLIPPAGE_MAX_BPS = 0.0010            # 最大滑点 10bps（小盘币熔断）

# — 执行约束（容量上限）—
MAX_POSITION_ADV_PCT = 0.05          # 单币持仓不超过日成交量 5%
MAX_ORDER_ADV_PCT = 0.01             # 单次下单不超过日成交量 1%

# — 资金费率（计入成本，不做alpha）—
FUNDING_COST_ENABLED = True          # 持仓期间扣减funding rate

# — 风控（资金级）—
RISK_PORTFOLIO_STOP = -0.10          # 组合回撤 < -10% → 杠杆减半
RISK_VOL_SHOCK = 5.0                 # realized_vol > target 5x → 清仓 (2x太紧，crypto天然波动大)
RISK_CONSECUTIVE_LOSS = 5            # 连续亏损N笔 → 暂停交易
RISK_PAUSE_BARS = 48                 # 暂停K线数（48h）

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
COOLING_BARS = 3               # 调仓冷却K线数（等级变化后冷却）
SMOOTH_WEIGHT_EXEC = 0.8       # 执行层权重平滑
# NO_TRADE_ZONE 已在生产配置区定义 (0.02)，此处不再覆盖

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

# ---- 暴露管理 (exposure-based) [兼容旧版] ----
BASE_EXPOSURE = 0.30           # 中波内基础仓位
SIGNAL_BOOST_MID = 0.10       # 中等信号阈值 → 暴露×0.6
SIGNAL_BOOST_HIGH = 0.15      # 强信号阈值 → 暴露×1.0

# ---- Regime分层底仓 (v2.1) ----
BASE_LOW = 0.0                # 低波不参与 (风险预算实验结论)
BASE_MID = 1.0                # 中波满仓 (风险放大)
BASE_HIGH = 0.0               # 高波继续空仓

# ---- Regime-Specific Vol Targeting (v2.1) ----
TARGET_VOL_LOW = 0.03         # 低波目标波动率 (deprecated, 用 PORTFOLIO_VOL_TARGET)
TARGET_VOL_MID = 0.08         # 中波目标波动率 (deprecated, 用 PORTFOLIO_VOL_TARGET)
TARGET_VOL_HIGH = 0.00        # 高波不动

# ---- Portfolio Layer (deprecated: 生产配置已迁移到文件顶部冻结区) ----
# PORTFOLIO_VOL_TARGET 已在顶部定义

# ---- Vol Targeting 通用参数 ----
TARGET_VOL = 0.05             # 默认目标年化波动率
VOL_LOOKBACK = 24             # 回看K线数 (1h → 24h)
VOL_SCALE_CAP = 2.0           # 仓位放大上限
VOL_FLOOR = 0.02              # 已实现波动率下限 (防止无仓位时 vol→0 导致杠杆→∞)

# ---- 数据拉取 ----
DATA_DIR = 'data/cache'
START_DATE = '2024-01-01'

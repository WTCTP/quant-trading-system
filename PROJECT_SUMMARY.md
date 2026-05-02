# 量化交易系统 — 项目全流程总结

> 最后更新：2026-04-30
> 当前版本：v3.2 — V1 执行状态机 + 纸盘交易 + Streamlit 可视化仪表盘

---

## 一、项目现状

### 1.1 当前架构

```
lianghua/
├── config.py                  # 全局配置（~130个参数，顶部冻结区为生产配置）
├── alpha/
│   ├── __init__.py
│   ├── features.py            # 7个市场微观结构特征 + 标签构建
│   ├── model.py               # RegimeAlphaModel（按波动率分low/mid子模型）
│   ├── cross_sectional.py     # 横截面动量轮动 Alpha（默认关闭）
│   ├── funding_alpha.py       # 资金费率拥挤 Alpha（默认关闭）
│   └── pullback.py            # 回调入场 Alpha（已弃用）
├── portfolio/
│   ├── __init__.py
│   ├── optimizer.py           # 收益/协方差收缩 + EWMA协方差 + Σ⁻¹μ优化
│   └── executor.py            # PortfolioExecutor V1: 订单状态机 + Maker/Taker路由
├── backtest/
│   ├── __init__.py
│   └── engine.py              # 回测引擎（~580行，process_bar()可复用）
├── live/                      # 纸盘交易模块
│   ├── __init__.py
│   ├── data_feed.py           # 实时K线数据源（ccxt轮询 + bar闭合检测）
│   └── paper_engine.py        # 纸盘引擎（继承PortfolioBacktest，含自动保存）
├── dashboard.py               # ★ 新增：Streamlit 可视化仪表盘（6标签页）
├── data/
│   ├── __init__.py
│   ├── fetcher.py             # CCXT获取OHLCV + CSV缓存
│   └── funding_fetcher.py     # 永续合约资金费率获取 + K线时间对齐
├── risk/
│   ├── __init__.py
│   └── manager.py             # 回撤监控 + 20%清仓线
├── execution/
│   ├── __init__.py
│   └── executor.py            # 旧版执行器（已废弃，由portfolio/executor.py替代）
├── log/
│   ├── __init__.py
│   └── recorder.py            # 交易日志CSV
├── main_backtest.py           # 回测入口（多实验对比）
├── main_paper.py              # ★ 新增：纸盘交易入口
├── dual_alpha_test.py         # 双Alpha实验
├── funding_alpha_test.py      # Funding Alpha独立验证
├── xs_alpha_test.py           # Cross-Sectional Alpha验证
├── risk_budget_test.py        # 风险预算实验
├── low_vol_benchmark.py       # 低波市场结构分析
├── portfolio_vol_sweep.py     # 组合目标波动率参数扫描
└── PROJECT_SUMMARY.md         # 本文档
```

### 1.2 当前性能指标（v3.1 生产配置：2026-04-30）

**PORTFOLIO_VOL_TARGET=0.10, PORTFOLIO_LEVERAGE=1.0x, Breakout-Only**:

| 指标 | 值 |
|------|-----|
| 总收益 | **+25.35%** |
| 年化波动 | 8.23% |
| Sharpe | **1.61** |
| Calmar | **4.81** |
| 最大回撤 | -5.27% |
| 交易次数 | **437** |
| 总手续费 | $314.90 |
| 总滑点 | $252.28 |
| 胜率 | 8.7% |
| 盈亏比 | 1.15 |

**Regime 分解**:
| Regime | K线数 | 累计收益 | Sharpe |
|--------|-------|---------|--------|
| 低波 | 7,118 | +8.71% | 1.51 |
| 中波 | 3,681 | +11.63% | 2.35 |
| 高波 | 1,951 | +0.89% | 0.74 |

**多Alpha对比**:
| 实验 | 收益 | Sharpe | Calmar | 交易 |
|------|------|--------|--------|------|
| Breakout-Only | **25.35%** | **1.61** | **4.81** | 437 |
| Breakout+Funding(10%) | 0.15% | 0.04 | 0.02 | 372 |
| Breakout+CrossSec(10%) | 22.36% | 1.58 | 4.0 | 418 |

> **结论：Breakout-Only 全维度最优。** V1 执行状态机将交易从 933 降至 437（-53%），同时收益和 Sharpe 均提升。

### 1.3 杠杆扫描结果

| 杠杆 | 收益 | DD | Sharpe | Calmar | 交易 |
|------|------|-----|--------|--------|------|
| 1.0x | 11.05% | -4.17% | 1.15 | 2.65 | 470 |
| 1.5x | 2.63% | -7.51% | 0.32 | 0.35 | 206 |
| **2.0x** | **18.96%** | **-3.63%** | **1.82** | **5.22** | 200 |
| 2.5x | -8.46% | -13.67% | -0.93 | -0.62 | 210 |
| 3.0x | -5.01% | -11.70% | -0.38 | -0.43 | 185 |

> **建议：Lev=2.0x 综合最优**（Sharpe 1.82, Calmar 5.22），但需配合实时风控。

---

## 二、架构演进全记录

### 阶段 0：原始设计（已废弃）
- **架构**：三因子加权（EMA趋势 + Z-score均值回归 + 动量）→ 信号 → 阈值决策
- **问题**：因子是手工设计的，没有统计学习

### 阶段 1：ML分类 + 组合优化（v1）
- **架构**：Logistic Regression → 预测方向概率 → Σ⁻¹μ 组合优化
- **Alpha**：7个行为特征 + 未来5bar二分类标签
- **诊断工具**：DISABLE_TRADING / SIGNAL_INVERT / SIGNAL_CONFIRM_BARS

### 阶段 2：Regime Switching（v1.5）
- **架构**：按波动率状态拆分为 low/mid 子模型
- **Regime**：vol_20/vol_100 → low(<0.90) / mid(0.90~1.10) / high(>1.10)

### 阶段 3：信号分层 + 诊断深化（v1.6）
- **关键发现**：alpha 集中在顶部 20%，Q8 Sharpe=10.66
- **问题**：NO_TRADE_ZONE 把样本空间砍没（0.7%通过）

### 阶段 4：暴露管理范式（v2.0）
- **范式转换**：从"是否交易" → "持多少仓"
- **问题**：3406笔交易，信号波动导致调仓爆炸

### 阶段 5：迟滞层（v2.1）
- **暴露等级迟滞** + **Regime EMA 平滑** + **非对称执行**
- **结果**：+5.14%, Sharpe 2.00, MaxDD -0.82%, 896笔交易

### 阶段 6：多Alpha融合 + 组合层风险预算（v3.0）
- Funding + CrossSectional 独立信息源融合
- 组合层 Vol Targeting
- **实验结论**：Breakout-Only 全维度最优

### 阶段 7：V1 执行状态机 + 纸盘交易（v3.1）← 当前版本

#### 7.1 PortfolioExecutor V1 执行状态机

将执行层从简单的一次性市价成交升级为完整订单状态机：

**OrderTask 数据结构**：
```
pending → working → filled (maker 成交)
                 → filled (taker 成交，maker超时后)
                 → cancelled（目标变化过大，重建订单）
```

**核心机制**：
- **Trade Buffer (5%)**：权重变化 < MIN_WEIGHT_DELTA 不执行，过滤噪声调仓
- **Maker 优先**：限价单在 close±0.1% 挂单，检查 OHLC 是否穿透
- **Timeout → Taker**：maker 挂单 2 根 K 线未成交 → 转 taker 市价（+5bps 额外滑点）
- **重试逻辑**：最多 3 次 maker 尝试，超时后强制 taker
- **卖出优先**：卖出单排在买入前执行，确保现金充足
- **跨 bar 订单追踪**：`pending_orders` 字典追踪未成交订单

**效果**：交易 933 → 437（-53%），成本大幅降低，收益不降反升。

#### 7.2 引擎重构：process_bar() 提取

从 `backtest/engine.py` 的 `run()` 方法中提取 `process_bar(i, time, df_dict)` 方法，使逐 bar 处理逻辑可被回测和纸盘共用。

#### 7.3 纸盘交易系统

新建 `live/` 模块，支持实时行情 + 模拟成交：

| 组件 | 功能 |
|------|------|
| `LiveDataFeed` | 从 cache 加载历史 + ccxt 轮询 Binance 最新 1h bar + bar 闭合检测 |
| `PaperTradingEngine` | 继承 PortfolioBacktest，`initialize()` 训练模型，`run_live()` 实时循环 |
| `main_paper.py` | 入口，Ctrl+C 安全退出并保存交易记录 |

---

## 三、关键问题与解决方案

### 问题1-8 见阶段 0-6 记录

### 问题9：交易过多（933笔），成本侵蚀利润（v3.1）
- **根因**：每根 K 线的信号波动触发微小调仓，无 trade buffer 过滤
- **解决**：V1 执行状态机 — 5% 权重变化阈值 + maker 优先 + timeout→taker
- **效果**：交易 933 → 437，收益 11% → 25%，Sharpe 1.15 → 1.61

### 问题10：RISK_VOL_SHOCK 过于敏感
- **根因**：2.0x * 5% target = 10% 熔断阈值，加密市场常规波动就触发
- **解决**：提升到 5.0x（25% 实际波动才熔断）
- **效果**：风险事件从 12,929 → 8,064

### 问题11：NO_TRADE_ZONE 重复定义
- **根因**：config.py 中两处定义（0.02 和 0.15），后者覆盖前者
- **解决**：删除重复定义，保留生产配置区 0.02

---

## 四、核心设计原则

1. **路径型 alpha，不要点触发** — 中波期间的持续暴露才是利润来源
2. **迟滞层必须存在** — 没有迟滞的定量系统是噪声放大器
3. **非对称执行** — 升仓慢（确认后进），降仓快（风控优先）
4. **价格确认要累积，不要逐根** — "至少一次" > "每根都要"
5. **先诊断，后治疗** — 禁用交易/信号反转/偏移测试等诊断工具价值巨大
6. **不要过早优化信号** — 同样的信号，好的执行能让收益翻倍
7. **简单优于复杂** — 三个 Alpha 融合不如单个 Breakout
8. **Maker 优先，Taker 兜底** — 0.2bps vs 5bps 滑点差异决定能否赚钱
9. **Trade Buffer 是必需品** — 5% 阈值过滤掉的都是噪声交易
10. **回测和实盘共用同一执行器** — process_bar() 提取使回测和纸盘零差异

---

## 五、当前执行层状态机

### 5.1 订单状态机（PortfolioExecutor V1）

```
信号 → 权重优化 → prepare_orders()
                        │
                计算 delta = target - current
                        │
                |delta| < 5%? → 跳过（trade buffer）
                        │
                创建 OrderTask (state=pending)
                        │
                卖出优先排序 → process_orders()
                        │
                ┌───────┴───────┐
                ↓               ↓
           _process_order()   跨bar追踪
                │            (pending_orders)
        pending → working
                    │
            _try_maker_fill()
            limit = close × 0.999 (buy)
                    × 1.001 (sell)
                    │
            OHLC 穿过限价？
          ┌─────┴─────┐
          ↓            ↓
        成交         超时? (2 bars)
     (maker,         ┌──┴──┐
     0.2bps)         ↓     ↓
                  重试<3  重试≥3
                    │     │
                 pending  taker
                          (5bps+)
```

### 5.2 风控触发器

```
风控规则 (check_risk_rules):
│
├─ 组合止损 (DD < -10%) ──→ 杠杆减半（不清仓）
│
├─ 波动率熔断 (realized_vol > 5x target) ──→ 清仓 + 暂停 48 bars
│
└─ 连续亏损 (≥5笔) ──→ 清仓 + 暂停 48 bars
```

---

## 六、调仓执行流程（中波）

```
1. Alpha预测 → mu_raw (n个币种的预测信号)
2. 信号分层 → target_level ∈ {base, mid, full}
3. 迟滞检查 → 等级变化? + 冷却期?
4. 升仓延迟 → ENTRY_DELAY_BARS(2) + 价格确认(累积)
5. 多Alpha融合:
   raw_w = optimize_weights(mu, cov)           # Breakout
   + Funding + CrossSectional (如启用)
6. 权重平滑 → smoothed = 0.8*old + 0.2*raw
7. Vol缩放 → final_w = smoothed * exposure * vol_scale * leverage
8. V1 执行:
   prepare_orders() → trade buffer过滤 → 卖出优先
   process_orders() → maker优先 → timeout→taker
```

---

## 七、纸盘交易系统

### 7.1 数据流

```
交易所 (Binance) ──ccxt fetch_ohlcv()──→ LiveDataFeed
                                              │
                                    检测 bar 闭合 (30s轮询)
                                              │
                                    追加到 df_dict
                                              │
                                    PaperTradingEngine
                                         process_bar()
                                              │
                                    PortfolioExecutor (模拟成交)
                                              │
                                    ├── 控制台输出（实时状态）
                                    ├── paper_records.csv（权益曲线，每bar保存）
                                    ├── paper_state.csv（当前状态快照）
                                    └── trades_paper.csv（交易记录，退出时保存）
                                              │
                                    Streamlit Dashboard（第6标签页读取）
```

### 7.2 启动方式

```bash
# 终端1：纸盘交易引擎
python main_paper.py

# 终端2：可视化仪表盘
streamlit run dashboard.py
```

流程：加载 cache → 训练模型 → 等待新 bar → 处理 → 输出状态 → 自动保存 CSV。
Ctrl+C 安全退出，交易记录保存到 `trades_paper.csv`。

### 7.3 实时监控（Dashboard 集成）

纸盘引擎每处理完一根 bar 自动保存：
- **paper_records.csv** — 完整权益曲线（time, capital, regime, signal_max）
- **paper_state.csv** — 当前状态快照（资金/Regime/信号/持仓/订单数）

仪表盘第 6 标签页 "📡 纸盘交易" 读取上述文件，提供：
- **状态面板**：当前资金、Regime、信号强度、持仓数、待成交订单
- **对比曲线**：纸盘权益叠加回测基准 + 偏离子图
- **交易标记**：maker/taker/force 分类标记在权益曲线上
- **统计指标**：纸盘收益/波动/Sharpe/交易笔数
- **交易明细**：最近 30 笔交易详情
- **手动刷新**：点击按钮获取最新数据

---

## 八、参数速查表

### 生产配置（config.py 顶部冻结区）

| 分类 | 参数 | 当前值 | 作用 |
|------|------|--------|------|
| Alpha | ALPHA_MODE | 'breakout_only' | 只跑 Breakout Alpha |
| 波动率 | PORTFOLIO_VOL_TARGET | 0.05 | 组合目标年化波动率 5% |
| 杠杆 | PORTFOLIO_LEVERAGE | 1.0 | 组合杠杆倍数 |
| 底仓 | BASE_MID | 1.0 | 中波满仓 |
| 底仓 | BASE_LOW / BASE_HIGH | 0.0 | 低波/高波不参与 |
| 过滤 | NO_TRADE_ZONE | 0.02 | 信号<此值且空仓→不开仓 |
| Alpha | USE_FUNDING | False | 默认关闭 |
| Alpha | USE_CROSS_SECTIONAL | False | 默认关闭 |

### 执行层状态机（V1）

| 分类 | 参数 | 当前值 | 作用 |
|------|------|--------|------|
| 缓冲 | MIN_WEIGHT_DELTA | 0.05 | trade buffer: 权重变化<5%不执行 |
| Maker | MAKER_TIMEOUT_BARS | 2 | maker挂单超时K线数 |
| Maker | MAKER_SLIP_BPS | 0.00002 | maker成交滑点 0.2bps |
| Taker | TAKER_SLIP_EXTRA_BPS | 0.0005 | taker额外滑点 5bps |
| 重试 | MAX_RETRIES | 3 | 单订单最大重试次数 |

### 成本模型

| 分类 | 参数 | 当前值 | 作用 |
|------|------|--------|------|
| 手续费 | FEE_RATE | 0.0004 | 双边手续费 |
| 滑点 | SLIPPAGE_K | 1.0 | 滑点系数 = k * |delta| / ADV |
| 滑点 | SLIPPAGE_MIN_BPS | 0.000005 | 最小滑点 0.05bps |
| 滑点 | SLIPPAGE_MAX_BPS | 0.0010 | 最大滑点 10bps |
| 容量 | MAX_POSITION_ADV_PCT | 0.05 | 单币持仓 ≤ 5% ADV |
| 容量 | MAX_ORDER_ADV_PCT | 0.01 | 单次下单 ≤ 1% ADV |
| 资金费 | FUNDING_COST_ENABLED | True | 持仓扣减funding rate |

### 风控（资金级）

| 分类 | 参数 | 当前值 | 作用 |
|------|------|--------|------|
| 止损 | RISK_PORTFOLIO_STOP | -0.10 | DD < -10% → 杠杆减半 |
| 熔断 | RISK_VOL_SHOCK | 5.0 | realized_vol > 5x target → 清仓暂停 |
| 连续亏损 | RISK_CONSECUTIVE_LOSS | 5 | 连亏N笔 → 暂停 |
| 暂停 | RISK_PAUSE_BARS | 48 | 暂停48h |

### Alpha 模型

| 分类 | 参数 | 当前值 | 作用 |
|------|------|--------|------|
| 标签 | FORWARD_BARS | 5 | 预测未来N根K线 |
| 训练 | TRAIN_WINDOW | 5000 | 训练窗口 (~7个月) |
| 训练 | RETRAIN_EVERY | 500 | 重训间隔 (~20天) |
| 特征 | EMA_SHORT/LONG | 20/50 | EMA参数 |
| 特征 | MOMENTUM_PERIOD | 10 | 动量回看 |
| 特征 | VOL_PERIOD | 20 | 波动率回看 |

### Regime 划分

| 分类 | 参数 | 当前值 | 作用 |
|------|------|--------|------|
| 阈值 | REGIME_LOW_THRESH | 0.90 | 低波/中波分界 |
| 阈值 | REGIME_HIGH_THRESH | 1.10 | 中波/高波分界 |
| 平滑 | VOL_REGIME_EMA_SPAN | 12 | vol_regime EMA平滑（防闪烁） |
| 确认 | REGIME_ENTRY_BARS | 2 | 进入中波确认K线数 |
| 缓冲 | EXIT_BUFFER_BARS | 6 | 退出中波缓冲K线数 |

### 半连续执行

| 分类 | 参数 | 当前值 | 作用 |
|------|------|--------|------|
| 迟滞 | SIGNAL_CHANGE_THRESH | 0.1 | 信号变化阈值 |
| 冷却 | COOLING_BARS | 3 | 调仓冷却期 |
| 平滑 | SMOOTH_WEIGHT_EXEC | 0.8 | 执行层权重平滑 |
| 过滤 | GLOBAL_VOL_FILTER | 1.3 | BTC vol > 此值 → 全部空仓 |

### 信号分层

| 分类 | 参数 | 当前值 | 作用 |
|------|------|--------|------|
| 分层 | SIGNAL_BOOST_MID | 0.10 | 中仓信号阈值 |
| 分层 | SIGNAL_BOOST_HIGH | 0.15 | 全仓信号阈值 |
| 入场 | SIGNAL_MUST_INCREASE | True | 信号必须增强才交易 |
| 入场 | ENTRY_PRICE_CONFIRM | True | BTC收盘价突破确认 |
| 入场 | ENTRY_DELAY_BARS | 2 | 升仓延迟确认K线数 |
| 持仓 | MIN_HOLD_BARS_EXEC | 6 | 最小持仓K线数 |

### 组合优化

| 分类 | 参数 | 当前值 | 作用 |
|------|------|--------|------|
| 收缩 | SHRINK_MU | 0.7 | 收益收缩强度 |
| 收缩 | SHRINK_COV | 0.3 | 协方差收缩强度 |
| 上限 | MAX_WEIGHT | 0.30 | 单资产最大权重 |
| 阈值 | MIN_WEIGHT_CHANGE | 0.05 | 最小调仓阈值 |

### Vol Targeting

| 分类 | 参数 | 当前值 | 作用 |
|------|------|--------|------|
| 回看 | VOL_LOOKBACK | 24 | 已实现波动率回看K线数 |
| 上限 | VOL_SCALE_CAP | 2.0 | 仓位缩放上限 |
| 下限 | VOL_FLOOR | 0.02 | 已实现波动率下限 |

### 诊断开关

| 分类 | 参数 | 当前值 | 作用 |
|------|------|--------|------|
| 诊断 | DISABLE_TRADING | False | 禁止所有交易 |
| 诊断 | SIGNAL_INVERT | False | 信号取反 |
| 诊断 | SIGNAL_CONFIRM_BARS | 0 | 延迟确认K线数 |

---

## 九、后续提升方向

### 优先级1：纸盘观测 → 真盘过渡
- 运行纸盘 1-2 周，观察虚拟订单是否合理
- 接入 Binance 真实 API 下单
- 单币 BTC 先跑，验证后扩展到 8 币

### 优先级2：信号 → 仓位连续映射
- 当前：信号强度 → 离散三档（30%/60%/100%）
- 方向：连续映射函数，减少等级跳变

### 优先级3：退出时机优化
- 降仓也加轻度延迟（2-3 bar），减少假突破过早退出

### 优先级4：杠杆实盘验证
- Lev=2.0x 回测最优（Sharpe 1.82），需实盘验证风控

### 优先级5：特征工程
- 加入波动率偏度、成交量剖面、跨币种相关性

### 优先级6：退出规则非对称化
- exit_threshold < entry_threshold（让利润跑）

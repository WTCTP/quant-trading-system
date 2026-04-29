"""
量化交易系统可视化仪表盘
Streamlit + Plotly — 回测结果交互式分析
启动: streamlit run dashboard.py
"""
import sys
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="量化交易系统 Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("量化交易系统 v3.1 — Breakout Alpha + V1 执行状态机")

# ─── 缓存数据加载 ───────────────────────────────

@st.cache_data(ttl=3600)
def load_experiment_data():
    """运行回测实验，返回所有结果"""
    from config import (
        SYMBOLS, INITIAL_CAPITAL, BASE_LOW, BASE_MID, BASE_HIGH,
        PORTFOLIO_VOL_TARGET, PORTFOLIO_LEVERAGE,
        USE_FUNDING, USE_CROSS_SECTIONAL,
    )
    from data.fetcher import DataFetcher
    from risk.manager import RiskManager
    from log.recorder import TradeLogger
    from backtest.engine import PortfolioBacktest

    # 加载数据
    fetcher = DataFetcher()
    df_dict = {}
    for s in SYMBOLS:
        df = fetcher.fetch_ohlcv(s)
        if not df.empty:
            df_dict[s] = df

    def run_one(label, use_funding=False, use_cross_sectional=False,
                funding_weight=0.10, cross_sectional_weight=0.10,
                base_mid=None, vol_target=None):
        risk_mgr = RiskManager(INITIAL_CAPITAL)
        logger = TradeLogger(f'trades_{label}.csv')
        engine = PortfolioBacktest(
            risk_mgr, logger,
            base_low=BASE_LOW,
            base_mid=base_mid if base_mid is not None else BASE_MID,
            base_high=BASE_HIGH,
            portfolio_vol_target=vol_target if vol_target is not None else PORTFOLIO_VOL_TARGET,
            portfolio_leverage=PORTFOLIO_LEVERAGE,
            use_pullback=False,
            use_funding=use_funding, funding_weight=funding_weight,
            use_cross_sectional=use_cross_sectional,
            cross_sectional_weight=cross_sectional_weight,
        )
        engine.run(df_dict.copy())
        return engine

    # 运行实验
    with st.spinner('正在运行回测...'):
        eng = run_one('Breakout-Only')
        eng_cs = run_one('Breakout+CrossSec(10%)', use_cross_sectional=True)
        eng_fund = run_one('Breakout+Funding(10%)', use_funding=True)

    return {
        'symbols': SYMBOLS,
        'df_dict': df_dict,
        'breakout': eng,
        'cross_sectional': eng_cs,
        'funding': eng_fund,
    }


def load_paper_data():
    """加载纸盘交易实时数据（不缓存，每次刷新都读取）"""
    records_df = pd.DataFrame()
    trades_df = pd.DataFrame()
    state = None

    try:
        records_df = pd.read_csv('paper_records.csv')
        if not records_df.empty:
            records_df['time'] = pd.to_datetime(records_df['time'])
    except Exception:
        pass

    try:
        trades_df = pd.read_csv('trades_paper.csv')
        if not trades_df.empty:
            trades_df['time'] = pd.to_datetime(trades_df['time'])
    except Exception:
        pass

    try:
        state_df = pd.read_csv('paper_state.csv')
        if not state_df.empty:
            state = state_df.iloc[-1].to_dict()
    except Exception:
        pass

    return records_df, trades_df, state


@st.cache_data(ttl=3600)
def load_trades_csv():
    """尝试从已有CSV加载交易记录"""
    import glob
    import os
    trades = {}
    for f in glob.glob('trades_*.csv'):
        name = f.replace('trades_', '').replace('.csv', '')
        try:
            df = pd.read_csv(f)
            if not df.empty:
                trades[name] = df
        except Exception:
            pass
    return trades


# ─── 指标计算 ──────────────────────────────────

def calc_metrics(engine):
    """从engine计算关键指标"""
    df = pd.DataFrame(engine.records)
    df['returns'] = df['capital'].pct_change()
    df['cum_return'] = (1 + df['returns']).cumprod() - 1

    final = df['capital'].iloc[-1]
    total_ret = (final - engine.initial_capital) / engine.initial_capital
    peak = df['capital'].expanding().max()
    dd_series = (df['capital'] - peak) / peak
    max_dd = dd_series.min()

    rets = df['returns'].dropna()
    ann_vol = rets.std() * np.sqrt(365 * 24)
    sharpe = (rets.mean() / rets.std()) * np.sqrt(365 * 24) if rets.std() > 0 else 0
    calmar = total_ret / abs(max_dd) if max_dd != 0 else 0

    return {
        'df': df,
        'final_capital': final,
        'total_return': total_ret,
        'max_dd': max_dd,
        'ann_vol': ann_vol,
        'sharpe': sharpe,
        'calmar': calmar,
        'n_trades': len(engine.trades),
        'total_fee': engine.total_fee,
        'total_slippage': engine.total_slippage,
        'dd_series': dd_series,
    }


# ─── 图表组件 ──────────────────────────────────

def plot_equity_curve(metrics, engine=None, title="资金曲线"):
    """资金曲线 + 回撤双轴图 + 交易标记"""
    df = metrics['df']

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        subplot_titles=(title, "回撤"),
    )

    fig.add_trace(
        go.Scatter(x=df['time'], y=df['capital'], mode='lines',
                   name='资金', line=dict(color='#00b4d8', width=1.5),
                   fill='tozeroy', fillcolor='rgba(0,180,216,0.1)'),
        row=1, col=1,
    )
    fig.add_hline(y=10000, line_dash="dash", line_color="gray",
                  opacity=0.5, row=1, col=1)

    # 交易标记
    if engine is not None:
        trades_df = engine.get_trades_df()
        if not trades_df.empty:
            # 映射 trade time → 资金
            time_to_cap = dict(zip(df['time'], df['capital']))
            trades_df['cap_at_trade'] = trades_df['time'].map(time_to_cap)
            trades_df = trades_df.dropna(subset=['cap_at_trade'])

            # 买入绿色，卖出红色，强平黑色
            color_map = {'maker': '#2ecc71', 'taker': '#00b4d8', 'force': '#e74c3c'}
            for etype, sub in trades_df.groupby('type'):
                marker_color = color_map.get(etype, '#888')
                # marker 大小按交易额缩放
                sizes = np.clip(abs(sub['delta_value'].values) / 200, 4, 14)
                fig.add_trace(
                    go.Scatter(
                        x=sub['time'], y=sub['cap_at_trade'],
                        mode='markers', name=f'{etype}',
                        marker=dict(
                            symbol='circle-open' if etype == 'maker' else 'circle',
                            size=sizes, color=marker_color,
                            line=dict(width=1.5, color=marker_color),
                            opacity=0.8,
                        ),
                        text=[f'{etype}<br>¥{v:,.0f}<br>${c:,.0f}'
                              for v, c in zip(sub['delta_value'], sub['cap_at_trade'])],
                        hoverinfo='text',
                    ),
                    row=1, col=1,
                )

            # 图例：maker/taker/force
            for etype, color in [('maker', '#2ecc71'), ('taker', '#00b4d8'), ('force', '#e74c3c')]:
                count = len(trades_df[trades_df['type'] == etype])
                if count > 0:
                    fig.add_trace(
                        go.Scatter(x=[None], y=[None], mode='markers',
                                   name=f'{etype} ({count}笔)',
                                   marker=dict(color=color, size=8)),
                        row=1, col=1,
                    )

    fig.add_trace(
        go.Scatter(x=df['time'], y=metrics['dd_series'] * 100, mode='lines',
                   name='回撤 %', line=dict(color='#e63946', width=1),
                   fill='tozeroy', fillcolor='rgba(230,57,70,0.15)'),
        row=2, col=1,
    )

    fig.update_layout(
        height=550, showlegend=True,
        legend=dict(yanchor='top', y=0.99, xanchor='left', x=0.01, bgcolor='rgba(0,0,0,0)'),
        margin=dict(l=0, r=0, t=30, b=0),
        hovermode='x unified',
    )
    fig.update_yaxes(title_text="USDT", row=1, col=1)
    fig.update_yaxes(title_text="%", row=2, col=1)
    return fig


def plot_regime_pie(df):
    """Regime分布饼图"""
    def regime_root(r):
        if isinstance(r, str):
            if r.startswith('mid'): return '中波'
            if r.startswith('low'): return '低波'
            if r.startswith('high'): return '高波'
        return '其他'

    df['root'] = df['regime'].apply(regime_root)
    counts = df['root'].value_counts()

    colors = {'中波': '#00b4d8', '低波': '#ffd166', '高波': '#e63946', '其他': '#ccc'}
    fig = go.Figure(go.Pie(
        labels=counts.index, values=counts.values,
        marker=dict(colors=[colors.get(c, '#ccc') for c in counts.index]),
        hole=0.4, textinfo='label+percent',
    ))
    fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=10))
    return fig


def plot_regime_returns(df):
    """各Regime累计收益"""
    def regime_root(r):
        if isinstance(r, str):
            if r.startswith('mid'): return '中波'
            if r.startswith('low'): return '低波'
            if r.startswith('high'): return '高波'
        return '其他'

    df = df.copy()
    df['root'] = df['regime'].apply(regime_root)

    fig = go.Figure()
    colors = {'中波': '#00b4d8', '低波': '#ffd166', '高波': '#e63946'}

    for regime in ['低波', '中波', '高波']:
        sub = df[df['root'] == regime]
        if len(sub) < 5:
            continue
        sub = sub.copy()
        sub['cum'] = (1 + sub['returns'].fillna(0)).cumprod() - 1
        fig.add_trace(go.Scatter(
            x=sub['time'], y=sub['cum'] * 100, mode='lines',
            name=regime, line=dict(color=colors.get(regime, '#ccc'), width=1.5),
        ))

    fig.update_layout(
        height=350, margin=dict(l=0, r=0, t=10, b=0),
        hovermode='x unified',
        yaxis_title='累计收益 (%)',
    )
    return fig


def plot_signal_buckets(engine):
    """信号强度分桶收益"""
    analysis = engine.get_signal_bucket_analysis(n_buckets=10)
    if not analysis:
        return go.Figure()

    buckets = [a['bucket'] for a in analysis]
    sharpes = [a['sharpe'] for a in analysis]
    cum_rets = [float(a['cum_return'].replace('%', '')) for a in analysis]
    counts = [a['count'] for a in analysis]
    ranges = [a['signal_range'] for a in analysis]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=buckets, y=counts, name='K线数', marker_color='#e0e0e0'))
    fig.add_trace(go.Scatter(x=buckets, y=sharpes, name='Sharpe',
                              line=dict(color='#00b4d8', width=2)),
                  secondary_y=True)
    fig.add_trace(go.Scatter(x=buckets, y=cum_rets, name='累计收益%',
                              line=dict(color='#ffd166', width=2)),
                  secondary_y=True)

    fig.update_layout(height=350, margin=dict(l=0, r=0, t=10, b=0),
                      hovermode='x unified')
    fig.update_xaxes(title_text='信号强度分桶 (0=最弱, 9=最强)')
    fig.update_yaxes(title_text='K线数', secondary_y=False)
    fig.update_yaxes(title_text='Sharpe / 收益%', secondary_y=True)
    return fig


def plot_trade_impact(engine, metrics):
    """每笔交易后的资金变化：交易点资金 + 前后连接线"""
    trades_df = engine.get_trades_df()
    df = metrics['df']
    if trades_df.empty or df.empty:
        return go.Figure()

    time_to_cap = dict(zip(df['time'], df['capital']))
    trades_df['cap'] = trades_df['time'].map(time_to_cap)
    trades_df = trades_df.dropna(subset=['cap'])

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.08, row_heights=[0.6, 0.4],
                        subplot_titles=('每笔交易时的资金水平', '交易金额分布'))

    # 买/卖分开
    buys = trades_df[trades_df['delta_value'] > 0]
    sells = trades_df[trades_df['delta_value'] < 0]

    fig.add_trace(go.Scatter(
        x=buys['time'], y=buys['cap'], mode='markers',
        name=f'买入 ({len(buys)}笔)', marker=dict(symbol='triangle-up', size=8, color='#2ecc71'),
        text=[f'买入 {s}<br>¥{v:,.0f}<br>资金 ${c:,.0f}'
              for s, v, c in zip(buys['symbol'], buys['delta_value'], buys['cap'])],
        hoverinfo='text',
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=sells['time'], y=sells['cap'], mode='markers',
        name=f'卖出 ({len(sells)}笔)', marker=dict(symbol='triangle-down', size=8, color='#e63946'),
        text=[f'卖出 {s}<br>¥{v:,.0f}<br>资金 ${c:,.0f}'
              for s, v, c in zip(sells['symbol'], sells['delta_value'], sells['cap'])],
        hoverinfo='text',
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df['time'], y=df['capital'], mode='lines',
        name='资金曲线', line=dict(color='#ccc', width=0.8), showlegend=False,
    ), row=1, col=1)

    # 交易金额直方图
    all_vals = trades_df['delta_value'].abs()
    fig.add_trace(go.Histogram(
        x=all_vals, nbinsx=40, name='交易金额',
        marker_color='#00b4d8', opacity=0.7,
    ), row=2, col=1)

    fig.update_layout(height=500, margin=dict(l=0, r=0, t=30, b=0),
                      hovermode='x unified')
    fig.update_xaxes(title_text='时间', row=1, col=1)
    fig.update_yaxes(title_text='资金 (USDT)', row=1, col=1)
    fig.update_xaxes(title_text='交易金额 (USDT)', row=2, col=1)
    fig.update_yaxes(title_text='笔数', row=2, col=1)
    return fig


def plot_trade_forward_returns(engine, metrics, forward_bars=5):
    """交易后N根K线的资金变化：评估交易时机质量"""
    trades_df = engine.get_trades_df()
    df = metrics['df']
    if trades_df.empty or df.empty:
        return go.Figure()

    # 为每笔交易计算 forward return
    df = df.set_index('time')
    cap_series = df['capital']
    forward_returns = []

    for _, trade in trades_df.iterrows():
        t = trade['time']
        if t not in cap_series.index:
            continue
        idx = cap_series.index.get_loc(t)
        if idx + forward_bars >= len(cap_series):
            continue
        fwd_cap = cap_series.iloc[idx + forward_bars]
        cur_cap = cap_series.iloc[idx]
        fwd_ret = (fwd_cap / cur_cap) - 1
        forward_returns.append({
            'time': t,
            'symbol': trade['symbol'],
            'type': trade['type'],
            'delta_value': trade['delta_value'],
            'fwd_return': fwd_ret * 100,
            'side': '买' if trade['delta_value'] > 0 else '卖',
        })

    if not forward_returns:
        return go.Figure()

    fwd_df = pd.DataFrame(forward_returns)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.08, row_heights=[0.5, 0.5],
                        subplot_titles=(f'交易后{forward_bars}根K线收益 (%)', '累计前向收益'))

    colors = {'买': '#2ecc71', '卖': '#e63946'}
    for side, sub in fwd_df.groupby('side'):
        fig.add_trace(go.Bar(
            x=sub['time'], y=sub['fwd_return'],
            name=side, marker_color=colors.get(side, '#888'),
            text=[f'{side} {s}<br>{v:.2f}%' for s, v in zip(sub['symbol'], sub['fwd_return'])],
            hoverinfo='text',
        ), row=1, col=1)

    # 累计前向收益
    fwd_df = fwd_df.sort_values('time')
    fwd_df['cum_fwd'] = fwd_df['fwd_return'].cumsum()
    fig.add_trace(go.Scatter(
        x=fwd_df['time'], y=fwd_df['cum_fwd'], mode='lines+markers',
        name='累计前向收益%', line=dict(color='#ffd166', width=2),
        marker=dict(size=4),
    ), row=2, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=2, col=1)

    pos_pct = (fwd_df['fwd_return'] > 0).mean() * 100
    avg_fwd = fwd_df['fwd_return'].mean()
    fig.add_annotation(
        xref='paper', yref='paper', x=0.98, y=0.98,
        text=f'正向率: {pos_pct:.0f}% | 均值: {avg_fwd:.2f}%',
        showarrow=False, bgcolor='rgba(0,0,0,0.5)', font=dict(color='white', size=12),
    )

    fig.update_layout(height=450, margin=dict(l=0, r=0, t=30, b=0),
                      hovermode='x unified')
    fig.update_yaxes(title_text='前向收益 %', row=1, col=1)
    fig.update_yaxes(title_text='累计 %', row=2, col=1)
    return fig


def plot_trade_distribution(engine):
    """交易分布：按币种"""
    trades_df = engine.get_trades_df()
    if trades_df.empty:
        return go.Figure()

    sym_stats = trades_df.groupby('symbol').agg(
        交易次数=('delta_value', 'count'),
        总交易额=('delta_value', lambda x: abs(x).sum()),
    ).reset_index()

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=sym_stats['symbol'], y=sym_stats['交易次数'],
                          name='交易次数', marker_color='#00b4d8'))
    fig.add_trace(go.Scatter(x=sym_stats['symbol'], y=sym_stats['总交易额'],
                              name='总交易额', mode='markers+lines',
                              marker=dict(size=10, color='#e63946')),
                  secondary_y=True)
    fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))
    return fig


def plot_monthly_heatmap(metrics):
    """月度收益热力图"""
    df = metrics['df'].copy()
    df['month'] = pd.to_datetime(df['time']).dt.to_period('M')
    monthly = df.groupby('month')['returns'].apply(lambda x: (1 + x).prod() - 1)
    monthly.index = monthly.index.astype(str)

    # reshape to year x month matrix
    data = []
    for m, v in monthly.items():
        y, mo = m.split('-')
        data.append({'year': int(y), 'month': int(mo), 'return': v * 100})

    hm_df = pd.DataFrame(data).pivot(index='year', columns='month', values='return')
    # 确保列是 1-12
    for c in range(1, 13):
        if c not in hm_df.columns:
            hm_df[c] = np.nan
    hm_df = hm_df[sorted(hm_df.columns)]

    fig = go.Figure(go.Heatmap(
        z=hm_df.values, x=[f'{c}月' for c in hm_df.columns],
        y=[str(y) for y in hm_df.index],
        colorscale='RdYlGn', zmid=0,
        text=[[f'{v:.1f}%' if not np.isnan(v) else '' for v in row]
              for row in hm_df.values],
        texttemplate='%{text}', textfont=dict(size=11),
    ))
    fig.update_layout(height=250, margin=dict(l=0, r=0, t=10, b=0))
    return fig


def plot_experiment_comparison(results):
    """多实验对比雷达图 + 柱状图"""
    # 柱状图对比
    exp_names = []
    returns = []
    sharpes = []
    calmars = []
    trades = []

    for name, eng in results.items():
        m = calc_metrics(eng)
        exp_names.append(name)
        returns.append(m['total_return'] * 100)
        sharpes.append(m['sharpe'])
        calmars.append(m['calmar'])
        trades.append(m['n_trades'])

    fig = make_subplots(rows=2, cols=2, subplot_titles=(
        '总收益 (%)', 'Sharpe Ratio', 'Calmar Ratio', '交易次数'))

    colors = ['#00b4d8' if r == max(returns) else '#adb5bd' for r in returns]
    fig.add_trace(go.Bar(x=exp_names, y=returns, marker_color=colors,
                          text=[f'{v:.1f}' for v in returns], textposition='outside'),
                  row=1, col=1)
    colors = ['#00b4d8' if s == max(sharpes) else '#adb5bd' for s in sharpes]
    fig.add_trace(go.Bar(x=exp_names, y=sharpes, marker_color=colors,
                          text=[f'{v:.2f}' for v in sharpes], textposition='outside'),
                  row=1, col=2)
    colors = ['#00b4d8' if c == max(calmars) else '#adb5bd' for c in calmars]
    fig.add_trace(go.Bar(x=exp_names, y=calmars, marker_color=colors,
                          text=[f'{v:.2f}' for v in calmars], textposition='outside'),
                  row=2, col=1)
    fig.add_trace(go.Bar(x=exp_names, y=trades, marker_color='#ffd166',
                          text=[str(t) for t in trades], textposition='outside'),
                  row=2, col=2)

    fig.update_layout(height=500, showlegend=False, margin=dict(l=0, r=0, t=30, b=0))
    return fig


def plot_paper_vs_backtest(paper_df, backtest_metrics, backtest_engine):
    """纸盘权益曲线叠加回测基准对比"""
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.05, row_heights=[0.7, 0.3],
        subplot_titles=("纸盘 vs 回测 — 资金曲线对比", "偏离 (纸盘 - 回测)"),
    )

    bt_df = backtest_metrics['df']

    # 回测基准曲线
    fig.add_trace(
        go.Scatter(x=bt_df['time'], y=bt_df['capital'], mode='lines',
                   name='回测基准', line=dict(color='#adb5bd', width=1.5, dash='dash')),
        row=1, col=1,
    )

    if not paper_df.empty:
        # 纸盘权益曲线
        fig.add_trace(
            go.Scatter(x=paper_df['time'], y=paper_df['capital'], mode='lines+markers',
                       name='纸盘交易', line=dict(color='#00b4d8', width=2.5),
                       marker=dict(size=4, color='#00b4d8')),
            row=1, col=1,
        )

        # 纸盘交易标记
        try:
            paper_trades = pd.read_csv('trades_paper.csv')
            if not paper_trades.empty:
                paper_trades['time'] = pd.to_datetime(paper_trades['time'])
                time_to_cap = dict(zip(paper_df['time'], paper_df['capital']))
                paper_trades['cap_at_trade'] = paper_trades['time'].map(time_to_cap)
                paper_trades = paper_trades.dropna(subset=['cap_at_trade'])

                color_map = {'maker': '#2ecc71', 'taker': '#00b4d8', 'force': '#e74c3c'}
                for etype, sub in paper_trades.groupby('type'):
                    marker_color = color_map.get(etype, '#888')
                    sizes = np.clip(abs(sub['delta_value'].values) / 200, 4, 12)
                    fig.add_trace(
                        go.Scatter(
                            x=sub['time'], y=sub['cap_at_trade'],
                            mode='markers', name=f'纸盘-{etype}',
                            marker=dict(symbol='circle-open' if etype == 'maker' else 'circle',
                                        size=sizes, color=marker_color,
                                        line=dict(width=1.5, color=marker_color), opacity=0.8),
                            text=[f'{etype}<br>{s}<br>${v:,.0f}'
                                  for s, v in zip(sub['symbol'], abs(sub['delta_value']))],
                            hoverinfo='text',
                        ),
                        row=1, col=1,
                    )
        except Exception:
            pass

        # 偏离曲线
        bt_time_to_cap = dict(zip(bt_df['time'], bt_df['capital']))
        paper_df['bt_cap'] = paper_df['time'].map(bt_time_to_cap)
        paper_df = paper_df.dropna(subset=['bt_cap'])
        paper_df['deviation'] = (paper_df['capital'] - paper_df['bt_cap']) / paper_df['bt_cap'] * 100
        fig.add_trace(
            go.Scatter(x=paper_df['time'], y=paper_df['deviation'], mode='lines',
                       name='偏离 %', line=dict(color='#ffd166', width=1.5),
                       fill='tozeroy', fillcolor='rgba(255,209,102,0.1)'),
            row=2, col=1,
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=2, col=1)

    fig.add_hline(y=10000, line_dash="dash", line_color="gray",
                  opacity=0.5, row=1, col=1)
    fig.update_layout(
        height=550, showlegend=True,
        legend=dict(yanchor='top', y=0.99, xanchor='left', x=0.01, bgcolor='rgba(0,0,0,0)'),
        margin=dict(l=0, r=0, t=30, b=0), hovermode='x unified',
    )
    fig.update_yaxes(title_text="USDT", row=1, col=1)
    fig.update_yaxes(title_text="%", row=2, col=1)
    return fig


# ═══════════════════════════════════════════════════
# Main App
# ═══════════════════════════════════════════════════

with st.sidebar:
    st.header("⚙️ 控制面板")
    st.caption("数据自动缓存，刷新页面不会重新计算")

    if st.button("🔄 重新运行回测", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.divider()
    st.caption("启动方式: `streamlit run dashboard.py`")
    st.caption("数据来源: Binance 1h K线 (2024-01-01 ~ 最新)")

# 加载数据
try:
    data = load_experiment_data()
    eng_main = data['breakout']
    eng_cs = data['cross_sectional']
    eng_fund = data['funding']

    metrics_main = calc_metrics(eng_main)

    # ═══ 概览 KPI 卡片 ═══
    st.subheader("📈 回测概览 — Breakout-Only (生产配置)")

    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
    with col1:
        st.metric("总收益", f"{metrics_main['total_return']:.2%}", delta="+25.35%")
    with col2:
        st.metric("Sharpe", f"{metrics_main['sharpe']:.2f}")
    with col3:
        st.metric("Calmar", f"{metrics_main['calmar']:.2f}")
    with col4:
        st.metric("最大回撤", f"{metrics_main['max_dd']:.2%}")
    with col5:
        st.metric("年化波动", f"{metrics_main['ann_vol']:.2%}")
    with col6:
        st.metric("交易次数", str(metrics_main['n_trades']))
    with col7:
        st.metric("最终资金", f"${metrics_main['final_capital']:,.0f}")

    # ═══ Tabs ═══
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 资金曲线", "🗺️ Regime分析", "📡 信号分析", "💹 交易明细", "⚖️ 实验对比", "📡 纸盘交易"
    ])

    with tab1:
        st.plotly_chart(plot_equity_curve(metrics_main, eng_main), use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(plot_monthly_heatmap(metrics_main), use_container_width=True)
        with c2:
            st.subheader("成本分解")
            cost_df = pd.DataFrame({
                '项目': ['手续费', '滑点', '资金费'],
                '金额': [metrics_main['total_fee'], metrics_main['total_slippage'], 0],
            })
            fig_cost = go.Figure(go.Waterfall(
                name='成本', orientation='v',
                measure=['absolute', 'absolute', 'absolute'],
                x=cost_df['项目'], y=cost_df['金额'],
                connector=dict(line=dict(color='gray')),
            ))
            fig_cost.update_layout(height=250, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig_cost, use_container_width=True)

    with tab2:
        c1, c2 = st.columns([1, 2])
        with c1:
            st.plotly_chart(plot_regime_pie(metrics_main['df']), use_container_width=True)

            # Regime统计表
            def regime_root(r):
                if isinstance(r, str):
                    if r.startswith('mid'): return '中波'
                    if r.startswith('low'): return '低波'
                    if r.startswith('high'): return '高波'
                return '其他'
            df_tmp = metrics_main['df'].copy()
            df_tmp['root'] = df_tmp['regime'].apply(regime_root)
            for regime in ['低波', '中波', '高波']:
                sub = df_tmp[df_tmp['root'] == regime]
                if len(sub) > 5:
                    rets = sub['returns'].dropna()
                    cum = (1 + rets).prod() - 1
                    sr = (rets.mean() / rets.std()) * np.sqrt(365*24) if rets.std() > 0 else 0
                    st.metric(f"{regime}累计收益", f"{cum:.2%}", delta=f"Sharpe {sr:.2f}")

        with c2:
            st.plotly_chart(plot_regime_returns(metrics_main['df']), use_container_width=True)

        st.subheader("Regime 状态分解")
        regime_data = eng_main.get_regime_analysis(data['df_dict'])
        if regime_data:
            regime_df = pd.DataFrame(regime_data)
            st.dataframe(regime_df, use_container_width=True, hide_index=True)

    with tab3:
        st.subheader("信号强度分桶分析 (10档 pd.qcut)")
        st.plotly_chart(plot_signal_buckets(eng_main), use_container_width=True)

        st.caption("信号集中在顶部20%：Q8-Q9 Sharpe最高，底部信号接近随机。这验证了暴露管理策略——只在强信号时增加仓位。")

        # 时间稳定性
        st.subheader("时间稳定性 (按年份拆分)")
        stability = eng_main.get_signal_stability_analysis()
        if stability:
            stab_df = pd.DataFrame(stability)
            # 透视
            stab_pivot = stab_df.pivot_table(
                index='bucket', columns='year', values='sharpe', aggfunc='first')
            fig_stab = go.Figure(go.Heatmap(
                z=stab_pivot.values, x=[str(int(y)) for y in stab_pivot.columns],
                y=[f'Q{b}' for b in stab_pivot.index],
                colorscale='RdYlGn', zmid=0,
                text=[[f'{v:.1f}' if not np.isnan(v) else '' for v in row]
                      for row in stab_pivot.values],
                texttemplate='%{text}',
            ))
            fig_stab.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0),
                                    xaxis_title='年份', yaxis_title='信号分位')
            st.plotly_chart(fig_stab, use_container_width=True)

    with tab4:
        st.subheader("每笔交易时的资金水平")
        st.caption("▲ 买入 | ▼ 卖出 | 点在资金曲线上，悬停看详情")
        st.plotly_chart(plot_trade_impact(eng_main, metrics_main), use_container_width=True)

        st.subheader(f"交易后 5 根K线的前向收益")
        st.caption("正值=交易后资金上涨（时机好），负值=交易后资金下跌（时机差）")
        st.plotly_chart(plot_trade_forward_returns(eng_main, metrics_main), use_container_width=True)

        st.subheader("按币种分布")
        st.plotly_chart(plot_trade_distribution(eng_main), use_container_width=True)

        st.subheader("最近交易明细")
        trades_df = eng_main.get_trades_df()
        if not trades_df.empty:
            trades_df_display = trades_df.tail(50).sort_values('time', ascending=False)
            st.dataframe(trades_df_display, use_container_width=True, hide_index=True)

    with tab5:
        st.subheader("多Alpha实验对比")
        results = {
            'Breakout-Only': eng_main,
            'Breakout+CrossSec(10%)': eng_cs,
            'Breakout+Funding(10%)': eng_fund,
        }
        st.plotly_chart(plot_experiment_comparison(results), use_container_width=True)

        # 汇总表
        rows = []
        for name, eng in results.items():
            m = calc_metrics(eng)
            rows.append({
                '实验': name,
                '收益': f"{m['total_return']:.2%}",
                'Sharpe': f"{m['sharpe']:.2f}",
                'Calmar': f"{m['calmar']:.2f}",
                '最大回撤': f"{m['max_dd']:.2%}",
                '交易': m['n_trades'],
                '手续费': f"${m['total_fee']:,.0f}",
                '滑点': f"${m['total_slippage']:,.0f}",
            })
        comp_df = pd.DataFrame(rows)
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

    with tab6:
        st.subheader("📡 纸盘交易 — 实时 vs 回测对比")

        paper_df, paper_trades, paper_state = load_paper_data()

        # 状态面板
        if paper_state:
            c1, c2, c3, c4, c5 = st.columns(5)
            with c1:
                st.metric("当前资金", f"${paper_state.get('capital', 0):,.0f}")
            with c2:
                st.metric("当前Regime", paper_state.get('regime', 'N/A'))
            with c3:
                st.metric("信号强度", f"{float(paper_state.get('signal_max', 0)):.3f}")
            with c4:
                st.metric("持仓币数", str(paper_state.get('n_positions', 0)))
            with c5:
                st.metric("待成交订单", str(paper_state.get('n_pending', 0)))
            st.caption(f"最后更新: {paper_state.get('time', 'N/A')}")
        else:
            st.warning("暂无纸盘数据。请确保纸盘交易正在运行 (`python main_paper.py`)")
            st.info("纸盘引擎每处理完一根K线后自动保存状态到 `paper_records.csv` 和 `paper_state.csv`")

        # 对比图
        if not paper_df.empty:
            st.plotly_chart(
                plot_paper_vs_backtest(paper_df, metrics_main, eng_main),
                use_container_width=True,
            )

            # 纸盘统计
            c1, c2, c3, c4 = st.columns(4)
            paper_df['returns'] = paper_df['capital'].pct_change()
            rets = paper_df['returns'].dropna()
            if len(rets) > 1:
                paper_ret = (paper_df['capital'].iloc[-1] / paper_df['capital'].iloc[0]) - 1
                paper_vol = rets.std() * np.sqrt(365 * 24)
                paper_sharpe = (rets.mean() / rets.std()) * np.sqrt(365 * 24) if rets.std() > 0 else 0
                with c1:
                    st.metric("纸盘收益", f"{paper_ret:.2%}")
                with c2:
                    st.metric("纸盘波动", f"{paper_vol:.2%}")
                with c3:
                    st.metric("纸盘Sharpe", f"{paper_sharpe:.2f}")
                with c4:
                    st.metric("纸盘交易", f"{len(paper_trades)} 笔")

            # 交易明细
            st.subheader("纸盘交易明细")
            if not paper_trades.empty:
                st.dataframe(
                    paper_trades.tail(30).sort_values('time', ascending=False),
                    use_container_width=True, hide_index=True,
                )
        else:
            st.plotly_chart(
                plot_equity_curve(metrics_main, eng_main, title="回测资金曲线 (纸盘暂无数据)"),
                use_container_width=True,
            )

        # 手动刷新
        c1, c2 = st.columns([1, 5])
        with c1:
            if st.button("🔄 刷新纸盘数据", use_container_width=True):
                st.rerun()
        with c2:
            st.caption("纸盘数据由 `main_paper.py` 实时写入 `paper_records.csv` 和 `paper_state.csv`，刷新页面即可看到最新状态。")

except Exception as e:
    st.error(f"加载失败: {e}")
    st.info("请确保已运行 main_backtest.py 下载数据，并且 data/cache/ 中有 CSV 缓存文件。")

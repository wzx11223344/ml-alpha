"""
report.py - HTML报告与图表生成模块

使用matplotlib生成可视化图表，并将图表以base64编码嵌入HTML报告。

生成的图表包括：
1. 累积收益曲线（各组 + 多空对冲）
2. 分组月度收益热力图
3. 特征重要性条形图
4. 滚动IC曲线
5. 模型预测vs实际收益散点图

最终输出为一个自包含的HTML文件。
"""

import os
import base64
import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 使用非交互式后端，避免GUI依赖
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from typing import Dict, Any, Optional

# 设置中文字体支持
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

# 定义配色方案
COLORS = {
    "group_colors": ["#d32f2f", "#f57c00", "#fbc02d", "#689f38", "#1e88e5"],
    "long_short_color": "#7b1fa2",
    "ic_color": "#00897b",
    "scatter_color": "#5c6bc0",
    "importance_color": "#3f51b5",
}


# ============================================================
# 1. 图表生成函数
# ============================================================

def plot_cumulative_returns(
    cumulative: pd.DataFrame,
    cumulative_long_short: pd.Series,
    title: str = "分组累积收益曲线",
) -> str:
    """
    绘制累积收益曲线

    参数:
        cumulative: 各组累积收益DataFrame
        cumulative_long_short: 多空对冲累积收益Series
        title: 图表标题

    返回:
        base64编码的PNG图片字符串
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # 绘制各分组累积收益
    n_groups = len(cumulative.columns)
    colors = COLORS["group_colors"][:n_groups]

    for i, col in enumerate(cumulative.columns):
        color = colors[i % len(colors)]
        label = f"Q{int(col)}" if col != n_groups else f"Q{int(col)} (Top)"
        if col == 1:
            label = f"Q{int(col)} (Bottom)"
        ax.plot(cumulative.index, cumulative[col], label=label, color=color, linewidth=1.5)

    # 绘制多空对冲累积收益
    ax.plot(
        cumulative_long_short.index,
        cumulative_long_short,
        label="多空对冲",
        color=COLORS["long_short_color"],
        linewidth=2.5,
        linestyle="--",
    )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("日期")
    ax.set_ylabel("累积收益率")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="black", linewidth=0.5)

    # 格式化x轴日期
    fig.autofmt_xdate(rotation=30)

    plt.tight_layout()
    return _fig_to_base64(fig)


def plot_group_heatmap(
    group_returns: pd.DataFrame,
    title: str = "分组月度收益热力图",
) -> str:
    """
    绘制分组月度收益热力图

    参数:
        group_returns: 分组收益DataFrame
        title: 图表标题

    返回:
        base64编码的PNG图片字符串
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    # 转换为百分比
    data = (group_returns * 100).T

    # 创建自定义颜色映射（红-白-绿）
    cmap = LinearSegmentedColormap.from_list(
        "rg", ["#d32f2f", "#ffffff", "#2e7d32"], N=100
    )

    im = ax.imshow(data.values, aspect="auto", cmap=cmap, vmin=-5, vmax=5)

    # 设置刻度
    ax.set_yticks(range(len(data.index)))
    ax.set_yticklabels([f"Q{int(g)}" for g in data.index])

    # x轴只显示部分标签避免拥挤
    n_dates = len(data.columns)
    step = max(1, n_dates // 15)
    ax.set_xticks(range(0, n_dates, step))
    date_labels = [d.strftime("%Y-%m") for d in data.columns[::step]]
    ax.set_xticklabels(date_labels, rotation=45, ha="right")

    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("月度收益率 (%)")

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("调仓日期")
    ax.set_ylabel("分组")

    plt.tight_layout()
    return _fig_to_base64(fig)


def plot_feature_importance(
    feature_importance: pd.DataFrame,
    top_n: int = 15,
    title: str = "特征重要性排名",
) -> str:
    """
    绘制特征重要性条形图

    参数:
        feature_importance: 特征重要性DataFrame，包含 feature 和 mean_importance 列
        top_n: 显示前N个特征
        title: 图表标题

    返回:
        base64编码的PNG图片字符串
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # 取前N个特征
    top_features = feature_importance.head(top_n).sort_values("mean_importance", ascending=True)

    # 绘制水平条形图
    bars = ax.barh(
        top_features["feature"],
        top_features["mean_importance"],
        color=COLORS["importance_color"],
        alpha=0.8,
    )

    # 添加数值标签
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + 0.001,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.4f}",
            ha="left",
            va="center",
            fontsize=9,
        )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("平均重要性")
    ax.set_ylabel("特征名称")
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    return _fig_to_base64(fig)


def plot_rolling_ic(
    rolling_ic: pd.Series,
    ic_series: pd.Series = None,
    title: str = "滚动IC曲线",
) -> str:
    """
    绘制滚动IC曲线

    参数:
        rolling_ic: 滚动IC Series
        ic_series: 原始IC Series（可选，用于背景显示）
        title: 图表标题

    返回:
        base64编码的PNG图片字符串
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    # 绘制原始IC（浅色背景）
    if ic_series is not None and len(ic_series) > 0:
        ax.bar(
            ic_series.index,
            ic_series.values,
            alpha=0.2,
            color=COLORS["ic_color"],
            label="单期IC",
            width=2,
        )

    # 绘制滚动IC
    ax.plot(
        rolling_ic.index,
        rolling_ic.values,
        color=COLORS["ic_color"],
        linewidth=2,
        label="滚动IC (6期均值)",
    )

    # 添加零线
    ax.axhline(y=0, color="black", linewidth=0.5)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("日期")
    ax.set_ylabel("IC值")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.autofmt_xdate(rotation=30)

    plt.tight_layout()
    return _fig_to_base64(fig)


def plot_prediction_vs_actual(
    pred_actual: pd.DataFrame,
    title: str = "预测值 vs 实际收益散点图",
) -> str:
    """
    绘制预测值 vs 实际收益散点图

    参数:
        pred_actual: 包含 prediction 和 actual_return 两列的DataFrame
        title: 图表标题

    返回:
        base64编码的PNG图片字符串
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # 为了避免数据点过多，随机采样（这不是生成业务数据，只是可视化采样）
    sample = pred_actual
    if len(sample) > 5000:
        # 使用固定种子采样，仅用于可视化，不影响模型
        sample = pred_actual.sample(n=5000, random_state=42)

    ax.scatter(
        sample["prediction"],
        sample["actual_return"],
        alpha=0.3,
        s=5,
        color=COLORS["scatter_color"],
    )

    # 添加零线
    ax.axhline(y=0, color="black", linewidth=0.5, linestyle="--")
    ax.axvline(x=0, color="black", linewidth=0.5, linestyle="--")

    # 添加拟合线
    if len(sample) > 10:
        x = sample["prediction"].values
        y = sample["actual_return"].values
        # 简单线性回归拟合
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() > 10:
            x_clean = x[mask]
            y_clean = y[mask]
            z = np.polyfit(x_clean, y_clean, 1)
            p = np.poly1d(z)
            x_fit = np.linspace(x_clean.min(), x_clean.max(), 100)
            ax.plot(x_fit, p(x_fit), "r--", linewidth=1.5, label=f"拟合线 (斜率={z[0]:.4f})")
            ax.legend(fontsize=10)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("模型预测值")
    ax.set_ylabel("实际收益率")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return _fig_to_base64(fig)


# ============================================================
# 2. 辅助函数
# ============================================================

def _fig_to_base64(fig) -> str:
    """
    将matplotlib Figure转换为base64编码的PNG字符串

    参数:
        fig: matplotlib Figure对象

    返回:
        base64编码的字符串
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    return img_base64


def _base64_to_html_img(img_base64: str, alt: str = "") -> str:
    """
    将base64图片转换为HTML img标签

    参数:
        img_base64: base64编码的图片
        alt: 替代文本

    返回:
        HTML img标签字符串
    """
    return f'<img src="data:image/png;base64,{img_base64}" alt="{alt}" style="width:100%;max-width:1000px;margin:10px 0;">'


# ============================================================
# 3. HTML报告生成
# ============================================================

def generate_html_report(
    eval_result: Dict[str, Any],
    predictions: pd.DataFrame,
    panel: pd.DataFrame,
    target_col: str,
    model_name: str = "XGBoost",
    output_path: str = "report.html",
) -> str:
    """
    生成完整的HTML报告

    参数:
        eval_result: 评估结果字典（由 evaluation.evaluate_model 生成）
        predictions: 预测结果
        panel: 面板数据
        target_col: 目标收益率列名
        model_name: 模型名称
        output_path: 输出文件路径

    返回:
        HTML报告文件路径
    """
    print("=" * 60)
    print("生成HTML报告...")
    print("=" * 60)

    # --- 生成所有图表 ---
    print("  [1/5] 生成累积收益曲线...")
    img_cumulative = plot_cumulative_returns(
        eval_result["cumulative"],
        eval_result["cumulative_long_short"],
    )

    print("  [2/5] 生成分组月度收益热力图...")
    img_heatmap = plot_group_heatmap(eval_result["group_returns"])

    print("  [3/5] 生成特征重要性图...")
    img_importance = plot_feature_importance(eval_result["feature_importance"])

    print("  [4/5] 生成滚动IC曲线...")
    img_rolling_ic = plot_rolling_ic(
        eval_result["rolling_ic"],
        eval_result["ic_series"],
    )

    print("  [5/5] 生成散点图...")
    from .evaluation import get_prediction_vs_actual_data
    pred_actual = get_prediction_vs_actual_data(predictions, panel, target_col)
    img_scatter = plot_prediction_vs_actual(pred_actual)

    # --- 构建HTML内容 ---
    ic_summary = eval_result["ic_summary"]
    monotonicity = eval_result["monotonicity"]
    mdd = eval_result["max_drawdown"]

    html = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ML选股模型评估报告 - {model_name}</title>
    <style>
        body {{
            font-family: "Microsoft YaHei", "SimHei", Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
            color: #333;
        }}
        h1 {{
            color: #1e88e5;
            text-align: center;
            border-bottom: 3px solid #1e88e5;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #1565c0;
            border-left: 4px solid #1e88e5;
            padding-left: 10px;
            margin-top: 30px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: white;
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #1e88e5;
        }}
        .metric-label {{
            font-size: 14px;
            color: #666;
            margin-top: 5px;
        }}
        .chart-container {{
            background: white;
            border-radius: 8px;
            padding: 15px;
            margin: 15px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .table th {{
            background: #1e88e5;
            color: white;
            padding: 10px;
            text-align: left;
        }}
        .table td {{
            padding: 8px 10px;
            border-bottom: 1px solid #eee;
        }}
        .table tr:nth-child(even) {{
            background: #f9f9f9;
        }}
        .summary {{
            background: #e3f2fd;
            border-radius: 8px;
            padding: 15px;
            margin: 15px 0;
        }}
        .positive {{ color: #2e7d32; }}
        .negative {{ color: #c62828; }}
    </style>
</head>
<body>
    <h1>ML选股模型评估报告</h1>
    <p style="text-align:center;color:#666;">模型: {model_name} | 生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}</p>

    <div class="summary">
        <strong>报告概述：</strong>本报告展示了基于{model_name}的A股多因子选股模型的评估结果。
        模型使用滚动窗口训练方式，在历史数据上预测未来收益率，并通过分组回测验证选股效果。
    </div>

    <h2>1. 核心指标</h2>
    <div class="metrics-grid">
        <div class="metric-card">
            <div class="metric-value">{ic_summary['mean_ic']:.4f}</div>
            <div class="metric-label">平均IC</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{ic_summary['ic_ir']:.4f}</div>
            <div class="metric-label">IC信息比率</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{ic_summary['ic_win_rate']:.1%}</div>
            <div class="metric-label">IC胜率</div>
        </div>
        <div class="metric-card">
            <div class="metric-value {'positive' if eval_result['annual_return'] > 0 else 'negative'}">{eval_result['annual_return']:.2%}</div>
            <div class="metric-label">多空年化收益</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{eval_result['sharpe_ratio']:.2f}</div>
            <div class="metric-label">夏普比率</div>
        </div>
        <div class="metric-card">
            <div class="metric-value negative">{mdd['max_drawdown']:.2%}</div>
            <div class="metric-label">最大回撤</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{eval_result['avg_turnover']:.1%}</div>
            <div class="metric-label">平均换手率</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{monotonicity['monotonic_ratio']:.1%}</div>
            <div class="metric-label">分组单调性</div>
        </div>
    </div>

    <h2>2. 累积收益曲线</h2>
    <p>下图展示各分组（Q1为预测最低组，Q5为预测最高组）及多空对冲组合的累积收益。
    理想情况下，Q5收益最高，Q1收益最低，多空对冲收益稳定上升。</p>
    <div class="chart-container">
        {_base64_to_html_img(img_cumulative, "累积收益曲线")}
    </div>

    <h2>3. 分组月度收益热力图</h2>
    <p>热力图展示各分组在不同月份的收益率。红色表示亏损，绿色表示盈利。
    理想情况下，从Q1到Q5颜色应从红渐变为绿。</p>
    <div class="chart-container">
        {_base64_to_html_img(img_heatmap, "分组月度收益热力图")}
    </div>

    <h2>4. 特征重要性排名</h2>
    <p>下图展示模型中各特征的平均重要性排名，帮助理解哪些因子对预测贡献最大。</p>
    <div class="chart-container">
        {_base64_to_html_img(img_importance, "特征重要性")}
    </div>

    <h2>5. 滚动IC曲线</h2>
    <p>IC（信息系数）衡量预测值与实际收益的秩相关性。滚动IC反映模型预测能力的稳定性。
    IC持续为正且波动小，说明模型具有稳定的预测能力。</p>
    <div class="chart-container">
        {_base64_to_html_img(img_rolling_ic, "滚动IC曲线")}
    </div>

    <h2>6. 预测值 vs 实际收益散点图</h2>
    <p>散点图展示模型预测值与实际收益的关系。理想情况下，点应沿对角线分布，
    拟合线斜率为正。</p>
    <div class="chart-container">
        {_base64_to_html_img(img_scatter, "预测vs实际散点图")}
    </div>

    <h2>7. 分组收益明细</h2>
    <table class="table">
        <tr>
            <th>组别</th>
            <th>平均月度收益</th>
            <th>收益描述</th>
        </tr>
"""

    # 添加分组收益表格
    avg_returns = eval_result["group_returns"].mean()
    for group, ret in avg_returns.items():
        css_class = "positive" if ret > 0 else "negative"
        label = f"Q{int(group)}"
        if int(group) == len(avg_returns):
            label += " (Top组 - 做多)"
        elif int(group) == 1:
            label += " (Bottom组 - 做空)"
        html += f"""
        <tr>
            <td>{label}</td>
            <td class="{css_class}">{ret:.4f}</td>
            <td>{"盈利" if ret > 0 else "亏损"}</td>
        </tr>
"""

    html += f"""
    </table>

    <h2>8. 最大回撤详情</h2>
    <table class="table">
        <tr><th>指标</th><th>值</th></tr>
        <tr><td>最大回撤幅度</td><td class="negative">{mdd['max_drawdown']:.2%}</td></tr>
        <tr><td>回撤开始日期</td><td>{mdd['max_drawdown_start']}</td></tr>
        <tr><td>回撤结束日期</td><td>{mdd['max_drawdown_end']}</td></tr>
        <tr><td>恢复天数</td><td>{mdd['recovery_days']}天 {'(未恢复)' if mdd['recovery_days'] < 0 else ''}</td></tr>
    </table>

    <hr style="margin-top:40px;">
    <p style="text-align:center;color:#999;font-size:12px;">
        本报告由 ml-alpha 机器学习选股模型自动生成 | 数据来源: akshare
    </p>
</body>
</html>
"""

    # 写入文件
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\n  [完成] HTML报告已生成: {output_path}")
    return output_path

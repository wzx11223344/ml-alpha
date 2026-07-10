"""
backtest.py - ML选股回测模块

基于机器学习预测结果进行分组回测，包括：
- 按预测分数分5组（quintile分组）
- 做多Top组、做空Bottom组构建多空对冲组合
- 月度调仓
- 组合收益计算
- 换手率计算
- 多空对冲收益

所有收益基于真实行情数据计算。
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

from .data import get_month_end_dates


# ============================================================
# 1. 分组构建
# ============================================================

def assign_groups(
    predictions: pd.DataFrame,
    n_groups: int = 5,
) -> pd.DataFrame:
    """
    按预测分数将股票分为N组

    在每个调仓日（截面），根据模型预测的收益率对股票进行分组。
    第1组（Q1）是预测收益率最低的组，第N组是预测收益率最高的组。

    参数:
        predictions: 预测结果DataFrame，包含 date, ticker, prediction 列
        n_groups: 分组数量，默认5组

    返回:
        添加了 group 列的预测DataFrame，group取值1~n_groups
    """
    result = predictions.copy()

    def assign_group(s):
        """在截面内按预测值分组"""
        # 使用qcut进行等数量分组
        try:
            groups = pd.qcut(s, q=n_groups, labels=False, duplicates="drop")
            # 调整为1~n_groups
            return groups + 1
        except ValueError:
            # 如果分组失败（如预测值全相同），全部归为中间组
            mid = (n_groups + 1) // 2
            return pd.Series(mid, index=s.index)

    # 按日期分组，在截面内进行分组
    result["group"] = result.groupby("date")["prediction"].transform(assign_group)

    return result


# ============================================================
# 2. 分组回测
# ============================================================

def group_backtest(
    predictions: pd.DataFrame,
    panel: pd.DataFrame,
    target_col: str,
    n_groups: int = 5,
) -> Dict[str, pd.DataFrame]:
    """
    分组回测

    在每个调仓日，按模型预测将股票分为N组，计算每组的实际平均收益。
    同时计算多空对冲收益（Top组 - Bottom组）。

    参数:
        predictions: 预测结果DataFrame，包含 date, ticker, prediction 列
        panel: 原始面板数据，包含目标收益率列
        target_col: 目标收益率列名（如 forward_return_5d）
        n_groups: 分组数量

    返回:
        回测结果字典:
        {
            "group_returns": 每组每个调仓日的收益率DataFrame
            "long_short": 多空对冲收益Series
            "cumulative": 累积收益DataFrame
        }
    """
    # 分组
    grouped_preds = assign_groups(predictions, n_groups)

    # 获取每只股票在调仓日的实际未来收益率
    # 从panel中提取目标变量
    target_data = panel[["date", "ticker", target_col]].copy()
    target_data = target_data.rename(columns={target_col: "actual_return"})

    # 合并预测和实际收益
    merged = pd.merge(
        grouped_preds,
        target_data,
        on=["date", "ticker"],
        how="inner",
    )

    # 去除实际收益为NaN的记录
    merged = merged.dropna(subset=["actual_return"])

    # 按日期和组别计算平均收益
    group_returns = merged.groupby(["date", "group"])["actual_return"].mean().unstack("group")

    # 确保列名为整数
    group_returns.columns = [int(c) for c in group_returns.columns]

    # 计算多空对冲收益 = Top组收益 - Bottom组收益
    top_group = group_returns.columns.max()
    bottom_group = group_returns.columns.min()
    long_short = group_returns[top_group] - group_returns[bottom_group]
    long_short.name = "long_short"

    # 计算累积收益
    cumulative = (1 + group_returns).cumprod() - 1
    cumulative_ls = (1 + long_short).cumprod() - 1

    return {
        "group_returns": group_returns,
        "long_short": long_short,
        "cumulative": cumulative,
        "cumulative_long_short": cumulative_ls,
    }


# ============================================================
# 3. 换手率计算
# ============================================================

def calculate_turnover(
    predictions: pd.DataFrame,
    n_groups: int = 5,
) -> pd.Series:
    """
    计算组合换手率

    换手率衡量组合持仓的变化程度：
    turnover = |新持仓权重 - 旧持仓权重| / 2

    参数:
        predictions: 预测结果DataFrame，包含 date, ticker, prediction, group 列
        n_groups: 分组数量

    返回:
        换手率Series（按调仓日索引）
    """
    grouped = predictions.copy()
    if "group" not in grouped.columns:
        grouped = assign_groups(grouped, n_groups)

    # 只看Top组和Bottom组的换手率
    dates = sorted(grouped["date"].unique())
    turnover_records = []

    for i in range(1, len(dates)):
        prev_date = dates[i - 1]
        curr_date = dates[i]

        # 获取前一调仓日和当前调仓日的持仓
        prev_stocks = set(
            grouped[
                (grouped["date"] == prev_date) &
                (grouped["group"] == n_groups)
            ]["ticker"]
        )
        curr_stocks = set(
            grouped[
                (grouped["date"] == curr_date) &
                (grouped["group"] == n_groups)
            ]["ticker"]
        )

        # 计算换手率
        if len(prev_stocks) > 0 and len(curr_stocks) > 0:
            # 换入和换出的股票数量
            bought = len(curr_stocks - prev_stocks)
            sold = len(prev_stocks - curr_stocks)
            avg_holding = (len(prev_stocks) + len(curr_stocks)) / 2
            turnover = (bought + sold) / (2 * avg_holding) if avg_holding > 0 else 0
        else:
            turnover = 0.0

        turnover_records.append({
            "date": curr_date,
            "turnover": turnover,
        })

    return pd.DataFrame(turnover_records).set_index("date")["turnover"]


# ============================================================
# 4. 完整回测流程
# ============================================================

def run_backtest(
    predictions: pd.DataFrame,
    panel: pd.DataFrame,
    target_col: str,
    n_groups: int = 5,
) -> Dict[str, pd.DataFrame]:
    """
    完整的回测流程

    包含分组回测和换手率计算。

    参数:
        predictions: 预测结果DataFrame
        panel: 面板数据
        target_col: 目标收益率列名
        n_groups: 分组数量

    返回:
        完整回测结果字典:
        {
            "group_returns": 分组收益,
            "long_short": 多空对冲收益,
            "cumulative": 累积收益,
            "cumulative_long_short": 多空累积收益,
            "turnover": 换手率,
        }
    """
    print("=" * 60)
    print("开始分组回测")
    print(f"  分组数: {n_groups}")
    print("=" * 60)

    # 分组回测
    bt_result = group_backtest(predictions, panel, target_col, n_groups)

    # 换手率
    turnover = calculate_turnover(predictions, n_groups)

    print(f"\n  [完成] 分组回测完成")
    print(f"  调仓次数: {len(bt_result['group_returns'])}")
    print(f"  平均换手率: {turnover.mean():.2%}")
    print(f"  多空对冲累计收益: {bt_result['cumulative_long_short'].iloc[-1]:.2%}")

    result = {
        "group_returns": bt_result["group_returns"],
        "long_short": bt_result["long_short"],
        "cumulative": bt_result["cumulative"],
        "cumulative_long_short": bt_result["cumulative_long_short"],
        "turnover": turnover,
    }

    return result

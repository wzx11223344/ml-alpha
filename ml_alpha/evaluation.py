"""
evaluation.py - 模型评估模块

提供全面的ML选股模型评估指标，包括：
- 模型IC（Spearman秩相关系数）
- 分组单调性检验
- Top组 vs Bottom组收益差
- 夏普比率
- 最大回撤
- 换手率分析
- 特征重要性排名
- 模型稳定性评估（滚动IC）

所有评估基于真实预测结果和实际收益率计算。
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from typing import Dict, List, Optional, Tuple, Any


# ============================================================
# 1. IC（信息系数）计算
# ============================================================

def calculate_ic(
    predictions: pd.DataFrame,
    actual_returns: pd.DataFrame,
    method: str = "spearman",
) -> pd.Series:
    """
    计算每个截面的IC（信息系数）

    IC衡量模型预测值与实际收益率之间的相关性。
    通常使用Spearman秩相关系数，因为它对异常值不敏感。

    IC > 0 表示模型预测方向正确，IC越高预测能力越强。
    通常 IC > 0.03 被认为是有预测力的因子。

    参数:
        predictions: 预测结果，包含 date, ticker, prediction 列
        actual_returns: 实际收益，包含 date, ticker, actual_return 列
        method: 相关性计算方法，'spearman' 或 'pearson'

    返回:
        每个截面的IC值Series（按日期索引）
    """
    # 合并预测和实际收益
    merged = pd.merge(
        predictions,
        actual_returns,
        on=["date", "ticker"],
        how="inner",
    )
    merged = merged.dropna(subset=["prediction", "actual_return"])

    ic_records = []

    for date, group in merged.groupby("date"):
        if len(group) < 5:
            ic_records.append({"date": date, "ic": np.nan})
            continue

        pred = group["prediction"].values
        actual = group["actual_return"].values

        if method == "spearman":
            ic, _ = spearmanr(pred, actual)
        else:
            ic, _ = pearsonr(pred, actual)

        if np.isnan(ic):
            ic = 0.0

        ic_records.append({"date": date, "ic": ic})

    ic_series = pd.DataFrame(ic_records).set_index("date")["ic"]
    return ic_series


def calculate_ic_summary(ic_series: pd.Series) -> Dict[str, float]:
    """
    计算IC统计摘要

    参数:
        ic_series: IC时间序列

    返回:
        IC统计字典:
        {
            "mean_ic": 平均IC,
            "ic_std": IC标准差,
            "ic_ir": IC信息比率 (mean/std),
            "ic_positive_ratio": IC为正的比例,
            "ic_win_rate": IC胜率 (IC>0的比例),
        }
    """
    ic_clean = ic_series.dropna()

    if len(ic_clean) == 0:
        return {
            "mean_ic": 0.0,
            "ic_std": 0.0,
            "ic_ir": 0.0,
            "ic_positive_ratio": 0.0,
            "ic_win_rate": 0.0,
        }

    mean_ic = ic_clean.mean()
    ic_std = ic_clean.std()
    ic_ir = mean_ic / (ic_std + 1e-10)  # IC信息比率
    ic_positive_ratio = (ic_clean > 0).mean()
    ic_win_rate = ic_positive_ratio  # 胜率 = IC为正的比例

    return {
        "mean_ic": float(mean_ic),
        "ic_std": float(ic_std),
        "ic_ir": float(ic_ir),
        "ic_positive_ratio": float(ic_positive_ratio),
        "ic_win_rate": float(ic_win_rate),
    }


# ============================================================
# 2. 滚动IC
# ============================================================

def calculate_rolling_ic(
    ic_series: pd.Series,
    window: int = 12,
) -> pd.Series:
    """
    计算滚动IC

    用于评估模型预测能力的稳定性。
    滚动IC的波动越小，说明模型越稳定。

    参数:
        ic_series: IC时间序列
        window: 滚动窗口大小

    返回:
        滚动IC均值Series
    """
    return ic_series.rolling(window=window, min_periods=1).mean()


# ============================================================
# 3. 分组单调性检验
# ============================================================

def group_monotonicity_test(
    group_returns: pd.DataFrame,
) -> Dict[str, Any]:
    """
    分组单调性检验

    检验分组收益是否呈现单调递增（或递减）趋势。
    如果模型预测有效，从低分组到高分组的收益应该单调递增。

    参数:
        group_returns: 分组收益DataFrame（行为日期，列为组别1~N）

    返回:
        单调性检验结果字典:
        {
            "is_monotonic": 是否单调,
            "monotonic_ratio": 单调性比例（满足单调的截面比例）,
            "avg_group_returns": 各组平均收益,
            "spread": Top组-Bottom组平均收益差,
        }
    """
    groups = sorted(group_returns.columns)
    n_groups = len(groups)

    # 各组平均收益
    avg_returns = group_returns.mean()

    # 检查每个截面是否单调
    monotonic_count = 0
    total_count = 0

    for date in group_returns.index:
        row = group_returns.loc[date]
        if row.isna().any():
            continue

        total_count += 1
        # 检查是否单调递增
        is_mono = True
        for j in range(n_groups - 1):
            if row.iloc[j] > row.iloc[j + 1]:
                is_mono = False
                break
        if is_mono:
            monotonic_count += 1

    monotonic_ratio = monotonic_count / max(total_count, 1)

    # Top组 - Bottom组收益差
    spread = avg_returns.iloc[-1] - avg_returns.iloc[0]

    return {
        "is_monotonic": monotonic_ratio > 0.5,
        "monotonic_ratio": float(monotonic_ratio),
        "avg_group_returns": avg_returns.to_dict(),
        "spread": float(spread),
    }


# ============================================================
# 4. 风险指标
# ============================================================

def sharpe_ratio(
    returns: pd.Series,
    annualization_factor: float = 12,
    risk_free_rate: float = 0.0,
) -> float:
    """
    计算夏普比率

    Sharpe = (mean_return - risk_free) / std_return * sqrt(annualization_factor)

    参数:
        returns: 收益率序列
        annualization_factor: 年化因子（月度=12，周度=52，日度=252）
        risk_free_rate: 无风险利率（年化）

    返回:
        年化夏普比率
    """
    if len(returns) == 0 or returns.std() == 0:
        return 0.0

    excess_returns = returns - risk_free_rate / annualization_factor
    sharpe = excess_returns.mean() / (returns.std() + 1e-10)
    annualized_sharpe = sharpe * np.sqrt(annualization_factor)

    return float(annualized_sharpe)


def max_drawdown(
    returns: pd.Series,
) -> Dict[str, float]:
    """
    计算最大回撤

    最大回撤衡量从历史最高点到后续最低点的最大跌幅。

    参数:
        returns: 收益率序列

    返回:
        最大回撤结果字典:
        {
            "max_drawdown": 最大回撤幅度,
            "max_drawdown_start": 回撤开始日期,
            "max_drawdown_end": 回撤结束日期,
            "recovery_days": 回撤恢复天数,
        }
    """
    # 计算累积净值
    cumulative = (1 + returns).cumprod()
    # 计算历史最高点
    running_max = cumulative.cummax()
    # 计算回撤
    drawdown = (cumulative - running_max) / running_max

    max_dd = drawdown.min()
    max_dd_end = drawdown.idxmin()

    # 找到回撤开始的日期
    if isinstance(max_dd_end, pd.Timestamp):
        peak_mask = cumulative.loc[:max_dd_end] == running_max.loc[:max_dd_end]
        if peak_mask.any():
            max_dd_start = peak_mask[peak_mask].index[-1]
        else:
            max_dd_start = returns.index[0]
    else:
        max_dd_start = returns.index[0]

    # 检查是否恢复
    recovery_mask = (cumulative.loc[max_dd_end:] >= running_max.loc[max_dd_end])
    if recovery_mask.any():
        recovery_date = recovery_mask[recovery_mask].index[0]
        if isinstance(recovery_date, pd.Timestamp) and isinstance(max_dd_end, pd.Timestamp):
            recovery_days = (recovery_date - max_dd_end).days
        else:
            recovery_days = 0
    else:
        recovery_days = -1  # 未恢复

    return {
        "max_drawdown": float(abs(max_dd)),
        "max_drawdown_start": str(max_dd_start),
        "max_drawdown_end": str(max_dd_end),
        "recovery_days": int(recovery_days),
    }


def calculate_annual_return(
    returns: pd.Series,
    annualization_factor: float = 12,
) -> float:
    """
    计算年化收益率

    参数:
        returns: 收益率序列
        annualization_factor: 年化因子

    返回:
        年化收益率
    """
    if len(returns) == 0:
        return 0.0

    total_return = (1 + returns).prod() - 1
    n_periods = len(returns)
    if n_periods == 0:
        return 0.0

    annual_return = (1 + total_return) ** (annualization_factor / n_periods) - 1
    return float(annual_return)


# ============================================================
# 5. 特征重要性分析
# ============================================================

def analyze_feature_importance(
    models_list: List[Any],
    feature_names: List[str],
) -> pd.DataFrame:
    """
    分析特征重要性

    对滚动窗口训练的多个模型，聚合特征重要性排名。

    参数:
        models_list: 模型列表（滚动窗口训练产生）
        feature_names: 特征名称列表

    返回:
        特征重要性DataFrame，包含列：
        [feature, mean_importance, std_importance, rank]
    """
    all_importance = []

    for model in models_list:
        if hasattr(model, "feature_importances_"):
            importance = model.feature_importances_
        elif hasattr(model, "coef_"):
            importance = np.abs(model.coef_)
        else:
            continue

        if len(importance) == len(feature_names):
            all_importance.append(importance)

    if len(all_importance) == 0:
        return pd.DataFrame(columns=["feature", "mean_importance", "std_importance", "rank"])

    importance_df = pd.DataFrame(all_importance, columns=feature_names)

    result = pd.DataFrame({
        "feature": feature_names,
        "mean_importance": importance_df.mean(),
        "std_importance": importance_df.std(),
    })

    # 按平均重要性排序
    result = result.sort_values("mean_importance", ascending=False)
    result["rank"] = range(1, len(result) + 1)
    result = result.reset_index(drop=True)

    return result


# ============================================================
# 6. 综合评估
# ============================================================

def evaluate_model(
    predictions: pd.DataFrame,
    panel: pd.DataFrame,
    target_col: str,
    backtest_result: Dict[str, pd.DataFrame],
    models_list: List[Any],
    feature_names: List[str],
    annualization_factor: float = 12,
) -> Dict[str, Any]:
    """
    综合模型评估

    整合所有评估指标，生成完整的评估报告。

    参数:
        predictions: 预测结果
        panel: 原始面板数据
        target_col: 目标收益率列名
        backtest_result: 回测结果字典
        models_list: 模型列表
        feature_names: 特征名称列表
        annualization_factor: 年化因子

    返回:
        综合评估结果字典
    """
    print("=" * 60)
    print("开始综合模型评估")
    print("=" * 60)

    # 提取实际收益
    actual_data = panel[["date", "ticker", target_col]].copy()
    actual_data = actual_data.rename(columns={target_col: "actual_return"})

    # 1. IC计算
    print("  [1/6] 计算IC...")
    ic_series = calculate_ic(predictions, actual_data)
    ic_summary = calculate_ic_summary(ic_series)

    # 2. 滚动IC
    print("  [2/6] 计算滚动IC...")
    rolling_ic = calculate_rolling_ic(ic_series, window=6)

    # 3. 分组单调性
    print("  [3/6] 分组单调性检验...")
    group_returns = backtest_result["group_returns"]
    monotonicity = group_monotonicity_test(group_returns)

    # 4. 风险指标（基于多空对冲收益）
    print("  [4/6] 计算风险指标...")
    long_short = backtest_result["long_short"]

    sharpe = sharpe_ratio(long_short, annualization_factor)
    mdd = max_drawdown(long_short)
    annual_ret = calculate_annual_return(long_short, annualization_factor)

    # 5. 换手率
    print("  [5/6] 换手率分析...")
    turnover = backtest_result.get("turnover", pd.Series(dtype=float))
    avg_turnover = float(turnover.mean()) if len(turnover) > 0 else 0.0

    # 6. 特征重要性
    print("  [6/6] 特征重要性分析...")
    feature_importance = analyze_feature_importance(models_list, feature_names)

    print(f"\n{'=' * 60}")
    print("评估结果摘要:")
    print(f"  平均IC:        {ic_summary['mean_ic']:.4f}")
    print(f"  IC信息比率:    {ic_summary['ic_ir']:.4f}")
    print(f"  IC胜率:        {ic_summary['ic_win_rate']:.2%}")
    print(f"  多空年化收益:  {annual_ret:.2%}")
    print(f"  夏普比率:      {sharpe:.2f}")
    print(f"  最大回撤:      {mdd['max_drawdown']:.2%}")
    print(f"  平均换手率:    {avg_turnover:.2%}")
    print(f"  分组单调性:    {monotonicity['monotonic_ratio']:.2%}")
    print(f"{'=' * 60}")

    return {
        "ic_series": ic_series,
        "ic_summary": ic_summary,
        "rolling_ic": rolling_ic,
        "monotonicity": monotonicity,
        "sharpe_ratio": sharpe,
        "max_drawdown": mdd,
        "annual_return": annual_ret,
        "avg_turnover": avg_turnover,
        "feature_importance": feature_importance,
        "group_returns": group_returns,
        "long_short": long_short,
        "cumulative": backtest_result["cumulative"],
        "cumulative_long_short": backtest_result["cumulative_long_short"],
        "turnover": turnover,
    }


# ============================================================
# 7. 预测 vs 实际收益散点数据
# ============================================================

def get_prediction_vs_actual_data(
    predictions: pd.DataFrame,
    panel: pd.DataFrame,
    target_col: str,
) -> pd.DataFrame:
    """
    获取预测值与实际收益的散点图数据

    参数:
        predictions: 预测结果
        panel: 面板数据
        target_col: 目标收益率列名

    返回:
        包含 prediction 和 actual_return 两列的DataFrame
    """
    actual_data = panel[["date", "ticker", target_col]].copy()
    actual_data = actual_data.rename(columns={target_col: "actual_return"})

    merged = pd.merge(
        predictions,
        actual_data,
        on=["date", "ticker"],
        how="inner",
    )
    merged = merged.dropna(subset=["prediction", "actual_return"])

    return merged[["prediction", "actual_return"]]

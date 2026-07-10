"""
models.py - 机器学习模型模块

提供三种回归模型用于预测未来N日股票收益率：
- XGBoost回归：梯度提升树模型，处理非线性关系
- LightGBM回归：轻量级梯度提升树，训练速度快
- Ridge回归：线性回归baseline，防止过拟合

支持：
- 滚动窗口训练（train on past N months, predict next month）
- 时间序列交叉验证（TimeSeriesSplit）
- 超参数配置
- 模型持久化（joblib保存/加载）
- 预测排名输出

注意：模型训练中的随机种子 random_state=42 用于结果可复现性。
"""

import os
import numpy as np
import pandas as pd
import joblib
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

import xgboost as xgb
import lightgbm as lgb


# ============================================================
# 1. 模型超参数配置
# ============================================================

# XGBoost默认超参数
XGBOOST_PARAMS = {
    "n_estimators": 300,        # 树的数量
    "max_depth": 5,             # 树的最大深度
    "learning_rate": 0.05,      # 学习率
    "subsample": 0.8,           # 行采样比例
    "colsample_bytree": 0.8,    # 列采样比例
    "min_child_weight": 5,      # 最小叶子节点权重
    "reg_alpha": 0.1,           # L1正则化
    "reg_lambda": 1.0,         # L2正则化
    "random_state": 42,         # 随机种子（仅用于可复现性，不用于生成业务数据）
    "n_jobs": -1,               # 并行计算
}

# LightGBM默认超参数
LIGHTGBM_PARAMS = {
    "n_estimators": 300,        # 树的数量
    "max_depth": 5,             # 树的最大深度
    "learning_rate": 0.05,      # 学习率
    "num_leaves": 31,           # 叶子节点数
    "subsample": 0.8,           # 行采样比例
    "colsample_bytree": 0.8,    # 列采样比例
    "min_child_samples": 20,    # 叶子节点最小样本数
    "reg_alpha": 0.1,           # L1正则化
    "reg_lambda": 1.0,          # L2正则化
    "random_state": 42,         # 随机种子（仅用于可复现性）
    "n_jobs": -1,               # 并行计算
    "verbose": -1,              # 静默模式
}

# Ridge回归默认超参数
RIDGE_PARAMS = {
    "alpha": 1.0,               # L2正则化强度
    "fit_intercept": True,      # 是否拟合截距
}


# ============================================================
# 2. 模型包装类
# ============================================================

@dataclass
class ModelConfig:
    """模型配置类"""
    model_type: str = "xgboost"             # 模型类型: xgboost, lightgbm, ridge
    horizon: int = 5                        # 预测周期（天）
    train_months: int = 12                  # 训练窗口（月）
    params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """根据模型类型设置默认超参数"""
        if not self.params:
            if self.model_type == "xgboost":
                self.params = XGBOOST_PARAMS.copy()
            elif self.model_type == "lightgbm":
                self.params = LIGHTGBM_PARAMS.copy()
            elif self.model_type == "ridge":
                self.params = RIDGE_PARAMS.copy()


def create_model(config: ModelConfig):
    """
    根据配置创建模型实例

    参数:
        config: 模型配置

    返回:
        模型实例
    """
    if config.model_type == "xgboost":
        return xgb.XGBRegressor(**config.params)
    elif config.model_type == "lightgbm":
        return lgb.LGBMRegressor(**config.params)
    elif config.model_type == "ridge":
        return Ridge(**config.params)
    else:
        raise ValueError(f"不支持的模型类型: {config.model_type}，"
                         f"支持: xgboost, lightgbm, ridge")


# ============================================================
# 3. 模型训练与预测
# ============================================================

def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: ModelConfig,
):
    """
    训练单个模型

    参数:
        X_train: 训练特征
        y_train: 训练目标（未来N日收益率）
        config: 模型配置

    返回:
        训练好的模型
    """
    model = create_model(config)
    model.fit(X_train.values, y_train.values)
    return model


def predict(
    model,
    X: pd.DataFrame,
) -> np.ndarray:
    """
    使用模型进行预测

    参数:
        model: 训练好的模型
        X: 预测特征

    返回:
        预测的收益率数组
    """
    return model.predict(X.values)


def predict_rank(
    model,
    X: pd.DataFrame,
) -> pd.Series:
    """
    预测并对结果进行排名

    参数:
        model: 训练好的模型
        X: 预测特征

    返回:
        预测收益率排名（0~1之间的百分位排名）
    """
    pred = predict(model, X)
    # 转换为百分位排名
    rank = pd.Series(pred).rank(pct=True)
    return rank


def get_feature_importance(
    model,
    feature_names: List[str],
) -> pd.Series:
    """
    获取模型特征重要性

    参数:
        model: 训练好的模型
        feature_names: 特征名称列表

    返回:
        特征重要性Series（按重要性降序排列）
    """
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    elif hasattr(model, "coef_"):
        # 对于线性模型，使用系数绝对值作为重要性
        importance = np.abs(model.coef_)
    else:
        return pd.Series(dtype=float)

    return pd.Series(importance, index=feature_names).sort_values(ascending=False)


# ============================================================
# 4. 时间序列交叉验证
# ============================================================

def time_series_cross_validate(
    X: pd.DataFrame,
    y: pd.Series,
    config: ModelConfig,
    n_splits: int = 5,
) -> Dict[str, List[float]]:
    """
    时间序列交叉验证

    使用TimeSeriesSplit进行交叉验证，确保训练数据始终在验证数据之前，
    避免数据泄露（look-ahead bias）。

    参数:
        X: 特征数据（必须按时间排序）
        y: 目标变量
        config: 模型配置
        n_splits: 交叉验证折数

    返回:
        包含每折验证MSE和IC的字典:
        {
            "mse": [mse1, mse2, ...],
            "ic": [ic1, ic2, ...],
        }
    """
    from scipy.stats import spearmanr

    tscv = TimeSeriesSplit(n_splits=n_splits)
    results = {"mse": [], "ic": []}

    X_values = X.values
    y_values = y.values

    for train_idx, val_idx in tscv.split(X_values):
        X_train, X_val = X_values[train_idx], X_values[val_idx]
        y_train, y_val = y_values[train_idx], y_values[val_idx]

        # 训练模型
        model = create_model(config)
        model.fit(X_train, y_train)

        # 预测
        y_pred = model.predict(X_val)

        # 计算MSE
        mse = mean_squared_error(y_val, y_pred)
        results["mse"].append(mse)

        # 计算Spearman IC
        if len(y_val) > 2:
            ic, _ = spearmanr(y_pred, y_val)
            results["ic"].append(ic if not np.isnan(ic) else 0.0)
        else:
            results["ic"].append(0.0)

    return results


# ============================================================
# 5. 滚动窗口训练
# ============================================================

def rolling_window_train_predict(
    panel: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    config: ModelConfig,
    rebalance_freq: str = "M",
) -> Tuple[pd.DataFrame, List[Any]]:
    """
    滚动窗口训练与预测

    核心逻辑：
    1. 在每个调仓日（月末），使用过去N个月的数据训练模型
    2. 用训练好的模型预测下一个月的收益率
    3. 记录每只股票的预测值

    这种方法避免了数据泄露，模拟真实的投资决策流程。

    参数:
        panel: 面板数据，包含特征列和目标列
        feature_cols: 特征列名列表
        target_col: 目标列名（如 forward_return_5d）
        config: 模型配置
        rebalance_freq: 调仓频率，'M'为月度，'W'为周度

    返回:
        (predictions_df, models_list)
        - predictions_df: 预测结果DataFrame，包含 date, ticker, prediction, rank 列
        - models_list: 每个调仓窗口训练的模型列表
    """
    print("=" * 60)
    print(f"开始滚动窗口训练 [{config.model_type}]")
    print(f"  训练窗口: {config.train_months} 个月")
    print(f"  预测周期: {config.horizon} 天")
    print(f"  调仓频率: {rebalance_freq}")
    print("=" * 60)

    # 确保面板数据按日期排序
    panel = panel.sort_values(["date", "ticker"]).reset_index(drop=True)

    # 获取调仓日期列表
    dates = panel["date"].drop_duplicates().sort_values()

    # 按调仓频率分组，取每组最后一个交易日
    if rebalance_freq == "M":
        rebalance_dates = dates.groupby(dates.dt.to_period("M")).max().tolist()
    elif rebalance_freq == "W":
        rebalance_dates = dates.groupby(dates.dt.to_period("W")).max().tolist()
    else:
        rebalance_dates = dates.tolist()

    # 训练窗口大小（月数转交易日数，近似每月21个交易日）
    train_window_days = config.train_months * 21

    all_predictions = []
    models_list = []

    print(f"  总调仓次数: {len(rebalance_dates)}")

    for i, rebalance_date in enumerate(rebalance_dates):
        # 训练数据：rebalance_date之前 train_window_days 个交易日的数据
        # 注意：必须确保训练数据的目标变量在未来已经实现（避免数据泄露）
        # 即目标变量 forward_return_Nd 在 rebalance_date - N 天之前的数据

        # 预测日数据：rebalance_date 当天的数据
        predict_data = panel[panel["date"] == rebalance_date]

        if len(predict_data) == 0:
            continue

        # 训练数据截止日期：需要减去预测周期（确保训练目标的未来收益已实现）
        # 例如 horizon=5，则在 rebalance_date - 5 天之前的数据才能用于训练
        train_end_date = rebalance_date - pd.Timedelta(days=config.horizon)

        # 训练数据起始日期
        train_start_idx = dates.searchsorted(rebalance_date) - train_window_days
        if train_start_idx < 0:
            # 训练数据不足，跳过
            continue

        train_start_date = dates.iloc[train_start_idx]

        # 获取训练数据
        train_data = panel[
            (panel["date"] >= train_start_date) &
            (panel["date"] <= train_end_date)
        ]

        if len(train_data) < 100:
            # 训练数据太少，跳过
            continue

        # 提取特征和目标
        X_train = train_data[feature_cols].fillna(0)
        y_train = train_data[target_col].fillna(0)

        # 移除目标变量为NaN的行
        valid_mask = train_data[target_col].notna()
        X_train = X_train[valid_mask]
        y_train = y_train[valid_mask]

        if len(X_train) < 50:
            continue

        # 训练模型
        try:
            model = train_model(X_train, y_train, config)
            models_list.append(model)
        except Exception as e:
            print(f"  [警告] 第{i+1}次训练失败: {e}")
            continue

        # 预测
        X_predict = predict_data[feature_cols].fillna(0)
        predictions = predict(model, X_predict)

        # 构建预测结果
        pred_df = pd.DataFrame({
            "date": rebalance_date,
            "ticker": predict_data["ticker"].values,
            "prediction": predictions,
        })

        # 计算截面排名（0~1百分位排名）
        pred_df["rank"] = pred_df["prediction"].rank(pct=True)

        all_predictions.append(pred_df)

        if (i + 1) % 5 == 0 or i == 0:
            print(f"  [{i+1}/{len(rebalance_dates)}] 调仓日期: {rebalance_date.date()}, "
                  f"训练样本: {len(X_train)}, 预测股票: {len(pred_df)}")

    if len(all_predictions) == 0:
        raise ValueError("滚动窗口训练未能产生任何预测结果，请检查数据量是否充足")

    predictions_df = pd.concat(all_predictions, ignore_index=True)

    print(f"\n{'=' * 60}")
    print(f"滚动窗口训练完成:")
    print(f"  调仓次数: {len(all_predictions)}")
    print(f"  训练模型数: {len(models_list)}")
    print(f"  预测记录数: {len(predictions_df)}")
    print(f"{'=' * 60}")

    return predictions_df, models_list


# ============================================================
# 6. 模型持久化
# ============================================================

def save_model(model, filepath: str):
    """
    保存模型到文件

    参数:
        model: 训练好的模型
        filepath: 保存路径
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    print(f"  [完成] 模型已保存到: {filepath}")


def load_model(filepath: str):
    """
    从文件加载模型

    参数:
        filepath: 模型文件路径

    返回:
        加载的模型
    """
    model = joblib.load(filepath)
    print(f"  [完成] 模型已从 {filepath} 加载")
    return model


# ============================================================
# 7. 多模型训练
# ============================================================

def train_all_models(
    panel: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    horizon: int = 5,
    train_months: int = 12,
    rebalance_freq: str = "M",
) -> Dict[str, Tuple[pd.DataFrame, List[Any]]]:
    """
    训练所有模型（XGBoost + LightGBM + Ridge）

    参数:
        panel: 面板数据
        feature_cols: 特征列名
        target_col: 目标列名
        horizon: 预测周期
        train_months: 训练窗口
        rebalance_freq: 调仓频率

    返回:
        模型结果字典:
        {
            "xgboost": (predictions_df, models_list),
            "lightgbm": (predictions_df, models_list),
            "ridge": (predictions_df, models_list),
        }
    """
    results = {}

    for model_type in ["xgboost", "lightgbm", "ridge"]:
        print(f"\n{'=' * 60}")
        print(f"训练模型: {model_type}")
        print(f"{'=' * 60}")

        config = ModelConfig(
            model_type=model_type,
            horizon=horizon,
            train_months=train_months,
        )

        predictions, models = rolling_window_train_predict(
            panel=panel,
            feature_cols=feature_cols,
            target_col=target_col,
            config=config,
            rebalance_freq=rebalance_freq,
        )

        results[model_type] = (predictions, models)

    return results

"""
ml_alpha - 机器学习选股模型包

基于XGBoost/LightGBM的A股多因子选股预测系统。
支持15+特征工程、滚动窗口训练、分组回测、IC评估。

模块:
    data       - 数据获取与清洗（akshare真实数据）
    features   - 多因子特征构建（技术/量价/基本面/动量/波动率）
    models     - ML模型（XGBoost/LightGBM/Ridge）
    backtest   - ML选股分组回测
    evaluation - 模型评估（IC/分组收益/换手率/夏普比率）
    report     - HTML报告与图表生成
"""

__version__ = "1.0.0"
__author__ = "ml-alpha"

# 导出核心模块，方便外部调用
from . import data
from . import features
from . import models
from . import backtest
from . import evaluation
from . import report

__all__ = [
    "data",
    "features",
    "models",
    "backtest",
    "evaluation",
    "report",
]

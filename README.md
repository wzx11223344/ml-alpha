# ML-Alpha: 机器学习选股模型

基于XGBoost/LightGBM的A股多因子选股预测系统。

## 项目简介

ML-Alpha 是一个完整的量化选股框架，使用机器学习模型预测A股未来收益率。系统实现了从数据获取、特征工程、模型训练、分组回测到评估报告的完整流程。

所有数据均来自 akshare 真实接口，不使用任何随机或伪造数据。

## 核心特性

- **20个选股因子**：覆盖技术、量价、基本面、动量、波动率五大类别
- **三种ML模型**：XGBoost、LightGBM、Ridge回归，支持对比分析
- **滚动窗口训练**：train on past N months, predict next month，避免数据泄露
- **分组回测**：5组quintile分组，多空对冲组合
- **全面评估**：IC、ICIR、夏普比率、最大回撤、换手率、分组单调性
- **可视化报告**：5种图表，自包含HTML报告

## 项目结构

```
ml-alpha/
├── predict.py                 # CLI入口
├── ml_alpha/
│   ├── __init__.py            # 包初始化
│   ├── data.py                # 数据获取与特征工程
│   ├── features.py            # 多因子特征构建
│   ├── models.py              # ML模型（XGBoost/LightGBM/Ridge）
│   ├── backtest.py            # ML选股回测
│   ├── evaluation.py          # 模型评估（IC/分组收益/换手率）
│   └── report.py              # HTML报告生成
├── SKILL.md                   # 技能描述文档
├── README.md                  # 本文件
└── requirements.txt           # Python依赖
```

## 快速开始

### 安装

```bash
cd ml-alpha
pip install -r requirements.txt
```

### 运行

```bash
# 使用指定股票池运行XGBoost模型
python predict.py --tickers 600519,000858,601318 --start 20220101 --end 20250101

# 使用沪深300成分股，5日预测
python predict.py --index hs300 --model xgboost --horizon 5

# 训练所有模型并对比
python predict.py --index zz500 --model all --horizon 10
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--tickers` | 股票代码列表（逗号分隔） | - |
| `--index` | 指数简称（hs300/zz500/sz50/zz1000） | - |
| `--start` | 开始日期 | 20200101 |
| `--end` | 结束日期 | 20250101 |
| `--model` | 模型类型 | xgboost |
| `--horizon` | 预测周期（1/5/10/20日） | 5 |
| `--train-months` | 训练窗口（月） | 12 |
| `--n-groups` | 分组数量 | 5 |
| `--output-dir` | 输出目录 | ./output |
| `--save-models` | 保存训练好的模型 | False |

## 因子列表

| 类别 | 因子名称 | 说明 |
|------|----------|------|
| 技术 | ma5_ratio | 收盘价相对5日均线偏离度 |
| 技术 | ma20_ratio | 收盘价相对20日均线偏离度 |
| 技术 | ma60_ratio | 收盘价相对60日均线偏离度 |
| 技术 | rsi_14 | 14日RSI指标 |
| 技术 | macd_hist | MACD直方图 |
| 技术 | boll_pos | 布林带位置 |
| 量价 | volume_ratio | 量比 |
| 量价 | turnover_change | 换手率变化 |
| 量价 | amt_price_ratio | 成交额/量比率 |
| 基本面 | pe_percentile | PE分位数 |
| 基本面 | pb_percentile | PB分位数 |
| 基本面 | roe | 净资产收益率 |
| 基本面 | gross_margin | 毛利率 |
| 动量 | mom_1m | 1月动量 |
| 动量 | mom_3m | 3月动量 |
| 动量 | mom_6m | 6月动量 |
| 动量 | reversal_5d | 5日反转 |
| 波动率 | vol_20d | 20日波动率 |
| 波动率 | skew_20d | 20日偏度 |
| 波动率 | kurt_20d | 20日峰度 |

## 输出文件

运行完成后在 `output/` 目录下生成：

- `report_{model}.html` - HTML评估报告（含5种图表）
- `predictions_{model}.csv` - 预测结果
- `models/{model}_latest.pkl` - 训练好的模型（--save-models）

## 评估指标

| 指标 | 说明 | 参考标准 |
|------|------|----------|
| IC | 预测值与实际收益的Spearman秩相关 | >0.03为有预测力 |
| ICIR | IC信息比率 (mean(IC)/std(IC)) | >0.5为可接受 |
| IC胜率 | IC为正的比例 | >55%为良好 |
| 夏普比率 | 多空组合风险调整收益 | >1.0为良好 |
| 最大回撤 | 历史最大回撤幅度 | <20%为良好 |
| 分组单调性 | 分组收益单调递增比例 | >50%为良好 |

## 技术细节

### 滚动窗口训练

在每个月末调仓日：
1. 取过去N个月（默认12个月）的数据作为训练集
2. 训练ML模型（XGBoost/LightGBM/Ridge）
3. 用模型预测下个月各股票的收益率
4. 按预测值分5组，构建多空组合

训练数据截止日期需减去预测周期（避免使用未实现的目标变量）。

### 截面标准化

所有特征在每个截面日期内进行：
1. 去极值（1%~99%分位数截断）
2. z-score标准化

### 数据来源

所有数据通过 akshare 获取：
- `stock_zh_a_hist`: 个股日线行情（前复权）
- `stock_a_indicator_lg`: 估值指标（PE/PB等）
- `index_stock_cons_csindex`: 指数成分股

## 依赖

- Python 3.9+
- akshare >= 1.12.0
- numpy, pandas, scipy
- scikit-learn, xgboost, lightgbm
- matplotlib, rich, joblib

## 许可证

MIT License

## 免责声明

本工具仅用于学术研究和量化分析，不构成任何投资建议。历史回测结果不代表未来收益。投资有风险，入市需谨慎。

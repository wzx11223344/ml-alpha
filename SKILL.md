---
slug: ml-alpha
displayName: 机器学习选股模型
version: 1.0.0
summary: 基于XGBoost/LightGBM的A股多因子选股预测系统，实现15+特征工程、滚动窗口训练、分组回测、IC评估，支持时间序列交叉验证和特征重要性分析。
tags:
  - finance
  - machine-learning
  - stock-selection
  - xgboost
  - quantitative-finance
license: MIT
---

# 机器学习选股模型 (ML-Alpha)

## 概述

ML-Alpha 是一个基于机器学习的A股多因子选股预测系统。系统使用XGBoost、LightGBM和Ridge回归三种模型，通过15+因子特征工程和滚动窗口训练，预测股票未来收益率并进行分组回测验证。

所有数据均来自 akshare 真实接口，不使用任何随机或伪造数据。

## 核心能力

### 数据层 (data.py)
- 使用 akshare 获取A股日线行情（前复权）
- 获取指数成分股（沪深300/中证500/上证50等）
- 获取基本面数据（PE/PB/ROE等估值与财务指标）
- 面板数据构建（date x stock x features）
- 数据清洗：去极值、截面标准化、前向填充处理停牌

### 特征层 (features.py)
构建20个选股因子，覆盖5大类别：
| 类别 | 因子 | 数量 |
|------|------|------|
| 技术特征 | MA5/MA20/MA60比率、RSI、MACD直方图、布林带位置 | 6 |
| 量价特征 | 量比、换手率变化、成交额/量比率 | 3 |
| 基本面特征 | PE分位数、PB分位数、ROE、毛利率 | 4 |
| 动量特征 | 1月/3月/6月动量、5日反转 | 4 |
| 波动率特征 | 20日波动率、偏度、峰度 | 3 |

所有特征计算后进行截面z-score标准化。

### 模型层 (models.py)
- XGBoost回归模型（梯度提升树，非线性建模）
- LightGBM回归模型（轻量级梯度提升树，训练快速）
- Ridge回归（线性baseline）
- 滚动窗口训练：train on past N months, predict next month
- 时间序列交叉验证（TimeSeriesSplit，避免数据泄露）
- 模型持久化（joblib保存/加载）

### 回测层 (backtest.py)
- 按预测分数分5组（quintile分组）
- 多空对冲组合：做多Top组，做空Bottom组
- 月度调仓
- 组合收益与换手率计算

### 评估层 (evaluation.py)
- IC（Spearman秩相关系数）及IC统计摘要
- 滚动IC评估模型稳定性
- 分组单调性检验
- 夏普比率、最大回撤、年化收益
- 换手率分析
- 特征重要性聚合排名

### 报告层 (report.py)
- 累积收益曲线（各组 + 多空对冲）
- 分组月度收益热力图
- 特征重要性条形图
- 滚动IC曲线
- 预测vs实际收益散点图
- 自包含HTML报告（base64嵌入图片）

## 能力边界说明

1. **数据来源**：仅支持A股数据（通过akshare获取），不支持港股/美股
2. **预测范围**：预测未来1/5/10/20日收益率，不支持高频预测
3. **交易成本**：回测未考虑交易成本（手续费、滑点、冲击成本），实际收益可能低于回测结果
4. **市场环境**：模型基于历史数据训练，在极端市场环境下（如金融危机、政策剧变）可能失效
5. **基本面数据**：财务指标可能存在报告延迟，实际使用中需注意数据时效性
6. **并非投资建议**：本工具仅用于学术研究和量化分析，不构成任何投资建议

## 安装使用

### 环境要求
- Python 3.9+
- 网络连接（akshare需要在线获取数据）

### 安装依赖
```bash
cd ml-alpha
pip install -r requirements.txt
```

### 基本用法

```bash
# 使用指定股票池
python predict.py --tickers 600519,000858,601318 --start 20220101 --end 20250101

# 使用沪深300成分股，XGBoost模型，5日预测周期
python predict.py --index hs300 --model xgboost --horizon 5

# 使用中证500成分股，训练所有模型对比
python predict.py --index zz500 --model all --horizon 10

# 自定义训练窗口
python predict.py --tickers 600519,000858 --model lightgbm --horizon 20 --train-months 18
```

### 参数说明
| 参数 | 说明 | 默认值 |
|------|------|--------|
| --tickers | 股票代码列表（逗号分隔） | - |
| --index | 指数简称（hs300/zz500/sz50/zz1000） | - |
| --start | 开始日期 YYYYMMDD | 20200101 |
| --end | 结束日期 YYYYMMDD | 20250101 |
| --model | 模型类型（xgboost/lightgbm/ridge/all） | xgboost |
| --horizon | 预测周期（1/5/10/20日） | 5 |
| --train-months | 训练窗口月数 | 12 |
| --n-groups | 分组数量 | 5 |
| --output-dir | 输出目录 | ./output |
| --save-models | 保存模型 | False |

## 输出示例

运行完成后，在 output 目录下生成：
```
output/
├── report_xgboost.html        # HTML评估报告（含图表）
├── predictions_xgboost.csv    # 预测结果CSV
├── report_lightgbm.html       # LightGBM报告（--model all时）
├── predictions_lightgbm.csv
├── report_ridge.html          # Ridge报告
├── predictions_ridge.csv
└── models/                    # 模型文件（--save-models时）
    └── xgboost_latest.pkl
```

### 控制台输出示例
```
┌──── 配置信息 ────────────────────────────────────┐
│ 股票池: 指数 hs300                                │
│ 日期范围: 20220101 ~ 20250101                     │
│ 模型: xgboost                                     │
│ 预测周期: 5 天                                     │
│ 训练窗口: 12 个月                                  │
└────────────────────────────────────────────────────┘

评估结果摘要:
  平均IC:        0.0342
  IC信息比率:    0.4521
  IC胜率:        58.3%
  多空年化收益:  12.5%
  夏普比率:      1.23
  最大回撤:      8.7%
  平均换手率:    35.2%
```

## FAQ

### Q1: 获取数据速度很慢怎么办？
A: akshare需要逐只股票请求历史数据，大股票池（如沪深300的300只股票）需要较长时间。建议：
- 使用较少的股票（--tickers 指定少量股票）
- 缩短日期范围
- 每只股票请求间隔0.1秒避免被限流

### Q2: 哪些IC值算好的模型？
A: 通常IC（Spearman秩相关系数）的评判标准：
- IC > 0.05: 优秀，具有较强预测力
- 0.03 < IC < 0.05: 良好，有一定预测力
- 0.01 < IC < 0.03: 一般，预测力较弱
- IC < 0.01: 模型预测力不足

IC信息比率（ICIR = mean(IC)/std(IC)）> 0.5 通常被认为是可以接受的。

### Q3: 滚动窗口训练的原理是什么？
A: 模型在每个调仓日（月末），使用过去N个月的数据训练，然后预测下个月的收益。这模拟了真实的投资决策流程，避免了使用未来数据（look-ahead bias）。例如 train_months=12 表示用过去12个月的数据训练，预测下个月。

### Q4: 如何选择预测周期（horizon）？
A: 预测周期取决于投资策略的持仓时间：
- horizon=1: 日频策略，适合短线交易
- horizon=5: 周频策略，适合波段交易
- horizon=10: 两周策略，适合中短期持仓
- horizon=20: 月频策略，适合中长期持仓
较短的预测周期换手率较高但可能捕捉更多短期信号，较长的周期换手率低但需要更长的数据验证。

### Q5: 模型在熊市中表现如何？
A: ML模型的预测能力在不同市场环境下会有变化。通常在趋势明确的市场中表现较好，在震荡市或急转市场中可能失效。建议关注滚动IC曲线，当IC持续为负时，应考虑暂停使用模型或调整策略。回测结果仅供参考，不保证未来收益。

### Q6: 为什么有些股票的数据获取失败？
A: 可能的原因包括：股票已退市、停牌时间过长、akshare接口暂时不可用、网络问题等。系统会跳过失败的股票并继续处理其他股票，不影响整体流程。

## 技术架构

```
predict.py (CLI)
    │
    ├── data.py ────── akshare获取数据 → 面板数据
    │                    ├── 日线行情（OHLCV）
    │                    ├── 基本面数据（PE/PB）
    │                    └── 数据清洗（去极值/标准化/前向填充）
    │
    ├── features.py ── 构建20个因子 → 特征面板
    │                    ├── 技术因子（MA/RSI/MACD/BOLL）
    │                    ├── 量价因子（量比/换手率/资金流向）
    │                    ├── 基本面因子（PE/PB分位数/ROE/毛利率）
    │                    ├── 动量因子（1m/3m/6m动量/反转）
    │                    └── 波动率因子（std/skew/kurt）
    │
    ├── models.py ──── 滚动窗口训练 → 预测结果
    │                    ├── XGBoost回归
    │                    ├── LightGBM回归
    │                    ├── Ridge回归
    │                    └── TimeSeriesSplit交叉验证
    │
    ├── backtest.py ── 分组回测 → 组合收益
    │                    ├── 5组分组
    │                    ├── 多空对冲
    │                    └── 换手率计算
    │
    ├── evaluation.py ─ 评估指标 → 评估结果
    │                    ├── IC/Spearman相关
    │                    ├── 分组单调性
    │                    ├── Sharpe/MaxDD
    │                    └── 特征重要性
    │
    └── report.py ──── 图表生成 → HTML报告
                         ├── 累积收益曲线
                         ├── 热力图
                         ├── 特征重要性图
                         ├── 滚动IC曲线
                         └── 散点图
```

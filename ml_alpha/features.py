"""
features.py - 多因子特征构建模块

基于真实行情和基本面数据构建15+个选股因子，包括：
- 技术特征（5个）：MA5/MA20/MA60比率、RSI、MACD直方图、布林带位置
- 量价特征（3个）：量价比、换手率变化、资金流向比率
- 基本面特征（4个）：PE分位数、PB分位数、ROE、毛利率
- 动量特征（3个）：1月动量、3月动量、6月动量、反转因子
- 波动率特征（3个）：20日波动率、偏度、峰度

所有特征基于真实akshare数据计算，不使用任何随机/伪造数据。
特征构建完成后进行截面z-score标准化。
"""

import numpy as np
import pandas as pd
from typing import List

from .data import winsorize, standardize


# ============================================================
# 1. 技术特征
# ============================================================

def compute_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算技术分析特征

    参数:
        df: 单只股票的日线数据，按日期排序
            必须包含列: close, high, low, volume

    返回:
        添加了技术特征的DataFrame，新增列：
        - ma5_ratio:   收盘价相对5日均线的偏离度
        - ma20_ratio:  收盘价相对20日均线的偏离度
        - ma60_ratio:  收盘价相对60日均线的偏离度
        - rsi_14:      14日RSI指标
        - macd_hist:   MACD直方图（DIF-DEA）
        - boll_pos:    布林带位置（在上下轨之间的相对位置，0~1）
    """
    result = df.copy()

    close = result["close"]
    high = result["high"]
    low = result["low"]

    # --- 移动平均线比率 ---
    # MA5/MA20/MA60 相对收盘价的偏离度
    ma5 = close.rolling(window=5, min_periods=1).mean()
    ma20 = close.rolling(window=20, min_periods=1).mean()
    ma60 = close.rolling(window=60, min_periods=1).mean()

    result["ma5_ratio"] = (close - ma5) / ma5
    result["ma20_ratio"] = (close - ma20) / ma20
    result["ma60_ratio"] = (close - ma60) / ma60

    # --- RSI（相对强弱指标） ---
    # RSI = 100 - 100/(1 + RS)，其中 RS = 平均涨幅/平均跌幅
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    # 使用指数移动平均计算平均涨幅和跌幅
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()

    # 避免除零
    rs = avg_gain / (avg_loss + 1e-10)
    result["rsi_14"] = 100 - (100 / (1 + rs))

    # --- MACD直方图 ---
    # DIF = EMA(12) - EMA(26)
    # DEA = EMA(DIF, 9)
    # MACD直方图 = 2 * (DIF - DEA)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()
    result["macd_hist"] = 2 * (dif - dea)

    # --- 布林带位置 ---
    # 布林带上下轨 = MA(20) ± 2*std(20)
    # 位置 = (close - 下轨) / (上轨 - 下轨)
    bb_std = close.rolling(window=20, min_periods=20).std()
    bb_upper = ma20 + 2 * bb_std
    bb_lower = ma20 - 2 * bb_std
    bb_width = bb_upper - bb_lower
    # 避免除零
    result["boll_pos"] = (close - bb_lower) / (bb_width + 1e-10)

    return result


# ============================================================
# 2. 量价特征
# ============================================================

def compute_volume_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算量价特征

    参数:
        df: 单只股票的日线数据，按日期排序
            必须包含列: close, volume, amount, turnover_rate

    返回:
        添加了量价特征的DataFrame，新增列：
        - volume_ratio:   量比（当日成交量 / 5日均量）
        - turnover_change: 换手率变化（当日换手率 / 5日均换手率）
        - amt_price_ratio: 成交额/成交量的比率变化（反映大单活跃度）
    """
    result = df.copy()

    volume = result["volume"]

    # --- 量比 ---
    # 量比 = 当日成交量 / 过去5日平均成交量
    vol_ma5 = volume.rolling(window=5, min_periods=1).mean()
    result["volume_ratio"] = volume / (vol_ma5 + 1e-10)

    # --- 换手率变化 ---
    # 换手率变化 = 当日换手率 / 过去5日平均换手率
    if "turnover_rate" in result.columns:
        turnover = result["turnover_rate"]
        turnover_ma5 = turnover.rolling(window=5, min_periods=1).mean()
        result["turnover_change"] = turnover / (turnover_ma5 + 1e-10)
    else:
        # 如果没有换手率数据，用成交量近似
        result["turnover_change"] = result["volume_ratio"]

    # --- 成交额/成交量比率变化 ---
    # 反映大单活跃度：成交额相对成交量的变化趋势
    if "amount" in result.columns:
        # 单笔均额 = 成交额 / 成交量
        avg_trade_size = result["amount"] / (volume + 1e-10)
        avg_trade_ma5 = avg_trade_size.rolling(window=5, min_periods=1).mean()
        result["amt_price_ratio"] = avg_trade_size / (avg_trade_ma5 + 1e-10)
    else:
        result["amt_price_ratio"] = 1.0

    return result


# ============================================================
# 3. 基本面特征
# ============================================================

def compute_fundamental_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算基本面特征

    参数:
        df: 单只股票的数据，包含估值指标（PE/PB等）和财务指标（ROE等）

    返回:
        添加了基本面特征的DataFrame，新增列：
        - pe_percentile:   PE分位数（截面内排名百分位）
        - pb_percentile:   PB分位数（截面内排名百分位）
        - roe:             净资产收益率
        - gross_margin:    毛利率
    """
    result = df.copy()

    # PE分位数将在截面处理阶段计算（因为分位数需要截面比较）
    # 这里先保留原始PE/PB，后续在build_all_features中进行截面分位数计算
    if "pe" in result.columns:
        result["pe_raw"] = result["pe"]
    if "pb" in result.columns:
        result["pb_raw"] = result["pb"]

    # ROE 和 毛利率直接使用原始值（如果存在）
    # 如果不存在，填充为NaN，后续会在截面处理中处理
    if "roe" not in result.columns:
        result["roe"] = np.nan
    if "gross_margin" not in result.columns:
        result["gross_margin"] = np.nan

    return result


# ============================================================
# 4. 动量特征
# ============================================================

def compute_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算动量与反转特征

    参数:
        df: 单只股票的日线数据，按日期排序
            必须包含列: close

    返回:
        添加了动量特征的DataFrame，新增列：
        - mom_1m:   1月动量（过去21日收益率）
        - mom_3m:   3月动量（过去63日收益率）
        - mom_6m:   6月动量（过去126日收益率）
        - reversal_5d: 5日反转（过去5日收益率的负值，反转因子）
    """
    result = df.copy()

    close = result["close"]

    # 1月动量 = 过去21个交易日收益率
    result["mom_1m"] = close.pct_change(periods=21)

    # 3月动量 = 过去63个交易日收益率
    result["mom_3m"] = close.pct_change(periods=63)

    # 6月动量 = 过去126个交易日收益率
    result["mom_6m"] = close.pct_change(periods=126)

    # 5日反转因子 = 过去5日收益率的负值
    # 短期反转效应：近期涨幅大的股票未来可能下跌
    result["reversal_5d"] = -close.pct_change(periods=5)

    return result


# ============================================================
# 5. 波动率特征
# ============================================================

def compute_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算波动率特征

    参数:
        df: 单只股票的日线数据，按日期排序
            必须包含列: close

    返回:
        添加了波动率特征的DataFrame，新增列：
        - vol_20d:    20日收益率波动率（标准差）
        - skew_20d:   20日收益率偏度
        - kurt_20d:   20日收益率峰度
    """
    result = df.copy()

    # 计算日收益率
    ret = result["close"].pct_change()

    # 20日波动率
    result["vol_20d"] = ret.rolling(window=20, min_periods=10).std()

    # 20日偏度
    result["skew_20d"] = ret.rolling(window=20, min_periods=10).skew()

    # 20日峰度
    result["kurt_20d"] = ret.rolling(window=20, min_periods=10).kurt()

    return result


# ============================================================
# 6. 特征构建主函数
# ============================================================

# 所有特征列名列表
FEATURE_COLUMNS = [
    # 技术特征（6个）
    "ma5_ratio",
    "ma20_ratio",
    "ma60_ratio",
    "rsi_14",
    "macd_hist",
    "boll_pos",
    # 量价特征（3个）
    "volume_ratio",
    "turnover_change",
    "amt_price_ratio",
    # 基本面特征（4个）
    "pe_percentile",
    "pb_percentile",
    "roe",
    "gross_margin",
    # 动量特征（4个）
    "mom_1m",
    "mom_3m",
    "mom_6m",
    "reversal_5d",
    # 波动率特征（3个）
    "vol_20d",
    "skew_20d",
    "kurt_20d",
]

# 特征分类
FEATURE_CATEGORIES = {
    "technical": ["ma5_ratio", "ma20_ratio", "ma60_ratio", "rsi_14", "macd_hist", "boll_pos"],
    "volume_price": ["volume_ratio", "turnover_change", "amt_price_ratio"],
    "fundamental": ["pe_percentile", "pb_percentile", "roe", "gross_margin"],
    "momentum": ["mom_1m", "mom_3m", "mom_6m", "reversal_5d"],
    "volatility": ["vol_20d", "skew_20d", "kurt_20d"],
}


def build_all_features(panel: pd.DataFrame) -> pd.DataFrame:
    """
    构建全部特征的主函数

    对面板数据中的每只股票依次计算所有特征，然后进行截面标准化。

    参数:
        panel: 面板数据，包含 date, ticker, close, high, low, volume, amount 等列

    返回:
        添加了所有特征列的面板数据
    """
    print("=" * 60)
    print("开始构建多因子特征")
    print("=" * 60)

    result_parts = []

    # 按股票代码分组，逐只计算特征
    tickers = panel["ticker"].unique()
    for i, ticker in enumerate(tickers):
        if (i + 1) % 50 == 0:
            print(f"  进度: {i+1}/{len(tickers)}")

        stock_data = panel[panel["ticker"] == ticker].sort_values("date").copy()

        # 1. 技术特征
        stock_data = compute_technical_features(stock_data)

        # 2. 量价特征
        stock_data = compute_volume_price_features(stock_data)

        # 3. 基本面特征
        stock_data = compute_fundamental_features(stock_data)

        # 4. 动量特征
        stock_data = compute_momentum_features(stock_data)

        # 5. 波动率特征
        stock_data = compute_volatility_features(stock_data)

        result_parts.append(stock_data)

    panel_features = pd.concat(result_parts, ignore_index=True)
    panel_features = panel_features.sort_values(["date", "ticker"]).reset_index(drop=True)

    # --- 截面处理：计算PE/PB分位数 ---
    # PE分位数：在每个截面日期内，PE值越低越好（低估值的股票排名越高）
    if "pe_raw" in panel_features.columns:
        # 使用排名百分位，PE越低排名越靠前
        panel_features["pe_percentile"] = panel_features.groupby("date")["pe_raw"].rank(pct=True)
        # PE分位数取反：低PE得分高
        panel_features["pe_percentile"] = 1 - panel_features["pe_percentile"]

    if "pb_raw" in panel_features.columns:
        panel_features["pb_percentile"] = panel_features.groupby("date")["pb_raw"].rank(pct=True)
        panel_features["pb_percentile"] = 1 - panel_features["pb_percentile"]

    # 如果没有pe_raw/pb_raw（基本面数据获取失败），用NaN填充
    if "pe_percentile" not in panel_features.columns:
        panel_features["pe_percentile"] = np.nan
    if "pb_percentile" not in panel_features.columns:
        panel_features["pb_percentile"] = np.nan

    # --- 截面去极值和标准化 ---
    # 对所有特征列进行截面z-score标准化
    available_features = [f for f in FEATURE_COLUMNS if f in panel_features.columns]

    print(f"\n  对 {len(available_features)} 个特征进行截面标准化...")

    for col in available_features:
        # 先去极值，再标准化
        panel_features[col] = panel_features.groupby("date")[col].transform(
            lambda s: standardize(winsorize(s))
        )

    # 填充剩余NaN为0（标准化后NaN表示无数据，填充为均值0）
    panel_features[available_features] = panel_features[available_features].fillna(0)

    print(f"\n{'=' * 60}")
    print(f"特征构建完成:")
    print(f"  特征数量: {len(available_features)}")
    print(f"  特征列表: {available_features}")
    print(f"{'=' * 60}")

    return panel_features


def get_feature_columns(panel: pd.DataFrame) -> List[str]:
    """
    获取面板数据中可用的特征列名

    参数:
        panel: 面板数据

    返回:
        特征列名列表
    """
    return [col for col in FEATURE_COLUMNS if col in panel.columns]

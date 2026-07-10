"""
data.py - 数据获取与清洗模块

使用akshare获取A股真实数据，包括：
- 指数成分股（沪深300/中证500等）
- 个股日线行情（开高低收量额）
- 个股基本面数据（PE/PB/ROE等）
- 构建面板数据（panel data: date x stock x features）
- 数据清洗：去极值、标准化、缺失值处理、前向填充

所有数据均来自akshare真实接口，禁止任何伪造数据。
"""

import time
import numpy as np
import pandas as pd
import akshare as ak

from typing import List, Optional, Dict, Union
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn


# ============================================================
# 1. 股票池获取
# ============================================================

# 指数代码映射表（akshare使用的指数代码格式）
INDEX_CODE_MAP = {
    "hs300": "000300",   # 沪深300
    "zz500": "000905",   # 中证500
    "zz1000": "000852",  # 中证1000
    "sz50": "000016",    # 上证50
    "cyb": "399006",     # 创业板指
    "zz2000": "932000",  # 中证2000
}


def get_index_constituents(index: str) -> List[str]:
    """
    获取指数成分股列表

    参数:
        index: 指数简称，如 'hs300', 'zz500', 'sz50'

    返回:
        成分股代码列表，如 ['600519', '000858', ...]
    """
    code = INDEX_CODE_MAP.get(index.lower())
    if code is None:
        raise ValueError(f"不支持的指数: {index}，支持的指数: {list(INDEX_CODE_MAP.keys())}")

    # 使用akshare获取指数成分股
    # 沪深300使用csindex接口
    if index.lower() in ("hs300", "zz500", "zz1000", "sz50", "zz2000"):
        try:
            df = ak.index_stock_cons_csindex(symbol=code)
            tickers = df["成分券代码"].tolist()
            # 标准化为6位代码
            tickers = [str(t).zfill(6) for t in tickers]
            return tickers
        except Exception:
            # 备用接口
            df = ak.index_stock_cons(symbol=code)
            tickers = df["品种代码"].tolist()
            tickers = [str(t).zfill(6) for t in tickers]
            return tickers
    else:
        df = ak.index_stock_cons(symbol=code)
        tickers = df["品种代码"].tolist()
        tickers = [str(t).zfill(6) for t in tickers]
        return tickers


def get_stock_pool(
    index: Optional[str] = None,
    tickers: Optional[List[str]] = None,
) -> List[str]:
    """
    获取股票池

    两种模式：
    1. 通过指数成分股获取（index参数）
    2. 直接指定股票代码列表（tickers参数）

    参数:
        index: 指数简称，如 'hs300'
        tickers: 股票代码列表，如 ['600519', '000858']

    返回:
        股票代码列表
    """
    if tickers is not None and len(tickers) > 0:
        # 标准化代码为6位
        tickers = [str(t).zfill(6) for t in tickers]
        return tickers

    if index is not None:
        return get_index_constituents(index)

    # 默认返回沪深300成分股
    return get_index_constituents("hs300")


# ============================================================
# 2. 日线行情数据获取
# ============================================================

def get_daily_data(
    tickers: List[str],
    start_date: str = "20200101",
    end_date: str = "20250101",
    adjust: str = "qfq",
) -> pd.DataFrame:
    """
    获取多只股票的日线行情数据

    使用akshare的stock_zh_a_hist接口获取前复权日线数据。

    参数:
        tickers: 股票代码列表
        start_date: 开始日期，格式 YYYYMMDD
        end_date: 结束日期，格式 YYYYMMDD
        adjust: 复权方式，'qfq'前复权, 'hfq'后复权, ''不复权

    返回:
        面板数据DataFrame，包含列：
        [date, ticker, open, high, low, close, volume, amount, turnover_rate]
    """
    all_data = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        transient=True,
    ) as progress:
        task = progress.add_task("获取日线行情数据...", total=len(tickers))

        for ticker in tickers:
            progress.update(task, description=f"获取 {ticker} 日线数据...")
            try:
                # 使用akshare获取个股历史行情（前复权）
                df = ak.stock_zh_a_hist(
                    symbol=ticker,
                    period="daily",
                    start_date=start_date,
                    end_date=end_date,
                    adjust=adjust,
                )

                if df is None or len(df) == 0:
                    progress.advance(task)
                    continue

                # 重命名列名为英文，便于后续处理
                df = df.rename(columns={
                    "日期": "date",
                    "开盘": "open",
                    "收盘": "close",
                    "最高": "high",
                    "最低": "low",
                    "成交量": "volume",
                    "成交额": "amount",
                    "振幅": "amplitude",
                    "涨跌幅": "pct_change",
                    "涨跌额": "change",
                    "换手率": "turnover_rate",
                })

                # 添加股票代码列
                df["ticker"] = ticker

                # 确保日期列为datetime类型
                df["date"] = pd.to_datetime(df["date"])

                all_data.append(df)

            except Exception as e:
                # 某些股票可能获取失败（如已退市），跳过即可
                print(f"  [警告] 获取 {ticker} 数据失败: {e}")

            progress.advance(task)

            # 避免请求过快被限流
            time.sleep(0.1)

    if len(all_data) == 0:
        raise ValueError("未能获取任何股票数据，请检查股票代码和日期范围")

    # 合并所有股票数据
    panel = pd.concat(all_data, ignore_index=True)

    # 按日期和股票代码排序
    panel = panel.sort_values(["date", "ticker"]).reset_index(drop=True)

    print(f"  [完成] 成功获取 {panel['ticker'].nunique()} 只股票，共 {len(panel)} 条记录")

    return panel


# ============================================================
# 3. 基本面数据获取
# ============================================================

def get_fundamental_data(
    tickers: List[str],
    dates: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    获取股票基本面数据（PE/PB/ROE等）

    使用akshare的stock_a_indicator_lg接口获取估值指标，
    使用stock_financial_analysis_indicator接口获取财务指标。

    参数:
        tickers: 股票代码列表
        dates: 需要获取的日期列表（可选）

    返回:
        基本面DataFrame，包含列：
        [date, ticker, pe, pe_ttm, pb, ps, dv_ratio, dv_ttm, total_mv]
    """
    all_fund = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        transient=True,
    ) as progress:
        task = progress.add_task("获取基本面数据...", total=len(tickers))

        for ticker in tickers:
            progress.update(task, description=f"获取 {ticker} 基本面数据...")
            try:
                # 使用akshare获取个股估值指标（PE/PB/PS等）
                df = ak.stock_a_indicator_lg(symbol=ticker)

                if df is None or len(df) == 0:
                    progress.advance(task)
                    continue

                # 重命名列
                df = df.rename(columns={
                    "trade_date": "date",
                    "pe": "pe",
                    "pe_ttm": "pe_ttm",
                    "pb": "pb",
                    "ps": "ps",
                    "ps_ttm": "ps_ttm",
                    "dv_ratio": "dividend_yield",
                    "dv_ttm": "dividend_yield_ttm",
                    "total_mv": "total_mv",
                })

                # 确保日期列为datetime类型
                df["date"] = pd.to_datetime(df["date"])
                df["ticker"] = ticker

                all_fund.append(df)

            except Exception as e:
                print(f"  [警告] 获取 {ticker} 基本面数据失败: {e}")

            progress.advance(task)
            time.sleep(0.1)

    if len(all_fund) == 0:
        # 如果全部获取失败，返回空DataFrame
        print("  [警告] 未能获取基本面数据，将使用空值填充")
        return pd.DataFrame(columns=["date", "ticker", "pe", "pe_ttm", "pb", "ps",
                                     "dividend_yield", "dividend_yield_ttm", "total_mv"])

    fund = pd.concat(all_fund, ignore_index=True)
    fund = fund.sort_values(["date", "ticker"]).reset_index(drop=True)

    print(f"  [完成] 成功获取 {fund['ticker'].nunique()} 只股票的基本面数据")

    return fund


def get_financial_indicators(
    tickers: List[str],
) -> pd.DataFrame:
    """
    获取财务分析指标（ROE、毛利率等）

    使用akshare的stock_financial_analysis_indicator接口。

    参数:
        tickers: 股票代码列表

    返回:
        财务指标DataFrame，包含ROE、毛利率等
    """
    all_fin = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        transient=True,
    ) as progress:
        task = progress.add_task("获取财务指标...", total=len(tickers))

        for ticker in tickers:
            progress.update(task, description=f"获取 {ticker} 财务指标...")
            try:
                df = ak.stock_financial_analysis_indicator(symbol=ticker)

                if df is None or len(df) == 0:
                    progress.advance(task)
                    continue

                # 重命名关键列
                rename_map = {}
                for col in df.columns:
                    if "净资产收益率" in col or "ROE" in col.upper():
                        rename_map[col] = "roe"
                    elif "毛利率" in col:
                        rename_map[col] = "gross_margin"
                    elif "净利率" in col:
                        rename_map[col] = "net_margin"
                    elif "资产负债率" in col:
                        rename_map[col] = "debt_ratio"
                    elif "流动比率" in col:
                        rename_map[col] = "current_ratio"
                    elif "速动比率" in col:
                        rename_map[col] = "quick_ratio"

                df = df.rename(columns=rename_map)

                # 确保有日期列
                if "日期" in df.columns:
                    df["date"] = pd.to_datetime(df["日期"])
                elif "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"])

                df["ticker"] = ticker

                # 只保留重命名后的列和必要的列
                keep_cols = ["date", "ticker"]
                for c in ["roe", "gross_margin", "net_margin", "debt_ratio",
                           "current_ratio", "quick_ratio"]:
                    if c in df.columns:
                        keep_cols.append(c)

                df = df[keep_cols]
                all_fin.append(df)

            except Exception as e:
                print(f"  [警告] 获取 {ticker} 财务指标失败: {e}")

            progress.advance(task)
            time.sleep(0.1)

    if len(all_fin) == 0:
        print("  [警告] 未能获取财务指标数据")
        return pd.DataFrame(columns=["date", "ticker", "roe", "gross_margin"])

    fin = pd.concat(all_fin, ignore_index=True)
    fin = fin.sort_values(["date", "ticker"]).reset_index(drop=True)

    print(f"  [完成] 成功获取 {fin['ticker'].nunique()} 只股票的财务指标")

    return fin


# ============================================================
# 4. 数据清洗
# ============================================================

def winsorize(series: pd.Series, limits: tuple = (0.01, 0.99)) -> pd.Series:
    """
    去极值（缩尾处理）

    将超出分位数范围的值截断到边界值。
    例如 limits=(0.01, 0.99) 表示将低于1%分位数的值设为1%分位数值，
    高于99%分位数的值设为99%分位数值。

    参数:
        series: 输入序列
        limits: (下分位数, 上分位数)

    返回:
        去极值后的序列
    """
    lower = series.quantile(limits[0])
    upper = series.quantile(limits[1])
    return series.clip(lower=lower, upper=upper)


def standardize(series: pd.Series) -> pd.Series:
    """
    标准化（z-score）

    将序列转换为均值为0、标准差为1的标准化值。

    参数:
        series: 输入序列

    返回:
        标准化后的序列
    """
    mean = series.mean()
    std = series.std()
    if std == 0 or pd.isna(std):
        return series - mean
    return (series - mean) / std


def cross_sectional_process(
    panel: pd.DataFrame,
    feature_cols: List[str],
    method: str = "winsorize_standardize",
) -> pd.DataFrame:
    """
    截面数据处理（按日期分组）

    对每个截面日期内的特征进行去极值和标准化处理。
    这是多因子选股中的标准流程，确保不同截面之间的特征可比。

    参数:
        panel: 面板数据
        feature_cols: 需要处理的特征列名列表
        method: 处理方法
            - 'winsorize_standardize': 先去极值再标准化（推荐）
            - 'winsorize': 仅去极值
            - 'standardize': 仅标准化

    返回:
        处理后的面板数据
    """
    result = panel.copy()

    for col in feature_cols:
        if col not in result.columns:
            continue

        def process_group(s):
            if method == "winsorize_standardize":
                s = winsorize(s)
                s = standardize(s)
            elif method == "winsorize":
                s = winsorize(s)
            elif method == "standardize":
                s = standardize(s)
            return s

        # 按日期分组进行截面处理
        result[col] = result.groupby("date")[col].transform(process_group)

    return result


def forward_fill_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """
    面板数据前向填充

    对每只股票按时间排序后进行前向填充，处理停牌导致的缺失值。
    停牌期间使用停牌前最后一天的数据填充。

    参数:
        panel: 面板数据

    返回:
        前向填充后的面板数据
    """
    result = panel.copy()
    # 按股票代码分组，按日期排序后前向填充
    result = result.sort_values(["ticker", "date"])
    # 对数值列进行前向填充
    numeric_cols = result.select_dtypes(include=[np.number]).columns.tolist()
    result[numeric_cols] = result.groupby("ticker")[numeric_cols].ffill()
    # 后向填充剩余的NaN（通常出现在序列开头）
    result[numeric_cols] = result.groupby("ticker")[numeric_cols].bfill()
    return result


def clean_panel_data(panel: pd.DataFrame) -> pd.DataFrame:
    """
    面板数据清洗主函数

    执行以下清洗步骤：
    1. 去除成交量为0的交易日（停牌日）
    2. 前向填充缺失值
    3. 去除全NaN的行

    参数:
        panel: 原始面板数据

    返回:
        清洗后的面板数据
    """
    result = panel.copy()

    # 确保日期列格式正确
    result["date"] = pd.to_datetime(result["date"])

    # 去除成交量为0或NaN的记录（停牌日）
    if "volume" in result.columns:
        result = result[result["volume"].notna() & (result["volume"] > 0)]

    # 前向填充处理停牌导致的缺失值
    result = forward_fill_panel(result)

    # 去除仍然有NaN的关键列的行
    key_cols = ["close", "volume"]
    for col in key_cols:
        if col in result.columns:
            result = result[result[col].notna()]

    # 重置索引
    result = result.reset_index(drop=True)

    print(f"  [完成] 数据清洗完成，剩余 {len(result)} 条记录")
    return result


# ============================================================
# 5. 面板数据构建
# ============================================================

def build_panel(
    tickers: List[str],
    start_date: str = "20200101",
    end_date: str = "20250101",
    include_fundamentals: bool = True,
) -> pd.DataFrame:
    """
    构建完整的面板数据

    整合日线行情和基本面数据，构建统一的面板数据结构。

    参数:
        tickers: 股票代码列表
        start_date: 开始日期
        end_date: 结束日期
        include_fundamentals: 是否包含基本面数据

    返回:
        完整的面板数据DataFrame
    """
    print("=" * 60)
    print("开始构建面板数据")
    print("=" * 60)

    # 1. 获取日线行情数据
    print("\n[步骤1/3] 获取日线行情数据...")
    daily = get_daily_data(tickers, start_date, end_date)

    # 2. 数据清洗
    print("\n[步骤2/3] 数据清洗...")
    daily = clean_panel_data(daily)

    # 3. 获取基本面数据并合并
    if include_fundamentals:
        print("\n[步骤3/3] 获取基本面数据...")
        try:
            fund = get_fundamental_data(tickers)
            if len(fund) > 0:
                # 合并基本面数据到日线数据
                # 使用merge_asof进行时间对齐（使用最近的基本面数据）
                fund = fund.sort_values(["ticker", "date"])
                daily = daily.sort_values(["ticker", "date"])

                # 为每只股票合并估值数据
                merged_parts = []
                for ticker in daily["ticker"].unique():
                    d_sub = daily[daily["ticker"] == ticker].sort_values("date")
                    f_sub = fund[fund["ticker"] == ticker].sort_values("date")

                    if len(f_sub) > 0:
                        # 使用merge_asof进行时间对齐
                        merged = pd.merge_asof(
                            d_sub,
                            f_sub.drop(columns=["ticker"]),
                            on="date",
                            direction="backward",
                        )
                        merged_parts.append(merged)
                    else:
                        merged_parts.append(d_sub)

                panel = pd.concat(merged_parts, ignore_index=True)
            else:
                panel = daily
        except Exception as e:
            print(f"  [警告] 基本面数据获取失败: {e}，将仅使用行情数据")
            panel = daily
    else:
        panel = daily

    # 确保数据按日期和股票代码排序
    panel = panel.sort_values(["date", "ticker"]).reset_index(drop=True)

    print(f"\n{'=' * 60}")
    print(f"面板数据构建完成:")
    print(f"  股票数量: {panel['ticker'].nunique()}")
    print(f"  日期范围: {panel['date'].min().date()} ~ {panel['date'].max().date()}")
    print(f"  总记录数: {len(panel)}")
    print(f"{'=' * 60}")

    return panel


# ============================================================
# 6. 收益率计算
# ============================================================

def calculate_returns(
    panel: pd.DataFrame,
    horizons: List[int] = [1, 5, 10, 20],
) -> pd.DataFrame:
    """
    计算未来N日收益率

    为每只股票计算不同预测周期（horizon）的未来收益率，
    作为机器学习模型的目标变量。

    参数:
        panel: 面板数据，必须包含 close 列
        horizons: 预测周期列表，如 [1, 5, 10, 20]

    返回:
        添加了收益率列的面板数据
        新增列名格式: forward_return_{N}d
    """
    result = panel.copy()
    result = result.sort_values(["ticker", "date"])

    for h in horizons:
        col_name = f"forward_return_{h}d"
        # 计算未来N日收益率: (close[t+N] / close[t]) - 1
        result[col_name] = result.groupby("ticker")["close"].pct_change(h).shift(-h)

    return result


def get_month_end_dates(panel: pd.DataFrame) -> List[pd.Timestamp]:
    """
    获取面板数据中的月末日期列表

    用于月度调仓回测，识别每个月的最后一个交易日。

    参数:
        panel: 面板数据

    返回:
        月末日期列表（按时间排序）
    """
    dates = panel["date"].drop_duplicates().sort_values()
    # 转换为月末标识
    month_end = dates.groupby(dates.dt.to_period("M")).max()
    return sorted(month_end.tolist())

#!/usr/bin/env python3
"""
predict.py - ML选股模型 CLI入口

命令行接口，用于运行完整的机器学习选股流程：
1. 数据获取（akshare真实数据）
2. 特征工程（15+因子）
3. 模型训练（滚动窗口）
4. 分组回测
5. 模型评估
6. HTML报告生成

使用示例:
    # 指定股票池
    python predict.py --tickers 600519,000858,601318 --start 20220101 --end 20250101

    # 使用指数成分股
    python predict.py --index hs300 --model xgboost --horizon 5

    # 训练所有模型
    python predict.py --index zz500 --model all --horizon 10

    # 自定义训练参数
    python predict.py --tickers 600519,000858 --model lightgbm --horizon 20 --train-months 18
"""

import os
import sys
import argparse
import time
import warnings
from typing import Dict, List

import numpy as np
import pandas as pd

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

# 将当前目录加入路径（确保从项目根目录运行时能找到ml_alpha包）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ml_alpha.data import get_stock_pool, build_panel, calculate_returns
from ml_alpha.features import build_all_features, get_feature_columns, FEATURE_COLUMNS
from ml_alpha.models import ModelConfig, rolling_window_train_predict, train_all_models
from ml_alpha.backtest import run_backtest
from ml_alpha.evaluation import evaluate_model, get_prediction_vs_actual_data
from ml_alpha.report import generate_html_report

# 抑制不必要的警告
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

console = Console()


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="ML选股模型 - 基于XGBoost/LightGBM的A股多因子选股预测系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python predict.py --tickers 600519,000858,601318 --start 20220101 --end 20250101
  python predict.py --index hs300 --model xgboost --horizon 5
  python predict.py --index zz500 --model all --horizon 10
        """,
    )

    # 股票池参数
    pool_group = parser.add_mutually_exclusive_group(required=True)
    pool_group.add_argument(
        "--tickers",
        type=str,
        default=None,
        help="股票代码列表，逗号分隔，如 600519,000858,601318",
    )
    pool_group.add_argument(
        "--index",
        type=str,
        default=None,
        help="指数简称，如 hs300(沪深300), zz500(中证500), sz50(上证50), zz1000(中证1000)",
    )

    # 日期参数
    parser.add_argument(
        "--start",
        type=str,
        default="20200101",
        help="开始日期，格式 YYYYMMDD，默认 20200101",
    )
    parser.add_argument(
        "--end",
        type=str,
        default="20250101",
        help="结束日期，格式 YYYYMMDD，默认 20250101",
    )

    # 模型参数
    parser.add_argument(
        "--model",
        type=str,
        default="xgboost",
        choices=["xgboost", "lightgbm", "ridge", "all"],
        help="模型类型: xgboost, lightgbm, ridge, all，默认 xgboost",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=5,
        choices=[1, 5, 10, 20],
        help="预测周期（天）: 1, 5, 10, 20，默认 5",
    )
    parser.add_argument(
        "--train-months",
        type=int,
        default=12,
        help="训练窗口（月），默认 12",
    )
    parser.add_argument(
        "--n-groups",
        type=int,
        default=5,
        help="分组数量，默认 5",
    )

    # 输出参数
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="输出目录，默认 ./output",
    )
    parser.add_argument(
        "--save-models",
        action="store_true",
        help="是否保存训练好的模型",
    )

    return parser.parse_args()


def print_banner(console: Console, args):
    """打印启动横幅"""
    console.print(Panel.fit(
        f"[bold blue]ML选股模型 - 机器学习选股预测系统[/bold blue]\n"
        f"[cyan]版本: 1.0.0[/cyan]\n"
        f"{'─' * 50}\n"
        f"股票池: {args.tickers or f'指数 {args.index}'}\n"
        f"日期范围: {args.start} ~ {args.end}\n"
        f"模型: {args.model}\n"
        f"预测周期: {args.horizon} 天\n"
        f"训练窗口: {args.train_months} 个月\n"
        f"分组数: {args.n_groups}",
        title="配置信息",
        border_style="blue",
    ))


def print_results(console: Console, eval_result: Dict, model_name: str):
    """打印评估结果表格"""
    ic_summary = eval_result["ic_summary"]
    mdd = eval_result["max_drawdown"]
    monotonicity = eval_result["monotonicity"]

    table = Table(title=f"{model_name} 模型评估结果", show_header=True, header_style="bold blue")
    table.add_column("指标", style="cyan", width=25)
    table.add_column("值", style="white", width=20)
    table.add_column("评价", style="green", width=15)

    # IC指标
    ic_rating = "优秀" if ic_summary["mean_ic"] > 0.05 else "良好" if ic_summary["mean_ic"] > 0.03 else "一般"
    table.add_row("平均IC", f"{ic_summary['mean_ic']:.4f}", ic_rating)

    table.add_row("IC信息比率", f"{ic_summary['ic_ir']:.4f}",
                  "优秀" if ic_summary["ic_ir"] > 0.5 else "良好" if ic_summary["ic_ir"] > 0.3 else "一般")

    table.add_row("IC胜率", f"{ic_summary['ic_win_rate']:.1%}",
                  "优秀" if ic_summary["ic_win_rate"] > 0.6 else "良好" if ic_summary["ic_win_rate"] > 0.5 else "一般")

    # 收益指标
    table.add_row("多空年化收益", f"{eval_result['annual_return']:.2%}",
                  "优秀" if eval_result["annual_return"] > 0.1 else "良好" if eval_result["annual_return"] > 0.05 else "一般")

    table.add_row("夏普比率", f"{eval_result['sharpe_ratio']:.2f}",
                  "优秀" if eval_result["sharpe_ratio"] > 1.5 else "良好" if eval_result["sharpe_ratio"] > 1.0 else "一般")

    table.add_row("最大回撤", f"{mdd['max_drawdown']:.2%}",
                  "优秀" if mdd["max_drawdown"] < 0.1 else "良好" if mdd["max_drawdown"] < 0.2 else "一般")

    # 换手率
    table.add_row("平均换手率", f"{eval_result['avg_turnover']:.1%}",
                  "优秀" if eval_result["avg_turnover"] < 0.3 else "良好" if eval_result["avg_turnover"] < 0.5 else "偏高")

    # 单调性
    table.add_row("分组单调性", f"{monotonicity['monotonic_ratio']:.1%}",
                  "优秀" if monotonicity["monotonic_ratio"] > 0.6 else "良好" if monotonicity["monotonic_ratio"] > 0.4 else "一般")

    console.print(table)


def run_single_model(
    model_type: str,
    panel: pd.DataFrame,
    feature_cols: List,
    target_col: str,
    horizon: int,
    train_months: int,
    n_groups: int,
    output_dir: str,
    save_models: bool,
    console: Console,
):
    """运行单个模型的完整流程"""
    console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
    console.print(f"[bold cyan]训练模型: {model_type}[/bold cyan]")
    console.print(f"[bold cyan]{'='*60}[/bold cyan]")

    # 1. 模型配置
    config = ModelConfig(
        model_type=model_type,
        horizon=horizon,
        train_months=train_months,
    )

    # 2. 滚动窗口训练
    console.print("\n[yellow][步骤1/4] 滚动窗口训练...[/yellow]")
    predictions, models_list = rolling_window_train_predict(
        panel=panel,
        feature_cols=feature_cols,
        target_col=target_col,
        config=config,
        rebalance_freq="M",
    )

    # 3. 分组回测
    console.print("\n[yellow][步骤2/4] 分组回测...[/yellow]")
    backtest_result = run_backtest(predictions, panel, target_col, n_groups)

    # 4. 模型评估
    console.print("\n[yellow][步骤3/4] 模型评估...[/yellow]")
    eval_result = evaluate_model(
        predictions=predictions,
        panel=panel,
        target_col=target_col,
        backtest_result=backtest_result,
        models_list=models_list,
        feature_names=feature_cols,
    )

    # 5. 生成HTML报告
    console.print("\n[yellow][步骤4/4] 生成HTML报告...[/yellow]")
    report_path = os.path.join(output_dir, f"report_{model_type}.html")
    generate_html_report(
        eval_result=eval_result,
        predictions=predictions,
        panel=panel,
        target_col=target_col,
        model_name=model_type.upper(),
        output_path=report_path,
    )

    # 打印结果
    print_results(console, eval_result, model_type.upper())

    # 保存模型
    if save_models and len(models_list) > 0:
        from ml_alpha.models import save_model
        model_dir = os.path.join(output_dir, "models")
        os.makedirs(model_dir, exist_ok=True)
        save_model(models_list[-1], os.path.join(model_dir, f"{model_type}_latest.pkl"))

    # 保存预测结果
    pred_path = os.path.join(output_dir, f"predictions_{model_type}.csv")
    predictions.to_csv(pred_path, index=False)
    console.print(f"\n[green]预测结果已保存: {pred_path}[/green]")
    console.print(f"[green]HTML报告已生成: {report_path}[/green]")

    return eval_result


def main():
    """主函数"""
    # 解析参数
    args = parse_args()

    # 打印横幅
    print_banner(console, args)

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 获取股票池
    console.print("\n[bold yellow]获取股票池...[/bold yellow]")
    if args.tickers:
        tickers = [t.strip() for t in args.tickers.split(",")]
    else:
        tickers = get_stock_pool(index=args.index)
    console.print(f"  股票池: {len(tickers)} 只股票")

    # 限制股票数量（避免请求过多）
    if len(tickers) > 100:
        console.print(f"  [yellow]股票数量较多({len(tickers)})，为演示取前100只[/yellow]")
        tickers = tickers[:100]

    # 构建面板数据
    console.print("\n[bold yellow]构建面板数据...[/bold yellow]")
    panel = build_panel(
        tickers=tickers,
        start_date=args.start,
        end_date=args.end,
        include_fundamentals=True,
    )

    # 计算未来收益率
    console.print("\n[bold yellow]计算未来收益率...[/bold yellow]")
    target_col = f"forward_return_{args.horizon}d"
    panel = calculate_returns(panel, horizons=[args.horizon])

    # 构建特征
    console.print("\n[bold yellow]构建多因子特征...[/bold yellow]")
    panel = build_all_features(panel)

    # 获取特征列
    feature_cols = get_feature_columns(panel)
    console.print(f"  特征数量: {len(feature_cols)}")

    # 运行模型
    if args.model == "all":
        # 训练所有模型
        all_results = {}
        for model_type in ["xgboost", "lightgbm", "ridge"]:
            eval_result = run_single_model(
                model_type=model_type,
                panel=panel,
                feature_cols=feature_cols,
                target_col=target_col,
                horizon=args.horizon,
                train_months=args.train_months,
                n_groups=args.n_groups,
                output_dir=args.output_dir,
                save_models=args.save_models,
                console=console,
            )
            all_results[model_type] = eval_result

        # 打印模型对比
        console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
        console.print("[bold cyan]模型对比[/bold cyan]")
        console.print(f"[bold cyan]{'='*60}[/bold cyan]")

        compare_table = Table(title="模型对比", show_header=True, header_style="bold blue")
        compare_table.add_column("指标", style="cyan")
        for mt in ["xgboost", "lightgbm", "ridge"]:
            compare_table.add_column(mt.upper(), style="white", justify="center")

        metrics = [
            ("平均IC", "mean_ic", ".4f"),
            ("IC信息比率", "ic_ir", ".4f"),
            ("IC胜率", "ic_win_rate", ".1%"),
            ("多空年化收益", "annual_return", ".2%"),
            ("夏普比率", "sharpe_ratio", ".2f"),
            ("最大回撤", "max_drawdown", ".2%"),
            ("平均换手率", "avg_turnover", ".1%"),
        ]

        for label, key, fmt in metrics:
            row = [label]
            for mt in ["xgboost", "lightgbm", "ridge"]:
                val = all_results[mt]
                if key == "max_drawdown":
                    row.append(f"{val['max_drawdown']['max_drawdown']:{fmt}}")
                else:
                    row.append(f"{val[key]:{fmt}}")
            compare_table.add_row(*row)

        console.print(compare_table)

    else:
        # 训练单个模型
        run_single_model(
            model_type=args.model,
            panel=panel,
            feature_cols=feature_cols,
            target_col=target_col,
            horizon=args.horizon,
            train_months=args.train_months,
            n_groups=args.n_groups,
            output_dir=args.output_dir,
            save_models=args.save_models,
            console=console,
        )

    console.print(f"\n[bold green]全部完成！输出目录: {args.output_dir}[/bold green]")


if __name__ == "__main__":
    main()

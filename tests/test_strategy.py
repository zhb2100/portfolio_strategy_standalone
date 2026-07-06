from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from portfolio_strategy_standalone.backtest import run_portfolio_backtest
from portfolio_strategy_standalone.cli import _describe_rebalance_schedule, _print_backtest_summary
from portfolio_strategy_standalone.config import BacktestConfig, ClusterConfig, PortfolioConfig
from portfolio_strategy_standalone.portfolio import _attack_score, _base_score, _defense_score


def test_describe_rebalance_schedule_is_monthly_and_weekly() -> None:
    assert _describe_rebalance_schedule(BacktestConfig()) == "调仓节奏: 底仓=M, 防御仓=M, 进攻仓=W-FRI"


def test_backtest_writes_outputs_and_reports_rebalance_types(tmp_path: Path, capsys) -> None:
    clean_daily = _build_clean_daily_frame()
    summary = run_portfolio_backtest(
        clean_daily,
        tmp_path / "strategy",
        BacktestConfig(transaction_cost_bps=5.0, min_listing_days=60, min_avg_amount_20=30_000_000),
        ClusterConfig(corr_threshold=0.99, min_overlap_days=20),
        PortfolioConfig(base_slots=2, defense_slots=2, attack_slots=2),
    )

    output_dir = tmp_path / "strategy"
    assert (output_dir / "backtest_nav.csv").exists()
    assert (output_dir / "backtest_positions.csv").exists()
    assert (output_dir / "backtest_trades.csv").exists()
    assert (output_dir / "backtest_summary.json").exists()
    assert "annual_return" in summary
    assert summary["rebalance_count"] > 0

    trades = pd.read_csv(output_dir / "backtest_trades.csv")
    assert {"base_defense", "attack"}.issubset(set(trades["rebalance_type"].unique()))

    positions = pd.read_csv(output_dir / "backtest_positions.csv")
    assert {"base", "defense", "attack"}.issubset(set(positions["pool"].unique()))
    assert {"base_weight", "defense_weight", "attack_weight", "total_weight"}.issubset(set(positions.columns))

    _print_backtest_summary(summary, output_dir, 1)
    out = capsys.readouterr().out
    assert "年度收益:" in out
    assert "最近调仓" in out


def test_three_pool_scores_follow_the_expected_momentum_ordering() -> None:
    frame = pd.DataFrame(
        [
            {"momentum_120": 0.50, "momentum_60": 0.40, "close_vs_sma200": 0.20, "avg_amount_20": 3.0, "volatility_120": 0.10, "momentum_20": 0.10, "momentum_acceleration": 0.10, "drawdown_20": 0.05},
            {"momentum_120": 0.30, "momentum_60": 0.25, "close_vs_sma200": 0.15, "avg_amount_20": 2.0, "volatility_120": 0.12, "momentum_20": 0.35, "momentum_acceleration": 0.30, "drawdown_20": 0.04},
            {"momentum_120": 0.10, "momentum_60": 0.15, "close_vs_sma200": 0.05, "avg_amount_20": 1.0, "volatility_120": 0.15, "momentum_20": 0.60, "momentum_acceleration": 0.55, "drawdown_20": 0.02},
        ]
    )
    base_order = _base_score(frame).sort_values(ascending=False).index.tolist()
    defense_order = _defense_score(frame).sort_values(ascending=False).index.tolist()
    attack_order = _attack_score(frame).sort_values(ascending=False).index.tolist()
    assert base_order == [0, 1, 2]
    assert defense_order == [0, 1, 2]
    assert attack_order == [2, 1, 0]


def _build_clean_daily_frame() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=220)
    rows: list[dict[str, object]] = []

    symbol_specs = [
        ("510300", "沪深300ETF华夏", "broad_based", 3.0, 0.0020, 120_000_000),
        ("512760", "芯片ETF", "industry", 1.5, 0.0030, 110_000_000),
        ("518880", "黄金ETF", "commodity", 4.0, 0.0004, 100_000_000),
        ("511010", "国债ETF", "bond", 1.0, 0.0003, 100_000_000),
    ]

    for symbol, name, category, start, slope, amount in symbol_specs:
        closes = start + np.arange(len(dates)) * slope
        for date, close in zip(dates, closes, strict=False):
            rows.append(
                {
                    "date": date.date(),
                    "symbol": symbol,
                    "code": symbol,
                    "name": name,
                    "category": category,
                    "dedup_key": symbol,
                    "close": float(close),
                    "open": float(close),
                    "high": float(close + 0.01),
                    "low": float(max(close - 0.01, 0.01)),
                    "amount": float(amount),
                    "zero_turnover": False,
                    "suspicious_jump": False,
                }
            )
    return pd.DataFrame(rows)

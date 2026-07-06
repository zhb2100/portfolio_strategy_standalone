from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

from .backtest import annual_returns_from_nav, run_portfolio_backtest_from_files
from .config import BacktestConfig, ClusterConfig, PortfolioConfig


def main() -> None:
    defaults = BacktestConfig()
    portfolio_defaults = PortfolioConfig()
    parser = argparse.ArgumentParser(description="Standalone base/defense monthly and attack weekly ETF strategy.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    backtest = subparsers.add_parser("backtest", help="Run the standalone strategy backtest.")
    backtest.add_argument("--clean-daily-csv", type=Path, default=Path("clean_etf_daily.csv"))
    backtest.add_argument("--output-dir", type=Path, default=Path("backtest"))
    backtest.add_argument("--start-date")
    backtest.add_argument("--end-date")
    backtest.add_argument("--transaction-cost-bps", type=float, default=defaults.transaction_cost_bps)
    backtest.add_argument("--trailing-stop-pct", type=float, default=defaults.trailing_stop_pct)
    backtest.add_argument("--slippage-bps", type=float, default=defaults.slippage_bps)
    backtest.add_argument("--impact-bps", type=float, default=defaults.impact_bps)
    backtest.add_argument("--impact-participation", type=float, default=defaults.impact_participation)
    backtest.add_argument("--min-listing-days", type=int, default=defaults.min_listing_days)
    backtest.add_argument("--min-avg-amount-20", type=float, default=defaults.min_avg_amount_20)
    backtest.add_argument("--min-valid-ratio-60", type=float, default=defaults.min_valid_ratio_60)
    _add_portfolio_args(backtest, portfolio_defaults)
    backtest.add_argument("--print-recent-rebalances", type=int, default=5)

    args = parser.parse_args()
    if args.command == "backtest":
        backtest_config = BacktestConfig(
            start_date=args.start_date,
            end_date=args.end_date,
            transaction_cost_bps=args.transaction_cost_bps,
            trailing_stop_pct=args.trailing_stop_pct,
            slippage_bps=args.slippage_bps,
            impact_bps=args.impact_bps,
            impact_participation=args.impact_participation,
            min_listing_days=args.min_listing_days,
            min_avg_amount_20=args.min_avg_amount_20,
            min_valid_ratio_60=args.min_valid_ratio_60,
        )
        cluster_config = ClusterConfig()
        portfolio_config = PortfolioConfig(
            enabled=True,
            base_slots=args.base_slots,
            defense_slots=args.defense_slots,
            attack_slots=args.attack_slots,
            bull_base_weight=args.bull_base_weight,
            bull_defense_weight=args.bull_defense_weight,
            bull_attack_weight=args.bull_attack_weight,
            neutral_base_weight=args.neutral_base_weight,
            neutral_defense_weight=args.neutral_defense_weight,
            neutral_attack_weight=args.neutral_attack_weight,
            bear_base_weight=args.bear_base_weight,
            bear_defense_weight=args.bear_defense_weight,
            bear_attack_weight=args.bear_attack_weight,
            risk_on_breadth_threshold=args.risk_on_breadth_threshold,
            market_regime_confirm_weeks=args.market_regime_confirm_weeks,
            market_bull_confirm_weeks=args.market_bull_confirm_weeks,
            market_bull_trend_threshold=args.market_bull_trend_threshold,
            market_fast_trend_threshold=args.market_fast_trend_threshold,
            local_fast_breadth_threshold=args.local_fast_breadth_threshold,
            local_fast_trend_threshold=args.local_fast_trend_threshold,
            market_volatility_veto_threshold=args.market_volatility_veto_threshold,
            bear_defensive_boost=args.bear_defensive_boost,
        )
        print(_describe_rebalance_schedule(backtest_config))
        summary = run_portfolio_backtest_from_files(
            args.clean_daily_csv,
            args.output_dir,
            backtest_config,
            cluster_config,
            portfolio_config,
        )
        _print_backtest_summary(summary, args.output_dir, args.print_recent_rebalances)
def _add_portfolio_args(parser: argparse.ArgumentParser, defaults: PortfolioConfig) -> None:
    parser.add_argument("--base-slots", type=int, default=defaults.base_slots)
    parser.add_argument("--defense-slots", type=int, default=defaults.defense_slots)
    parser.add_argument("--attack-slots", type=int, default=defaults.attack_slots)
    parser.add_argument("--bull-base-weight", type=float, default=defaults.bull_base_weight)
    parser.add_argument("--bull-defense-weight", type=float, default=defaults.bull_defense_weight)
    parser.add_argument("--bull-attack-weight", type=float, default=defaults.bull_attack_weight)
    parser.add_argument("--neutral-base-weight", type=float, default=defaults.neutral_base_weight)
    parser.add_argument("--neutral-defense-weight", type=float, default=defaults.neutral_defense_weight)
    parser.add_argument("--neutral-attack-weight", type=float, default=defaults.neutral_attack_weight)
    parser.add_argument("--bear-base-weight", type=float, default=defaults.bear_base_weight)
    parser.add_argument("--bear-defense-weight", type=float, default=defaults.bear_defense_weight)
    parser.add_argument("--bear-attack-weight", type=float, default=defaults.bear_attack_weight)
    parser.add_argument("--risk-on-breadth-threshold", type=float, default=defaults.risk_on_breadth_threshold)
    parser.add_argument("--market-regime-confirm-weeks", type=int, default=defaults.market_regime_confirm_weeks)
    parser.add_argument("--market-bull-confirm-weeks", type=int, default=defaults.market_bull_confirm_weeks)
    parser.add_argument("--market-bull-trend-threshold", type=float, default=defaults.market_bull_trend_threshold)
    parser.add_argument("--market-fast-trend-threshold", type=float, default=defaults.market_fast_trend_threshold)
    parser.add_argument("--local-fast-breadth-threshold", type=float, default=defaults.local_fast_breadth_threshold)
    parser.add_argument("--local-fast-trend-threshold", type=float, default=defaults.local_fast_trend_threshold)
    parser.add_argument("--market-volatility-veto-threshold", type=float, default=defaults.market_volatility_veto_threshold)
    parser.add_argument("--bear-defensive-boost", type=float, default=defaults.bear_defensive_boost)


def _describe_rebalance_schedule(backtest_config: BacktestConfig) -> str:
    return f"调仓节奏: 底仓={backtest_config.base_rebalance}, 防御仓={backtest_config.defense_rebalance}, 进攻仓={backtest_config.attack_rebalance}"


def _print_backtest_summary(summary: dict, output_dir: Path, recent_rebalances: int) -> None:
    print(f"输出目录: {output_dir}")
    if summary.get("error"):
        print(f"回测错误: {summary['error']}")
        return
    print(f"回测区间: {summary.get('start_date', '-')} -> {summary.get('end_date', '-')}")
    print(f"总收益: {_format_pct(summary.get('total_return'))}")
    print(f"年化收益: {_format_pct(summary.get('annual_return'))}")
    print(f"最大回撤: {_format_pct(summary.get('max_drawdown'))}")
    print(f"年化波动率: {_format_pct(summary.get('annual_volatility'))}")
    print(f"夏普比率(无风险利率按0): {_format_number(summary.get('sharpe_no_rf'))}")
    print(f"Calmar比率: {_format_number(summary.get('calmar'))}")
    print(f"调仓次数: {summary.get('rebalance_count', 0)}")
    print(f"平均换手率: {_format_pct(summary.get('avg_turnover'))}")
    print(f"累计交易成本: {_format_pct(summary.get('total_transaction_cost'))}")
    _print_annual_returns(output_dir / "backtest_nav.csv")
    _print_final_positions(output_dir / "backtest_positions.csv")
    _print_recent_rebalances(output_dir / "backtest_trades.csv", recent_rebalances)


def _print_final_positions(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        print("最终持仓: 无")
        return
    positions = pd.read_csv(path)
    if positions.empty or "date" not in positions:
        print("最终持仓: 无")
        return
    latest_date = positions["date"].max()
    latest = positions.loc[positions["date"] == latest_date].copy()
    if latest.empty:
        print("最终持仓: 无")
        return
    latest = latest.sort_values("total_weight", ascending=False) if "total_weight" in latest else latest
    latest["symbol"] = latest["symbol"].map(_format_symbol)
    for column in ("base_weight", "defense_weight", "attack_weight", "total_weight"):
        if column in latest:
            latest[column] = latest[column].map(_format_pct)
    columns = [column for column in ("symbol", "name", "category", "bucket", "pool", "base_weight", "defense_weight", "attack_weight", "total_weight") if column in latest]
    print(f"最终持仓 ({latest_date}):")
    print(latest[columns].to_string(index=False))


def _print_recent_rebalances(path: Path, rows: int) -> None:
    if rows <= 0:
        return
    if not path.exists() or path.stat().st_size == 0:
        print("最近调仓: 无")
        return
    trades = pd.read_csv(path)
    if trades.empty:
        print("最近调仓: 无")
        return
    recent = trades.tail(rows)
    print(f"最近调仓 (最近 {len(recent)} 次):")
    for record in recent.to_dict("records"):
        print(
            f"- {record.get('date', '-')} {_format_rebalance_type(record.get('rebalance_type'))} | 换手率 {_format_pct(record.get('turnover'))} | 成本 {_format_pct(record.get('transaction_cost'))} | 目标仓位 {_format_pct(record.get('target_weight_sum'))}"
        )
        print(f"  调入: {_format_text(record.get('buy_detail'), '无')}")
        print(f"  调出: {_format_text(record.get('sell_detail'), '无')}")
        print(f"  调仓后持仓: {_format_text(record.get('target_detail'), '无')}")


def _print_annual_returns(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        print("年度收益: 无")
        return
    nav = pd.read_csv(path)
    annual = annual_returns_from_nav(nav)
    if annual.empty:
        print("年度收益: 无")
        return
    print("年度收益:")
    for row in annual.itertuples(index=False):
        print(f"  {row.year}: {_format_pct(row.annual_return)} | 最大回撤 {_format_pct(row.max_drawdown)}")


def _format_pct(value: object) -> str:
    number = _to_float(value)
    if number is None:
        return "-"
    return f"{number * 100:.2f}%"


def _format_number(value: object) -> str:
    number = _to_float(value)
    if number is None:
        return "-"
    return f"{number:.2f}"


def _format_rebalance_type(value: object) -> str:
    mapping = {"base_defense": "底仓/防御仓调仓", "attack": "进攻仓调仓", "base_defense_attack": "底仓/防御仓/进攻仓调仓"}
    return mapping.get(str(value), str(value))


def _format_text(value: object, default: str) -> str:
    if value is None or pd.isna(value):
        return default
    text = str(value).strip()
    return text if text else default


def _format_symbol(value: object) -> str:
    text = str(value).strip()
    if "." in text:
        text = text.split(".", 1)[0]
    return text.zfill(6)


def _to_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number

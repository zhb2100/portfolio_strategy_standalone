from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import BacktestConfig, ClusterConfig, PortfolioConfig
from .metrics import compute_quality_metrics
from .normalization import normalize_symbol as _normalize_symbol
from .portfolio import (
    _max_drawdown,
    build_portfolio_candidates,
    combine_target_portfolio,
)


def run_portfolio_backtest(
    clean_daily: pd.DataFrame,
    output_dir: Path,
    backtest_config: BacktestConfig | None = None,
    cluster_config: ClusterConfig | None = None,
    portfolio_config: PortfolioConfig | None = None,
) -> dict[str, Any]:
    backtest_config = backtest_config or BacktestConfig()
    cluster_config = cluster_config or ClusterConfig()
    portfolio_config = portfolio_config or PortfolioConfig()
    output_dir.mkdir(parents=True, exist_ok=True)

    daily = _prepare_daily(clean_daily, backtest_config)
    if daily.empty:
        return _write_empty_backtest(output_dir, backtest_config, cluster_config, portfolio_config)

    prices = daily.pivot_table(index="date", columns="symbol", values="close", aggfunc="last").sort_index()
    amounts = daily.pivot_table(index="date", columns="symbol", values="amount", aggfunc="last").sort_index()
    dates = _nav_dates(list(prices.index), backtest_config)
    base_dates = set(_rebalance_dates(dates, backtest_config.base_rebalance))
    defense_dates = set(_rebalance_dates(dates, backtest_config.defense_rebalance))
    attack_dates = set(_rebalance_dates(dates, backtest_config.attack_rebalance))
    monthly_dates = base_dates | defense_dates
    weekly_dates = attack_dates
    rebalance_dates = monthly_dates | weekly_dates

    nav = float(backtest_config.initial_nav)
    current_weights: dict[str, float] = {}
    current_names: dict[str, str] = {}
    current_peaks: dict[str, float] = {}
    current_base = pd.DataFrame()
    current_defense = pd.DataFrame()
    current_attack = pd.DataFrame()
    market_state: dict[str, object] | None = None
    pending_signal_date: object | None = None
    pending_target_weights: dict[str, float] = {}
    pending_target_names: dict[str, str] = {}
    pending_target: pd.DataFrame = pd.DataFrame()
    pending_base: pd.DataFrame = pd.DataFrame()
    pending_defense: pd.DataFrame = pd.DataFrame()
    pending_attack: pd.DataFrame = pd.DataFrame()
    nav_rows: list[dict[str, object]] = []
    position_rows: list[dict[str, object]] = []
    trade_rows: list[dict[str, object]] = []

    for idx, date in enumerate(dates):
        gross_return = 0.0
        turnover = 0.0
        cost = 0.0

        if pending_signal_date is not None:
            signal_date = pending_signal_date
            target_weights = pending_target_weights
            target_names = pending_target_names
            target = pending_target
            execution_price_date = date
            turnover = _turnover(current_weights, target_weights)
            cost, _ = _execution_cost(current_weights, target_weights, nav, amounts.loc[date], backtest_config)
            nav *= 1.0 - cost
            trade_rows.append(
                {
                    "date": date,
                    "signal_date": signal_date,
                    "execution_price_date": execution_price_date,
                    "rebalance_type": _rebalance_type(signal_date, monthly_dates, weekly_dates),
                    "turnover": turnover,
                    "transaction_cost": cost,
                    "positions": len(target_weights),
                    "target_weight_sum": sum(target_weights.values()),
                    "buy_detail": _format_weight_changes(current_weights, target_weights, current_names, target_names, is_buy=True),
                    "sell_detail": _format_weight_changes(current_weights, target_weights, current_names, target_names, is_buy=False),
                    "target_detail": _format_target_detail(target),
                }
            )
            for row in _position_records(date, target):
                position_rows.append(row)
            current_weights = target_weights
            current_names = target_names
            current_base = pending_base
            current_defense = pending_defense
            current_attack = pending_attack
            current_peaks = {symbol: float(prices.loc[date].get(symbol)) for symbol in current_weights if pd.notna(prices.loc[date].get(symbol))}
            pending_signal_date = None

        if idx > 0:
            prev_date = dates[idx - 1]
            daily_returns = prices.loc[date].div(prices.loc[prev_date]).sub(1.0)
            gross_return = _portfolio_return(current_weights, daily_returns)
            nav *= 1.0 + gross_return
            current_peaks = _updated_peaks(current_weights, prices.loc[date], current_peaks)

        symbol_vols: dict[str, float] = {}
        if backtest_config.enable_adaptive_stop and idx >= 20:
            lookback_start = max(0, idx - 20)
            for symbol in current_weights:
                prices_series = prices.iloc[lookback_start:idx + 1][symbol].dropna().astype(float)
                if len(prices_series) >= 10:
                    daily_ret = prices_series.pct_change(fill_method=None).dropna()
                    vol = float(daily_ret.std() * np.sqrt(252))
                    symbol_vols[symbol] = vol

        stop_weights, stop_names = _stop_target(current_weights, current_names, prices.loc[date], current_peaks, backtest_config, symbol_vols if backtest_config.enable_adaptive_stop else None)
        has_stop_signal = stop_weights != current_weights

        if date in rebalance_dates:
            asof_daily = daily.loc[daily["date"] <= date].copy()
            universe = _build_asof_universe(asof_daily, backtest_config)
            if not universe.empty:
                candidate = build_portfolio_candidates(asof_daily, universe, cluster_config, portfolio_config, market_state)
                market_state = dict(candidate.risk_report)
                if date in monthly_dates or current_base.empty:
                    pending_base = candidate.base_positions
                    pending_defense = candidate.defense_positions
                elif date in weekly_dates:
                    pending_attack = candidate.attack_positions
                pending_target = combine_target_portfolio(pending_base, pending_defense, pending_attack, portfolio_config, candidate.risk_report)
            else:
                pending_target = pd.DataFrame()
            pending_signal_date = date
            pending_target_weights = _weights_from_target(pending_target)
            pending_target_names = _names_from_target(pending_target)
        elif has_stop_signal:
            pending_base = _filter_positions_by_weights(current_base, stop_weights)
            pending_defense = _filter_positions_by_weights(current_defense, stop_weights)
            pending_attack = _filter_positions_by_weights(current_attack, stop_weights)
            pending_signal_date = date
            pending_target = _target_from_weights(stop_weights, stop_names, pending_base, pending_defense, pending_attack)
            pending_target_weights = stop_weights
            pending_target_names = stop_names

        nav_rows.append(
            {
                "date": date,
                "nav": nav,
                "gross_return": gross_return,
                "turnover": turnover if not pending_signal_date else 0.0,
                "transaction_cost": cost if not pending_signal_date else 0.0,
                "cash_weight": max(1.0 - sum(current_weights.values()), 0.0),
            }
        )

    nav_df = pd.DataFrame(nav_rows)
    positions_df = pd.DataFrame(position_rows)
    trades_df = pd.DataFrame(trade_rows)
    summary = _summary(nav_df, trades_df, backtest_config, cluster_config, portfolio_config)

    nav_df.to_csv(output_dir / "backtest_nav.csv", index=False)
    positions_df.to_csv(output_dir / "backtest_positions.csv", index=False)
    trades_df.to_csv(output_dir / "backtest_trades.csv", index=False)
    (output_dir / "backtest_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return summary


def run_portfolio_backtest_from_files(
    clean_daily_csv: Path,
    output_dir: Path,
    backtest_config: BacktestConfig | None = None,
    cluster_config: ClusterConfig | None = None,
    portfolio_config: PortfolioConfig | None = None,
) -> dict[str, Any]:
    return run_portfolio_backtest(pd.read_csv(clean_daily_csv, low_memory=False), output_dir, backtest_config, cluster_config, portfolio_config)


def _prepare_daily(clean_daily: pd.DataFrame, config: BacktestConfig) -> pd.DataFrame:
    df = clean_daily.copy()
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df = df.dropna(subset=["date", "symbol", "close"]).copy()
    df["symbol"] = df["symbol"].map(lambda value: str(value).split(".", 1)[0].zfill(6))
    if "code" in df.columns:
        df["code"] = df["code"].map(lambda value: str(value).split(".", 1)[0].zfill(6) if pd.notna(value) else value)
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["close"])
    end = pd.to_datetime(config.end_date).date() if config.end_date else None
    if end is not None:
        df = df.loc[df["date"] <= end].copy()
    return df.sort_values(["date", "symbol"]).reset_index(drop=True)


def _nav_dates(dates: list[object], config: BacktestConfig) -> list[object]:
    if config.start_date is None:
        return dates
    start = pd.to_datetime(config.start_date).date()
    return [date for date in dates if date >= start]


def _build_asof_universe(asof_daily: pd.DataFrame, config: BacktestConfig) -> pd.DataFrame:
    if asof_daily.empty:
        return pd.DataFrame()
    spot = asof_daily.sort_values("date").drop_duplicates("symbol", keep="last")[["symbol", "code", "name", "category", "dedup_key"]].reset_index(drop=True)
    quality = compute_quality_metrics(asof_daily, spot)
    df = quality.copy()
    df = df.loc[df["category"].isin(("broad_based", "industry", "theme", "strategy", "cross_border", "commodity", "bond"))]
    df = df.loc[pd.to_numeric(df["listing_days"], errors="coerce") >= config.min_listing_days]
    df = df.loc[pd.to_numeric(df["avg_amount_20"], errors="coerce") >= config.min_avg_amount_20]
    df = df.loc[pd.to_numeric(df["valid_ratio_60"], errors="coerce") >= config.min_valid_ratio_60]
    close_vs_sma = pd.to_numeric(df["close_vs_sma200"], errors="coerce")
    momentum = pd.to_numeric(df["momentum_score"], errors="coerce")
    df = df.loc[~(close_vs_sma.lt(0) & momentum.lt(0))]
    if df.empty:
        return df
    df = df.sort_values(["dedup_key", "avg_amount_20", "listing_days"], ascending=[True, False, False])
    df = df.drop_duplicates("dedup_key", keep="first")
    df = df.sort_values(["momentum_score", "avg_amount_20"], ascending=[False, False], na_position="last")
    df["universe_rank"] = range(1, len(df) + 1)
    return df.reset_index(drop=True)


def _rebalance_dates(dates: list[object], rule: str) -> list[object]:
    if not dates:
        return []
    index = pd.to_datetime(pd.Series(dates))
    frame = pd.DataFrame({"date": dates, "period": index.dt.to_period(_period_alias(rule))})
    return frame.groupby("period", sort=True)["date"].last().tolist()


def _period_alias(rule: str) -> str:
    if rule.upper().startswith("W"):
        return "W"
    return "M"


def _portfolio_return(weights: dict[str, float], daily_returns: pd.Series) -> float:
    result = 0.0
    for symbol, weight in weights.items():
        value = daily_returns.get(symbol, 0.0)
        if pd.notna(value):
            result += weight * float(value)
    return result


def _weights_from_target(target: pd.DataFrame) -> dict[str, float]:
    if target.empty or "total_weight" not in target:
        return {}
    return {_normalize_symbol(row.symbol): float(row.total_weight) for row in target.itertuples(index=False)}


def _names_from_target(target: pd.DataFrame) -> dict[str, str]:
    if target.empty or "symbol" not in target:
        return {}
    return {_normalize_symbol(row.symbol): str(getattr(row, "name", "")) for row in target.itertuples(index=False)}


def _turnover(current: dict[str, float], target: dict[str, float]) -> float:
    symbols = sorted(set(current) | set(target))
    return sum(abs(target.get(symbol, 0.0) - current.get(symbol, 0.0)) for symbol in symbols)


def _rebalance_type(date: object, monthly_dates: set[object], weekly_dates: set[object]) -> str:
    if date in monthly_dates and date in weekly_dates:
        return "base_defense_attack"
    if date in monthly_dates:
        return "base_defense"
    return "attack"


def _format_weight_changes(
    current: dict[str, float],
    target: dict[str, float],
    current_names: dict[str, str],
    target_names: dict[str, str],
    *,
    is_buy: bool,
) -> str:
    changes: list[tuple[str, str, float]] = []
    for symbol in sorted(set(current) | set(target)):
        delta = target.get(symbol, 0.0) - current.get(symbol, 0.0)
        if is_buy and delta <= 1e-12:
            continue
        if not is_buy and delta >= -1e-12:
            continue
        name = target_names.get(symbol) or current_names.get(symbol) or ""
        changes.append((symbol, name, delta))
    if not changes:
        return "无"
    changes.sort(key=lambda item: abs(item[2]), reverse=True)
    return "；".join(_format_detail_item(symbol, name, delta) for symbol, name, delta in changes)


def _format_target_detail(target: pd.DataFrame) -> str:
    if target.empty or "total_weight" not in target:
        return "空仓"
    target = target.sort_values("total_weight", ascending=False)
    items = []
    for row in target.itertuples(index=False):
        items.append(_format_detail_item(_normalize_symbol(row.symbol), str(row.name), float(row.total_weight), show_sign=False))
    return "；".join(items)


def _format_detail_item(symbol: str, name: str, weight: float, *, show_sign: bool = True) -> str:
    label = f"{symbol} {name}".strip()
    sign = "+" if show_sign and weight > 0 else ""
    return f"{label} {sign}{weight * 100:.2f}%"


def _position_records(date: object, target: pd.DataFrame) -> list[dict[str, object]]:
    if target.empty:
        return []
    records: list[dict[str, object]] = []
    for row in target.itertuples(index=False):
        records.append(
            {
                "date": date,
                "symbol": row.symbol,
                "name": row.name,
                "category": row.category,
                "bucket": row.bucket,
                "cluster_id": row.cluster_id,
                "pool": row.pool,
                "base_weight": row.base_weight,
                "defense_weight": row.defense_weight,
                "attack_weight": row.attack_weight,
                "total_weight": row.total_weight,
            }
        )
    return records


def _summary(nav: pd.DataFrame, trades: pd.DataFrame, backtest_config: BacktestConfig, cluster_config: ClusterConfig, portfolio_config: PortfolioConfig) -> dict[str, Any]:
    if nav.empty:
        return {}
    nav_values = pd.to_numeric(nav["nav"], errors="coerce")
    returns = nav_values.pct_change(fill_method=None).dropna()
    days = max(len(nav_values) - 1, 1)
    total_return = float(nav_values.iloc[-1] / nav_values.iloc[0] - 1.0)
    annual_return = float((nav_values.iloc[-1] / nav_values.iloc[0]) ** (252 / days) - 1.0)
    annual_vol = float(returns.std() * np.sqrt(252)) if len(returns) > 1 else 0.0
    max_drawdown = _max_drawdown(nav_values.to_numpy())
    sharpe = float(annual_return / annual_vol) if annual_vol > 0 else np.nan
    calmar = float(annual_return / max_drawdown) if max_drawdown > 0 else np.nan
    return {
        "start_date": nav["date"].iloc[0],
        "end_date": nav["date"].iloc[-1],
        "total_return": total_return,
        "annual_return": annual_return,
        "annual_volatility": annual_vol,
        "max_drawdown": max_drawdown,
        "sharpe_no_rf": sharpe,
        "calmar": calmar,
        "rebalance_count": int(len(trades)),
        "avg_turnover": float(trades["turnover"].mean()) if "turnover" in trades and not trades.empty else 0.0,
        "total_transaction_cost": float(trades["transaction_cost"].sum()) if "transaction_cost" in trades and not trades.empty else 0.0,
        "backtest_config": asdict(backtest_config),
        "cluster_config": asdict(cluster_config),
        "portfolio_config": asdict(portfolio_config),
    }


def annual_returns_from_nav(nav: pd.DataFrame) -> pd.DataFrame:
    if nav.empty or "date" not in nav or "nav" not in nav:
        return pd.DataFrame(columns=["year", "annual_return", "max_drawdown"])
    frame = nav.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["nav"] = pd.to_numeric(frame["nav"], errors="coerce")
    frame = frame.dropna(subset=["date", "nav"]).sort_values("date")
    if frame.empty:
        return pd.DataFrame(columns=["year", "annual_return", "max_drawdown"])
    rows: list[dict[str, object]] = []
    for year, group in frame.groupby(frame["date"].dt.year, sort=True):
        start_nav = float(group["nav"].iloc[0])
        end_nav = float(group["nav"].iloc[-1])
        annual_return = end_nav / start_nav - 1.0
        max_drawdown = float((group["nav"] / group["nav"].cummax() - 1.0).min())
        rows.append({"year": int(year), "annual_return": annual_return, "max_drawdown": abs(max_drawdown)})
    return pd.DataFrame(rows)


def _trade_cost(weights: dict[str, float], stop_weights: dict[str, float], amount: pd.Series, trade_value: float, config: BacktestConfig) -> tuple[float, dict[str, float]]:
    if trade_value <= 0:
        return 0.0, {"commission": 0.0, "slippage": 0.0, "impact": 0.0}
    commission = trade_value * config.transaction_cost_bps / 10_000.0
    slippage = trade_value * config.slippage_bps / 10_000.0
    impact = 0.0
    if config.impact_bps > 0 and config.impact_participation > 0 and not amount.empty:
        impacted_value = trade_value * config.impact_participation
        avg_amount = float(amount.mean()) if len(amount) > 0 else 1.0
        if avg_amount > 0:
            participation_rate = impacted_value / avg_amount
            impact = trade_value * config.impact_bps / 10_000.0 * min(participation_rate * 10, 1.0)
    total = commission + slippage + impact
    return total, {"commission": commission, "slippage": slippage, "impact": impact}


def _execution_cost(current: dict[str, float], target: dict[str, float], nav: float, amount_row: pd.Series, config: BacktestConfig) -> tuple[float, dict[str, float]]:
    turnover = _turnover(current, target)
    trade_value = turnover * nav
    return _trade_cost(current, target, amount_row, trade_value, config)


def _trailing_stop_symbols(weights: dict[str, float], price_row: pd.Series, peaks: dict[str, float], config: BacktestConfig, symbol_vols: dict[str, float] | None = None) -> list[str]:
    if config.trailing_stop_pct <= 0:
        return []
    stopped: list[str] = []
    for symbol in sorted(weights):
        price = price_row.get(symbol)
        peak = peaks.get(symbol)
        if price is None or peak is None or peak <= 0:
            continue
        stop_pct = config.trailing_stop_pct
        if config.enable_adaptive_stop and symbol_vols:
            vol = symbol_vols.get(symbol, 0.0)
            if vol > 0:
                adaptive_stop = config.volatility_stop_multiplier * vol
                stop_pct = max(stop_pct, min(adaptive_stop, 0.30))
        drawdown = float(price) / float(peak) - 1.0
        if drawdown <= -stop_pct:
            stopped.append(symbol)
    return stopped


def _updated_peaks(weights: dict[str, float], price_row: pd.Series, peaks: dict[str, float]) -> dict[str, float]:
    updated: dict[str, float] = {}
    for symbol in sorted(weights):
        price = price_row.get(symbol)
        if pd.isna(price):
            if symbol in peaks:
                updated[symbol] = peaks[symbol]
            continue
        updated[symbol] = max(float(price), float(peaks.get(symbol, price)))
    return updated


def _stop_target(current_weights: dict[str, float], current_names: dict[str, str], price_row: pd.Series, peaks: dict[str, float], config: BacktestConfig, symbol_vols: dict[str, float] | None = None) -> tuple[dict[str, float], dict[str, str]]:
    stopped_symbols = set(_trailing_stop_symbols(current_weights, price_row, peaks, config, symbol_vols))
    if not stopped_symbols:
        return current_weights, current_names
    target_weights = {symbol: weight for symbol, weight in current_weights.items() if symbol not in stopped_symbols}
    target_names = {symbol: name for symbol, name in current_names.items() if symbol not in stopped_symbols}
    return target_weights, target_names


def _target_from_weights(
    weights: dict[str, float],
    names: dict[str, str],
    current_base: pd.DataFrame,
    current_defense: pd.DataFrame,
    current_attack: pd.DataFrame,
) -> pd.DataFrame:
    combined = combine_target_portfolio(current_base, current_defense, current_attack, PortfolioConfig())
    if combined.empty:
        return pd.DataFrame(columns=["symbol", "name", "category", "bucket", "cluster_id", "pool", "base_weight", "defense_weight", "attack_weight", "total_weight"])
    target = combined.loc[combined["symbol"].map(_normalize_symbol).isin(weights)].copy()
    if target.empty:
        return pd.DataFrame(columns=combined.columns)
    target["symbol"] = target["symbol"].map(_normalize_symbol)
    target["name"] = target["symbol"].map(names).fillna(target["name"])
    target["total_weight"] = target["symbol"].map(weights).fillna(0.0)
    return target.loc[target["total_weight"] > 0].sort_values("total_weight", ascending=False).reset_index(drop=True)


def _filter_positions_by_weights(positions: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    if positions.empty:
        return positions
    filtered = positions.loc[positions["symbol"].map(_normalize_symbol).isin(weights)].copy()
    return filtered.reset_index(drop=True)


def _write_empty_backtest(output_dir: Path, backtest_config: BacktestConfig, cluster_config: ClusterConfig, portfolio_config: PortfolioConfig) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "start_date": backtest_config.start_date,
        "end_date": backtest_config.end_date,
        "rebalance_count": 0,
        "backtest_config": asdict(backtest_config),
        "cluster_config": asdict(cluster_config),
        "portfolio_config": asdict(portfolio_config),
    }
    pd.DataFrame(columns=["date", "nav", "gross_return", "turnover", "transaction_cost", "cash_weight"]).to_csv(output_dir / "backtest_nav.csv", index=False)
    pd.DataFrame(columns=["date", "symbol", "name", "category", "bucket", "cluster_id", "pool", "base_weight", "defense_weight", "attack_weight", "total_weight"]).to_csv(output_dir / "backtest_positions.csv", index=False)
    pd.DataFrame(columns=["date", "signal_date", "execution_price_date", "rebalance_type", "turnover", "transaction_cost", "positions", "target_weight_sum", "buy_detail", "sell_detail", "target_detail"]).to_csv(output_dir / "backtest_trades.csv", index=False)
    (output_dir / "backtest_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return summary

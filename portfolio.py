from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ClusterConfig, PortfolioConfig


@dataclass(frozen=True)
class PortfolioResult:
    bucket_map: pd.DataFrame
    base_positions: pd.DataFrame
    defense_positions: pd.DataFrame
    attack_positions: pd.DataFrame
    target_portfolio: pd.DataFrame
    risk_report: dict[str, object]


def build_portfolio_candidates(
    clean_daily: pd.DataFrame,
    universe: pd.DataFrame,
    cluster_config: ClusterConfig | None = None,
    portfolio_config: PortfolioConfig | None = None,
    market_state: dict[str, object] | None = None,
) -> PortfolioResult:
    cluster_config = cluster_config or ClusterConfig()
    portfolio_config = portfolio_config or PortfolioConfig()
    if not portfolio_config.enabled or universe.empty or clean_daily.empty:
        return _empty_portfolio_result()

    enriched = _add_price_factors(clean_daily, universe)
    bucket_map = _build_bucket_map(enriched)
    if bucket_map.empty:
        return _empty_portfolio_result()

    risk_state = _build_risk_state(bucket_map, portfolio_config, market_state)
    base_positions = _select_base_positions(bucket_map, portfolio_config)
    defense_positions = _select_defense_positions(bucket_map, portfolio_config)
    attack_positions = _select_attack_positions(bucket_map, portfolio_config, risk_state)
    target_portfolio = _combine_target_portfolio(base_positions, defense_positions, attack_positions, portfolio_config, risk_state)

    if portfolio_config.enable_macro_signal:
        macro_scale = _macro_risk_signal(clean_daily, portfolio_config)
        if macro_scale < 1.0 and not target_portfolio.empty:
            for column in ("base_weight", "defense_weight", "attack_weight"):
                target_portfolio[column] *= macro_scale
            target_portfolio["total_weight"] = (
                target_portfolio["base_weight"] + target_portfolio["defense_weight"] + target_portfolio["attack_weight"]
            )

    risk_report = {
        **risk_state,
        "risk_on": bool(_is_risk_on(bucket_map, portfolio_config, risk_state)),
        "risk_on_breadth": float(_risk_on_breadth(bucket_map)),
        "local_fast_on": bool(risk_state.get("local_fast_on", False)),
        "local_fast_buckets": list(risk_state.get("local_fast_buckets", [])) if isinstance(risk_state.get("local_fast_buckets", []), list) else [],
        "candidate_total": int(bucket_map["bucket"].nunique()) if "bucket" in bucket_map else 0,
        "cluster_total": int(bucket_map["bucket"].nunique()) if "bucket" in bucket_map else 0,
        "base_positions": int(len(base_positions)),
        "defense_positions": int(len(defense_positions)),
        "attack_positions": int(len(attack_positions)),
        "target_positions": int(len(target_portfolio)),
        "target_weight_sum": float(target_portfolio["total_weight"].sum()) if "total_weight" in target_portfolio else 0.0,
        "target_base_weight": float(_effective_base_weight(portfolio_config, risk_state)),
        "target_defense_weight": float(_effective_defense_weight(portfolio_config, risk_state)),
        "target_attack_weight": float(_effective_attack_weight(portfolio_config, risk_state)),
    }
    return PortfolioResult(bucket_map, base_positions, defense_positions, attack_positions, target_portfolio, risk_report)


def write_portfolio_outputs(result: PortfolioResult, output_dir: Path) -> None:
    result.bucket_map.to_csv(output_dir / "etf_bucket_map.csv", index=False)
    result.base_positions.to_csv(output_dir / "base_positions.csv", index=False)
    result.defense_positions.to_csv(output_dir / "defense_positions.csv", index=False)
    result.attack_positions.to_csv(output_dir / "attack_positions.csv", index=False)
    result.target_portfolio.to_csv(output_dir / "target_portfolio.csv", index=False)


def select_fast_positions_for_slow(
    cluster_map: pd.DataFrame,
    slow_positions: pd.DataFrame,
    portfolio_config: PortfolioConfig | None = None,
    market_state: dict[str, object] | None = None,
) -> pd.DataFrame:
    portfolio_config = portfolio_config or PortfolioConfig()
    return _select_attack_positions(cluster_map, portfolio_config, market_state)


def combine_target_portfolio(
    base_positions: pd.DataFrame,
    defense_positions: pd.DataFrame,
    attack_positions: pd.DataFrame,
    portfolio_config: PortfolioConfig | None = None,
    risk_state: dict[str, object] | None = None,
) -> pd.DataFrame:
    portfolio_config = portfolio_config or PortfolioConfig()
    return _combine_target_portfolio(base_positions, defense_positions, attack_positions, portfolio_config, risk_state)


def _add_price_factors(clean_daily: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    df = universe.copy()
    factor_records: list[dict[str, object]] = []
    daily = clean_daily.loc[clean_daily["symbol"].isin(df["symbol"])].sort_values(["symbol", "date"])
    for symbol, group in daily.groupby("symbol", sort=False):
        closes = group["close"].dropna().astype(float).to_numpy()
        returns = pd.Series(closes).pct_change(fill_method=None).dropna()
        volumes_raw = group["volume"].dropna().astype(float).to_numpy() if "volume" in group.columns else None
        factor_records.append(
            {
                "symbol": symbol,
                "momentum_20": _log_return(closes, 20),
                "momentum_60": _log_return(closes, 60),
                "momentum_120": _log_return(closes, 120),
                "momentum_252": _log_return(closes, 252),
                "volatility_20": float(returns.tail(20).std() * np.sqrt(252)) if len(returns) >= 20 else np.nan,
                "volatility_120": float(returns.tail(120).std() * np.sqrt(252)) if len(returns) >= 20 else np.nan,
                "drawdown_20": _max_drawdown(closes[-20:]) if len(closes) >= 2 else np.nan,
                "volume_trend_20": _volume_trend(volumes_raw, 20) if volumes_raw is not None and len(volumes_raw) >= 20 else np.nan,
                "momentum_acceleration": (_log_return(closes, 20) or 0.0) - (_log_return(closes, 60) or 0.0),
            }
        )
    factors = pd.DataFrame(factor_records)
    if factors.empty:
        return df
    return df.merge(factors, on="symbol", how="left")


def _build_bucket_map(universe: pd.DataFrame) -> pd.DataFrame:
    df = universe.copy()
    if df.empty:
        return df
    df = df.copy()
    df["pool"] = df.apply(lambda row: _pool_for_row(str(row.get("name", "")), str(row.get("category", ""))), axis=1)
    df["bucket"] = df["pool"]
    df["cluster_id"] = df["bucket"]
    df["cluster_size"] = df.groupby("bucket")["symbol"].transform("size")
    df["base_score"] = _base_score(df)
    df["defense_score"] = _defense_score(df)
    df["attack_score"] = _attack_score(df)
    return df.sort_values(["pool", "base_score", "defense_score", "attack_score"], ascending=[True, False, False, False], na_position="last").reset_index(drop=True)


def _pool_for_row(name: str, category: str) -> str:
    text = str(name)
    category = str(category)
    if category == "broad_based":
        return "base"
    if category == "bond":
        return "defense"
    if category == "commodity":
        return "defense"
    if category == "strategy" and any(keyword in text for keyword in ("红利", "低波", "防御", "高股息", "短债", "国债", "黄金")):
        return "defense"
    if category in {"industry", "theme", "cross_border"}:
        return "attack"
    return "base" if category == "strategy" else "attack"


def _build_cluster_map(clean_daily: pd.DataFrame, universe: pd.DataFrame, config: ClusterConfig) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    returns = _returns_matrix(clean_daily, universe["symbol"].tolist(), config.lookback_days)
    base = universe.copy()
    base["bucket"] = base["category"].map(_bucket_for_category).fillna("other")
    base["leader_score"] = _leader_score(base)
    cluster_counter = 1
    for bucket, part in base.groupby("bucket", sort=False):
        symbols = part["symbol"].tolist()
        corr = returns[symbols].corr(min_periods=config.min_overlap_days) if symbols else pd.DataFrame()
        components = _connected_components(symbols, corr, config.corr_threshold)
        for component in components:
            cluster_id = f"C{cluster_counter:03d}"
            cluster_counter += 1
            cluster = part.loc[part["symbol"].isin(component)].copy()
            cluster = cluster.sort_values(["leader_score", "avg_amount_20"], ascending=[False, False], na_position="last")
            leader_symbol = str(cluster.iloc[0]["symbol"])
            cluster["cluster_id"] = cluster_id
            cluster["bucket"] = bucket
            cluster["cluster_size"] = len(cluster)
            cluster["cluster_leader_symbol"] = leader_symbol
            cluster["cluster_rank"] = range(1, len(cluster) + 1)
            cluster["cluster_leader"] = cluster["symbol"].eq(leader_symbol)
            cluster["avg_corr_to_cluster"] = cluster["symbol"].map(lambda symbol: _avg_corr(symbol, component, corr))
            cluster["max_corr_to_leader"] = cluster["symbol"].map(lambda symbol: _corr_value(symbol, leader_symbol, corr))
            rows.append(cluster)
    if not rows:
        return pd.DataFrame()
    cluster_map = pd.concat(rows, ignore_index=True)
    cluster_map["cluster_momentum_60"] = cluster_map.groupby("cluster_id")["momentum_60"].transform("mean")
    cluster_map["relative_strength_vs_cluster_60"] = cluster_map["momentum_60"] - cluster_map["cluster_momentum_60"]
    cluster_map["slow_score"] = _slow_score(cluster_map)
    cluster_map["fast_score"] = _fast_score(cluster_map)
    return cluster_map.sort_values(["cluster_id", "cluster_rank"]).reset_index(drop=True)


def _select_base_positions(cluster_map: pd.DataFrame, config: PortfolioConfig) -> pd.DataFrame:
    candidates = cluster_map.loc[cluster_map["pool"].eq("base")].copy()
    candidates = _filter_weak_trend(candidates)
    if candidates.empty:
        return pd.DataFrame(columns=_position_columns_list("base"))
    candidates = candidates.sort_values(["base_score", "momentum_120", "momentum_60"], ascending=[False, False, False], na_position="last")
    result = candidates.head(max(int(config.base_slots), 0)).copy()
    if result.empty:
        return pd.DataFrame(columns=_position_columns_list("base"))
    result = result.reset_index(drop=True)
    result["base_rank"] = range(1, len(result) + 1)
    result["target_base_weight"] = config.bull_base_weight / len(result)
    return _position_columns(result, "base")


def _select_defense_positions(cluster_map: pd.DataFrame, config: PortfolioConfig) -> pd.DataFrame:
    candidates = cluster_map.loc[cluster_map["pool"].eq("defense")].copy()
    candidates = _filter_weak_trend(candidates)
    if candidates.empty:
        return pd.DataFrame(columns=_position_columns_list("defense"))
    candidates = candidates.sort_values(["defense_score", "momentum_120", "momentum_60"], ascending=[False, False, False], na_position="last")
    result = candidates.head(max(int(config.defense_slots), 0)).copy()
    if result.empty:
        return pd.DataFrame(columns=_position_columns_list("defense"))
    result = result.reset_index(drop=True)
    result["defense_rank"] = range(1, len(result) + 1)
    result["target_defense_weight"] = config.bull_defense_weight / len(result)
    return _position_columns(result, "defense")


def _select_attack_positions(cluster_map: pd.DataFrame, config: PortfolioConfig, risk_state: dict[str, object] | None = None) -> pd.DataFrame:
    candidates = cluster_map.loc[cluster_map["pool"].eq("attack")].copy()
    candidates = _filter_weak_trend(candidates)
    if candidates.empty:
        return pd.DataFrame(columns=_position_columns_list("attack"))
    if isinstance(risk_state, dict) and str(risk_state.get("market_regime", "bear")) != "bull":
        if float(risk_state.get("market_volatility", 0.0)) >= config.market_volatility_veto_threshold:
            return pd.DataFrame(columns=_position_columns_list("attack"))
    candidates = candidates.sort_values(["attack_score", "momentum_20", "momentum_60"], ascending=[False, False, False], na_position="last")
    result = candidates.head(max(int(config.attack_slots), 0)).copy()
    if result.empty:
        return pd.DataFrame(columns=_position_columns_list("attack"))
    result = result.reset_index(drop=True)
    result["attack_rank"] = range(1, len(result) + 1)
    result["target_attack_weight"] = config.bull_attack_weight / len(result)
    return _position_columns(result, "attack")


def _combine_target_portfolio(
    base_positions: pd.DataFrame,
    defense_positions: pd.DataFrame,
    attack_positions: pd.DataFrame,
    config: PortfolioConfig,
    risk_state: dict[str, object] | None = None,
) -> pd.DataFrame:
    if risk_state is None:
        base_weight = config.bull_base_weight
        defense_weight = config.bull_defense_weight
        attack_weight = config.bull_attack_weight
    else:
        base_weight = _effective_base_weight(config, risk_state)
        defense_weight = _effective_defense_weight(config, risk_state)
        attack_weight = _effective_attack_weight(config, risk_state)
    pieces: list[pd.DataFrame] = []
    if not base_positions.empty:
        base = base_positions[["symbol", "name", "category", "bucket", "cluster_id", "target_base_weight"]].copy()
        base["pool"] = "base"
        base["target_defense_weight"] = 0.0
        base["target_attack_weight"] = 0.0
        pieces.append(base)
    if not defense_positions.empty:
        defense = defense_positions[["symbol", "name", "category", "bucket", "cluster_id", "target_defense_weight"]].copy()
        defense["pool"] = "defense"
        defense["target_base_weight"] = 0.0
        defense["target_attack_weight"] = 0.0
        pieces.append(defense)
    if not attack_positions.empty:
        attack = attack_positions[["symbol", "name", "category", "bucket", "cluster_id", "target_attack_weight"]].copy()
        attack["pool"] = "attack"
        attack["target_base_weight"] = 0.0
        attack["target_defense_weight"] = 0.0
        pieces.append(attack)
    if not pieces:
        return pd.DataFrame(columns=["symbol", "name", "category", "bucket", "cluster_id", "pool", "base_weight", "defense_weight", "attack_weight", "total_weight"])
    combined = pd.concat(pieces, ignore_index=True)
    for col in ("target_base_weight", "target_defense_weight", "target_attack_weight"):
        if col in combined:
            combined[col] = combined[col].fillna(0.0)
    result = combined.groupby("symbol", as_index=False).agg(
        name=("name", "first"),
        category=("category", "first"),
        bucket=("bucket", "first"),
        cluster_id=("cluster_id", "first"),
        pool=("pool", "first"),
        base_weight=("target_base_weight", "sum"),
        defense_weight=("target_defense_weight", "sum"),
        attack_weight=("target_attack_weight", "sum"),
    )
    if not result.empty:
        base_total = float(result["base_weight"].sum())
        defense_total = float(result["defense_weight"].sum())
        attack_total = float(result["attack_weight"].sum())
        if base_total > 0 and base_weight > 0:
            result["base_weight"] = result["base_weight"] / base_total * base_weight
        else:
            result["base_weight"] = 0.0
        if defense_total > 0 and defense_weight > 0:
            result["defense_weight"] = result["defense_weight"] / defense_total * defense_weight
        else:
            result["defense_weight"] = 0.0
        if attack_total > 0 and attack_weight > 0:
            result["attack_weight"] = result["attack_weight"] / attack_total * attack_weight
        else:
            result["attack_weight"] = 0.0
    result["total_weight"] = result["base_weight"] + result["defense_weight"] + result["attack_weight"]
    if config.enable_risk_parity:
        for group_col, weight_col in [("base_weight", "base_weight"), ("defense_weight", "defense_weight"), ("attack_weight", "attack_weight")]:
            mask = result[group_col] > 0
            if mask.any() and "volatility_20" in result.columns:
                vols = result.loc[mask, "volatility_20"].fillna(0.0)
                valid = vols > 0
                if valid.any():
                    inv_vol = 1.0 / vols[valid]
                    total_inv = inv_vol.sum()
                    if total_inv > 0:
                        orig_total = float(result.loc[mask, weight_col].sum())
                        result.loc[mask, weight_col] = 0.0
                        result.loc[valid, weight_col] = inv_vol / total_inv * orig_total
    if config.target_volatility > 0 and "volatility_20" in result.columns:
        total_w = result["base_weight"] + result["defense_weight"] + result["attack_weight"]
        total_sum = total_w.sum()
        if total_sum > 0:
            vols = result["volatility_20"].fillna(0.0).to_numpy()
            w = total_w.to_numpy() / total_sum
            port_vol = float(np.sqrt(np.sum(w ** 2 * vols ** 2)))
            if port_vol > 0 and port_vol != config.target_volatility:
                scale = config.target_volatility / port_vol
                result["base_weight"] *= scale
                result["defense_weight"] *= scale
                result["attack_weight"] *= scale
    result["total_weight"] = result["base_weight"] + result["defense_weight"] + result["attack_weight"]
    return result.loc[result["total_weight"] > 0].sort_values("total_weight", ascending=False).reset_index(drop=True)


def _returns_matrix(clean_daily: pd.DataFrame, symbols: list[str], lookback_days: int) -> pd.DataFrame:
    if not symbols:
        return pd.DataFrame()
    daily = clean_daily.loc[clean_daily["symbol"].isin(symbols), ["date", "symbol", "close"]].copy()
    pivot = daily.pivot_table(index="date", columns="symbol", values="close", aggfunc="last").sort_index()
    returns = pivot.pct_change(fill_method=None)
    return returns.tail(lookback_days)


def _connected_components(symbols: list[str], corr: pd.DataFrame, threshold: float) -> list[list[str]]:
    remaining = set(symbols)
    components: list[list[str]] = []
    while remaining:
        start = remaining.pop()
        stack = [start]
        component = {start}
        while stack:
            current = stack.pop()
            for other in list(remaining):
                value = _corr_value(current, other, corr)
                if pd.notna(value) and value >= threshold:
                    remaining.remove(other)
                    component.add(other)
                    stack.append(other)
        components.append(sorted(component))
    return components


def _leader_score(df: pd.DataFrame) -> pd.Series:
    liquidity = _rank_score(df.get("avg_amount_20"), df.index)
    scale = _rank_score(df.get("total_value"), df.index)
    age = _rank_score(df.get("listing_days"), df.index)
    momentum = _rank_score(df.get("momentum_score"), df.index)
    return 0.40 * liquidity + 0.30 * scale + 0.20 * age + 0.10 * momentum


def _base_score(df: pd.DataFrame) -> pd.Series:
    return (
        0.50 * _rank_score(df.get("momentum_120"), df.index)
        + 0.35 * _rank_score(df.get("momentum_60"), df.index)
        + 0.10 * _rank_score(df.get("avg_amount_20"), df.index)
        - 0.05 * _rank_score(df.get("volatility_120"), df.index)
    )


def _defense_score(df: pd.DataFrame) -> pd.Series:
    return (
        0.40 * _rank_score(df.get("momentum_120"), df.index)
        + 0.30 * _rank_score(df.get("momentum_60"), df.index)
        + 0.15 * _rank_score(df.get("close_vs_sma200"), df.index)
        + 0.10 * _rank_score(df.get("avg_amount_20"), df.index)
        - 0.05 * _rank_score(df.get("drawdown_20"), df.index)
    )


def _attack_score(df: pd.DataFrame) -> pd.Series:
    return (
        0.55 * _rank_score(df.get("momentum_20"), df.index)
        + 0.30 * _rank_score(df.get("momentum_60"), df.index)
        + 0.10 * _rank_score(df.get("momentum_acceleration"), df.index)
        + 0.05 * _rank_score(df.get("avg_amount_20"), df.index)
        - 0.05 * _rank_score(df.get("drawdown_20"), df.index)
    )


def _slow_score(df: pd.DataFrame) -> pd.Series:
    return _base_score(df)


def _fast_score(df: pd.DataFrame) -> pd.Series:
    return _attack_score(df)


def _rank_score(values: pd.Series | None, index: pd.Index) -> pd.Series:
    if values is None:
        return pd.Series(0.5, index=index)
    series = pd.to_numeric(values, errors="coerce")
    if series.notna().sum() == 0:
        return pd.Series(0.5, index=series.index)
    return series.rank(pct=True, na_option="bottom")


def _filter_weak_trend(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    close = pd.to_numeric(df.get("close_vs_sma200"), errors="coerce")
    momentum = pd.to_numeric(df.get("momentum_score"), errors="coerce")
    weak = close.lt(0) & momentum.lt(0)
    return df.loc[~weak.fillna(False)].copy()


def _is_risk_on(cluster_map: pd.DataFrame, config: PortfolioConfig, risk_state: dict[str, object] | None = None) -> bool:
    breadth = _risk_on_breadth(cluster_map)
    trend_strength = _trend_strength(cluster_map)
    volatility = _market_volatility(cluster_map)
    if breadth < config.risk_on_breadth_threshold:
        return False
    if trend_strength < config.market_fast_trend_threshold:
        return False
    if volatility >= config.market_volatility_veto_threshold:
        return False
    if isinstance(risk_state, dict):
        if str(risk_state.get("market_regime", "bear")) != "bull":
            return False
        if float(risk_state.get("market_volatility", volatility)) >= config.market_volatility_veto_threshold:
            return False
    return True


def _risk_on_breadth(cluster_map: pd.DataFrame) -> float:
    values = pd.to_numeric(cluster_map.get("close_vs_sma200"), errors="coerce")
    if values.empty:
        return 0.0
    return float(values.gt(0).mean())


def _build_risk_state(cluster_map: pd.DataFrame, config: PortfolioConfig, market_state: dict[str, object] | None = None) -> dict[str, object]:
    proxy = cluster_map.loc[cluster_map["pool"].eq("base")].copy()
    if proxy.empty:
        proxy = cluster_map.copy()
    market_breadth = _risk_on_breadth(proxy)
    market_trend = _trend_strength(proxy)
    market_volatility = _market_volatility(proxy)
    raw_market_regime = _regime_from_signals(
        market_breadth,
        market_trend,
        config.market_bull_trend_threshold,
        config.risk_on_breadth_threshold,
        config.sideways_breadth_upper,
        config.sideways_breadth_lower,
    )
    confirmed = _update_market_regime_state(raw_market_regime, market_state, bull_confirm_weeks=config.market_bull_confirm_weeks, bear_confirm_weeks=config.market_regime_confirm_weeks)
    local_fast_state = {**confirmed, "market_volatility": market_volatility}
    local_fast_buckets = _local_fast_buckets(cluster_map, config, local_fast_state)
    return {
        "market_regime": confirmed["market_regime"],
        "market_regime_raw": raw_market_regime,
        "market_regime_candidate": confirmed["candidate"],
        "market_regime_streak": confirmed["streak"],
        "market_breadth": market_breadth,
        "market_trend_strength": market_trend,
        "market_volatility": market_volatility,
        "market_volatility_veto": bool(market_volatility >= config.market_volatility_veto_threshold),
        "market_regime_confirm_weeks": config.market_regime_confirm_weeks,
        "market_bull_confirm_weeks": config.market_bull_confirm_weeks,
        "local_fast_on": bool(local_fast_buckets) and str(confirmed["market_regime"]) == "bull" and market_volatility < config.market_volatility_veto_threshold,
        "local_fast_buckets": local_fast_buckets,
    }


def _regime_from_signals(
    breadth: float,
    trend_strength: float,
    bull_trend_threshold: float,
    risk_on_breadth_threshold: float,
    sideways_breadth_upper: float = 0.60,
    sideways_breadth_lower: float = 0.25,
) -> str:
    if trend_strength >= bull_trend_threshold and breadth >= risk_on_breadth_threshold:
        return "bull"
    elif breadth <= sideways_breadth_lower:
        return "bear"
    return "sideways"


def _update_market_regime_state(
    raw_regime: str,
    market_state: dict[str, object] | None,
    *,
    bull_confirm_weeks: int,
    bear_confirm_weeks: int,
) -> dict[str, object]:
    previous = market_state if isinstance(market_state, dict) else {}
    confirmed = str(previous.get("market_regime", "bear"))
    candidate = str(previous.get("market_regime_candidate", ""))
    streak = int(previous.get("market_regime_streak", 0))
    if raw_regime == confirmed:
        return {"market_regime": confirmed, "candidate": "", "streak": 0}
    if raw_regime == candidate:
        streak += 1
    else:
        candidate = raw_regime
        streak = 1
    confirm_weeks = bull_confirm_weeks if raw_regime == "bull" else bear_confirm_weeks
    if streak >= max(confirm_weeks, 1):
        return {"market_regime": raw_regime, "candidate": "", "streak": 0}
    return {"market_regime": confirmed, "candidate": candidate, "streak": streak}


def _trend_strength(df: pd.DataFrame) -> float:
    close_ratio = _positive_ratio(df.get("close_vs_sma200"))
    momentum_ratio = _positive_ratio(df.get("momentum_score"))
    return 0.5 * close_ratio + 0.5 * momentum_ratio


def _positive_ratio(values: pd.Series | None) -> float:
    if values is None:
        return 0.0
    series = pd.to_numeric(values, errors="coerce").dropna()
    if series.empty:
        return 0.0
    return float(series.gt(0).mean())


def _is_too_correlated_with_selected(row: pd.Series, selected: list[pd.Series], threshold: float, leader_corr: pd.DataFrame | None = None) -> bool:
    symbol = str(row.get("symbol", ""))
    for existing in selected:
        existing_symbol = str(existing.get("symbol", ""))
        corr = _corr_value(symbol, existing_symbol, leader_corr if leader_corr is not None else pd.DataFrame())
        if pd.notna(corr) and corr >= threshold:
            return True
        if pd.isna(corr) and str(row.get("bucket")) == str(existing.get("bucket")):
            return True
    return False


def _bucket_for_category(category: object) -> str:
    value = str(category)
    if value in {"broad_based", "strategy"}:
        return "equity_core"
    if value in {"industry", "theme"}:
        return "industry_theme"
    if value == "cross_border":
        return "cross_border"
    if value == "commodity":
        return "commodity"
    if value == "bond":
        return "bond"
    return "other"


def _avg_corr(symbol: str, component: list[str], corr: pd.DataFrame) -> float:
    others = [other for other in component if other != symbol]
    if not others:
        return 1.0
    values = [_corr_value(symbol, other, corr) for other in others]
    values = [value for value in values if pd.notna(value)]
    return float(np.mean(values)) if values else np.nan


def _corr_value(left: str, right: str, corr: pd.DataFrame) -> float:
    if corr.empty or left not in corr.index or right not in corr.columns:
        return np.nan
    return float(corr.loc[left, right])


def _log_return(closes: np.ndarray, window: int) -> float:
    if len(closes) < 2:
        return np.nan
    values = closes[-window:] if len(closes) >= window else closes
    if len(values) < 2 or values[0] <= 0 or values[-1] <= 0:
        return np.nan
    return float(np.log(values[-1] / values[0]))


def _volume_trend(volumes: np.ndarray, window: int = 20) -> float:
    if len(volumes) < window or len(volumes) < 5:
        return np.nan
    recent = float(np.mean(volumes[-5:]))
    full = float(np.mean(volumes[-window:]))
    if full <= 0:
        return np.nan
    return recent / full


def _max_drawdown(closes: np.ndarray) -> float:
    values = np.asarray(closes, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    if len(values) < 2:
        return np.nan
    running_max = np.maximum.accumulate(values)
    drawdowns = values / running_max - 1.0
    return float(abs(drawdowns.min()))


def _position_columns(df: pd.DataFrame, leg: str) -> pd.DataFrame:
    columns = _position_columns_list(leg)
    return df[[column for column in columns if column in df.columns]].copy()


def _position_columns_list(leg: str) -> list[str]:
    rank = f"{leg}_rank"
    weight = f"target_{leg}_weight"
    score = f"{leg}_score"
    return [
        "symbol",
        "name",
        "category",
        "bucket",
        "dedup_key",
        "cluster_id",
        "cluster_size",
        "pool",
        rank,
        score,
        weight,
        "momentum_score",
        "momentum_20",
        "momentum_60",
        "momentum_120",
        "momentum_252",
        "close_vs_sma200",
        "avg_amount_20",
        "total_value",
        "volatility_20",
        "volume_trend_20",
    ]


def _empty_risk_state() -> dict[str, object]:
    return {
        "market_regime": "bear",
        "market_regime_raw": "bear",
        "market_regime_candidate": "",
        "market_regime_streak": 0,
        "market_breadth": 0.0,
        "market_trend_strength": 0.0,
        "market_volatility": 0.0,
        "market_volatility_veto": False,
        "risk_on": False,
        "local_fast_on": False,
        "local_fast_buckets": [],
        "risk_on_breadth": 0.0,
        "candidate_total": 0,
        "cluster_total": 0,
    }


def _effective_base_weight(config: PortfolioConfig, risk_state: dict[str, object]) -> float:
    regime = str(risk_state.get("market_regime", "bear"))
    if regime == "bull":
        return config.bull_base_weight
    elif regime == "sideways":
        return config.neutral_base_weight
    return config.bear_base_weight


def _effective_defense_weight(config: PortfolioConfig, risk_state: dict[str, object]) -> float:
    regime = str(risk_state.get("market_regime", "bear"))
    if regime == "bull":
        return config.bull_defense_weight
    elif regime == "sideways":
        return config.neutral_defense_weight
    return config.bear_defense_weight


def _effective_attack_weight(config: PortfolioConfig, risk_state: dict[str, object]) -> float:
    regime = str(risk_state.get("market_regime", "bear"))
    if regime == "bull":
        return config.bull_attack_weight
    elif regime == "sideways":
        return config.neutral_attack_weight
    return config.bear_attack_weight


def _market_volatility(cluster_map: pd.DataFrame) -> float:
    raw_values = cluster_map.get("volatility_20")
    if raw_values is None:
        return 0.0
    values = pd.to_numeric(pd.Series(raw_values), errors="coerce")
    values = values.replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return 0.0
    return float(values.median())


def _local_fast_buckets(cluster_map: pd.DataFrame, config: PortfolioConfig, risk_state: dict[str, object] | None = None) -> list[str]:
    if cluster_map.empty:
        return []
    if isinstance(risk_state, dict) and str(risk_state.get("market_regime", "bear")) != "bull":
        return []
    if isinstance(risk_state, dict) and float(risk_state.get("market_volatility", 0.0)) >= config.market_volatility_veto_threshold:
        return []
    active: list[tuple[str, float, float]] = []
    for bucket, part in cluster_map.groupby("bucket", sort=False):
        breadth = _risk_on_breadth(part)
        trend_strength = _trend_strength(part)
        volatility = _market_volatility(part)
        if breadth < config.local_fast_breadth_threshold:
            continue
        if trend_strength < config.local_fast_trend_threshold:
            continue
        if volatility >= config.market_volatility_veto_threshold:
            continue
        active.append((str(bucket), trend_strength, breadth))
    active.sort(key=lambda item: (item[1], item[2], item[0]), reverse=True)
    return [bucket for bucket, _, _ in active]


def _macro_risk_signal(clean_daily: pd.DataFrame, config: PortfolioConfig) -> float:
    if not config.enable_macro_signal:
        return 1.0
    lookback = max(config.macro_lookback_days, 5)
    needed = {config.macro_bond_symbol, config.macro_gold_symbol}
    macro_data = clean_daily.loc[clean_daily["symbol"].isin(needed)].copy()
    if macro_data.empty:
        return 1.0
    signals: list[float] = []
    for symbol in sorted(needed):
        series = macro_data.loc[macro_data["symbol"] == symbol, "close"].dropna()
        if len(series) >= lookback:
            mom = float(series.iloc[-1]) / float(series.iloc[-lookback]) - 1.0
            signals.append(mom)
    if not signals:
        return 1.0
    avg_signal = sum(signals) / len(signals)
    if avg_signal < -0.02 and all(s < 0 for s in signals):
        return config.macro_risk_off_scale
    return 1.0


def _bear_defensive_priority(row: pd.Series, config: PortfolioConfig) -> float:
    category = str(row.get("category", ""))
    name = str(row.get("name", ""))
    dedup_key = str(row.get("dedup_key", ""))
    text = f"{name} {dedup_key}"
    if category in {"bond", "commodity"}:
        return config.bear_defensive_boost
    if category in {"strategy", "broad_based", "industry", "theme"} and any(keyword in text for keyword in ("红利", "低波", "防御", "高股息", "短债", "国债", "黄金")):
        return config.bear_defensive_boost * 0.8
    return 0.0


def _empty_portfolio_result() -> PortfolioResult:
    return PortfolioResult(
        bucket_map=pd.DataFrame(),
        base_positions=pd.DataFrame(),
        defense_positions=pd.DataFrame(),
        attack_positions=pd.DataFrame(),
        target_portfolio=pd.DataFrame(),
        risk_report={
            "market_regime": "bear",
            "market_regime_raw": "bear",
            "market_regime_candidate": "",
            "market_regime_streak": 0,
            "market_breadth": 0.0,
            "market_trend_strength": 0.0,
            "market_volatility": 0.0,
            "market_volatility_veto": False,
            "risk_on": False,
            "local_fast_on": False,
            "local_fast_buckets": [],
            "risk_on_breadth": 0.0,
            "candidate_total": 0,
            "cluster_total": 0,
            "target_weight_sum": 0.0,
            "target_base_weight": 0.0,
            "target_defense_weight": 0.0,
            "target_attack_weight": 0.0,
        },
    )

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_quality_metrics(clean_daily: pd.DataFrame, spot: pd.DataFrame) -> pd.DataFrame:
    base_cols = ["symbol", "code", "name", "category", "dedup_key"]
    extra_cols = [col for col in ("amount", "total_value", "float_value", "latest_shares", "data_date") if col in spot.columns]
    base = spot[base_cols + extra_cols].copy()
    base = base.rename(columns={"amount": "spot_amount"})

    if clean_daily.empty:
        for col in (
            "first_date",
            "last_date",
            "listing_days",
            "rows",
            "avg_amount_20",
            "avg_amount_60",
            "valid_ratio_60",
            "zero_turnover_days",
            "suspicious_jump_days",
            "global_missing_days_60",
            "close_vs_sma200",
            "ma_window_used",
            "momentum_score",
            "momentum_source",
        ):
            base[col] = pd.NA
        return base

    global_dates = sorted(clean_daily["date"].dropna().unique())
    last_60_dates = set(global_dates[-60:])
    grouped = clean_daily.groupby("symbol", sort=False)
    metrics = grouped.agg(
        first_date=("date", "min"),
        last_date=("date", "max"),
        listing_days=("date", "nunique"),
        rows=("date", "size"),
        zero_turnover_days=("zero_turnover", "sum"),
        suspicious_jump_days=("suspicious_jump", "sum"),
    )
    metrics["avg_amount_20"] = grouped["amount"].apply(lambda s: s.tail(20).mean())
    metrics["avg_amount_60"] = grouped["amount"].apply(lambda s: s.tail(60).mean())
    valid_last_60 = clean_daily.loc[clean_daily["date"].isin(last_60_dates)].groupby("symbol")["date"].nunique()
    metrics["valid_ratio_60"] = valid_last_60.div(max(len(last_60_dates), 1))
    metrics["global_missing_days_60"] = 0
    metrics = metrics.reset_index()
    result = base.merge(metrics, on="symbol", how="left")
    if "spot_amount" in result.columns:
        result["avg_amount_20"] = result["avg_amount_20"].fillna(result["spot_amount"])
        result["avg_amount_60"] = result["avg_amount_60"].fillna(result["spot_amount"])
    tech = _compute_technical_metrics(clean_daily)
    if not tech.empty:
        result = result.merge(tech, on="symbol", how="left")
    return result


def _compute_technical_metrics(clean_daily: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for symbol, group in clean_daily.sort_values("date").groupby("symbol", sort=False):
        closes = group["close"].dropna().values.astype(float)
        ma_window_used = len(closes)
        if len(closes) >= 200:
            sma200 = float(np.mean(closes[-200:]))
            close_vs_sma200 = (closes[-1] - sma200) / sma200
        else:
            close_vs_sma200 = pd.NA
        if len(closes) < 60:
            records.append({"symbol": symbol, "close_vs_sma200": close_vs_sma200, "ma_window_used": len(closes), "momentum_score": pd.NA})
            continue
        values = closes[-60:]
        momentum_score = _log_momentum_score(values)
        records.append(
            {
                "symbol": symbol,
                "close_vs_sma200": close_vs_sma200,
                "ma_window_used": ma_window_used,
                "momentum_score": momentum_score,
                "momentum_source": "absolute",
            }
        )
    return pd.DataFrame(records)


def _log_momentum_score(values: np.ndarray) -> float:
    y = np.asarray(values, dtype=float)
    y = y[np.isfinite(y) & (y > 0)]
    if len(y) < 2:
        return float("nan")
    y = np.log(y)
    x = np.arange(len(y), dtype=float)
    slope = np.polyfit(x, y, 1)[0]
    if float(np.std(y)) == 0.0:
        return 0.0
    corr = np.corrcoef(x, y)[0, 1]
    r2 = 0.0 if not np.isfinite(corr) else corr * corr
    return float(slope * r2)

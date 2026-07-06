from __future__ import annotations

from dataclasses import dataclass


DEFAULT_CLUSTER_LOOKBACK_DAYS = 252
DEFAULT_TRAILING_STOP_PCT = 0.15
DEFAULT_SLIPPAGE_BPS = 2.0
DEFAULT_IMPACT_BPS = 5.0
DEFAULT_IMPACT_PARTICIPATION = 0.10


@dataclass(frozen=True)
class ClusterConfig:
    lookback_days: int = DEFAULT_CLUSTER_LOOKBACK_DAYS
    min_overlap_days: int = 60
    corr_threshold: float = 0.85


@dataclass(frozen=True)
class PortfolioConfig:
    enabled: bool = True
    base_slots: int = 2
    defense_slots: int = 2
    attack_slots: int = 2
    bull_base_weight: float = 0.60
    bull_defense_weight: float = 0.15
    bull_attack_weight: float = 0.25
    neutral_base_weight: float = 0.50
    neutral_defense_weight: float = 0.35
    neutral_attack_weight: float = 0.15
    bear_base_weight: float = 0.35
    bear_defense_weight: float = 0.65
    bear_attack_weight: float = 0.00
    risk_on_breadth_threshold: float = 0.40
    market_regime_confirm_weeks: int = 2
    market_bull_confirm_weeks: int = 1
    market_bull_trend_threshold: float = 0.60
    market_fast_trend_threshold: float = 0.55
    local_fast_breadth_threshold: float = 0.60
    local_fast_trend_threshold: float = 0.60
    market_volatility_veto_threshold: float = 0.30
    bear_defensive_boost: float = 0.30
    target_volatility: float = 0.0
    enable_risk_parity: bool = False
    sideways_breadth_upper: float = 0.60
    sideways_breadth_lower: float = 0.25
    enable_sector_momentum_filter: bool = False
    sector_momentum_filter_pct: float = 0.50
    enable_macro_signal: bool = False
    macro_bond_symbol: str = "511010"
    macro_gold_symbol: str = "518880"
    macro_lookback_days: int = 20
    macro_risk_off_scale: float = 0.80


@dataclass(frozen=True)
class BacktestConfig:
    start_date: str | None = None
    end_date: str | None = None
    initial_nav: float = 1.0
    transaction_cost_bps: float = 5.0
    min_listing_days: int = 120
    min_avg_amount_20: float = 30_000_000
    min_valid_ratio_60: float = 0.90
    base_rebalance: str = "M"
    defense_rebalance: str = "M"
    attack_rebalance: str = "W-FRI"
    trailing_stop_pct: float = DEFAULT_TRAILING_STOP_PCT
    enable_adaptive_stop: bool = False
    volatility_stop_multiplier: float = 2.5
    slow_stop_pct: float = 0.0
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS
    impact_bps: float = DEFAULT_IMPACT_BPS
    impact_participation: float = DEFAULT_IMPACT_PARTICIPATION

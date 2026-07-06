# portfolio_strategy_standalone

Self-contained A-share ETF three-pool rotation strategy: `base` / `defense` / `attack`.

## Quick Start

```bash
pip install pandas numpy

python run.py backtest --clean-daily-csv clean_etf_daily.csv --output-dir backtest

python -m portfolio_strategy_standalone.cli backtest --clean-daily-csv clean_etf_daily.csv --output-dir backtest
```

## Run Tests

```bash
PYTHONPATH=. pytest portfolio_strategy_standalone/tests/ -v
```

## Architecture

```
cli.py        -> argparse front-end, Chinese-formatted reports
backtest.py   -> daily loop, T+1 execution, trailing stops, NAV tracking
portfolio.py  -> 3-pool selection, market regime detection, weight combine
metrics.py    -> ETF quality metrics (momentum, SMA200, listing days)
config.py     -> 3 frozen dataclasses (BacktestConfig, ClusterConfig, PortfolioConfig)
```

## Backtest Outputs

- `backtest_nav.csv` - daily NAV, returns, turnover, costs, cash weight
- `backtest_positions.csv` - per-date position snapshots with pool/weight columns
- `backtest_trades.csv` - per-rebalance trade logs (buy/sell details in Chinese)
- `backtest_summary.json` - total_return, annual_return, sharpe, calmar, max_drawdown

from __future__ import annotations

from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from portfolio_strategy_standalone.cli import main


if __name__ == "__main__":
    main()

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ag_foundation.cli import main as cli_main


def main() -> None:
    cli_main(["create-catalog", *sys.argv[1:]], enable_logging=True)


if __name__ == "__main__":
    main()

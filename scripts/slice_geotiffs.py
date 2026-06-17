from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path = [path for path in sys.path if path != str(SRC_ROOT)]
sys.path.insert(0, str(SRC_ROOT))


def main() -> None:
    from ag_foundation.cli import main as cli_main

    cli_main(["slice-geotiffs", *sys.argv[1:]], enable_logging=True)


if __name__ == "__main__":
    main()

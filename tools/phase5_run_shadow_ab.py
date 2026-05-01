#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase5_tranche import main  # noqa: E402


if __name__ == "__main__":
    sys.argv = [sys.argv[0], "run-shadow-ab", *sys.argv[1:]]
    raise SystemExit(main())

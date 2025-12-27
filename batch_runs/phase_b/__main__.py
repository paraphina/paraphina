"""
__main__.py

Allows running Phase B as a module:
    python3 -m batch_runs.phase_b --help
"""

import sys
from batch_runs.phase_b.cli import main

if __name__ == "__main__":
    sys.exit(main())


"""Allow running as: python3 -m batch_runs.phase_a"""
from .cli import main
import sys

if __name__ == "__main__":
    sys.exit(main())


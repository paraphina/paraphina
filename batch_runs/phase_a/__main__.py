"""
Allow running as: python3 -m batch_runs.phase_a

Default entry point is promote_pipeline for Phase A-2.
For legacy cli.py interface, use: python3 -m batch_runs.phase_a.cli
"""
import sys

# Default to promote_pipeline as the Phase A-2 entry point
from .promote_pipeline import main

if __name__ == "__main__":
    sys.exit(main())


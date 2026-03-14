#!/usr/bin/env python3
"""Mint a fresh Paradex JWT using the official paradex-py SDK.

Required env:
- PARADEX_L2_ADDRESS
- PARADEX_L2_PRIVATE_KEY

Optional env:
- PARADEX_PY_ENV=prod|testnet

The script prints the JWT to stdout and nothing else on success.
"""

from __future__ import annotations

import argparse
import os
import sys

VALID_ENVS = ("prod", "testnet", "nightly")


def _require(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(f"missing required env: {name}")
    return value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mint a fresh Paradex JWT using paradex_py."
    )
    parser.add_argument(
        "--env",
        choices=VALID_ENVS,
        default=os.environ.get("PARADEX_PY_ENV", "prod").strip().lower() or "prod",
        help="Paradex environment. Defaults to PARADEX_PY_ENV or prod.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        from paradex_py import ParadexSubkey
    except ImportError:
        print(
            "paradex_py is not installed. Install it with: python3 -m pip install paradex_py",
            file=sys.stderr,
        )
        return 2

    client = ParadexSubkey(
        env=args.env,
        l2_address=_require("PARADEX_L2_ADDRESS"),
        l2_private_key=_require("PARADEX_L2_PRIVATE_KEY"),
    )
    token = client.account.jwt_token
    if not token:
        print("failed to obtain Paradex JWT", file=sys.stderr)
        return 1
    print(token)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

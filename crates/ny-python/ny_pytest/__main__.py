# Copyright 2026 Andrew Yates
# Author: Andrew Yates <andrewyates.name@gmail.com>
# Licensed under the Apache License, Version 2.0

"""Entry point for ny_pytest.

Provides a minimal CLI so the pytest plugin package registers as a
module entry point for integration audits and for `python -m` usage.
"""

from __future__ import annotations

import argparse

from ny_pytest import __version__ as ny_version


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="ny_pytest",
        description="pytest plugin for ny neural network verification",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"ny_pytest {ny_version}",
    )
    parser.parse_args()
    print("ny_pytest is a pytest plugin. Install ny[pytest] and run pytest.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

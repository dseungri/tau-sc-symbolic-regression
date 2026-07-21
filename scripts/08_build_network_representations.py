#!/usr/bin/env python3
"""Thin wrapper for `tau-sr build-networks`."""

from __future__ import annotations

import sys

from tau_sr.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["build-networks", *sys.argv[1:]]))

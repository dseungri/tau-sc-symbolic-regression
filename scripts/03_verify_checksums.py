#!/usr/bin/env python3
"""Thin wrapper for `tau-sr verify-checksums`."""

from __future__ import annotations

import sys

from tau_sr.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["verify-checksums", *sys.argv[1:]]))

#!/usr/bin/env python3
"""Deprecated: use evaluate_sam3.py instead."""
import warnings
import sys
from pathlib import Path

warnings.warn(
    "evaluate_model.py is deprecated. Use evaluate_sam3.py instead.",
    FutureWarning,
    stacklevel=1,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_sam3 import main

if __name__ == "__main__":
    main()

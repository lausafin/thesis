#!/usr/bin/env python3
"""Regenerate thesis results tables and figures.

Full step-by-step instructions: REPRODUCTION.md at the repository root.
"""

import sys
from pathlib import Path

THESIS_ROOT = Path(__file__).resolve().parents[1]
if str(THESIS_ROOT) not in sys.path:
    sys.path.insert(0, str(THESIS_ROOT))

from scripts.thesis_results.export import main

if __name__ == "__main__":
    main()

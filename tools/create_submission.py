"""Compatibility wrapper for the HW4 submission exporter.

The canonical implementation lives in ``src/infer_hw4.py``. This wrapper keeps
older commands that call ``tools/create_submission.py`` working with the same
arguments and behavior.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.infer_hw4 import main


if __name__ == "__main__":
    main()

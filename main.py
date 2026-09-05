"""Backwards-compatible entry point.

The original PythonEmotionDetecter was this single script. The project is now
the ``affectlab`` package, and ``python main.py`` simply launches
``affectlab live``. Any extra arguments are passed through, for example::

    python main.py --backend facs --mirror
"""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from affectlab.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(["live", *sys.argv[1:]]))

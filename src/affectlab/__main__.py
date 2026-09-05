"""Allow ``python -m affectlab``."""

from affectlab.cli import main

if __name__ == "__main__":
    raise SystemExit(main())

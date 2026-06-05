"""Enable ``python -m tdmr2d ...``."""
import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())

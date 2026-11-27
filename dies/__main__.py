"""`python -m dies ...`: the command line of DiES without installing the `dies` command (see dies/cli.py)."""

import sys

from dies.cli import main

sys.exit(main())

"""File-executed example import bridge to the project package.

The actual implementation remains in the sibling project directory.  This
bridge lets ``py examples/run_smoke.py`` work without installation.
"""

from pathlib import Path

__path__ = [str(Path(__file__).resolve().parents[2] / "structural_experiment")]

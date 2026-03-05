"""Root conftest: remove project root from sys.path to avoid shadowing stdlib modules.

profile.py lives at the project root for use as a script (python profile.py).
Without this conftest, pytest adds the project root to sys.path, which causes
``import profile`` inside cProfile to resolve to our profile.py instead of stdlib.
"""
import sys
from pathlib import Path

_project_root = str(Path(__file__).parent)

# Remove the project root entry that pytest may have inserted.
# Tests already have src/ on sys.path via tests/conftest.py.
if _project_root in sys.path:
    sys.path.remove(_project_root)

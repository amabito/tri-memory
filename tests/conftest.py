import sys
from pathlib import Path
# Ensure src/ is on the path for editable installs fallback
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

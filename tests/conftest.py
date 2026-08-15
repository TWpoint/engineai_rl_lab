from __future__ import annotations

import sys
from pathlib import Path

# The development environment may have the main checkout installed editable.
# Always exercise the source tree belonging to this worktree.
SOURCE_ROOT = Path(__file__).parents[1] / "source" / "engineai_rl_lab"
sys.path.insert(0, str(SOURCE_ROOT))

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Tests always use the development engines and never touch real model weights.
os.environ.setdefault("ASR_ENGINE", "none")
os.environ.setdefault("BRANCH_A_ENGINE", "dev-rules")
os.environ.setdefault("BRANCH_B_ENGINE", "dev-rules")
os.environ.setdefault("PRELOAD_MODELS", "false")

"""Pre-build the TRAID-Cascade set with the injection model (runs in parallel with the judge queue)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from traid.llm import make_llm
import run
subset = sys.argv[1] if len(sys.argv) > 1 else "alg"
inj = make_llm("ministral-8b-latest", max_tokens=4096, temperature=0.7, min_interval_s=31)
out = run.load_cascade(subset, inj, 0)
print("cascade", subset, len(out))

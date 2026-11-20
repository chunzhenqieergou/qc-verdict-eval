"""Post-hoc truncation sensitivity; identical batches for prefix comparisons."""
from run import ROOT, evaluate
evaluate(ROOT/"sensitivity"/"base-256",None,budget=256)

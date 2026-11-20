"""Sensitivity to semicolon, comma and sentence-delimited explicit verdicts."""
from collections import defaultdict
from pathlib import Path
import re
import sys
import unicodedata

here=Path(__file__).resolve().parent
project=here.parents[1]/"experiments/scenario-sft"
sys.path.insert(0,str(project))
from data import save
from score import load
cases=load(project/"data/far.json")
def measure(f):
    by_id={r["id"]:r for r in load(f)}
    pairs=defaultdict(list)
    correct=unparsed=0
    for row in cases:
        text=unicodedata.normalize("NFKC",by_id[row["id"]]["text"])
        chunks=re.split(r"[\n;,。.!！]",text)
        labels={t.strip() for t in chunks if t.strip() in ("不成立","成立")}
        answer=(next(iter(labels))=="成立") if len(labels)==1 else None
        unparsed+=answer is None
        hit=answer==row["positive"]
        correct+=hit
        pairs[row["pair"]].append(hit)
    return dict(correct=correct,pair=sum(all(v) for v in pairs.values()),unparsed=unparsed)
out={f.relative_to(project).as_posix():measure(f) for folder in ("runs","order-probe","order-training") for f in (project/folder).rglob("predictions.json")}
save(here/"delimiter-sensitivity.json",out)
print(out)

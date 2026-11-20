"""Independent verification of per-rule paired outcomes and follow-through."""
import json, re, sys, unicodedata
from collections import defaultdict
from pathlib import Path
import numpy as np
here=Path(__file__).resolve().parent
proj=here.parents[1]/"experiments/scenario-sft"
sys.path.insert(0,str(proj))
from data import save, RULES
from score import load, field
cases=load(proj/"data/far.json")
allids=[r["id"] for r in load(proj/"data/near.json")+cases]
def measure(path):
    preds=load(path)
    assert [r["id"] for r in preds]==allids
    pairs=defaultdict(list)
    hits=ordered=positive=unclear=capped=contradictions=numeric_correct=0
    for c,p in zip(cases,preds[-len(cases):]):
        s=unicodedata.normalize("NFKC",p["text"])
        lines=s.splitlines()
        clean=["".join(ch for ch in line if not ch.isspace() and not unicodedata.category(ch).startswith("P")) for line in lines]
        labels=[(i,v) for i,v in enumerate(clean) if v in ("成立","不成立")]
        valid=len({v for _,v in labels})==1
        value=(labels[0][1]=="成立") if valid else None
        if valid:
            before="\n".join(lines[:labels[0][0]])
            ordered+=int(all(field(before,k) is not None for k in ("统计量","阈值")))
        else: unclear+=1
        positive+=int(value is True)
        correct=value==c["positive"]
        hits+=int(correct)
        numbers=all((f:=field(s,k)) is not None and abs(f-c[key])<1e-7 for k,key in [("统计量","value"),("阈值","limit")])
        numeric_correct+=int(numbers)
        contradictions+=int(numbers and valid and not correct)
        capped+=p["capped"]
        pairs[(c["rule"],c["pair"])].append(correct)
    assert len(pairs)==60 and all(len(p)==2 for p in pairs.values())
    by_rule={r:sum(all(v) for (rr,_),v in pairs.items() if rr==r) for r in RULES}
    return dict(correct=hits,both=sum(by_rule.values()),by_rule=by_rule,observed_numeric_first=ordered,predicted_positive=positive,unparsed=unclear,capped=capped,numeric_correct=numeric_correct,contradictions=contradictions)
results={}
for s in ["base-0"]+[f"{a}-{seed}" for a in ["paraphrase","scenario"] for seed in [42,43,44]]:
    for key,prefix in [("original","runs"),("reordered","order-probe")]:
        results[f"{s}/{key}"]=measure(proj/prefix/s/"predictions.json")
    if s.startswith("scenario"):
        for prompt in ["original","reordered"]:
            results[f"trained-{s}/{prompt}"]=measure(proj/"order-training"/s/prompt/"predictions.json")
rng=np.random.default_rng(20261118)
ix=rng.integers(0,12,(10000,12))
def contrast(arm,other):
    deltas=np.array([[results[other.format(seed=s)]["by_rule"][r]-results[f"{arm}-{s}/original"]["by_rule"][r] for r in RULES] for s in [42,43,44]],dtype=float)/5
    dr=deltas.mean(axis=0)
    return dict(percentage_points=round(100*dr.mean(),3),ci=(100*np.quantile(dr[ix].mean(axis=1),[.025,.975])).round(3).tolist())
comparisons={arm:contrast(arm,arm+"-{seed}/reordered") for arm in ["paraphrase","scenario"]}
comparisons["matched_retraining"]=contrast("scenario","trained-scenario-{seed}/reordered")
save(here/"verified-scores.json",dict(groups=results,paired_changes=comparisons))
for name,m in results.items():
    print(name,*[m[k] for k in ["correct","both","observed_numeric_first","predicted_positive","unparsed","capped","numeric_correct","contradictions"]])
print(comparisons)


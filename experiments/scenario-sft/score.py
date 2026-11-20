"""Score predeclared fields and paired decisions; keep raw generations for semantic review."""
import hashlib
import json
import re
import unicodedata
from collections import defaultdict
import numpy as np
from data import DATA, ROOT, RULES, save

def load(path):
    return json.loads(path.read_text(encoding="utf8"))

def field(text,label):
    pattern = label + r"\s*=\s*([+-]?\d+(?:\.\d+)?)(?=\s*(?:[,;。\n]|$))"
    match = re.search(pattern,text)
    return float(match[1]) if match else None

def grade(case,pred):
    t=unicodedata.normalize("NFKC",pred["text"]).strip()
    line=t.splitlines()[0].strip().rstrip("。.!！") if t else ""
    decision={"成立":True,"不成立":False}.get(line)
    v=field(t,"统计量")
    limit=field(t,"阈值")
    hit=decision==case["positive"]
    vh=v is not None and np.isclose(v,case["value"])
    th=limit is not None and np.isclose(limit,case["limit"])
    return dict(id=case["id"],rule=case["rule"],pair=case["pair"],split=case["split"],decision=int(hit),value=int(vh),threshold=int(th),points=int(hit)+int(vh)+int(th),all_three=int(hit and vh and th),explicit_wrong=int(decision is not None and not hit),unparsed_decision=int(decision is None),unparsed_value=int(v is None),unparsed_threshold=int(limit is None),cap=int(pred["capped"]))

def summarize(rows):
    pairs=defaultdict(list)
    for r in rows: pairs[r["pair"]].append(r["decision"])
    assert all(len(v)==2 for v in pairs.values())
    n=len(rows)
    return dict(n=n,decision=sum(r["decision"] for r in rows)/n,pair=sum(all(v) for v in pairs.values())/len(pairs),coverage=sum(r["points"] for r in rows)/(3*n),all_three=sum(r["all_three"] for r in rows)/n,**{k:sum(r[k] for r in rows) for k in ("explicit_wrong","unparsed_decision","unparsed_value","unparsed_threshold","cap")})

def score():
    cases=load(DATA/"near.json")+load(DATA/"far.json")
    scores={}
    details={}
    for name in ["base-0"]+[f"{arm}-{s}" for arm in ["paraphrase","scenario"] for s in [42,43,44]]:
        preds=load(ROOT/"runs"/name/"predictions.json")
        assert [r["id"] for r in cases]==[r["id"] for r in preds]
        rows=[grade(c,p) for c,p in zip(cases,preds)]
        scores[name]={s:summarize([r for r in rows if r["split"]==s]) for s in ["near","far"]}
        details[name]=rows
    rng=np.random.default_rng(20261118)
    ix=rng.integers(0,len(RULES),(10000,len(RULES)))
    contrasts={}
    for metric in ["decision","pair","coverage","all_three"]:
        d=[]
        for rule in RULES:
            vals=[]
            for seed in [42,43,44]:
                per_arm=[summarize([r for r in details[f"{a}-{seed}"] if r["rule"]==rule and r["split"]=="far"])[metric] for a in ["scenario","paraphrase"]]
                vals.append(per_arm[0]-per_arm[1])
            d.append(np.mean(vals))
        contrasts[metric]={"difference":float(np.mean(d)),"rule_cluster_ci":np.quantile(np.array(d)[ix].mean(axis=1),[.025,.975]).tolist(),"rule_delta":dict(zip(RULES,d))}
    save(ROOT/"scores.json",dict(runs=scores,far_contrasts=contrasts))
    save(ROOT/"item-scores.json",details)
    print(json.dumps(scores,indent=1))
    print(json.dumps({k:[v["difference"],v["rule_cluster_ci"]] for k,v in contrasts.items()}))
if __name__=="__main__":
    score()

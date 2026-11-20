import hashlib,json,random,re,string
from collections import Counter
from pathlib import Path
R=Path(__file__).resolve().parent
OLD=R.parent/"scenario-sft"
def dump(name,obj):
    out=R/name
    text=json.dumps(obj,ensure_ascii=False,indent=2)
    if out.exists() and out.read_text(encoding="utf8")!=text:
        raise ValueError(f"Frozen file changed: {out}")
    out.write_text(text,encoding="utf8")
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def build():
    evt=[json.loads(s) for s in (R/"gemini.jsonl").read_text(encoding="utf-8-sig").splitlines() if s.strip()]
    result=evt[-1]["result"]
    assert result["status"]=="SUCCESS" and result["response"]
    assert not any(e.get("step_update",{}).get("step_type")=="tool" for e in evt), "author accessed tools"
    templates=json.loads(re.search(r"\[\s*\{.*\}\s*\]",result["response"],re.S)[0])
    counts=Counter(t["rule"] for t in templates)
    assert counts==dict.fromkeys(("single","cp","cpu","cpl","cpk","zone"),4),counts
    sources={"single":"https://support.minitab.com/en-us/minitab/help-and-how-to/quality-and-process-improvement/acceptance-sampling/how-to/attributes-acceptance-sampling/interpret-the-results/all-statistics-and-graphs/","capability":"https://support.minitab.com/en-us/minitab/help-and-how-to/quality-and-process-improvement/capability-analysis/how-to/capability-analysis/normal-capability-analysis/interpret-the-results/all-statistics-and-graphs/potential-within-capability/","zone":"https://www.jmp.com/content/dam/jmp/documents/en/support/jmp19/quality-and-process-methods.pdf"}
    cases=[]
    for i,t in enumerate(templates):
        rule=t["rule"]
        for k in range(2):
            j=2*i+k
            for yes in [True,False]:
                target=[1.25,1.75,2,1.5][j%4]
                value=target+([0,.25,.50][j%3] if yes else -[.25,.50][j%2])
                sd=[.4,.8,1.2,1.6][j%4]
                mean=30+j*3
                lo=mean-3*sd*(value if rule=="cpl" or (rule=="cpk" and k==1) else target+1.5)
                hi=mean+3*sd*(value if rule=="cpu" or (rule=="cpk" and k==0) else target+1.5)
                if rule=="cp":hi=lo+6*sd*value
                c=1+j%7
                x=c-(j%2) if yes else c+1+j%3
                seq=[.2+j%4*.2, (2.15+j%4*.12 if yes else 1.12+j%4*.10),2.3+j%5*.1]
                d=dict(n=65+j,c=c,x=x,mean=mean,sd=sd,lo=lo,hi=hi,target=target,seq="、".join(f"{a:+.2f}" for a in seq))
                fields={f for _,f,_,_ in string.Formatter().parse(t["text"]) if f}
                required={"single":"n c x","cp":"lo hi sd target","cpu":"mean hi sd target","cpl":"mean lo sd target","cpk":"mean hi lo sd target","zone":"seq"}[rule].split()
                assert fields==set(required),(t,fields)
                d={key:(float(f"{val:.5f}") if isinstance(val,float) else val) for key,val in d.items() if key in fields}
                q=t["text"].format(**d)
                if rule=="single": v,threshold=x,c
                elif rule=="zone": v,threshold=sum(float(a)>2 for a in d["seq"].split("、")),2
                else:
                    if rule=="cp": v=(d["hi"]-d["lo"])/(6*d["sd"])
                    elif rule=="cpl": v=(d["mean"]-d["lo"])/(3*d["sd"])
                    elif rule=="cpu": v=(d["hi"]-d["mean"])/(3*d["sd"])
                    else:v=min(d["hi"]-d["mean"],d["mean"]-d["lo"])/(3*d["sd"])
                    threshold=d["target"]
                v=round(v,6)
                assert (v<=threshold if rule=="single" else v>=threshold)==yes
                pair=f"{i:02}-{k}"
                cases.append(dict(id=f"B{len(cases):03}",rule=rule,pair=pair,question=q,positive=yes,value=v,limit=threshold,inputs=d,source=sources.get(rule,sources["capability"])))
    for a,b in zip(cases[::2],cases[1::2]):
        assert sum(a["inputs"][k]!=b["inputs"][k] for k in a["inputs"])==1
    oldq={r["question"] for f in ["near","far","paraphrase","scenario"] for r in json.loads((OLD/"data"/(f+".json")).read_text(encoding="utf8"))}
    assert len({r["question"] for r in cases})==96 and all(r["question"] not in oldq for r in cases)
    random.Random(1284).shuffle(cases)
    dump("cases.json",cases)
    weights=[f for s in [42,43,44] for group in ["runs","order-training"] for f in [(OLD/group/f"scenario-{s}"/"adapter"/"adapter_model.safetensors")]]
    dump("manifest.json",dict(cases=sha(R/"cases.json"),protocol=sha(R/"protocol.md"),reviewer=evt[0]["init"]["model"],sources=sources,weights={str(f.relative_to(OLD)):sha(f) for f in weights}))
if __name__=="__main__":build()

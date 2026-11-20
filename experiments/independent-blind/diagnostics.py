import json,re,unicodedata
from collections import defaultdict
from pathlib import Path
from freeze import OLD,R
from score import verdict,load

rules={"cp":"cp","cpu":"cpu","cpl":"cpl","cpk":"cpk","single":"single","zone":"weco2"}
def field(text,name):
    text=unicodedata.normalize("NFKC",text)
    m=re.search(name+r"\s*=\s*([+-]?\d+(?:\.\d+)?)(?=\s*(?:[,;。\n]|$))",text)
    return float(m[1]) if m else None
def audit(cases,ps):
    byid={p["id"]:p for p in ps}
    pairs=defaultdict(list); good=wrong=unknown=0
    for c in cases:
        p=byid[c["id"]];label=verdict(p["text"],False)
        pairs[(c["rule"],c["pair"])].append(label==c["positive"])
        if all((v:=field(p["text"],n)) is not None and abs(v-c[k])<1e-6 for n,k in [("统计量","value"),("阈值","limit")]):
            good+=1
            wrong+=int(label is not None and label!=c["positive"])
            unknown+=label is None
    assert all(len(v)==2 for v in pairs.values())
    return dict(pair_hits=sum(all(v) for v in pairs.values()),pairs=len(pairs),num_correct=good,contradiction=wrong,unparsed_given_num_correct=unknown)
data={}
for phase in ["historical","blind"]:
    cases=load(R/"cases.json") if phase=="blind" else [c for c in load(OLD/"data/far.json") if c["rule"] in rules.values()]
    for s in [42,43,44]:
        for o in ["original","reordered"]:
            f=R/"outputs"/f"{s}-{o}.json" if phase=="blind" else (OLD/"runs"/f"scenario-{s}"/"predictions.json" if o=="original" else OLD/"order-training"/f"scenario-{s}"/"reordered"/"predictions.json")
            data[f"{phase}/{s}/{o}"]=audit(cases,load(f))
(R/"diagnostics.json").write_text(json.dumps(data,indent=2),encoding="utf8")
lines=["","## 相同规则范围的事后诊断","","| 题库与顺序 | 成对全对（汇总三种子） | 数值与阈值均对 | 其中明确判错 | 其中无可解析结论 |","|---|---:|---:|---:|---:|"]
for phase in ["historical","blind"]:
    for o in ["original","reordered"]:
        group=[data[f"{phase}/{s}/{o}"] for s in [42,43,44]]
        sums={k:sum(r[k] for r in group) for k in group[0]}
        lines.append("| "+" | ".join([f"{phase} / {o}",f"{sums['pair_hits']}/{sums['pairs']}",str(sums["num_correct"]),str(sums["contradiction"]),str(sums["unparsed_given_num_correct"])])+" |")
lines+=["","两个题库都限定同六项规则，配对题数量不同。次数保留同题跨种子的重复回答。部分数值是人为指定的题设，未经现场工程可行性校核。"]
p=R/"RESULTS.md"
prefix=p.read_text(encoding="utf8").split("\n## 相同规则范围的事后诊断")[0].rstrip()
p.write_text(prefix+"\n"+"\n".join(lines),encoding="utf8")

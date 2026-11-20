from collections import defaultdict
import json,re,unicodedata
from pathlib import Path
import numpy as np
from freeze import R,sha
def load(p):return json.loads(p.read_text(encoding="utf-8-sig"))
def verdict(s,loose):
    text=unicodedata.normalize("NFKC",s)
    pieces=re.split(r"[\n;,。.!！]",text) if loose else text.splitlines()
    labels={"".join(c for c in line if not c.isspace() and not unicodedata.category(c).startswith("P")) for line in pieces}&{"成立","不成立"}
    return next(iter(labels))=="成立" if len(labels)==1 else None
def main():
    assert sha(R/"cases.json")==load(R/"manifest.json")["cases"]
    cases=load(R/"cases.json")
    metrics={}
    for f in sorted((R/"outputs").glob("*.json")):
        ps=load(f);assert [r["id"] for r in cases]==[p["id"] for p in ps]
        data={}
        for loose in [False,True]:
            pairs=defaultdict(list);right=missing=0
            for c,p in zip(cases,ps):
                d=verdict(p["text"],loose);hit=d==c["positive"];right+=hit;missing+=d is None
                pairs[(c["rule"],c["pair"])].append(hit)
            assert len(pairs)==48 and all(len(v)==2 for v in pairs.values())
            hits={r:sum(all(v) for (rule,k),v in pairs.items() if rule==r) for r in sorted({k[0] for k in pairs})}
            data["loose" if loose else "strict"]=dict(right=right,pairs=sum(hits.values()),missing=missing,by_rule=hits,cap=sum(p["capped"] for p in ps))
        metrics[f.stem]=data
    assert len(metrics)==8
    def avg(names,k,mode="strict"):
        return np.mean([metrics[n][mode][k] for n in names])
    groups=[("底座，先结论",["base-original"]),("底座，先数值",["base-reordered"]),("微调，先结论",[f"{i}-original" for i in [42,43,44]]),("微调，先数值",[f"{i}-reordered" for i in [42,43,44]])]
    rules=list(metrics["42-original"]["strict"]["by_rule"])
    def by(o,r):return sum(metrics[f"{i}-{o}"]["strict"]["by_rule"][r] for i in [42,43,44])
    diffs=np.array([(by("reordered",r)-by("original",r))/24 for r in rules])
    ix=np.random.default_rng(1284).integers(0,6,(10000,6));lo,hi=100*np.quantile(diffs[ix].mean(axis=1),[.025,.975])
    lines=["# 独立命题盲测结果","","冻结 24 个 Gemini 新编中文模板、六项规则、每模板两套数字，共 96 题（48 对）。模型权重固定。每个表格百分比在底座上来自一次评测，在微调组上是种子 42、43、44 的平均。","","| 模型与作答顺序 | 单题正确 | 成对全对 | 放宽分隔符后的成对全对 | 独立行无法解析次数 | 截断次数 |","|---|---:|---:|---:|---:|---:|"]
    for title,ns in groups:lines.append(f"| {title} | {100*avg(ns,'right')/96:.1f}% | {100*avg(ns,'pairs')/48:.1f}% | {100*avg(ns,'pairs','loose')/48:.1f}% | {sum(metrics[n]['strict']['missing'] for n in ns)} | {sum(metrics[n]['strict']['cap'] for n in ns)} |")
    lines += ["",f"主比较中，数值先行减结论先行的成对正确率差值为 {100*diffs.mean():+.2f} 个百分点。配对按规则簇重采样的描述性 95% 区间为 [{lo:+.2f}, {hi:+.2f}]。只有六个规则簇，三个固定种子共享题库。","","| 规则 | 先结论（对/24） | 先数值（对/24） |","|---|---:|---:|"]+[f"| {r} | {by('original',r)} | {by('reordered',r)} |" for r in rules]
    lines+=["","比较同时切换了训练与推理的前两行顺序。复用的是六种已学过的规则；题面、工件、数值配置由本轮新建。四种能力指数占到三分之二的规则。该设计检验合成质量判断任务的模板迁移，未覆盖真实维修工单、其他底座或人类独立评审。未测这批新题在预训练语料中的概念重叠。详细预测和解析缺失均保留。","", "[冻结清单与来源](manifest.json) · [题库](cases.json) · [逐模型统计](scores.json) · [原始回答](outputs/)"]
    (R/"scores.json").write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding="utf8")
    (R/"RESULTS.md").write_text("\n".join(lines),encoding="utf8")
if __name__=="__main__":main()

import json, sys
from pathlib import Path

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root.parent / "independent-blind"))
from score import verdict, load

cases = load(root / "cases.json")
rows = ["# 公开检验记录的外部核验", "", "表中数字为独立行判定正确的模型次数。两个底座列各运行一次，两组微调各运行种子 42、43、44。每行只有一个检验个案。", "", "|记录|来源类型|实测/接收数|应判|底座先结论|底座先数值|微调先结论|微调先数值|","|---|---|---:|---|---:|---:|---:|---:|"]
groups = [["base-original"], ["base-reordered"], [f"{s}-original" for s in (42,43,44)], [f"{s}-reordered" for s in (42,43,44)]]
detail = {}
for group in groups:
    for name in group:
        preds = load(root / "outputs" / (name+".json"))
        assert [p["id"] for p in preds] == [c["id"] for c in cases]
        detail[name] = [dict(id=c["id"], verdict=verdict(p["text"],False), correct=verdict(p["text"],False)==c["positive"], capped=p["capped"]) for p,c in zip(preds,cases)]
for i,c in enumerate(cases):
    counts = [f"{sum(detail[name][i]['correct'] for name in g)}/{len(g)}" for g in groups]
    rows.append("|"+"|".join([c["id"],c["tier"],f"{c['x']}/{c['ac']}","接收" if c["positive"] else "拒收",*counts])+"|")
rows.extend(["","接收只涉及轻微不合格这一栏。真实抽样子集只有一条接收记录；唯一拒收记录是批量一台的全检。两份 VICC 报告均标为示例且共用编号。公开副本未经出具机构二次认证，预训练语料是否包含报告未知。本次核验不足以估计真实批次上的泛化收益。","","## 来源",""])
rows.extend(f"- [{c['id']}]({c['source']})" for c in cases)
(root/"results.md").write_text("\n".join(rows),encoding="utf8")
(root/"scores.json").write_text(json.dumps(detail,indent=2,ensure_ascii=False),encoding="utf8")

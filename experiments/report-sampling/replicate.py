import json, sys
from pathlib import Path

root = Path(__file__).resolve().parent
prior = root.parent / "independent-blind"
sys.path.insert(0, str(prior))
import evaluate
from freeze import sha

def save(name, data):
    path = root / name
    content = json.dumps(data, ensure_ascii=False, indent=2)
    if path.exists() and path.read_text(encoding="utf8") != content:
        raise RuntimeError(f"Frozen file differs: {path}")
    path.write_text(content, encoding="utf8")

records = [
    ("3268288","普拉提训练器",5,0,0,"批量11，抽检5","https://shopcdnpro.grainajz.com/category/345655/2633/01d539d97d40b33bdd67c5766d13d8d7/Inspection%20report.pdf"),
    ("3234769","数控雕刻机",1,1,0,"批量1，全检","https://www.scribd.com/document/946774180/StyleCNC-Inspection"),
    ("vicc-cleaner","高压清洗机",80,4,7,"供应商示例","https://www.vicc.com/wp-content/sample-report/Hardline-electric-product-inspection-sample-report.pdf"),
    ("vicc-motor","车窗升降电机",315,0,21,"供应商示例","https://www.vicc.com/wp-content/sample-report/Hardline-auto-part-inspection-sample-report.pdf"),
]
cases = []
for key, product, n, x, ac, tier, url in records:
    q = f"对{product}检验了{n}件。轻微不合格项按件计，接收数为{ac}，实际记录{ x }件。待审核的结论是：该次检验的轻微不合格项符合接收条件。"
    cases.append(dict(id=key,question=q,n=n,x=x,ac=ac,positive=x<=ac,tier=tier,source=url))
save("cases.json",cases)
save("manifest.json",dict(cases=sha(root/"cases.json"),protocol=sha(root/"protocol.md"),weights=evaluate.load(prior/"manifest.json")["weights"]))
evaluate.R = root
evaluate.main()

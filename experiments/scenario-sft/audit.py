"""Reproduce the anonymous spot-check packet for one fixed seed and the base model."""
import json
import random
from data import ROOT, RULES, DATA, save

cases=json.loads((DATA/"far.json").read_text(encoding="utf8"))
chosen={f"far-{r}-{i%5}-{i%2}" for i,r in enumerate(RULES)}
cases={r["id"]:r for r in cases if r["id"] in chosen}
assert len(cases)==12
packet=[]
for run in ["base-0","paraphrase-42","scenario-42"]:
    for row in json.loads((ROOT/"runs"/run/"predictions.json").read_text(encoding="utf8")):
        if row["id"] in cases:
            packet.append((run,row["id"],row["text"]))
random.Random(731).shuffle(packet)
save(ROOT/"audit-packet.json",[dict(code=i+1,id=q,reference=cases[q]["answer"],question=cases[q]["question"],output=t) for i,(_,q,t) in enumerate(packet)])
save(ROOT/"audit-key.json",[dict(code=i+1,run=run) for i,(run,q,t) in enumerate(packet)])

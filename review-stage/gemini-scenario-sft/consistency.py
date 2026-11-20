"""Post-hoc numeric/decision consistency on the frozen far split."""
import json
import re
import sys
import unicodedata
from pathlib import Path

here = Path(__file__).resolve().parent
experiment = here.parents[1] / "experiments/scenario-sft"
sys.path.insert(0, str(experiment))
from score import field, load, grade
from data import save

def decide(rule, value, threshold):
    if rule in {"single", "first_accept", "second_accept"}:
        return value <= threshold
    if rule == "weco3":
        return value > threshold
    return value >= threshold

rows = load(experiment / "data/far.json")
results = {}
for path in sorted((experiment / "runs").glob("*/predictions.json")):
    predictions = {r["id"]:r for r in load(path)}
    both = contradictory = pairs = fixed = 0
    output_labels = {}
    for row in rows:
        p = predictions[row["id"]]
        score = grade(row,p)
        if score["value"] and score["threshold"]:
            both += 1
            contradictory += 1-score["decision"]
        text = unicodedata.normalize("NFKC",p["text"])
        v,t = field(text,"统计量"),field(text,"阈值")
        calculated = None if v is None or t is None else decide(row["rule"],v,t)
        correct = calculated == row["positive"]
        fixed += int(correct)
        output_labels.setdefault(row["pair"],[]).append(correct)
    pairs = sum(all(x) for x in output_labels.values())
    results[path.parent.name] = dict(both_numerical_fields_correct=both,wrong_decision_with_both_fields_correct=contradictory,oracle_rule_id_recomputed_correct=fixed,oracle_rule_id_recomputed_pairs_correct=pairs,n=120,pairs=60)
save(here / "consistency.json",results)
print(json.dumps(results))

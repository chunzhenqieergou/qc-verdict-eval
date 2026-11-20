"""Boundary-relation audit (below / at / above) on dev_v2 predictions.

Quantifies whether fine-tuned adapters learned an equality shortcut (`==` instead of `<=`)
or conflated strict (`>`) vs inclusive (`>=`) boundaries.
"""

from collections import Counter
import json
from pathlib import Path

from rules import RULE_SPECS, RULES_ORDER
from scorer import load_json, parse_verdict

ROOT = Path(__file__).resolve().parent
CASES = {c["id"]: c for c in load_json(ROOT / "data" / "dev_v2.json")}
OUT_DIR = ROOT / "dev_outputs"

COND_FILES = {
    "base_VF": ["base_VF"],
    "base_NF": ["base_NF"],
    "VF_train_VF_infer": [f"VF_train_VF_infer_s{s}" for s in (42, 43, 44)],
    "NF_train_NF_infer": [f"NF_train_NF_infer_s{s}" for s in (42, 43, 44)],
}


def main() -> None:
    report = {}
    lines = [
        "# `dev_v2` 边界位置（below / at / above）捷径诊断报告",
        "",
        "本诊断检验模型是否把包容不等式 `<=` 退化为等号捷径 `==`，或把严格不等式 `>` 误判为 `>=`。",
        "",
    ]

    for cond, stems in COND_FILES.items():
        report[cond] = {}
        lines += [
            f"## 条件：`{cond}`",
            "",
            "| 规则 | 算子 | `below` 正确率 (预测成立) | `at` 正确率 (预测成立) | `above` 正确率 (预测成立) |",
            "|---|---|---|---|---|",
        ]
        preds_all = [load_json(OUT_DIR / f"{st}.json") for st in stems]
        for rule in RULES_ORDER:
            op = RULE_SPECS[rule].op
            tot = Counter()
            ok = Counter()
            pred_t = Counter()
            for preds in preds_all:
                for p in preds:
                    c = CASES[p["id"]]
                    if c["rule"] != rule:
                        continue
                    rel = c["meta"]["boundary_relation"]
                    v = parse_verdict(p["text"])
                    tot[rel] += 1
                    ok[rel] += int(v == c["positive"])
                    pred_t[rel] += int(v is True)
            report[cond][rule] = {
                rel: {"ok": ok[rel], "pred_true": pred_t[rel], "n": tot[rel]}
                for rel in ("below", "at", "above")
                if tot[rel]
            }
            cells = []
            for rel in ("below", "at", "above"):
                if tot[rel]:
                    cells.append(f"{ok[rel]}/{tot[rel]} ({100*ok[rel]/tot[rel]:.0f}%, T={pred_t[rel]})")
                else:
                    cells.append("—")
            lines.append(f"| `{rule}` | `{op}` | " + " | ".join(cells) + " |")
        lines.append("")

    (ROOT / "dev_boundary_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf8"
    )
    (ROOT / "DEV_BOUNDARY_AUDIT.md").write_text("\n".join(lines), encoding="utf8")
    print("Wrote:", ROOT / "DEV_BOUNDARY_AUDIT.md")


if __name__ == "__main__":
    main()

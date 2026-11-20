"""Post-hoc diagnostic audit: deterministic rule-oracle comparator and full error decomposition
on archived scenario-sft (12 rules, 60 pairs) and independent-blind (6 rules, 48 pairs) outputs.

NOTE (per submission framework §4 row 4):
Existing archived predictions do not include a model-predicted comparison operator field.
Therefore, replacing the model verdict with `compare(reference_rule, model_stat, model_limit)`
is explicitly an oracle-rule diagnostic control (uses reference rule identity, only numbers
come from the model). It separates numerical reporting errors from comparison/verdict errors.
"""

from collections import defaultdict
import json
from pathlib import Path
import re
import unicodedata

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "experiments" / "posthoc-comparator-audit"
OLD_DIR = Path(r"d:\Users\Administrator\Desktop\project1\experiments\scenario-sft")
BLIND_DIR = Path(r"d:\Users\Administrator\Desktop\project1\experiments\independent-blind")

RULE_OP = {
    "single": "<=",
    "first_accept": "<=",
    "first_reject": ">=",
    "second_accept": "<=",
    "weco3": ">",
    "weco2": ">=",
    "zone": ">=",  # alias of weco2 in independent-blind
    "weco1": ">=",
    "weco8": ">=",
    "cp": ">=",
    "cpu": ">=",
    "cpl": ">=",
    "cpk": ">=",
}

RULE_FAMILY = {
    "single": "sampling",
    "first_accept": "sampling",
    "first_reject": "sampling",
    "second_accept": "sampling",
    "weco3": "control_chart",
    "weco2": "control_chart",
    "zone": "control_chart",
    "weco1": "control_chart",
    "weco8": "control_chart",
    "cp": "capability",
    "cpu": "capability",
    "cpl": "capability",
    "cpk": "capability",
}

EPS = 1e-6


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def apply_rule_comparator(rule: str, stat: float | None, limit: float | None) -> bool | None:
    if stat is None or limit is None:
        return None
    op = RULE_OP[rule]
    if op == "<=":
        return stat <= limit + 1e-9
    if op == ">=":
        return stat >= limit - 1e-9
    if op == ">":
        return stat > limit + 1e-9
    raise ValueError(f"Unknown op: {op}")


def parse_verdict(text: str) -> bool | None:
    norm = unicodedata.normalize("NFKC", text)
    labels = set()
    for line in norm.splitlines():
        cleaned = "".join(
            ch for ch in line if not ch.isspace() and not unicodedata.category(ch).startswith("P")
        )
        if cleaned in ("成立", "不成立"):
            labels.add(cleaned)
    if len(labels) == 1:
        return next(iter(labels)) == "成立"
    return None


def parse_number(text: str, name: str) -> float | None:
    norm = unicodedata.normalize("NFKC", text)
    m = re.search(name + r"\s*=\s*([+-]?\d+(?:\.\d+)?)(?=\s*(?:[,;。\n]|$))", norm)
    return float(m.group(1)) if m else None


def verify_ground_truth_consistency(cases: list[dict], split_name: str) -> None:
    for c in cases:
        pred = apply_rule_comparator(c["rule"], float(c["value"]), float(c["limit"]))
        assert pred == c["positive"], (
            f"Ground-truth mismatch in {split_name} item {c['id']}: "
            f"rule={c['rule']} val={c['value']} lim={c['limit']} ref={c['positive']} comp={pred}"
        )


def eval_predictions(cases: list[dict], preds: list[dict]) -> dict:
    by_id = {p["id"]: p for p in preds}
    item_buckets = defaultdict(int)
    by_rule = defaultdict(lambda: defaultdict(int))
    by_family = defaultdict(lambda: defaultdict(int))

    pair_orig = defaultdict(dict)
    pair_comp = defaultdict(dict)
    pair_rule = {}

    orig_right = 0
    comp_right = 0
    fixed_items = 0
    broken_items = 0
    both_right_items = 0
    both_wrong_items = 0
    unparsed_verdict = 0
    missing_num_items = 0

    for c in cases:
        p = by_id[c["id"]]
        text = p["text"]
        ref_pos = bool(c["positive"])
        ref_val = float(c["value"])
        ref_lim = float(c["limit"])
        rule = c["rule"]
        fam = RULE_FAMILY[rule]
        pair_key = (rule, c["pair"])
        pair_rule[pair_key] = rule

        v_model = parse_verdict(text)
        stat = parse_number(text, "统计量")
        limit = parse_number(text, "阈值")
        v_comp = apply_rule_comparator(rule, stat, limit)

        stat_ok = stat is not None and abs(stat - ref_val) < EPS
        lim_ok = limit is not None and abs(limit - ref_lim) < EPS
        both_num_ok = stat_ok and lim_ok

        model_hit = v_model == ref_pos
        comp_hit = v_comp == ref_pos

        orig_right += int(model_hit)
        comp_right += int(comp_hit)
        unparsed_verdict += int(v_model is None)
        missing_num_items += int(stat is None or limit is None)

        if model_hit and comp_hit:
            both_right_items += 1
        elif (not model_hit) and comp_hit:
            fixed_items += 1
        elif model_hit and (not comp_hit):
            broken_items += 1
        else:
            both_wrong_items += 1

        # Mutually exclusive item-level taxonomy
        if both_num_ok:
            if model_hit:
                bucket = "all_three_ok"
            elif v_model is None:
                bucket = "nums_ok_verdict_unparsed"
            else:
                bucket = "nums_ok_verdict_wrong"
        elif stat is None or limit is None:
            bucket = "num_missing"
        elif (not stat_ok) and lim_ok:
            bucket = "stat_wrong_limit_ok"
        elif stat_ok and (not lim_ok):
            bucket = "stat_ok_limit_wrong"
        else:
            bucket = "both_nums_wrong"

        item_buckets[bucket] += 1

        for store in (by_rule[rule], by_family[fam]):
            store["items"] += 1
            store["orig_item_ok"] += int(model_hit)
            store["comp_item_ok"] += int(comp_hit)
            store["fixed_items"] += int((not model_hit) and comp_hit)
            store["broken_items"] += int(model_hit and (not comp_hit))
            store["both_num_ok"] += int(both_num_ok)
            store["stat_ok"] += int(stat_ok)
            store["lim_ok"] += int(lim_ok)
            store[bucket] += 1

        pair_orig[pair_key][ref_pos] = v_model
        pair_comp[pair_key][ref_pos] = v_comp

    # Pair-level taxonomy (same denominator for model and comparator)
    def classify_pair(d: dict[bool, bool | None]) -> str:
        assert True in d and False in d, f"Incomplete pair: {d}"
        vt, vf = d[True], d[False]
        if vt is None or vf is None:
            return "any_unparseable"
        if vt is True and vf is False:
            return "both_correct"
        if vt is True and vf is True:
            return "both_true"
        if vt is False and vf is False:
            return "both_false"
        if vt is False and vf is True:
            return "reversed_wrong"
        raise AssertionError("unreachable")

    orig_pair_tax = defaultdict(int)
    comp_pair_tax = defaultdict(int)
    fixed_pairs = 0
    broken_pairs = 0

    for pk in sorted(pair_orig):
        rule = pair_rule[pk]
        fam = RULE_FAMILY[rule]
        co = classify_pair(pair_orig[pk])
        cc = classify_pair(pair_comp[pk])
        orig_pair_tax[co] += 1
        comp_pair_tax[cc] += 1

        o_ok = co == "both_correct"
        c_ok = cc == "both_correct"
        if (not o_ok) and c_ok:
            fixed_pairs += 1
        elif o_ok and (not c_ok):
            broken_pairs += 1

        for store in (by_rule[rule], by_family[fam]):
            store["pairs"] += 1
            store["orig_pair_ok"] += int(o_ok)
            store["comp_pair_ok"] += int(c_ok)
            store["fixed_pairs"] += int((not o_ok) and c_ok)
            store["broken_pairs"] += int(o_ok and (not c_ok))

    n_items = len(cases)
    n_pairs = len(pair_orig)
    assert sum(item_buckets.values()) == n_items
    assert sum(orig_pair_tax.values()) == n_pairs
    assert sum(comp_pair_tax.values()) == n_pairs

    return {
        "n_items": n_items,
        "n_pairs": n_pairs,
        "orig_item_ok": orig_right,
        "comp_item_ok": comp_right,
        "fixed_items": fixed_items,
        "broken_items": broken_items,
        "both_right_items": both_right_items,
        "both_wrong_items": both_wrong_items,
        "unparsed_verdict": unparsed_verdict,
        "missing_num_items": missing_num_items,
        "item_buckets": dict(item_buckets),
        "orig_pair_ok": orig_pair_tax["both_correct"],
        "comp_pair_ok": comp_pair_tax["both_correct"],
        "fixed_pairs": fixed_pairs,
        "broken_pairs": broken_pairs,
        "orig_pair_tax": dict(orig_pair_tax),
        "comp_pair_tax": dict(comp_pair_tax),
        "by_family": {k: dict(v) for k, v in sorted(by_family.items())},
        "by_rule": {k: dict(v) for k, v in sorted(by_rule.items())},
    }


def aggregate_seeds(run_dicts: list[dict]) -> dict:
    """Sum counts across matched seeds (also retains per-seed details)."""
    out = {
        "n_seeds": len(run_dicts),
        "n_items_total": sum(r["n_items"] for r in run_dicts),
        "n_pairs_total": sum(r["n_pairs"] for r in run_dicts),
    }
    for k in (
        "orig_item_ok",
        "comp_item_ok",
        "fixed_items",
        "broken_items",
        "unparsed_verdict",
        "missing_num_items",
        "orig_pair_ok",
        "comp_pair_ok",
        "fixed_pairs",
        "broken_pairs",
    ):
        out[k] = sum(r[k] for r in run_dicts)

    for sub in ("item_buckets", "orig_pair_tax", "comp_pair_tax"):
        merged = defaultdict(int)
        for r in run_dicts:
            for k, v in r[sub].items():
                merged[k] += v
        out[sub] = dict(merged)

    for nested in ("by_family", "by_rule"):
        merged = defaultdict(lambda: defaultdict(int))
        for r in run_dicts:
            for grp, stats in r[nested].items():
                for k, v in stats.items():
                    merged[grp][k] += v
        out[nested] = {g: dict(s) for g, s in sorted(merged.items())}

    out["per_seed_orig_pair_pct"] = [round(100 * r["orig_pair_ok"] / r["n_pairs"], 2) for r in run_dicts]
    out["per_seed_comp_pair_pct"] = [round(100 * r["comp_pair_ok"] / r["n_pairs"], 2) for r in run_dicts]
    return out


def pct(num: int, den: int) -> str:
    return f"{100.0 * num / den:.1f}%" if den else "0.0%"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    far_cases = load_json(OLD_DIR / "data" / "far.json")
    blind_cases = load_json(BLIND_DIR / "cases.json")
    verify_ground_truth_consistency(far_cases, "scenario-sft/far")
    verify_ground_truth_consistency(blind_cases, "independent-blind/cases")

    # Define condition -> list of prediction files
    dev_specs = {
        "base_VF": [OLD_DIR / "runs" / "base-0" / "predictions.json"],
        "base_NF": [OLD_DIR / "order-probe" / "base-0" / "predictions.json"],
        "SFT_VF_train_VF_infer": [OLD_DIR / "runs" / f"scenario-{s}" / "predictions.json" for s in (42, 43, 44)],
        "SFT_VF_train_NF_infer": [OLD_DIR / "order-probe" / f"scenario-{s}" / "predictions.json" for s in (42, 43, 44)],
        "SFT_NF_train_VF_infer": [OLD_DIR / "order-training" / f"scenario-{s}" / "original" / "predictions.json" for s in (42, 43, 44)],
        "SFT_NF_train_NF_infer": [OLD_DIR / "order-training" / f"scenario-{s}" / "reordered" / "predictions.json" for s in (42, 43, 44)],
    }

    blind_specs = {
        "base_VF": [BLIND_DIR / "outputs" / "base-original.json"],
        "base_NF": [BLIND_DIR / "outputs" / "base-reordered.json"],
        "SFT_VF_train_VF_infer": [BLIND_DIR / "outputs" / f"{s}-original.json" for s in (42, 43, 44)],
        "SFT_NF_train_NF_infer": [BLIND_DIR / "outputs" / f"{s}-reordered.json" for s in (42, 43, 44)],
    }

    results = {"dev_far_12rules": {}, "independent_blind_6rules": {}}

    for name, files in dev_specs.items():
        runs = [eval_predictions(far_cases, load_json(f)) for f in files]
        results["dev_far_12rules"][name] = {"aggregate": aggregate_seeds(runs), "runs": runs}

    for name, files in blind_specs.items():
        runs = [eval_predictions(blind_cases, load_json(f)) for f in files]
        results["independent_blind_6rules"][name] = {"aggregate": aggregate_seeds(runs), "runs": runs}

    (OUT_DIR / "audit_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf8"
    )

    # Render concise markdown report
    lines = [
        "# 现有预测的事后规则执行诊断与错误分解",
        "",
        "> **口径说明（对应方案 §4 第 4 行）**：现有存档输出不包含模型预测的比较算子字段，因此本诊断采用**规则标识已知的确定性比较器（oracle-rule comparator）**——数字完全取自模型原始回答的 `统计量` 与 `阈值`，比较方向取自参考规则。真值字段 + 真值规则在全部 120 题开发集与 96 题独立盲测集上已通过 100% 一致性校验。",
        "",
    ]

    for split_key, title in (
        ("dev_far_12rules", "## 1. 开发反转集（far.json，12 条规则，60 对 / 种子）"),
        ("independent_blind_6rules", "## 2. 独立盲测集（cases.json，6 条规则，48 对 / 种子）"),
    ):
        lines += [
            title,
            "",
            "### 1) 模型原判 vs. 模型字段 + 规则已知比较器（配对全对 & 单题改判净效应）",
            "",
            "| 条件 | 原单题正确 | 比较器单题正确 | 单题纠正 / 改错 | 原成对全对 | 比较器成对全对 | 成对纠正 / 改错 | 逐种子原对% → 比较器对% |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
        for cond, blob in results[split_key].items():
            a = blob["aggregate"]
            ni, np_ = a["n_items_total"], a["n_pairs_total"]
            seeds_str = ", ".join(
                f"{o:.1f}→{c:.1f}" for o, c in zip(a["per_seed_orig_pair_pct"], a["per_seed_comp_pair_pct"])
            )
            lines.append(
                f"| `{cond}` | {a['orig_item_ok']}/{ni} ({pct(a['orig_item_ok'], ni)}) "
                f"| {a['comp_item_ok']}/{ni} ({pct(a['comp_item_ok'], ni)}) "
                f"| +{a['fixed_items']} / -{a['broken_items']} "
                f"| {a['orig_pair_ok']}/{np_} ({pct(a['orig_pair_ok'], np_)}) "
                f"| {a['comp_pair_ok']}/{np_} ({pct(a['comp_pair_ok'], np_)}) "
                f"| +{a['fixed_pairs']} / -{a['broken_pairs']} "
                f"| {seeds_str} |"
            )

        lines += [
            "",
            "### 2) 互斥单题错误分解（定位数值瓶颈 vs. 比较判定瓶颈）",
            "",
            "| 条件 | 三项全对 | 双数对但判错 | 双数对但无结论 | 仅统计量错 | 仅阈值错 | 双数均错 | 数字缺失 | 总题数 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for cond, blob in results[split_key].items():
            a = blob["aggregate"]
            b = a["item_buckets"]
            cols = [
                b.get("all_three_ok", 0),
                b.get("nums_ok_verdict_wrong", 0),
                b.get("nums_ok_verdict_unparsed", 0),
                b.get("stat_wrong_limit_ok", 0),
                b.get("stat_ok_limit_wrong", 0),
                b.get("both_nums_wrong", 0),
                b.get("num_missing", 0),
                a["n_items_total"],
            ]
            lines.append(f"| `{cond}` | " + " | ".join(str(x) for x in cols) + " |")

        lines += [
            "",
            "### 3) 成对错误分类（模型原判 → 比较器替换后）",
            "",
            "| 条件 | 双对 | 双肯定 | 双否定 | 反向翻转 | 含不可解析 | 总对数 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for cond, blob in results[split_key].items():
            a = blob["aggregate"]
            po, pc = a["orig_pair_tax"], a["comp_pair_tax"]
            keys = ("both_correct", "both_true", "both_false", "reversed_wrong", "any_unparseable")
            vals = [f"{po.get(k,0)}→{pc.get(k,0)}" for k in keys]
            lines.append(f"| `{cond}` | " + " | ".join(vals) + f" | {a['n_pairs_total']} |")

        lines += [
            "",
            "### 4) 按规则族拆分（仅看核心 SFT 匹配组 `SFT_VF_train_VF_infer` 与 `SFT_NF_train_NF_infer`）",
            "",
            "| 条件 | 规则族 | 双数均对（题） | 仅统计量错（题） | 原成对全对 | 比较器成对全对 | 成对纠正 / 改错 |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
        for cond in ("SFT_VF_train_VF_infer", "SFT_NF_train_NF_infer"):
            fam_map = results[split_key][cond]["aggregate"]["by_family"]
            for fam, s in fam_map.items():
                lines.append(
                    f"| `{cond}` | {fam} "
                    f"| {s.get('both_num_ok',0)}/{s['items']} ({pct(s.get('both_num_ok',0), s['items'])}) "
                    f"| {s.get('stat_wrong_limit_ok',0)}/{s['items']} "
                    f"| {s.get('orig_pair_ok',0)}/{s['pairs']} ({pct(s.get('orig_pair_ok',0), s['pairs'])}) "
                    f"| {s.get('comp_pair_ok',0)}/{s['pairs']} ({pct(s.get('comp_pair_ok',0), s['pairs'])}) "
                    f"| +{s.get('fixed_pairs',0)} / -{s.get('broken_pairs',0)} |"
                )

        lines += [
            "",
            "### 5) 按逐条规则拆分（`SFT_NF_train_NF_infer`）",
            "",
            "| 规则 | 统计量对 | 阈值对 | 双数均对 | 原成对全对 | 比较器成对全对 | 成对纠正 / 改错 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        rule_map = results[split_key]["SFT_NF_train_NF_infer"]["aggregate"]["by_rule"]
        for rule, s in rule_map.items():
            lines.append(
                f"| `{rule}` | {s.get('stat_ok',0)}/{s['items']} "
                f"| {s.get('lim_ok',0)}/{s['items']} "
                f"| {s.get('both_num_ok',0)}/{s['items']} "
                f"| {s.get('orig_pair_ok',0)}/{s['pairs']} "
                f"| {s.get('comp_pair_ok',0)}/{s['pairs']} "
                f"| +{s.get('fixed_pairs',0)} / -{s.get('broken_pairs',0)} |"
            )
        lines.append("")

    (OUT_DIR / "REPORT.md").write_text("\n".join(lines), encoding="utf8")
    print("Audit complete. Wrote:", OUT_DIR / "REPORT.md")


if __name__ == "__main__":
    main()

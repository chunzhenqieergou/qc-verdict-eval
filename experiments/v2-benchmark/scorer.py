"""Unified scorer, error taxonomy, comparator evaluator, and template-cluster bootstrap
for v2-benchmark (per submission framework §4 & §5).
"""

from collections import defaultdict
import json
from pathlib import Path
import re
import sys
import unicodedata
from typing import Any

import numpy as np

from rules import RULE_SPECS, RULES_ORDER, compare_float

EPS = 1e-6

OP_MAP = {
    "<=": "<=",
    "≤": "<=",
    "=<": "<=",
    ">=": ">=",
    "≥": ">=",
    "=>": ">=",
    ">": ">",
    "<": "<",
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def parse_verdict(text: str, loose: bool = False) -> bool | None:
    norm = unicodedata.normalize("NFKC", text)
    pieces = re.split(r"[\n;,。.!！]", norm) if loose else norm.splitlines()
    labels = set()
    for line in pieces:
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


def parse_operator(text: str) -> str | None:
    """Parse structured comparison operator field, e.g. `算子=<=` or `比较=>=`."""
    norm = unicodedata.normalize("NFKC", text)
    m = re.search(r"(?:算子|比较|关系|方向)\s*=\s*(<=|>=|=<|=>|≤|≥|>|<)(?=\s*(?:[,;。\n]|$))", norm)
    if not m:
        return None
    return OP_MAP.get(m.group(1))


def _classify_cross_pair(v_true: bool | None, v_false: bool | None) -> str:
    if v_true is None or v_false is None:
        return "any_unparseable"
    if v_true is True and v_false is False:
        return "both_correct"
    if v_true is True and v_false is True:
        return "both_true"
    if v_true is False and v_false is False:
        return "both_false"
    return "reversed_wrong"


def _eval_verdict_map(cases: list[dict], verdicts: dict[str, bool | None]) -> dict[str, Any]:
    """Evaluate a verdict map {item_id -> bool|None} across cross, same-side, and paraphrase pairs."""
    groups: dict[str, dict[str, dict]] = defaultdict(dict)
    for c in cases:
        groups[c["group_id"]][c["role"]] = c

    item_ok = 0
    item_unparsed = 0
    for c in cases:
        v = verdicts[c["id"]]
        item_ok += int(v == c["positive"])
        item_unparsed += int(v is None)

    cross_tax = defaultdict(int)
    same_stats = {
        "all": defaultdict(int),
        "pos": defaultdict(int),
        "neg": defaultdict(int),
    }
    para_stats = {
        "all": defaultdict(int),
        "pos": defaultdict(int),
        "neg": defaultdict(int),
    }

    by_rule = defaultdict(lambda: defaultdict(int))
    by_family = defaultdict(lambda: defaultdict(int))
    by_template = defaultdict(lambda: defaultdict(int))

    for gid, roles in sorted(groups.items()):
        b, cr, sm, pa = roles["base"], roles["cross"], roles["same"], roles["para"]
        rule = b["rule"]
        fam = b["family"]
        tid = b["template_id"]
        side = b["same_side_label"]  # "pos" or "neg"

        vb, vcr, vsm, vpa = verdicts[b["id"]], verdicts[cr["id"]], verdicts[sm["id"]], verdicts[pa["id"]]

        # 1. Cross-boundary pair
        if b["positive"]:
            ct = _classify_cross_pair(vb, vcr)
        else:
            ct = _classify_cross_pair(vcr, vb)
        cross_tax[ct] += 1
        cross_hit = int(ct == "both_correct")

        # 2. Same-side pair (both_correct AND label_consistent, stratified by pos/neg)
        same_unparsed = int(vb is None or vsm is None)
        same_both_ok = int(vb == b["positive"] and vsm == sm["positive"])
        same_consistent = int((not same_unparsed) and (vb == vsm))
        for bucket in ("all", side):
            same_stats[bucket]["n"] += 1
            same_stats[bucket]["both_correct"] += same_both_ok
            same_stats[bucket]["label_consistent"] += same_consistent
            same_stats[bucket]["any_unparseable"] += same_unparsed

        # 3. Paraphrase pair
        para_unparsed = int(vb is None or vpa is None)
        para_both_ok = int(vb == b["positive"] and vpa == pa["positive"])
        para_consistent = int((not para_unparsed) and (vb == vpa))
        for bucket in ("all", side):
            para_stats[bucket]["n"] += 1
            para_stats[bucket]["both_correct"] += para_both_ok
            para_stats[bucket]["label_consistent"] += para_consistent
            para_stats[bucket]["any_unparseable"] += para_unparsed

        for store in (by_rule[rule], by_family[fam], by_template[tid]):
            store["groups"] += 1
            store["cross_ok"] += cross_hit
            store["same_ok"] += same_both_ok
            store["same_consistent"] += same_consistent
            store[f"same_{side}_n"] += 1
            store[f"same_{side}_ok"] += same_both_ok
            store["para_ok"] += para_both_ok
            store["para_consistent"] += para_consistent

    n_groups = len(groups)
    return {
        "n_items": len(cases),
        "n_groups": n_groups,
        "item_ok": item_ok,
        "item_unparsed": item_unparsed,
        "cross_ok": cross_tax["both_correct"],
        "cross_tax": dict(cross_tax),
        "same": {k: dict(v) for k, v in same_stats.items()},
        "para": {k: dict(v) for k, v in para_stats.items()},
        "by_rule": {k: dict(v) for k, v in sorted(by_rule.items())},
        "by_family": {k: dict(v) for k, v in sorted(by_family.items())},
        "by_template": {k: dict(v) for k, v in sorted(by_template.items())},
    }


def score_run(cases: list[dict], preds: list[dict], loose: bool = False) -> dict[str, Any]:
    """Score one prediction file under end-to-end, deployable comparator, and oracle-rule comparator."""
    by_id = {p["id"]: p for p in preds}
    assert len(by_id) == len(cases) and set(by_id.keys()) == {c["id"] for c in cases}

    v_e2e: dict[str, bool | None] = {}
    v_dep: dict[str, bool | None] = {}
    v_ora: dict[str, bool | None] = {}

    item_buckets = defaultdict(int)
    op_present = 0
    op_correct = 0

    for c in cases:
        text = by_id[c["id"]]["text"]
        vm = parse_verdict(text, loose=loose)
        stat = parse_number(text, "统计量")
        lim = parse_number(text, "阈值")
        op_pred = parse_operator(text)

        ref_val = float(c["value"])
        ref_lim = float(c["limit"])
        ref_pos = bool(c["positive"])
        ref_op = c["op"]

        stat_ok = stat is not None and abs(stat - ref_val) < EPS
        lim_ok = lim is not None and abs(lim - ref_lim) < EPS
        both_num_ok = stat_ok and lim_ok

        if op_pred is not None:
            op_present += 1
            op_correct += int(op_pred == ref_op)

        v_e2e[c["id"]] = vm
        v_dep[c["id"]] = compare_float(op_pred, stat, lim) if op_pred is not None else None
        v_ora[c["id"]] = compare_float(ref_op, stat, lim)

        # Mutually exclusive item buckets (based on end-to-end verdict + numeric fields)
        if both_num_ok:
            if vm == ref_pos:
                bucket = "all_three_ok"
            elif vm is None:
                bucket = "nums_ok_verdict_unparsed"
            else:
                bucket = "nums_ok_verdict_wrong"
        elif stat is None or lim is None:
            bucket = "num_missing"
        elif (not stat_ok) and lim_ok:
            bucket = "stat_wrong_limit_ok"
        elif stat_ok and (not lim_ok):
            bucket = "stat_ok_limit_wrong"
        else:
            bucket = "both_nums_wrong"
        item_buckets[bucket] += 1

    def net_effect(v_alt: dict[str, bool | None]) -> dict[str, int]:
        fixed_items = broken_items = 0
        for c in cases:
            o = v_e2e[c["id"]] == c["positive"]
            a = v_alt[c["id"]] == c["positive"]
            fixed_items += int((not o) and a)
            broken_items += int(o and (not a))

        groups = defaultdict(dict)
        for c in cases:
            groups[c["group_id"]][c["role"]] = c
        fixed_cross = broken_cross = 0
        for roles in groups.values():
            b, cr = roles["base"], roles["cross"]
            o_ok = (v_e2e[b["id"]] == b["positive"]) and (v_e2e[cr["id"]] == cr["positive"])
            a_ok = (v_alt[b["id"]] == b["positive"]) and (v_alt[cr["id"]] == cr["positive"])
            fixed_cross += int((not o_ok) and a_ok)
            broken_cross += int(o_ok and (not a_ok))
        return {
            "fixed_items": fixed_items,
            "broken_items": broken_items,
            "fixed_cross_pairs": fixed_cross,
            "broken_cross_pairs": broken_cross,
        }

    return {
        "end_to_end": _eval_verdict_map(cases, v_e2e),
        "deployable_comparator": {
            **_eval_verdict_map(cases, v_dep),
            "op_present": op_present,
            "op_correct": op_correct,
            "vs_e2e": net_effect(v_dep),
        },
        "oracle_rule_comparator": {
            **_eval_verdict_map(cases, v_ora),
            "vs_e2e": net_effect(v_ora),
        },
        "item_buckets": dict(item_buckets),
    }


def paired_template_cluster_bootstrap(
    runs_a: list[dict],
    runs_b: list[dict],
    metric_key: str = "cross_ok",
    procedure: str = "end_to_end",
    n_boot: int = 10000,
    seed: int = 20261119,
) -> dict[str, Any]:
    """Paired template-cluster bootstrap within fixed rules (§5).

    Computes (A - B) equal-weight averaged across the 12 fixed rules (in percentage points).
    All configs and matched seeds within a template move together.
    Also reports leave-one-template-out (LOTO) min/max across all templates.
    """
    assert len(runs_a) == len(runs_b) and len(runs_a) >= 1

    # Organize templates by rule
    tmpl_by_rule: dict[str, list[str]] = defaultdict(list)
    sample_tmpls = runs_a[0][procedure]["by_template"]
    for tid in sorted(sample_tmpls):
        # template_id format: {split}_{rule}_t{idx}
        parts = tid.split("_")
        rule = "_".join(parts[1:-1])
        tmpl_by_rule[rule].append(tid)

    # For each rule and template, compute mean score across matched seeds
    diff_matrix: dict[str, np.ndarray] = {}
    rule_means: dict[str, float] = {}
    for rule in RULES_ORDER:
        tids = sorted(tmpl_by_rule[rule])
        t_diffs = []
        for tid in tids:
            vals_a = [r[procedure]["by_template"][tid][metric_key] / r[procedure]["by_template"][tid]["groups"] for r in runs_a]
            vals_b = [r[procedure]["by_template"][tid][metric_key] / r[procedure]["by_template"][tid]["groups"] for r in runs_b]
            t_diffs.append(float(np.mean(vals_a) - np.mean(vals_b)))
        arr = np.array(t_diffs, dtype=float)
        diff_matrix[rule] = arr
        rule_means[rule] = round(100.0 * float(arr.mean()), 2)

    point_est = 100.0 * float(np.mean([arr.mean() for arr in diff_matrix.values()]))

    rng = np.random.default_rng(seed)
    boot_vals = np.zeros(n_boot, dtype=float)
    for rule in RULES_ORDER:
        arr = diff_matrix[rule]
        k = len(arr)
        idx = rng.integers(0, k, size=(n_boot, k))
        boot_vals += arr[idx].mean(axis=1)
    boot_vals = 100.0 * (boot_vals / len(RULES_ORDER))
    lo, hi = np.quantile(boot_vals, [0.025, 0.975])

    # Leave-one-template-out sensitivity (drop one template index from one rule at a time)
    loto_ests = []
    for rule in RULES_ORDER:
        arr = diff_matrix[rule]
        if len(arr) <= 1:
            continue
        for i in range(len(arr)):
            other_rules = [diff_matrix[r].mean() for r in RULES_ORDER if r != rule]
            reduced_rule = np.delete(arr, i).mean()
            loto_ests.append(100.0 * float(np.mean(other_rules + [reduced_rule])))

    return {
        "point_diff_pp": round(point_est, 2),
        "ci95_template_cluster_pp": [round(float(lo), 2), round(float(hi), 2)],
        "loto_range_pp": [round(min(loto_ests), 2), round(max(loto_ests), 2)] if loto_ests else None,
        "by_rule_diff_pp": rule_means,
    }


def simulate_ci_width(
    dev_runs_a: list[dict],
    dev_runs_b: list[dict],
    target_templates_per_rule: tuple[int, ...] = (2, 4, 6, 8),
    n_sims: int = 2000,
    seed: int = 20261119,
) -> dict[str, Any]:
    """Simulate expected 95% CI width of (A - B) cross-boundary accuracy as a function of
    templates per rule, using empirical template-level paired divergence from dev_v2 (§3).
    """
    tmpl_diffs = []
    for tid in sorted(dev_runs_a[0]["end_to_end"]["by_template"]):
        va = np.mean([r["end_to_end"]["by_template"][tid]["cross_ok"] / r["end_to_end"]["by_template"][tid]["groups"] for r in dev_runs_a])
        vb = np.mean([r["end_to_end"]["by_template"][tid]["cross_ok"] / r["end_to_end"]["by_template"][tid]["groups"] for r in dev_runs_b])
        tmpl_diffs.append(float(va - vb))
    arr = np.array(tmpl_diffs, dtype=float)
    rng = np.random.default_rng(seed)
    summary = {}
    for k in target_templates_per_rule:
        n_total_tmpls = 12 * k
        # Sample n_total_tmpls template differences with replacement and check sampling distribution width
        idx = rng.integers(0, len(arr), size=(n_sims, n_total_tmpls))
        means = 100.0 * arr[idx].mean(axis=1)
        lo, hi = np.quantile(means, [0.025, 0.975])
        summary[f"{k}_templates_per_rule_({n_total_tmpls}_tmpls)"] = {
            "sim_mean_pp": round(float(means.mean()), 2),
            "expected_ci95_width_pp": round(float(hi - lo), 2),
            "expected_half_width_pp": round(float((hi - lo) / 2), 2),
        }
    return {
        "empirical_dev_template_diff_sd_pp": round(100.0 * float(arr.std(ddof=1)), 2),
        "simulations": summary,
    }


def self_test() -> None:
    data_dir = Path(__file__).resolve().parent / "data"
    for split_file in ("dev_v2.json", "test_v2_candidate.json"):
        cases = load_json(data_dir / split_file)

        # 1. Oracle predictions (Row 5 truth check)
        oracle_preds = [
            {
                "id": c["id"],
                "text": f"统计量={c['value']:g}，阈值={c['limit']:g}，算子={c['op']}\n"
                        f"{'成立' if c['positive'] else '不成立'}\n理由。",
            }
            for c in cases
        ]
        res = score_run(cases, oracle_preds)
        for proc in ("end_to_end", "deployable_comparator", "oracle_rule_comparator"):
            r = res[proc]
            assert r["item_ok"] == len(cases), f"{split_file} {proc} item_ok failed"
            assert r["cross_ok"] == r["n_groups"], f"{split_file} {proc} cross_ok failed"
            assert r["same"]["all"]["both_correct"] == r["n_groups"]
            assert r["para"]["all"]["both_correct"] == r["n_groups"]
        assert res["item_buckets"] == {"all_three_ok": len(cases)}

        # 2. Constant-True dummy check (§5 verification)
        const_preds = [{"id": c["id"], "text": "成立\n统计量=0，阈值=0\n无。"} for c in cases]
        res_c = score_run(cases, const_preds)
        e2e = res_c["end_to_end"]
        assert e2e["cross_ok"] == 0, "Constant model must score 0 on cross pairs!"
        assert e2e["same"]["all"]["label_consistent"] == e2e["n_groups"], "Constant model gets 100% consistency"
        assert e2e["same"]["all"]["both_correct"] == e2e["n_groups"] // 2, "Constant model gets 50% same both_correct"
        assert e2e["same"]["pos"]["both_correct"] == e2e["same"]["pos"]["n"]
        assert e2e["same"]["neg"]["both_correct"] == 0

        # 3. Bootstrap check (oracle vs constant)
        boot = paired_template_cluster_bootstrap([res], [res_c])
        assert boot["point_diff_pp"] == 100.0 and boot["ci95_template_cluster_pp"] == [100.0, 100.0]

    print("scorer.py self-test PASSED on both dev_v2.json and test_v2_candidate.json!")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        self_test()

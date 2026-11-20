"""Diagnostic scorer comparing original and reordered prediction order on the far split."""

import re
import unicodedata
from collections import defaultdict

import numpy as np

from data import DATA, ROOT, RULES, save
import score

SEEDS = [42, 43, 44]
ARMS = ["paraphrase", "scenario"]
MODELS = ["base-0"] + [f"{arm}-{s}" for arm in ARMS for s in SEEDS]
GROUPS = {
    "base": ["base-0"],
    "paraphrase": [f"paraphrase-{s}" for s in SEEDS],
    "scenario": [f"scenario-{s}" for s in SEEDS],
}
BOOTSTRAP_SEED = 20261118
BOOTSTRAP_ITERATIONS = 10000

VALUE_RE = re.compile(r"统计量\s*=\s*([+-]?\d+(?:\.\d+)?)(?=\s*(?:[,;。\n]|$))")
LIMIT_RE = re.compile(r"阈值\s*=\s*([+-]?\d+(?:\.\d+)?)(?=\s*(?:[,;。\n]|$))")


def _is_ws_or_punct(ch):
    return ch.isspace() or unicodedata.category(ch).startswith("P")


def parse_decision_and_order(text):
    """Parse standalone decision lines uniformly and check whether both fields precede conclusion."""
    normalized = unicodedata.normalize("NFKC", text).strip()
    matches = []
    pos = 0
    for raw_line in normalized.splitlines(keepends=True):
        cleaned = "".join(ch for ch in raw_line if not _is_ws_or_punct(ch))
        if cleaned == "成立":
            matches.append((pos, True))
        elif cleaned == "不成立":
            matches.append((pos, False))
        pos += len(raw_line)

    distinct = {label for _, label in matches}
    conflicting = len(distinct) > 1
    decision = next(iter(distinct)) if len(distinct) == 1 else None
    first_pos = matches[0][0] if decision is not None else None

    m_val = VALUE_RE.search(normalized)
    m_lim = LIMIT_RE.search(normalized)
    values_before = bool(
        first_pos is not None
        and m_val is not None
        and m_lim is not None
        and m_val.end() <= first_pos
        and m_lim.end() <= first_pos
    )
    return decision, values_before, conflicting


def grade_item(case, pred):
    """Reuse score.grade for numeric/cap fields and apply unified decision and order parsing."""
    case_for_grade = case if "split" in case else {**case, "split": "far"}
    base = score.grade(case_for_grade, pred)
    decision, values_before, conflicting = parse_decision_and_order(pred["text"])
    hit = int(decision == case["positive"])
    return {
        "id": base["id"],
        "rule": base["rule"],
        "pair": base["pair"],
        "split": base["split"],
        "positive": bool(case["positive"]),
        "decision": hit,
        "parsed_decision": decision,
        "positive_prediction": int(decision is True),
        "unparsed_decision": int(decision is None),
        "conflicting_labels": int(conflicting),
        "explicit_wrong": int(decision is not None and not hit),
        "value": base["value"],
        "threshold": base["threshold"],
        "values_both_correct": int(bool(base["value"] and base["threshold"])),
        "unparsed_value": base["unparsed_value"],
        "unparsed_threshold": base["unparsed_threshold"],
        "values_before_decision": int(values_before),
        "cap": base["cap"],
    }


def load_far_cases():
    """Load near and far cases and verify disjointness and unique IDs."""
    near_cases = score.load(DATA / "near.json")
    far_cases = score.load(DATA / "far.json")
    near_ids = [c["id"] for c in near_cases]
    far_ids = [c["id"] for c in far_cases]
    assert len(near_ids) == len(set(near_ids)), "Duplicate IDs in near.json"
    assert len(far_ids) == len(set(far_ids)), "Duplicate IDs in far.json"
    assert set(near_ids).isdisjoint(set(far_ids)), "Overlapping IDs between near.json and far.json"
    for c in near_cases:
        if "split" in c:
            assert c["split"] == "near", f"Unexpected split in near.json: {c['split']}"
    for c in far_cases:
        if "split" in c:
            assert c["split"] == "far", f"Unexpected split in far.json: {c['split']}"
    return near_cases, far_cases


def load_aligned_far_predictions(path, near_cases, far_cases, require_all=False):
    """Verify all IDs in a predictions file and return predictions aligned to far_cases."""
    preds = score.load(path)
    pred_ids = [p["id"] for p in preds]
    assert len(pred_ids) == len(set(pred_ids)), f"Duplicate IDs in {path}"

    far_id_set = {c["id"] for c in far_cases}
    all_id_set = {c["id"] for c in near_cases} | far_id_set
    pred_id_set = set(pred_ids)

    if require_all:
        assert pred_id_set == all_id_set, (
            f"ID mismatch in {path}: expected all {len(all_id_set)} near+far IDs, got {len(pred_id_set)}"
        )
    else:
        assert pred_id_set == all_id_set or pred_id_set == far_id_set, (
            f"ID mismatch in {path}: expected {len(all_id_set)} (near+far) or "
            f"{len(far_id_set)} (far) IDs, got {len(pred_id_set)}"
        )

    by_id = {p["id"]: p for p in preds}
    scope = "near+far" if pred_id_set == all_id_set else "far"
    return [by_id[c["id"]] for c in far_cases], scope


def score_prediction_file(path, near_cases, far_cases, require_all=False):
    """Load, ID-verify, and grade far predictions from a prediction file."""
    far_preds, scope = load_aligned_far_predictions(path, near_cases, far_cases, require_all=require_all)
    rows = [grade_item(c, p) for c, p in zip(far_cases, far_preds)]
    return rows, scope


def summarize_rows(rows):
    """Compute paired and item-level far diagnostic metrics."""
    pairs = defaultdict(list)
    for r in rows:
        pairs[r["pair"]].append(r["decision"])
    assert pairs and all(len(v) == 2 for v in pairs.values()), "Each pair must contain exactly 2 items"
    n = len(rows)
    n_pairs = len(pairs)
    return {
        "n": int(n),
        "n_pairs": int(n_pairs),
        "pair": float(sum(all(v) for v in pairs.values()) / n_pairs),
        "decision": float(sum(r["decision"] for r in rows) / n),
        "values_before_decision": int(sum(r["values_before_decision"] for r in rows)),
        "unparsed_decision": int(sum(r["unparsed_decision"] for r in rows)),
        "positive_predictions": int(sum(r["positive_prediction"] for r in rows)),
        "cap": int(sum(r["cap"] for r in rows)),
        "conflicting_labels": int(sum(r["conflicting_labels"] for r in rows)),
        "unparsed_value": int(sum(r["unparsed_value"] for r in rows)),
        "unparsed_threshold": int(sum(r["unparsed_threshold"] for r in rows)),
        "values_both_correct": int(sum(r["values_both_correct"] for r in rows)),
    }


def summarize_polarity(rows_by_model):
    """Stratify accuracy by ground-truth positive/negative polarity when both numeric fields match."""
    def _bucket(rows, pos_flag):
        sub = [r for r in rows if r["values_both_correct"] and r["positive"] is pos_flag]
        correct = int(sum(r["decision"] for r in sub))
        total = int(len(sub))
        return {
            "correct": correct,
            "total": total,
            "accuracy": float(correct / total) if total else None,
        }

    pooled = [r for rows in rows_by_model.values() for r in rows]
    return {
        "diagnostic_note": "Descriptive counts/rates only; repeated items across seeds are not independent samples.",
        "positive": {
            **_bucket(pooled, True),
            "per_model": {m: _bucket(rows, True) for m, rows in rows_by_model.items()},
        },
        "negative": {
            **_bucket(pooled, False),
            "per_model": {m: _bucket(rows, False) for m, rows in rows_by_model.items()},
        },
    }


def _rule_pair_hits(rows):
    """Return per-rule correct pair counts and total pair counts in RULES order."""
    by_rule = {rule: defaultdict(list) for rule in RULES}
    for r in rows:
        assert r["rule"] in by_rule, f"Unknown rule in far data: {r['rule']}"
        by_rule[r["rule"]][r["pair"]].append(r["decision"])

    hits = []
    totals = []
    pair_ids_by_rule = {}
    for rule in RULES:
        pairs = by_rule[rule]
        assert pairs and all(len(v) == 2 for v in pairs.values()), f"Incomplete pair in rule {rule}"
        hits.append(sum(all(v) for v in pairs.values()))
        totals.append(len(pairs))
        pair_ids_by_rule[rule] = set(pairs.keys())
    return np.array(hits, dtype=float), np.array(totals, dtype=float), pair_ids_by_rule


def paired_rule_bootstrap(orig_rows_list, probe_rows_list, ix):
    """Compute paired condition pair-rate difference and 12-rule cluster bootstrap 95% CI."""
    assert len(orig_rows_list) == len(probe_rows_list) > 0
    n_seeds = len(orig_rows_list)

    orig_hits = []
    probe_hits = []
    ref_totals = None
    ref_pair_ids = None

    for orig_rows, probe_rows in zip(orig_rows_list, probe_rows_list):
        oh, ot, opids = _rule_pair_hits(orig_rows)
        ph, pt, ppids = _rule_pair_hits(probe_rows)
        assert np.array_equal(ot, pt), "Rule pair count mismatch between conditions"
        assert opids == ppids, "Pair ID mismatch between conditions"
        if ref_totals is None:
            ref_totals = ot
            ref_pair_ids = opids
        else:
            assert np.array_equal(ot, ref_totals), "Rule pair count mismatch across seeds"
            assert opids == ref_pair_ids, "Pair ID mismatch across seeds"
        orig_hits.append(oh)
        probe_hits.append(ph)

    orig_hits = np.stack(orig_hits, axis=0)
    probe_hits = np.stack(probe_hits, axis=0)
    diff_hits_mean = (probe_hits - orig_hits).mean(axis=0)

    boot = diff_hits_mean[ix].sum(axis=1) / ref_totals[ix].sum(axis=1)
    ci_95 = np.quantile(boot, [0.025, 0.975]).tolist()

    per_seed_orig = [summarize_rows(r)["pair"] for r in orig_rows_list]
    per_seed_probe = [summarize_rows(r)["pair"] for r in probe_rows_list]
    per_seed_diff = [p - o for o, p in zip(per_seed_orig, per_seed_probe)]

    return {
        "contrast": "probe_minus_original",
        "n_seeds": int(n_seeds),
        "n_rules": int(len(RULES)),
        "n_bootstrap": int(ix.shape[0]),
        "original_mean_pair": float(np.mean(per_seed_orig)),
        "probe_mean_pair": float(np.mean(per_seed_probe)),
        "mean_pair_difference": float(np.mean(per_seed_diff)),
        "per_seed_pair_difference": [float(x) for x in per_seed_diff],
        "rule_cluster_ci_95": [float(ci_95[0]), float(ci_95[1])],
        "rule_delta": {
            rule: float(diff_hits_mean[i] / ref_totals[i])
            for i, rule in enumerate(RULES)
        },
    }


def evaluate_order_training(near_cases, far_cases, ix):
    """Discover, verify, and score order-training/scenario-<seed>/{original,reordered} if present."""
    ot_root = ROOT / "order-training"
    if not ot_root.is_dir():
        return {"present": False, "seeds": {}, "group_summary": None}

    seed_names = [str(s) for s in SEEDS]
    for sub in sorted(ot_root.iterdir()):
        if sub.is_dir() and sub.name.startswith("scenario-"):
            s = sub.name[len("scenario-"):]
            if s not in seed_names:
                seed_names.append(s)

    details = {}
    summaries = {}
    for s in seed_names:
        seed_key = f"scenario-{s}"
        seed_dir = ot_root / seed_key
        if not seed_dir.is_dir():
            continue
        seed_details = {}
        seed_summary = {}
        for cond in ["original", "reordered"]:
            pred_path = seed_dir / cond / "predictions.json"
            if pred_path.exists():
                rows, scope = score_prediction_file(pred_path, near_cases, far_cases, require_all=False)
                seed_details[cond] = rows
                seed_summary[cond] = {
                    **summarize_rows(rows),
                    "id_scope": scope,
                    "polarity_when_values_correct": summarize_polarity({seed_key: rows}),
                }
        if seed_summary:
            if "original" in seed_summary and "reordered" in seed_summary:
                seed_summary["pair_difference_reordered_minus_original"] = float(
                    seed_summary["reordered"]["pair"] - seed_summary["original"]["pair"]
                )
            details[seed_key] = seed_details
            summaries[seed_key] = seed_summary

    canonical_keys = [f"scenario-{s}" for s in SEEDS]
    group_summary = None
    if summaries:
        orig_by_model = {
            k: details[k]["original"] for k in details if "original" in details[k]
        }
        reord_by_model = {
            k: details[k]["reordered"] for k in details if "reordered" in details[k]
        }
        polarity = {}
        if orig_by_model:
            polarity["original"] = summarize_polarity(orig_by_model)
        if reord_by_model:
            polarity["reordered"] = summarize_polarity(reord_by_model)

        order_effect = None
        if all(k in details and "original" in details[k] and "reordered" in details[k] for k in canonical_keys):
            order_effect = paired_rule_bootstrap(
                [details[k]["original"] for k in canonical_keys],
                [details[k]["reordered"] for k in canonical_keys],
                ix,
            )

        group_summary = {
            "order_effect": order_effect,
            "polarity_when_values_correct": polarity,
        }

    return {
        "present": bool(summaries),
        "seeds": summaries,
        "group_summary": group_summary,
    }


def _fmt_pol(bucket):
    acc = f"{bucket['accuracy']:.3f}" if bucket["accuracy"] is not None else "NA"
    return f"{bucket['correct']}/{bucket['total']} ({acc})"


def print_summary(report, n_far, n_pairs):
    """Print a concise diagnostic summary table to stdout."""
    print(f"[Diagnostic only | far split: n={n_far}, pairs={n_pairs}, rules={len(RULES)}]")
    print("Note: All metrics are diagnostics; repeated items across seeds are not treated as independent samples.")
    print()
    header = (
        f"{'model':<14} {'cond':<11} {'pair':>7} {'dec':>7} "
        f"{'val_pre':>7} {'unp_dec':>7} {'pos_cnt':>7} {'cap':>5}"
    )
    print(header)
    print("-" * len(header))
    for name in MODELS:
        for cond in ["original", "order_probe"]:
            m = report["models"][name][cond]
            print(
                f"{name:<14} {cond:<11} {m['pair']:>7.3f} {m['decision']:>7.3f} "
                f"{m['values_before_decision']:>7d} {m['unparsed_decision']:>7d} "
                f"{m['positive_predictions']:>7d} {m['cap']:>5d}"
            )

    print("\nGroup order effects (order_probe - original pair rate, 12-rule cluster 95% CI):")
    for grp in ["paraphrase", "scenario", "base"]:
        ge = report["groups"][grp]["order_effect"]
        lo, hi = ge["rule_cluster_ci_95"]
        print(
            f"  {grp:<10} orig={ge['original_mean_pair']:.3f} probe={ge['probe_mean_pair']:.3f} "
            f"diff={ge['mean_pair_difference']:+.4f} CI95=[{lo:+.4f}, {hi:+.4f}] (seeds={ge['n_seeds']})"
        )

    print("\nPolarity stratification when both numeric fields are correct (correct/total):")
    for grp in ["base", "paraphrase", "scenario"]:
        for cond in ["original", "order_probe"]:
            pol = report["groups"][grp]["polarity_when_values_correct"][cond]
            print(
                f"  {grp:<10} {cond:<11} pos={_fmt_pol(pol['positive']):<20} "
                f"neg={_fmt_pol(pol['negative'])}"
            )

    ot = report["order_training"]
    if not ot["present"]:
        print("\norder-training: not found (skipped)")
    else:
        print("\norder-training:")
        for seed_key, seed_res in ot["seeds"].items():
            for cond in ["original", "reordered"]:
                if cond in seed_res:
                    m = seed_res[cond]
                    print(
                        f"  {seed_key:<12} {cond:<10} pair={m['pair']:.3f} dec={m['decision']:.3f} "
                        f"val_pre={m['values_before_decision']} unp_dec={m['unparsed_decision']} "
                        f"pos_cnt={m['positive_predictions']} cap={m['cap']}"
                    )
        if ot["group_summary"] and ot["group_summary"]["order_effect"] is not None:
            ge = ot["group_summary"]["order_effect"]
            lo, hi = ge["rule_cluster_ci_95"]
            print(
                f"  3-seed diff (reordered - original)={ge['mean_pair_difference']:+.4f} "
                f"CI95=[{lo:+.4f}, {hi:+.4f}]"
            )


def main():
    near_cases, far_cases = load_far_cases()
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    ix = rng.integers(0, len(RULES), (BOOTSTRAP_ITERATIONS, len(RULES)))

    raw_details = {}
    model_summaries = {}

    for name in MODELS:
        orig_rows, orig_scope = score_prediction_file(
            ROOT / "runs" / name / "predictions.json",
            near_cases,
            far_cases,
            require_all=True,
        )
        probe_rows, probe_scope = score_prediction_file(
            ROOT / "order-probe" / name / "predictions.json",
            near_cases,
            far_cases,
            require_all=False,
        )
        raw_details[name] = {"original": orig_rows, "order_probe": probe_rows}

        orig_sum = summarize_rows(orig_rows)
        probe_sum = summarize_rows(probe_rows)
        model_summaries[name] = {
            "original": {**orig_sum, "id_scope": orig_scope},
            "order_probe": {**probe_sum, "id_scope": probe_scope},
            "pair_difference_probe_minus_original": float(probe_sum["pair"] - orig_sum["pair"]),
        }

    group_summaries = {}
    for grp, members in GROUPS.items():
        group_summaries[grp] = {
            "models": members,
            "order_effect": paired_rule_bootstrap(
                [raw_details[m]["original"] for m in members],
                [raw_details[m]["order_probe"] for m in members],
                ix,
            ),
            "polarity_when_values_correct": {
                cond: summarize_polarity({m: raw_details[m][cond] for m in members})
                for cond in ["original", "order_probe"]
            },
        }

    order_training = evaluate_order_training(near_cases, far_cases, ix)

    report = {
        "split": "far",
        "diagnostic_note": (
            "All metrics are diagnostic; repeated items across seeds are not independent samples. "
            "Rule-cluster bootstrap resamples the 12 rules with replacement, keeping all seeds and "
            "paired items within each rule together."
        ),
        "n_far": len(far_cases),
        "n_far_pairs": model_summaries["base-0"]["original"]["n_pairs"],
        "rules": list(RULES),
        "models": model_summaries,
        "groups": group_summaries,
        "order_training": order_training,
    }

    save(ROOT / "order-comparison.json", report)
    print_summary(report, len(far_cases), report["n_far_pairs"])


if __name__ == "__main__":
    main()

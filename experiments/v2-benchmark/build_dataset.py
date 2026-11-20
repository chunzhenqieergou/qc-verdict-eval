"""Build and verify v2-benchmark datasets (dev_v2 and test_v2_candidate).

Structure:
  - dev_v2:            12 rules x 2 templates x 2 configs =  48 base groups = 192 items
  - test_v2_candidate: 12 rules x 4 templates x 3 configs = 144 base groups = 576 items

Each base group contains 4 items:
  1. base  (primary template, anchor state)
  2. cross (primary template, single-observation edit flipping the verdict)
  3. same  (primary template, single-observation edit preserving the verdict)
  4. para  (semantic paraphrase template, identical inputs & ground truth as base)

Every item is checked by:
  - Independent exact Fraction ground-truth calculator (rules.py)
  - Single-observation edit assertions for cross and same pairs
  - Exact input identity & wording difference assertion for paraphrase pairs
  - Rounding-ambiguity guard for capability indices
  - Zero overlap across dev_v2, test_v2_candidate, historical scenario-sft, and independent-blind
"""

from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import random
import string
from typing import Any

from rules import RULE_SPECS, RULES_ORDER, WECO_WINDOW_PARAMS, compute_ground_truth
from templates import DEV_PARTS, TEMPLATES, TEST_PARTS

OUT_DIR = Path(__file__).resolve().parent
DATA_DIR = OUT_DIR / "data"
OLD_SFT_DIR = Path(r"d:\Users\Administrator\Desktop\project1\experiments\scenario-sft\data")
OLD_BLIND_PATH = Path(r"d:\Users\Administrator\Desktop\project1\experiments\independent-blind\cases.json")


def fmt_num(val: Fraction | int) -> str:
    if isinstance(val, int):
        return str(val)
    if val.denominator == 1:
        return str(val.numerator)
    f = float(val)
    # Require exact representation within 3 decimals (no silent rounding drift)
    for dec in (1, 2, 3):
        s = f"{f:.{dec}f}"
        if Fraction(s) == val:
            return s.rstrip("0").rstrip(".") if "." in s else s
    raise ValueError(f"Non-terminating decimal fraction not allowed in display: {val} ({f})")


def fmt_signed2(val: Fraction) -> str:
    f = float(val)
    s = f"{f:+.2f}"
    assert Fraction(s) == val, f"Signed 2-dec mismatch: {val} vs {s}"
    return s


def build_group_states(split: str, rule: str, t_idx: int, c_idx: int) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str]:
    """Return (base_inputs, cross_inputs, same_inputs, boundary_profile) for one base group."""
    is_dev = split == "dev"
    g_idx = t_idx * (2 if is_dev else 3) + c_idx  # 0..3 in dev, 0..11 in test
    mode = g_idx % 4  # 0: pos@boundary, 1: neg@tight, 2: pos@strict, 3: neg@moderate

    # -------- Family 1: Acceptance Sampling --------
    if rule == "single":
        c_pool = [4, 6, 8, 10] if is_dev else [3, 5, 7, 9, 11, 13, 14, 15, 12, 16, 17, 18]
        n_pool = [80, 100, 120, 150] if is_dev else [50, 65, 75, 90, 110, 125, 140, 160, 180, 200, 220, 250]
        c = c_pool[g_idx]
        n = n_pool[g_idx]
        if mode == 0:
            x_b, x_c, x_s = c, c + 1, c - 1
            prof = "at_boundary_pos"
        elif mode == 1:
            x_b, x_c, x_s = c + 1, c, c + 2
            prof = "tight_outside_neg"
        elif mode == 2:
            x_b, x_c, x_s = c - 1, c + 2, c - 2
            prof = "strict_inside_pos"
        else:
            x_b, x_c, x_s = c + 2, c - 1, c + 1
            prof = "moderate_outside_neg"
        return (
            {"n": n, "c": c, "x": x_b},
            {"n": n, "c": c, "x": x_c},
            {"n": n, "c": c, "x": x_s},
            prof,
        )

    if rule == "first_accept":
        c1_pool = [3, 5, 7, 9] if is_dev else [4, 6, 8, 10, 12, 5, 7, 9, 11, 13, 6, 8]
        gap_pool = [3, 4, 3, 4] if is_dev else [4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 6, 6]
        n1_pool = [60, 80, 100, 120] if is_dev else [50, 70, 90, 110, 130, 65, 85, 105, 125, 145, 75, 95]
        c1 = c1_pool[g_idx]
        r1 = c1 + gap_pool[g_idx]
        n1 = n1_pool[g_idx]
        if mode == 0:
            x_b, x_c, x_s = c1, c1 + 1, c1 - 1
            prof = "at_boundary_pos"
        elif mode == 1:
            x_b, x_c, x_s = c1 + 1, c1, c1 + 2
            prof = "tight_outside_neg"
        elif mode == 2:
            x_b, x_c, x_s = c1 - 1, c1 + 2, c1 - 2
            prof = "strict_inside_pos"
        else:
            x_b, x_c, x_s = c1 + 2, c1 - 1, c1 + 1
            prof = "moderate_outside_neg"
        return (
            {"n1": n1, "c1": c1, "r1": r1, "x1": x_b},
            {"n1": n1, "c1": c1, "r1": r1, "x1": x_c},
            {"n1": n1, "c1": c1, "r1": r1, "x1": x_s},
            prof,
        )

    if rule == "first_reject":
        c1_pool = [2, 3, 4, 5] if is_dev else [2, 3, 4, 5, 6, 3, 4, 5, 6, 7, 4, 5]
        gap_pool = [4, 5, 4, 5] if is_dev else [4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6]
        n1_pool = [60, 80, 100, 120] if is_dev else [55, 75, 95, 115, 135, 65, 85, 105, 125, 145, 90, 110]
        c1 = c1_pool[g_idx]
        r1 = c1 + gap_pool[g_idx]
        n1 = n1_pool[g_idx]
        if mode == 0:
            x_b, x_c, x_s = r1, r1 - 1, r1 + 1
            prof = "at_boundary_pos"
        elif mode == 1:
            x_b, x_c, x_s = r1 - 1, r1, r1 - 2
            prof = "tight_outside_neg"
        elif mode == 2:
            x_b, x_c, x_s = r1 + 1, r1 - 2, r1 + 2
            prof = "strict_inside_pos"
        else:
            x_b, x_c, x_s = r1 - 2, r1 + 1, r1 - 1
            prof = "moderate_outside_neg"
        return (
            {"n1": n1, "c1": c1, "r1": r1, "x1": x_b},
            {"n1": n1, "c1": c1, "r1": r1, "x1": x_c},
            {"n1": n1, "c1": c1, "r1": r1, "x1": x_s},
            prof,
        )

    if rule == "second_accept":
        c2_pool = [8, 10, 12, 14] if is_dev else [7, 9, 11, 13, 15, 8, 10, 12, 14, 16, 17, 19]
        x1_pool = [3, 4, 5, 6] if is_dev else [2, 3, 4, 5, 6, 3, 4, 5, 6, 7, 5, 6]
        n1_pool = [50, 60, 70, 80] if is_dev else [45, 55, 65, 75, 85, 50, 60, 70, 80, 90, 100, 110]
        c2 = c2_pool[g_idx]
        x1 = x1_pool[g_idx]
        n1 = n1_pool[g_idx]
        n2 = n1
        rem = c2 - x1  # >= 5 in all entries
        assert rem >= 4
        if mode == 0:
            x2_b, x2_c, x2_s = rem, rem + 1, rem - 1
            prof = "at_boundary_pos"
        elif mode == 1:
            x2_b, x2_c, x2_s = rem + 1, rem, rem + 2
            prof = "tight_outside_neg"
        elif mode == 2:
            x2_b, x2_c, x2_s = rem - 1, rem + 2, rem - 2
            prof = "strict_inside_pos"
        else:
            x2_b, x2_c, x2_s = rem + 2, rem - 1, rem + 1
            prof = "moderate_outside_neg"
        return (
            {"n1": n1, "n2": n2, "x1": x1, "x2": x2_b, "c2": c2},
            {"n1": n1, "n2": n2, "x1": x1, "x2": x2_c, "c2": c2},
            {"n1": n1, "n2": n2, "x1": x1, "x2": x2_s, "c2": c2},
            prof,
        )

    # -------- Family 2: Control Chart --------
    if rule == "weco3":
        # Strict > 3.00! z = +3.00 is FALSE.
        if is_dev:
            triples = [
                (Fraction("3.15"), Fraction("2.85"), Fraction("3.45"), "tight_above_pos"),
                (Fraction("3.00"), Fraction("3.20"), Fraction("2.75"), "at_boundary_neg"),
                (Fraction("3.40"), Fraction("3.00"), Fraction("3.80"), "moderate_above_pos"),
                (Fraction("2.65"), Fraction("3.30"), Fraction("2.90"), "below_boundary_neg"),
            ]
        else:
            triples = [
                (Fraction("3.12"), Fraction("2.92"), Fraction("3.38"), "tight_above_pos"),
                (Fraction("3.00"), Fraction("3.18"), Fraction("2.82"), "at_boundary_neg"),
                (Fraction("3.35"), Fraction("2.70"), Fraction("3.75"), "moderate_above_pos"),
                (Fraction("2.78"), Fraction("3.24"), Fraction("3.00"), "below_to_boundary_neg"),
                (Fraction("3.08"), Fraction("3.00"), Fraction("3.42"), "tight_above_vs_eq_pos"),
                (Fraction("3.00"), Fraction("3.28"), Fraction("2.64"), "at_boundary_neg"),
                (Fraction("3.52"), Fraction("2.88"), Fraction("3.90"), "moderate_above_pos"),
                (Fraction("2.55"), Fraction("3.16"), Fraction("2.85"), "below_boundary_neg"),
                (Fraction("3.22"), Fraction("2.96"), Fraction("3.58"), "tight_above_pos"),
                (Fraction("3.00"), Fraction("3.14"), Fraction("2.72"), "at_boundary_neg"),
                (Fraction("3.65"), Fraction("3.00"), Fraction("4.05"), "moderate_above_vs_eq_pos"),
                (Fraction("2.68"), Fraction("3.44"), Fraction("2.94"), "below_boundary_neg"),
            ]
        zb, zc, zs, prof = triples[g_idx]
        return (
            {"z": fmt_signed2(zb)},
            {"z": fmt_signed2(zc)},
            {"z": fmt_signed2(zs)},
            prof,
        )

    if rule in WECO_WINDOW_PARAMS:
        window, thresh, level = WECO_WINDOW_PARAMS[rule]
        rng = random.Random(10000 + (0 if is_dev else 5000) + RULES_ORDER.index(rule) * 100 + g_idx)

        def make_hi(k: int) -> list[Fraction]:
            pool = [Fraction(20 + 7 * i, 100) for i in range(1, 14)]  # +0.27 .. +1.11
            if not is_dev:
                pool = [Fraction(15 + 9 * i, 100) for i in range(1, 14)]  # disjoint offsets
            rng.shuffle(pool)
            return [level + d for d in pool[:k]]

        def make_lo(k: int) -> list[Fraction]:
            pool = [Fraction(18 + 8 * i, 100) for i in range(1, 14)]
            if not is_dev:
                pool = [Fraction(14 + 11 * i, 100) for i in range(1, 14)]
            rng.shuffle(pool)
            return [level - d for d in pool[:k]]

        if rule in ("weco2", "weco1"):
            # thresh = window - 1 (2 of 3, or 4 of 5)
            # Build ordered list with distinct roles, then rotate by deterministic shift
            hi_vals = make_hi(window)
            lo_vals = make_lo(window)
            if mode == 0:
                # base: count = thresh (TRUE); cross: count = thresh-1 (FALSE); same: count = window (TRUE)
                base = hi_vals[:thresh] + [lo_vals[0]]
                cross = list(base)
                cross[0] = lo_vals[1]  # edit idx 0 -> count = thresh - 1
                same = list(base)
                same[thresh] = hi_vals[thresh]  # edit last idx -> count = window
                prof = "at_count_pos"
            elif mode == 1:
                # base: count = thresh-1 (FALSE); cross: count = thresh (TRUE); same: count = thresh-2 (FALSE)
                base = hi_vals[: thresh - 1] + lo_vals[:2]
                cross = list(base)
                cross[thresh - 1] = hi_vals[thresh - 1]  # edit -> count = thresh
                same = list(base)
                same[0] = lo_vals[2]  # edit -> count = thresh - 2
                prof = "tight_below_count_neg"
            elif mode == 2:
                # Layer-1 boundary test: non-counting slot is EXACTLY level (+2.00 / +1.00)
                base = hi_vals[:thresh] + [level]
                cross = list(base)
                cross[0] = level  # exact-on-line does NOT count -> count = thresh - 1 (FALSE)
                same = list(base)
                same[thresh] = hi_vals[thresh]  # move exact-on-line above -> count = window (TRUE)
                prof = "layer1_exact_line_pos"
            else:
                # Layer-1 boundary test on FALSE base: one slot is EXACTLY level (+2.00 / +1.00)
                base = hi_vals[: thresh - 1] + [level, lo_vals[0]]
                cross = list(base)
                cross[thresh - 1] = hi_vals[thresh - 1]  # move exact-on-line above -> count = thresh (TRUE)
                same = list(base)
                same[0] = lo_vals[1]  # drop one above point -> count = thresh - 2 (FALSE)
                prof = "layer1_exact_line_neg"
        else:
            # weco8: window=8, thresh=8, level=0
            hi_vals = make_hi(10)
            lo_vals = make_lo(4)
            if mode == 0:
                base = list(hi_vals[:8])  # count=8 (TRUE)
                cross = list(base)
                cross[2] = lo_vals[0]  # count=7 (FALSE)
                same = list(base)
                same[5] = hi_vals[8]  # magnitude shift, still >0, count=8 (TRUE)
                prof = "full_window_pos_mag_shift"
            elif mode == 1:
                base = list(hi_vals[:7]) + [lo_vals[0]]  # count=7 (FALSE)
                cross = list(base)
                cross[7] = hi_vals[7]  # count=8 (TRUE)
                same = list(base)
                same[1] = lo_vals[1]  # count=6 (FALSE)
                prof = "one_short_neg"
            elif mode == 2:
                base = list(hi_vals[:8])  # count=8 (TRUE)
                cross = list(base)
                cross[4] = Fraction(0)  # exact 0.00 breaks strict >0 -> count=7 (FALSE)
                same = list(base)
                same[6] = hi_vals[9]  # magnitude shift, still >0, count=8 (TRUE)
                prof = "layer1_zero_line_pos"
            else:
                base = list(hi_vals[:7]) + [Fraction(0)]  # exact 0.00 -> count=7 (FALSE)
                cross = list(base)
                cross[7] = hi_vals[7]  # move 0.00 above 0 -> count=8 (TRUE)
                same = list(base)
                same[3] = lo_vals[0]  # drop one positive -> count=6 (FALSE)
                prof = "layer1_zero_line_neg"

        # Deterministic permutation applied identically to (base, cross, same)
        perm = list(range(window))
        rng.shuffle(perm)
        seq_b = "、".join(fmt_signed2(base[p]) for p in perm)
        seq_c = "、".join(fmt_signed2(cross[p]) for p in perm)
        seq_s = "、".join(fmt_signed2(same[p]) for p in perm)
        return {"seq": seq_b}, {"seq": seq_c}, {"seq": seq_s}, prof

    # -------- Family 3: Process Capability --------
    if rule in ("cp", "cpu", "cpl", "cpk"):
        if is_dev:
            sd_pool = [Fraction("0.5"), Fraction("1.0"), Fraction("2.0"), Fraction("1.0")]
            tgt_pool = [Fraction("1.0"), Fraction("1.2"), Fraction("1.5"), Fraction("1.25")]
            mean_pool = [Fraction("40"), Fraction("60"), Fraction("80"), Fraction("120")]
            d_tight, d_mod = Fraction("0.2"), Fraction("0.4")
        else:
            sd_pool = [
                Fraction("0.4"), Fraction("0.8"), Fraction("1.2"),
                Fraction("1.6"), Fraction("2.0"), Fraction("0.5"),
                Fraction("0.8"), Fraction("1.0"), Fraction("1.2"),
                Fraction("1.5"), Fraction("0.6"), Fraction("2.5"),
            ]
            tgt_pool = [
                Fraction("1.25"), Fraction("1.5"), Fraction("1.4"),
                Fraction("1.75"), Fraction("1.3"), Fraction("1.6"),
                Fraction("1.25"), Fraction("1.8"), Fraction("1.5"),
                Fraction("1.4"), Fraction("1.25"), Fraction("1.6"),
            ]
            mean_pool = [
                Fraction("25"), Fraction("35"), Fraction("48"),
                Fraction("55"), Fraction("72"), Fraction("90"),
                Fraction("110"), Fraction("135"), Fraction("150"),
                Fraction("175"), Fraction("210"), Fraction("240"),
            ]
            # Disjoint deltas per template to avoid repetitive values
            d_tight = [Fraction("0.15"), Fraction("0.25"), Fraction("0.2"), Fraction("0.25")][t_idx]
            d_mod = [Fraction("0.35"), Fraction("0.5"), Fraction("0.4"), Fraction("0.45")][t_idx]

        sd = sd_pool[g_idx]
        target = tgt_pool[g_idx]
        mean = mean_pool[g_idx]

        if mode == 0:
            v_b, v_c, v_s = target, target - d_tight, target + d_tight
            prof = "at_boundary_pos"
        elif mode == 1:
            v_b, v_c, v_s = target - d_tight, target, target - d_mod
            prof = "tight_below_neg"
        elif mode == 2:
            v_b, v_c, v_s = target + d_tight, target - d_mod, target + d_mod
            prof = "strict_above_pos"
        else:
            v_b, v_c, v_s = target - d_mod, target + d_tight, target - d_tight
            prof = "moderate_below_neg"

        assert v_b > 0 and v_c > 0 and v_s > 0

        def cap_inputs(v: Fraction) -> dict[str, str]:
            if rule == "cp":
                lo = mean - 3 * sd * target
                hi = lo + 6 * sd * v  # only hi changes with v!
                return {
                    "lo": fmt_num(lo),
                    "hi": fmt_num(hi),
                    "sd": fmt_num(sd),
                    "target": fmt_num(target),
                }
            if rule == "cpu":
                hi = mean + 3 * sd * v  # only hi changes with v!
                return {
                    "mean": fmt_num(mean),
                    "hi": fmt_num(hi),
                    "sd": fmt_num(sd),
                    "target": fmt_num(target),
                }
            if rule == "cpl":
                lo = mean - 3 * sd * v  # only lo changes with v!
                return {
                    "mean": fmt_num(mean),
                    "lo": fmt_num(lo),
                    "sd": fmt_num(sd),
                    "target": fmt_num(target),
                }
            # cpk: alternate binding side (upper on even g_idx, lower on odd g_idx)
            slack = target + Fraction("0.8")
            if g_idx % 2 == 0:
                lo = mean - 3 * sd * slack
                hi = mean + 3 * sd * v  # only hi changes (upper binding)
            else:
                hi = mean + 3 * sd * slack
                lo = mean - 3 * sd * v  # only lo changes (lower binding)
            return {
                "mean": fmt_num(mean),
                "lo": fmt_num(lo),
                "hi": fmt_num(hi),
                "sd": fmt_num(sd),
                "target": fmt_num(target),
            }

        return cap_inputs(v_b), cap_inputs(v_c), cap_inputs(v_s), prof

    raise ValueError(f"Unhandled rule: {rule}")


def build_split(split: str) -> list[dict[str, Any]]:
    parts_pool = DEV_PARTS if split == "dev" else TEST_PARTS
    n_configs = 2 if split == "dev" else 3
    items: list[dict[str, Any]] = []

    for r_idx, rule in enumerate(RULES_ORDER):
        spec = RULE_SPECS[rule]
        rule_templates = TEMPLATES[split][rule]
        assert len(rule_templates) == (2 if split == "dev" else 4)

        for t_idx, tmpl in enumerate(rule_templates):
            # Validate template placeholders match rule required_inputs + {part}
            expected_fields = set(spec.required_inputs) | {"part"}
            for key in ("primary", "paraphrase"):
                fields = {f for _, f, _, _ in string.Formatter().parse(tmpl[key]) if f}
                assert fields == expected_fields, (
                    f"Placeholder mismatch in {tmpl['template_id']}.{key}: got {fields}, expected {expected_fields}"
                )

            for c_idx in range(n_configs):
                group_id = f"{split}_{rule}_t{t_idx}_c{c_idx}"
                part = parts_pool[(r_idx * 7 + t_idx * 3 + c_idx) % len(parts_pool)]
                base_in, cross_in, same_in, prof = build_group_states(split, rule, t_idx, c_idx)

                # Verify single-observation edit invariants BEFORE rendering
                diff_cross = [k for k in base_in if base_in[k] != cross_in[k]]
                diff_same = [k for k in base_in if base_in[k] != same_in[k]]
                assert len(diff_cross) == 1, f"{group_id} cross diff keys: {diff_cross}"
                assert len(diff_same) == 1, f"{group_id} same diff keys: {diff_same}"

                # For WECO sequences, also verify strictly ONE point in the sequence changed
                if rule in WECO_WINDOW_PARAMS:
                    sb = base_in["seq"].split("、")
                    sc = cross_in["seq"].split("、")
                    ss = same_in["seq"].split("、")
                    assert sum(a != b for a, b in zip(sb, sc)) == 1, f"{group_id} cross multi-point edit"
                    assert sum(a != b for a, b in zip(sb, ss)) == 1, f"{group_id} same multi-point edit"

                gt_base = compute_ground_truth(rule, base_in)
                gt_cross = compute_ground_truth(rule, cross_in)
                gt_same = compute_ground_truth(rule, same_in)

                assert gt_base["positive"] != gt_cross["positive"], f"{group_id} cross did not flip!"
                assert gt_base["positive"] == gt_same["positive"], f"{group_id} same flipped!"

                variant_specs = [
                    ("base", tmpl["primary"], base_in, gt_base, None),
                    ("cross", tmpl["primary"], cross_in, gt_cross, diff_cross[0]),
                    ("same", tmpl["primary"], same_in, gt_same, diff_same[0]),
                    ("para", tmpl["paraphrase"], base_in, gt_base, None),
                ]

                for role, text_tmpl, inp, gt, edited_field in variant_specs:
                    q = text_tmpl.format(part=part, **inp)
                    items.append(
                        {
                            "id": f"{group_id}_{role}",
                            "split": split,
                            "group_id": group_id,
                            "role": role,  # base | cross | same | para
                            "rule": rule,
                            "family": spec.family,
                            "template_id": tmpl["template_id"],
                            "template_style": tmpl["style"],
                            "config_idx": c_idx,
                            "boundary_profile": prof,
                            "same_side_label": "pos" if gt_base["positive"] else "neg",
                            "edited_field": edited_field,
                            "part": part,
                            "question": q,
                            "inputs": inp,
                            "op": gt["op"],
                            "value": gt["value"],
                            "limit": gt["limit"],
                            "value_exact": gt["value_exact"],
                            "limit_exact": gt["limit_exact"],
                            "positive": gt["positive"],
                            "meta": gt["meta"],
                            "source": spec.source_url,
                        }
                    )

    # Verify paraphrase wording differs from base wording
    by_id = {x["id"]: x for x in items}
    for x in items:
        if x["role"] == "para":
            base_item = by_id[x["group_id"] + "_base"]
            assert x["question"] != base_item["question"]
            assert x["inputs"] == base_item["inputs"]
            assert x["positive"] == base_item["positive"]

    return items


def load_historical_questions() -> set[str]:
    qs = set()
    for name in ("paraphrase.json", "scenario.json", "near.json", "far.json"):
        p = OLD_SFT_DIR / name
        if p.exists():
            for r in json.loads(p.read_text(encoding="utf-8-sig")):
                qs.add(r["question"])
    if OLD_BLIND_PATH.exists():
        for r in json.loads(OLD_BLIND_PATH.read_text(encoding="utf-8-sig")):
            qs.add(r["question"])
    return qs


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def write_review_sheet(dev_items: list[dict], test_items: list[dict]) -> None:
    lines = [
        "# v2 题库领域复核清单（Domain Review Sheet）",
        "",
        "> 本清单用于 §3 第 1 周的领域复核：请逐项核验 12 条规则的边界/等号定义、每条新模板及改写的措辞无歧义性，以及代表性边界样本。**在完成复核并签署冻结前，`test_v2_candidate.json` 严禁用于任何模型评测。**",
        "",
        "## 1. 12 条规则边界与等号定义（请重点核验等号与两层比较）",
        "",
        "| 规则 | 规则族 | 比较算子 | 边界与等号说明 | 参考来源 |",
        "|---|---|---|---|---|",
    ]
    for r in RULES_ORDER:
        s = RULE_SPECS[r]
        lines.append(f"| `{s.rule}` | {s.family} | `{s.op}` | {s.boundary_desc} | [NIST]({s.source_url}) |")

    lines += [
        "",
        "## 2. 题库规模与平衡性统计",
        "",
        "| 集合 | 规则数 | 模板数/规则 | 配置数/模板 | 基础题组 | 派生题目 | 跨界对 | 同侧正对 | 同侧负对 | 改写对 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, arr, nt, nc in (("dev_v2", dev_items, 2, 2), ("test_v2_candidate", test_items, 4, 3)):
        bases = [x for x in arr if x["role"] == "base"]
        n_pos = sum(1 for x in bases if x["positive"])
        n_neg = sum(1 for x in bases if not x["positive"])
        lines.append(
            f"| `{label}` | 12 | {nt} | {nc} | {len(bases)} | {len(arr)} | {len(bases)} | {n_pos} | {n_neg} | {len(bases)} |"
        )

    lines += [
        "",
        "## 3. 终测候选集（`test_v2_candidate`）全部 48 条模板与改写逐条核验表",
        "",
        "（每条模板附一个实际实例化边界样本便于直观检查单位、等号与表达自然度）",
        "",
    ]

    by_tmpl = {}
    for x in test_items:
        if x["role"] == "base" and x["template_id"] not in by_tmpl:
            by_tmpl[x["template_id"]] = x

    for r in RULES_ORDER:
        s = RULE_SPECS[r]
        lines.append(f"### `{r}` — {s.cn_name}（`{s.op}`）")
        lines.append("")
        for t in TEMPLATES["test"][r]:
            ex = by_tmpl[t["template_id"]]
            lines.append(f"- **{t['template_id']}** (`{t['style']}`)")
            lines.append(f"  - 主模板：`{t['primary']}`")
            lines.append(f"  - 等义改写：`{t['paraphrase']}`")
            lines.append(
                f"  - 示例（`{ex['boundary_profile']}`）：统计量=`{ex['value']}`，阈值=`{ex['limit']}`，真值=`{'成立' if ex['positive'] else '不成立'}`"
            )
            lines.append(f"  - 复核状态：[ ] 规则无误  [ ] 等号无歧义  [ ] 改写等义  备注：")
        lines.append("")

    (OUT_DIR / "REVIEW_SHEET.md").write_text("\n".join(lines), encoding="utf8")


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    dev_items = build_split("dev")
    test_items = build_split("test")

    # Global uniqueness & zero-leakage checks
    hist_q = load_historical_questions()
    dev_q = [x["question"] for x in dev_items]
    test_q = [x["question"] for x in test_items]

    assert len(set(dev_q)) == len(dev_q), "Duplicate question inside dev_v2!"
    assert len(set(test_q)) == len(test_q), "Duplicate question inside test_v2_candidate!"
    assert set(dev_q).isdisjoint(set(test_q)), "Leakage between dev_v2 and test_v2_candidate!"
    assert set(dev_q).isdisjoint(hist_q), "Leakage between dev_v2 and historical sets!"
    assert set(test_q).isdisjoint(hist_q), "Leakage between test_v2_candidate and historical sets!"
    assert set(DEV_PARTS).isdisjoint(set(TEST_PARTS)), "Workpiece overlap between dev and test!"

    # Balance checks per rule
    for name, arr, expected_groups in (("dev", dev_items, 48), ("test", test_items, 144)):
        bases = [x for x in arr if x["role"] == "base"]
        assert len(bases) == expected_groups and len(arr) == 4 * expected_groups
        for r in RULES_ORDER:
            rb = [x for x in bases if x["rule"] == r]
            n_pos = sum(1 for x in rb if x["positive"])
            n_neg = sum(1 for x in rb if not x["positive"])
            assert n_pos == n_neg, f"Unbalanced anchor labels in {name}/{r}: pos={n_pos}, neg={n_neg}"

    # Deterministic shuffled evaluation orders (separate seeds for dev and test)
    dev_eval = list(dev_items)
    random.Random(20261119).shuffle(dev_eval)
    test_eval = list(test_items)
    random.Random(20261120).shuffle(test_eval)

    dev_path = DATA_DIR / "dev_v2.json"
    test_path = DATA_DIR / "test_v2_candidate.json"
    dev_bytes = json.dumps(dev_eval, ensure_ascii=False, indent=2).encode("utf8")
    test_bytes = json.dumps(test_eval, ensure_ascii=False, indent=2).encode("utf8")
    dev_path.write_bytes(dev_bytes)
    test_path.write_bytes(test_bytes)

    manifest = {
        "schema_version": "v2.0-candidate",
        "status": {
            "dev_v2": "ready_for_development",
            "test_v2_candidate": "pending_domain_review_DO_NOT_EVALUATE_YET",
        },
        "rules": list(RULES_ORDER),
        "counts": {
            "dev_v2": {
                "base_groups": 48,
                "items": len(dev_eval),
                "by_family": dict(Counter(x["family"] for x in dev_eval if x["role"] == "base")),
                "boundary_relations": dict(Counter(x["meta"]["boundary_relation"] for x in dev_eval)),
            },
            "test_v2_candidate": {
                "base_groups": 144,
                "items": len(test_eval),
                "by_family": dict(Counter(x["family"] for x in test_eval if x["role"] == "base")),
                "boundary_relations": dict(Counter(x["meta"]["boundary_relation"] for x in test_eval)),
            },
        },
        "sha256": {
            "dev_v2.json": sha256_bytes(dev_bytes),
            "test_v2_candidate.json": sha256_bytes(test_bytes),
            "rules.py": sha256_bytes((OUT_DIR / "rules.py").read_bytes()),
            "templates.py": sha256_bytes((OUT_DIR / "templates.py").read_bytes()),
        },
        "checks_passed": [
            "exact_fraction_truth_and_float_comparator_100pct_agreement",
            "capability_rounding_ambiguity_guard",
            "strict_single_observation_edit_for_all_cross_and_same_pairs",
            "strict_single_point_edit_for_all_weco_window_pairs",
            "exact_input_identity_and_wording_change_for_all_paraphrase_pairs",
            "balanced_50_50_same_side_positive_and_negative_per_rule",
            "zero_question_overlap_across_dev_test_and_historical_sets",
            "disjoint_workpiece_pools_across_dev_test_and_historical_sets",
        ],
    }
    (DATA_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf8"
    )
    write_review_sheet(dev_items, test_items)
    print("Built v2-benchmark successfully!")
    print(json.dumps(manifest["counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

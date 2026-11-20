"""Canonical 12-rule quality-control specification, exact Fraction truth calculators,
deterministic comparators, and rounding-ambiguity guards for v2-benchmark.

Rules (3 families x 4 rules = 12):
  sampling:       single, first_accept, first_reject, second_accept
  control_chart:  weco3, weco2, weco1, weco8
  capability:     cp, cpu, cpl, cpk
"""

from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from typing import Any


@dataclass(frozen=True)
class RuleSpec:
    rule: str
    family: str
    op: str  # "<=", ">=", ">"
    cn_name: str
    boundary_desc: str
    required_inputs: tuple[str, ...]
    source_url: str


RULE_SPECS: dict[str, RuleSpec] = {
    "single": RuleSpec(
        rule="single",
        family="sampling",
        op="<=",
        cn_name="一次计数抽样接收",
        boundary_desc="不合格品数 x <= 接收数 c 时判定成立（含等于）；x > c 不成立。",
        required_inputs=("n", "c", "x"),
        source_url="https://www.itl.nist.gov/div898/handbook/pmc/section2/pmc23.htm",
    ),
    "first_accept": RuleSpec(
        rule="first_accept",
        family="sampling",
        op="<=",
        cn_name="二次抽样首样直接接收",
        boundary_desc="首样不合格品数 x1 <= 首样接收数 c1 时首样直接接收成立（含等于）；x1 > c1 不成立。",
        required_inputs=("n1", "c1", "r1", "x1"),
        source_url="https://www.itl.nist.gov/div898/handbook/pmc/section2/pmc251.htm",
    ),
    "first_reject": RuleSpec(
        rule="first_reject",
        family="sampling",
        op=">=",
        cn_name="二次抽样首样直接拒收",
        boundary_desc="首样不合格品数 x1 >= 首样拒收数 r1 时首样直接拒收成立（含等于）；x1 < r1 不成立。",
        required_inputs=("n1", "c1", "r1", "x1"),
        source_url="https://www.itl.nist.gov/div898/handbook/pmc/section2/pmc251.htm",
    ),
    "second_accept": RuleSpec(
        rule="second_accept",
        family="sampling",
        op="<=",
        cn_name="二次抽样累计接收",
        boundary_desc="两样本累计不合格品数 x1 + x2 <= 累计接收数 c2 时接收成立（含等于）；x1 + x2 > c2 不成立。",
        required_inputs=("n1", "n2", "x1", "x2", "c2"),
        source_url="https://www.itl.nist.gov/div898/handbook/pmc/section2/pmc251.htm",
    ),
    "weco3": RuleSpec(
        rule="weco3",
        family="control_chart",
        op=">",
        cn_name="控制图上侧单点 3σ 越限",
        boundary_desc="单点标准化偏差 z > 3 时触发越限（严格大于，z = 3.00 不触发）。",
        required_inputs=("z",),
        source_url="https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc32.htm",
    ),
    "weco2": RuleSpec(
        rule="weco2",
        family="control_chart",
        op=">=",
        cn_name="控制图上侧 WECO 3点中至少2点高于+2σ",
        boundary_desc="两层比较：①逐点严格高于 +2.00σ（等于 +2.00 不计入）；②窗口3点中计入点数 >= 2 时触发（含等于）。",
        required_inputs=("seq",),
        source_url="https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc32.htm",
    ),
    "weco1": RuleSpec(
        rule="weco1",
        family="control_chart",
        op=">=",
        cn_name="控制图上侧 WECO 5点中至少4点高于+1σ",
        boundary_desc="两层比较：①逐点严格高于 +1.00σ（等于 +1.00 不计入）；②窗口5点中计入点数 >= 4 时触发（含等于）。",
        required_inputs=("seq",),
        source_url="https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc32.htm",
    ),
    "weco8": RuleSpec(
        rule="weco8",
        family="control_chart",
        op=">=",
        cn_name="控制图上侧 WECO 连续8点高于中心线(0σ)",
        boundary_desc="两层比较：①逐点严格高于 0.00σ（等于 0.00 不计入）；②窗口8点中计入点数 >= 8 时触发（即全8点均严格大于0）。",
        required_inputs=("seq",),
        source_url="https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc32.htm",
    ),
    "cp": RuleSpec(
        rule="cp",
        family="capability",
        op=">=",
        cn_name="过程能力指数 Cp 达标",
        boundary_desc="Cp = (USL - LSL) / (6 * sigma) >= target 时达标（含等于）；真值按未舍入精确值判定，并剔除舍入歧义题。",
        required_inputs=("lo", "hi", "sd", "target"),
        source_url="https://www.itl.nist.gov/div898/handbook/pmc/section1/pmc16.htm",
    ),
    "cpu": RuleSpec(
        rule="cpu",
        family="capability",
        op=">=",
        cn_name="单侧过程能力指数 Cpu 达标",
        boundary_desc="Cpu = (USL - mean) / (3 * sigma) >= target 时达标（含等于）；真值按未舍入精确值判定。",
        required_inputs=("mean", "hi", "sd", "target"),
        source_url="https://www.itl.nist.gov/div898/handbook/pmc/section1/pmc16.htm",
    ),
    "cpl": RuleSpec(
        rule="cpl",
        family="capability",
        op=">=",
        cn_name="单侧过程能力指数 Cpl 达标",
        boundary_desc="Cpl = (mean - LSL) / (3 * sigma) >= target 时达标（含等于）；真值按未舍入精确值判定。",
        required_inputs=("mean", "lo", "sd", "target"),
        source_url="https://www.itl.nist.gov/div898/handbook/pmc/section1/pmc16.htm",
    ),
    "cpk": RuleSpec(
        rule="cpk",
        family="capability",
        op=">=",
        cn_name="过程能力指数 Cpk 达标",
        boundary_desc="Cpk = min((USL - mean)/(3*sigma), (mean - LSL)/(3*sigma)) >= target 时达标（含等于）；真值按未舍入精确值判定。",
        required_inputs=("mean", "lo", "hi", "sd", "target"),
        source_url="https://www.itl.nist.gov/div898/handbook/pmc/section1/pmc16.htm",
    ),
}

RULES_ORDER: tuple[str, ...] = tuple(RULE_SPECS.keys())

WECO_WINDOW_PARAMS: dict[str, tuple[int, int, Fraction]] = {
    "weco2": (3, 2, Fraction(2)),
    "weco1": (5, 4, Fraction(1)),
    "weco8": (8, 8, Fraction(0)),
}

# Display precision for capability indices and z-scores
CAPABILITY_DISPLAY_DECIMALS = 2
CAPABILITY_AMBIGUITY_MARGIN = Fraction(1, 25)  # 0.04 minimum distance unless exact equality


def to_fraction(val: int | float | str | Decimal | Fraction) -> Fraction:
    if isinstance(val, Fraction):
        return val
    return Fraction(str(val))


def compare_exact(op: str, stat: Fraction, limit: Fraction) -> bool:
    if op == "<=":
        return stat <= limit
    if op == ">=":
        return stat >= limit
    if op == ">":
        return stat > limit
    if op == "<":
        return stat < limit
    raise ValueError(f"Unsupported operator: {op}")


def compare_float(op: str, stat: float | None, limit: float | None, eps: float = 1e-9) -> bool | None:
    if stat is None or limit is None or op is None:
        return None
    if op == "<=":
        return stat <= limit + eps
    if op == ">=":
        return stat >= limit - eps
    if op == ">":
        return stat > limit + eps
    if op == "<":
        return stat < limit - eps
    return None


def compute_ground_truth(rule: str, inputs: dict[str, Any]) -> dict[str, Any]:
    """Deterministically compute exact statistic, threshold, operator, and verdict from inputs.

    Also enforces domain sanity checks and rounding-ambiguity guards.
    """
    spec = RULE_SPECS[rule]
    missing = set(spec.required_inputs) - set(inputs.keys())
    if missing:
        raise ValueError(f"Missing inputs for {rule}: {missing}")

    if rule == "single":
        n, c, x = int(inputs["n"]), int(inputs["c"]), int(inputs["x"])
        assert 0 <= c < n and 0 <= x <= n, f"Invalid single sampling params: {inputs}"
        stat_f, lim_f = Fraction(x), Fraction(c)
        meta = {"boundary_relation": _rel(stat_f, lim_f)}

    elif rule == "first_accept":
        n1, c1, r1, x1 = int(inputs["n1"]), int(inputs["c1"]), int(inputs["r1"]), int(inputs["x1"])
        assert 0 <= c1 < r1 <= n1 and 0 <= x1 <= n1, f"Invalid first_accept params: {inputs}"
        stat_f, lim_f = Fraction(x1), Fraction(c1)
        meta = {"boundary_relation": _rel(stat_f, lim_f)}

    elif rule == "first_reject":
        n1, c1, r1, x1 = int(inputs["n1"]), int(inputs["c1"]), int(inputs["r1"]), int(inputs["x1"])
        assert 0 <= c1 < r1 <= n1 and 0 <= x1 <= n1, f"Invalid first_reject params: {inputs}"
        stat_f, lim_f = Fraction(x1), Fraction(r1)
        meta = {"boundary_relation": _rel(stat_f, lim_f)}

    elif rule == "second_accept":
        n1, n2 = int(inputs["n1"]), int(inputs["n2"])
        x1, x2, c2 = int(inputs["x1"]), int(inputs["x2"]), int(inputs["c2"])
        assert n1 > 0 and n2 > 0 and 0 <= x1 <= n1 and 0 <= x2 <= n2 and 0 <= c2 <= n1 + n2
        stat_f, lim_f = Fraction(x1 + x2), Fraction(c2)
        meta = {"boundary_relation": _rel(stat_f, lim_f)}

    elif rule == "weco3":
        z_str = str(inputs["z"])
        stat_f = to_fraction(z_str)
        lim_f = Fraction(3)
        meta = {"boundary_relation": _rel(stat_f, lim_f), "strict_point_check": True}

    elif rule in WECO_WINDOW_PARAMS:
        window, thresh, level = WECO_WINDOW_PARAMS[rule]
        raw_seq = inputs["seq"]
        parts = [p.strip() for p in raw_seq.split("、")] if isinstance(raw_seq, str) else [str(x) for x in raw_seq]
        assert len(parts) == window, f"Expected {window} points for {rule}, got {len(parts)}"
        z_vals = [to_fraction(p) for p in parts]
        above_flags = [z > level for z in z_vals]  # Layer 1: strict > level
        eq_boundary_pts = sum(z == level for z in z_vals)
        count = sum(above_flags)
        stat_f, lim_f = Fraction(count), Fraction(thresh)
        meta = {
            "boundary_relation": _rel(stat_f, lim_f),
            "layer1_level": float(level),
            "layer1_strict_above_count": count,
            "layer1_exact_on_line_points": eq_boundary_pts,
        }

    elif rule in ("cp", "cpu", "cpl", "cpk"):
        sd = to_fraction(inputs["sd"])
        target = to_fraction(inputs["target"])
        assert sd > 0 and target > 0
        if rule == "cp":
            lo, hi = to_fraction(inputs["lo"]), to_fraction(inputs["hi"])
            assert hi > lo
            stat_f = (hi - lo) / (6 * sd)
            binding = "width"
        elif rule == "cpu":
            mean, hi = to_fraction(inputs["mean"]), to_fraction(inputs["hi"])
            assert hi > mean
            stat_f = (hi - mean) / (3 * sd)
            binding = "upper"
        elif rule == "cpl":
            mean, lo = to_fraction(inputs["mean"]), to_fraction(inputs["lo"])
            assert mean > lo
            stat_f = (mean - lo) / (3 * sd)
            binding = "lower"
        else:
            mean, lo, hi = to_fraction(inputs["mean"]), to_fraction(inputs["lo"]), to_fraction(inputs["hi"])
            assert lo < mean < hi
            cpu = (hi - mean) / (3 * sd)
            cpl = (mean - lo) / (3 * sd)
            stat_f = min(cpu, cpl)
            binding = "upper" if cpu < cpl else ("lower" if cpl < cpu else "tied")
        lim_f = target

        # Rounding-ambiguity guard (§2):
        # Exact unrounded truth must either equal target OR be well separated from target,
        # and rounded display value must agree with exact unrounded truth.
        exact_pos = compare_exact(spec.op, stat_f, lim_f)
        if stat_f != lim_f:
            dist = abs(stat_f - lim_f)
            assert dist >= CAPABILITY_AMBIGUITY_MARGIN, (
                f"Ambiguous capability case for {rule}: stat={float(stat_f):.6f}, "
                f"target={float(lim_f):.6f}, dist={float(dist):.6f}"
            )
        disp_val = round(float(stat_f), CAPABILITY_DISPLAY_DECIMALS)
        disp_pos = compare_float(spec.op, disp_val, float(lim_f))
        assert disp_pos == exact_pos, (
            f"Display rounding flips verdict for {rule}: exact={float(stat_f)} ({exact_pos}) "
            f"vs disp={disp_val} ({disp_pos}) against {float(lim_f)}"
        )
        meta = {
            "boundary_relation": _rel(stat_f, lim_f),
            "binding_side": binding,
            "exact_fraction": f"{stat_f.numerator}/{stat_f.denominator}",
            "display_decimals": CAPABILITY_DISPLAY_DECIMALS,
        }
    else:
        raise ValueError(f"Unknown rule: {rule}")

    positive = compare_exact(spec.op, stat_f, lim_f)
    # Cross-check with float comparator (row 5 verification)
    assert compare_float(spec.op, float(stat_f), float(lim_f)) == positive

    return {
        "rule": rule,
        "family": spec.family,
        "op": spec.op,
        "value": round(float(stat_f), 6),
        "limit": round(float(lim_f), 6),
        "value_exact": f"{stat_f.numerator}/{stat_f.denominator}",
        "limit_exact": f"{lim_f.numerator}/{lim_f.denominator}",
        "positive": positive,
        "meta": meta,
    }


def _rel(stat: Fraction, limit: Fraction) -> str:
    if stat < limit:
        return "below"
    if stat == limit:
        return "at"
    return "above"

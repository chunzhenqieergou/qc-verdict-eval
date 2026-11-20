"""Publication-quality figure generator for the manuscript (PDF + 300-DPI PNG).

Supports two modes:
  --mode dev    Generate and preview all 4 figures from completed dev_v2 + transfer experiments
                (keeps test_v2_candidate completely untouched).
  --mode final  Generate final submission figures directly from `final_scores.json` after blind test.

Usage (system Python with matplotlib 3.11+):
  python make_figures.py --mode dev
  python make_figures.py --mode final
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT = Path(__file__).resolve().parent
EXP_ROOT = ROOT.parent

RULES_ORDER = [
    "single", "first_accept", "first_reject", "second_accept",
    "weco3", "weco2", "weco1", "weco8",
    "cp", "cpu", "cpl", "cpk",
]
RULE_SHORT = {
    "single": "single\n(<=)",
    "first_accept": "1st_acc\n(<=)",
    "first_reject": "1st_rej\n(>=)",
    "second_accept": "2nd_acc\n(<=)",
    "weco3": "weco3\n(>)",
    "weco2": "weco2\n(>=)",
    "weco1": "weco1\n(>=)",
    "weco8": "weco8\n(>=)",
    "cp": "cp\n(>=)",
    "cpu": "cpu\n(>=)",
    "cpl": "cpl\n(>=)",
    "cpk": "cpk\n(>=)",
}

# Colorblind-safe Okabe-Ito palette
C_BLUE = "#0072B2"
C_ORANGE = "#D55E00"
C_GREEN = "#009E73"
C_SKY = "#56B4E9"
C_PURPLE = "#CC79A7"
C_YELLOW = "#E69F00"
C_GRAY = "#666666"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9.5,
    "axes.titlesize": 10.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 9.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
})


def load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8-sig"))


def save_fig(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.png", bbox_inches="tight", dpi=300)
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_dir / name}.png/.pdf")


def load_bundle(mode: str) -> dict:
    transfer = load_json(EXP_ROOT / "v2-transfer-decomp" / "transfer_scores.json")
    if mode == "final":
        final = load_json(ROOT / "final_scores.json")
        return {
            "tag": "Frozen Blind Test (144 base groups / 576 items)",
            "qwen_scored": final["models"]["qwen"]["scored"],
            "phi4_scored": final["models"]["phi4"]["scored"],
            "qwen_boot": final["models"]["qwen"]["bootstrap"],
            "phi4_boot": final["models"]["phi4"]["bootstrap"],
            "qwen_ba": final["models"]["qwen"]["boundary_audits"],
            "phi4_ba": final["models"]["phi4"]["boundary_audits"],
            "transfer": transfer,
        }

    # mode == "dev"
    q_base = load_json(EXP_ROOT / "v2-benchmark" / "dev_scores.json")
    q_op = load_json(EXP_ROOT / "v2-operator-comparator" / "dev_op_scores.json")
    q_ba_base = load_json(EXP_ROOT / "v2-benchmark" / "dev_boundary_audit.json")
    phi = load_json(EXP_ROOT / "v2-cross-model-phi4" / "phi4_dev_scores.json")

    q_scored = {**q_base["scored"], **q_op["scored"]}
    # Normalize boundary audit formats into {rule: {rel: {"ok": ..., "n": ...}}}
    def norm_ba(d):
        out = {}
        for r, rels in d.items():
            out[r] = {}
            for rel, val in rels.items():
                if isinstance(val, dict):
                    out[r][rel] = {"ok": val["ok"], "n": val["n"]}
                else:
                    out[r][rel] = {"ok": val[0], "n": val[1]}
        return out

    q_ba = {
        "NF_NF_e2e": norm_ba({r: q_ba_base["NF_train_NF_infer"][r] for r in ("single", "first_accept", "first_reject", "weco3")}),
        "SFT_NF_op_e2e": norm_ba(q_op["boundary_audits"]["SFT_NF_op_e2e"]),
        "SFT_NF_op_deployable": norm_ba(q_op["boundary_audits"]["SFT_NF_op_deployable"]),
        "SFT_VF_op_deployable": norm_ba(q_op["boundary_audits"]["SFT_VF_op_deployable"]),
    }
    phi_ba = {k: norm_ba(v) for k, v in phi["boundary_audits"].items()}

    q_boot = {
        "order_NFNF_minus_VFVF_cross": q_base["primary_diff_NFNF_minus_VFVF"]["cross_boundary"],
        "dep_minus_e2e_VF_op_cross": q_op["bootstrap"]["SFT_VF_op_deployable_minus_e2e"],
        "dep_minus_e2e_NF_op_cross": q_op["bootstrap"]["SFT_NF_op_deployable_minus_e2e"],
    }
    phi_boot = {
        "order_NFNF_minus_VFVF_cross": phi["bootstrap"]["NFNF_minus_VFVF_e2e"],
        "dep_minus_e2e_VF_op_cross": phi["bootstrap"]["SFT_VF_op_deployable_minus_e2e"],
        "dep_minus_e2e_NF_op_cross": phi["bootstrap"]["SFT_NF_op_deployable_minus_e2e"],
    }
    return {
        "tag": "v2 Dev Set (48 base groups / 192 items)",
        "qwen_scored": q_scored,
        "phi4_scored": phi["scored"],
        "qwen_boot": q_boot,
        "phi4_boot": phi_boot,
        "qwen_ba": q_ba,
        "phi4_ba": phi_ba,
        "transfer": transfer,
    }


def make_fig1_protocol(out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.6, 4.3))
    ax.set_xlim(0, 11.6)
    ax.set_ylim(0, 4.3)
    ax.axis("off")

    def box(x, y, w, h, title, body, fc, ec, lw=1.3):
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.12",
            facecolor=fc, edgecolor=ec, linewidth=lw
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h - 0.18, title, ha="center", va="top", fontsize=8.5, fontweight="bold", color="#1a1a1a")
        ax.text(x + w / 2, y + (h - 0.34) / 2, body, ha="center", va="center", fontsize=7.9, color="#333333", linespacing=1.32)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>,head_width=0.22,head_length=0.35", color="#444444", lw=1.25))

    # Col 1: Base Group
    box(0.15, 1.32, 2.35, 1.72, "1. Base Config (12 Rules)",
        "Example: Single Sampling\nAccept limit c = 8 (<=)\nAnchor: x = 8 (True)\nExact Fraction Truth",
        "#F0F6FA", C_BLUE)

    # Col 2: Three Paired Probes
    box(3.05, 2.90, 2.55, 1.12, "Cross-Boundary Pair",
        "x = 8 -> 9 (True -> False)\nDetects label collapse &\nmissed boundary crossing",
        "#FFF4EC", C_ORANGE)
    box(3.05, 1.60, 2.55, 1.12, "Same-Side Pair (Pos/Neg)",
        "x = 8 -> 7 (True -> True)\nDetects '==' shortcut &\nover-reaction to edits",
        "#EDF8F4", C_GREEN)
    box(3.05, 0.30, 2.55, 1.12, "Paraphrase Control",
        "x = 8 -> 8 (reworded)\nDetects wording sensitivity\n(identical numbers & rule)",
        "#F5F0FA", C_PURPLE)

    for yy in (3.46, 2.16, 0.86):
        arrow(2.56, 2.18, 2.98, yy)

    # Col 3: Independent Queries & Outputs
    box(6.15, 1.12, 2.35, 2.12, "2. Independent Queries",
        "Each member queried\nseparately (greedy).\n\nParsed fields:\n• Statistic (e.g., 7)\n• Threshold (e.g., 8)\n• Operator (<=, >=, >)\n• Verdict (True/False)",
        "#F7F7F7", C_GRAY)

    for yy in (3.46, 2.16, 0.86):
        arrow(5.66, yy, 6.08, 2.18)

    # Col 4: Three-Stage Audit
    box(9.05, 2.90, 2.40, 1.12, "Stage 1: Statistic Error",
        "Oracle-Rule Comparator\nCeiling (counts / arithmetic)",
        "#EBF4FB", C_BLUE)
    box(9.05, 1.60, 2.40, 1.12, "Stage 2: Operator Error",
        "Deployable Comparator\ncompare(pred_op, stat, lim)",
        "#FFF9E6", C_YELLOW)
    box(9.05, 0.30, 2.40, 1.12, "Stage 3: Verdict Binding",
        "End-to-End Verdict\n(VF vs NF; == shortcuts)",
        "#FDEEE9", C_ORANGE)

    for yy in (3.46, 2.16, 0.86):
        arrow(8.56, 2.18, 8.98, yy)

    ax.set_title("Paired Behavioral Evaluation & Three-Stage Error Decomposition Protocol", pad=6, fontsize=11)
    save_fig(fig, out_dir, "fig1_protocol_overview")


def make_fig2_per_rule_seeds(bundle: dict, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11.0, 6.8), sharex=True)
    x = np.arange(len(RULES_ORDER))

    def shade_families(ax):
        ax.axvspan(-0.5, 3.5, color="#E8F3FA", alpha=0.35, zorder=0)
        ax.axvspan(3.5, 7.5, color="#FDF3E7", alpha=0.35, zorder=0)
        ax.axvspan(7.5, 11.5, color="#EEF7F1", alpha=0.35, zorder=0)
        ax.axhline(0, color="#444444", lw=0.9, ls="--", zorder=1)

    # Panel A: Order effect (NF-NF - VF-VF, end-to-end cross-boundary paired acc)
    ax = axes[0]
    shade_families(ax)
    for m_idx, (label, sc, color, offset) in enumerate([
        ("Qwen2.5-3B-Instruct", bundle["qwen_scored"], C_BLUE, -0.16),
        ("Phi-4-mini-instruct", bundle["phi4_scored"], C_ORANGE, +0.16),
    ]):
        means = []
        for r_i, rule in enumerate(RULES_ORDER):
            diffs = []
            for s_i in range(3):
                nf = sc["NF_train_NF_infer"][s_i]["end_to_end"]["by_rule"][rule]
                vf = sc["VF_train_VF_infer"][s_i]["end_to_end"]["by_rule"][rule]
                diffs.append(100.0 * (nf["cross_ok"] / nf["groups"] - vf["cross_ok"] / vf["groups"]))
            # Plot individual seed dots
            ax.scatter(
                [r_i + offset] * 3 + np.array([-0.03, 0.0, 0.03]),
                diffs, color=color, alpha=0.45, s=26, zorder=3, edgecolors="none"
            )
            means.append(np.mean(diffs))
        ax.scatter(x + offset, means, color=color, s=62, marker="D", edgecolors="white", lw=0.7, label=label, zorder=4)

    qb = bundle["qwen_boot"]["order_NFNF_minus_VFVF_cross"]
    pb = bundle["phi4_boot"]["order_NFNF_minus_VFVF_cross"]
    ax.set_ylabel("Cross-Boundary Pair Diff (pp)")
    ax.set_title(
        f"(a) Output-Order Effect (NF-NF minus VF-VF, End-to-End) — "
        f"Qwen: {qb['point_diff_pp']:+.1f} pp {qb['ci95_template_cluster_pp']}, "
        f"Phi-4: {pb['point_diff_pp']:+.1f} pp {pb['ci95_template_cluster_pp']}"
    )
    ax.set_ylim(-32, 126)
    ax.legend(loc="upper right", bbox_to_anchor=(0.995, 0.86), frameon=True, framealpha=0.95)
    for cx, flab in ((1.5, "Acceptance Sampling"), (5.5, "Control Chart (WECO)"), (9.5, "Process Capability")):
        ax.text(cx, 120, flab, ha="center", va="top", fontsize=8.8, fontweight="bold", color="#444444")

    # Panel B: Deployable Comparator net effect on SFT_VF_op (Deployable - End-to-End)
    ax = axes[1]
    shade_families(ax)
    for label, sc, color, offset in [
        ("Qwen2.5-3B (SFT_VF_op: Deployable - E2E)", bundle["qwen_scored"], C_BLUE, -0.16),
        ("Phi-4-mini (SFT_VF_op: Deployable - E2E)", bundle["phi4_scored"], C_ORANGE, +0.16),
    ]:
        means = []
        for r_i, rule in enumerate(RULES_ORDER):
            diffs = []
            for s_i in range(3):
                dep = sc["SFT_VF_op"][s_i]["deployable_comparator"]["by_rule"][rule]
                e2e = sc["SFT_VF_op"][s_i]["end_to_end"]["by_rule"][rule]
                diffs.append(100.0 * (dep["cross_ok"] / dep["groups"] - e2e["cross_ok"] / e2e["groups"]))
            ax.scatter(
                [r_i + offset] * 3 + np.array([-0.03, 0.0, 0.03]),
                diffs, color=color, alpha=0.45, s=26, zorder=3, edgecolors="none"
            )
            means.append(np.mean(diffs))
        ax.scatter(x + offset, means, color=color, s=62, marker="s", edgecolors="white", lw=0.7, label=label, zorder=4)

    qbd = bundle["qwen_boot"]["dep_minus_e2e_VF_op_cross"]
    pbd = bundle["phi4_boot"]["dep_minus_e2e_VF_op_cross"]
    ax.set_ylabel("Deployable Comparator Gain (pp)")
    ax.set_title(
        f"(b) Deployable Comparator Gain over End-to-End (SFT_VF_op) — "
        f"Qwen: {qbd['point_diff_pp']:+.1f} pp {qbd['ci95_template_cluster_pp']}, "
        f"Phi-4: {pbd['point_diff_pp']:+.1f} pp {pbd['ci95_template_cluster_pp']}"
    )
    ax.set_xticks(x)
    ax.set_xticklabels([RULE_SHORT[r] for r in RULES_ORDER], fontsize=8.5)
    ax.set_ylim(-30, 112)
    ax.legend(loc="upper right", frameon=True, framealpha=0.9)

    fig.suptitle(f"Per-Rule, Per-Seed Paired Accuracy Effects — {bundle['tag']}", fontsize=11.5, fontweight="bold", y=1.01)
    fig.tight_layout()
    save_fig(fig, out_dir, "fig2_per_rule_seed_effects")


def make_fig3_shortcut_and_stages(bundle: dict, out_dir: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.3), gridspec_kw={"width_ratios": [1.05, 1.15]})

    # Panel A: Equality shortcut & strict '>' audit on Qwen2.5-3B
    probes = [
        ("single\n(below, x<c)", "single", "below"),
        ("single\n(at, x=c)", "single", "at"),
        ("1st_acc\n(below, x<c)", "first_accept", "below"),
        ("1st_acc\n(at, x=c)", "first_accept", "at"),
        ("weco3\n(at, z=3.0)", "weco3", "at"),
        ("weco3\n(above, z>3)", "weco3", "above"),
    ]
    qba = bundle["qwen_ba"]
    vals_e2e = [100 * qba["NF_NF_e2e"][r][rel]["ok"] / qba["NF_NF_e2e"][r][rel]["n"] for _, r, rel in probes]
    vals_dep = [100 * qba["SFT_NF_op_deployable"][r][rel]["ok"] / qba["SFT_NF_op_deployable"][r][rel]["n"] for _, r, rel in probes]

    x1 = np.arange(len(probes))
    w = 0.36
    b1 = ax1.bar(x1 - w / 2, vals_e2e, width=w, color=C_ORANGE, label="NF-NF End-to-End")
    b2 = ax1.bar(x1 + w / 2, vals_dep, width=w, color=C_GREEN, label="SFT_NF_op Deployable Comp.")
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, h + 1.8, f"{h:.0f}%", ha="center", va="bottom", fontsize=7.6)
    ax1.set_xticks(x1)
    ax1.set_xticklabels([p[0] for p in probes], fontsize=8.0)
    ax1.set_ylim(0, 128)
    ax1.set_ylabel("Item Verdict Accuracy (%)")
    ax1.set_title("(a) Equality Shortcut & Strict Boundary Audit (Qwen2.5-3B)")
    ax1.legend(loc="upper center", ncol=2, fontsize=8.0, frameon=True)

    # Panel B: Three-Stage Bottleneck by Rule Family (Qwen & Phi-4 on SFT_NF_op)
    fams = ["sampling", "control_chart", "capability"]
    fam_labels = ["Sampling\n(Direct/Sum)", "Control Chart\n(WECO)", "Capability\n(Arithmetic)"]
    groups_x = np.arange(len(fams))
    bw = 0.13

    def fam_pct(sc, cond, proc, fam):
        ok = sum(r[proc]["by_family"][fam]["cross_ok"] for r in sc[cond])
        tot = sum(r[proc]["by_family"][fam]["groups"] for r in sc[cond])
        return 100.0 * ok / tot

    series = [
        ("Qwen E2E", bundle["qwen_scored"], "SFT_NF_op", "end_to_end", C_SKY, -2.5 * bw, ""),
        ("Qwen Deploy. Comp.", bundle["qwen_scored"], "SFT_NF_op", "deployable_comparator", C_BLUE, -1.5 * bw, ""),
        ("Qwen Oracle-Rule", bundle["qwen_scored"], "SFT_NF_op", "oracle_rule_comparator", "#003E6B", -0.5 * bw, "//"),
        ("Phi-4 E2E", bundle["phi4_scored"], "SFT_NF_op", "end_to_end", "#F4B183", +0.5 * bw, ""),
        ("Phi-4 Deploy. Comp.", bundle["phi4_scored"], "SFT_NF_op", "deployable_comparator", C_ORANGE, +1.5 * bw, ""),
        ("Phi-4 Oracle-Rule", bundle["phi4_scored"], "SFT_NF_op", "oracle_rule_comparator", "#8C3B00", +2.5 * bw, "//"),
    ]
    for lbl, sc, cond, proc, col, dx, hatch in series:
        ys = [fam_pct(sc, cond, proc, f) for f in fams]
        ax2.bar(groups_x + dx, ys, width=bw * 0.92, color=col, hatch=hatch, edgecolor="white" if hatch else "none", label=lbl)

    ax2.set_xticks(groups_x)
    ax2.set_xticklabels(fam_labels, fontsize=8.8)
    ax2.set_ylim(0, 110)
    ax2.set_ylabel("Cross-Boundary Paired Accuracy (%)")
    ax2.set_title("(b) Three-Stage Bottleneck by Rule Family (SFT_NF_op)")
    ax2.legend(loc="upper right", ncol=2, fontsize=7.6, frameon=True)

    fig.suptitle(f"Mechanistic Error Diagnosis — {bundle['tag']}", fontsize=11.2, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, out_dir, "fig3_boundary_shortcut_and_stages")


def make_fig4_transfer(bundle: dict, out_dir: Path) -> None:
    nf = bundle["transfer"]["NF_NF"]
    cells = ["old_word__old_num", "new_word__old_num", "old_word__new_num", "new_word__new_num"]
    cell_labels = ["Old Word\n× Old Num", "New Word\n× Old Num", "Old Word\n× New Num", "New Word\n× New Num"]

    fig, ax = plt.subplots(figsize=(8.8, 4.1))
    x = np.arange(len(cells))
    w = 0.19

    def get_vals(fam_or_all):
        if fam_or_all == "all":
            return [100.0 * nf["by_cell"][c]["e2e_pair_ok"] / nf["by_cell"][c]["pairs"] for c in cells]
        return [100.0 * nf["by_cell_fam"][f"{c}|{fam_or_all}"]["e2e_pair_ok"] / nf["by_cell_fam"][f"{c}|{fam_or_all}"]["pairs"] for c in cells]

    specs = [
        ("All 12 Rules", "all", "#222222", -1.5 * w),
        ("Sampling", "sampling", C_BLUE, -0.5 * w),
        ("Control Chart", "control_chart", C_YELLOW, +0.5 * w),
        ("Capability", "capability", C_ORANGE, +1.5 * w),
    ]
    for lbl, key, col, dx in specs:
        ys = get_vals(key)
        bars = ax.bar(x + dx, ys, width=w * 0.92, color=col, label=lbl)
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, h + 1.4, f"{h:.0f}%", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(cell_labels, fontsize=9.0)
    ax.set_ylim(0, 106)
    ax.set_ylabel("Cross-Boundary Paired Accuracy (%)")
    ax.set_title("2×2 Transfer Decomposition: Wording Shift vs. Numerical-Config Shift (Qwen2.5-3B NF-NF)")
    ax.legend(loc="upper right", ncol=2, fontsize=8.4, frameon=True)
    fig.tight_layout()
    save_fig(fig, out_dir, "fig4_transfer_2x2")


def make_fig5_per_template(bundle: dict, out_dir: Path) -> None:
    """Per-template dot plot of (NF-NF - VF-VF) cross-boundary diff within each fixed rule (§5)."""
    fig, ax = plt.subplots(figsize=(11.2, 4.4))
    ax.axvspan(-0.5, 3.5, color="#E8F3FA", alpha=0.35, zorder=0)
    ax.axvspan(3.5, 7.5, color="#FDF3E7", alpha=0.35, zorder=0)
    ax.axvspan(7.5, 11.5, color="#EEF7F1", alpha=0.35, zorder=0)
    ax.axhline(0, color="#444444", lw=0.9, ls="--", zorder=1)

    for label, sc, color, offset in [
        ("Qwen2.5-3B-Instruct (per template)", bundle["qwen_scored"], C_BLUE, -0.17),
        ("Phi-4-mini-instruct (per template)", bundle["phi4_scored"], C_ORANGE, +0.17),
    ]:
        # Group templates by rule
        sample_tids = sorted(sc["NF_train_NF_infer"][0]["end_to_end"]["by_template"].keys())
        by_rule = {r: [] for r in RULES_ORDER}
        for tid in sample_tids:
            parts = tid.split("_")
            rule = "_".join(parts[1:-1])
            va = np.mean([s["end_to_end"]["by_template"][tid]["cross_ok"] / s["end_to_end"]["by_template"][tid]["groups"] for s in sc["NF_train_NF_infer"]])
            vb = np.mean([s["end_to_end"]["by_template"][tid]["cross_ok"] / s["end_to_end"]["by_template"][tid]["groups"] for s in sc["VF_train_VF_infer"]])
            by_rule[rule].append(100.0 * (va - vb))

        first_dot = True
        for r_i, rule in enumerate(RULES_ORDER):
            vals = by_rule[rule]
            k = len(vals)
            jit = np.linspace(-0.07, 0.07, k) if k > 1 else np.array([0.0])
            ax.scatter(
                np.full(k, r_i + offset) + jit, vals,
                color=color, alpha=0.72, s=36, edgecolors="white", lw=0.5, zorder=3,
                label=label if first_dot else None,
            )
            first_dot = False
            ax.plot([r_i + offset - 0.11, r_i + offset + 0.11], [np.mean(vals), np.mean(vals)], color=color, lw=2.0, zorder=4)

    ax.set_xticks(np.arange(len(RULES_ORDER)))
    ax.set_xticklabels([RULE_SHORT[r] for r in RULES_ORDER], fontsize=8.5)
    ax.set_ylim(-35, 125)
    ax.set_ylabel("Template-Level Order Diff: NF-NF − VF-VF (pp)")
    for cx, flab in ((1.5, "Acceptance Sampling"), (5.5, "Control Chart (WECO)"), (9.5, "Process Capability")):
        ax.text(cx, 118, flab, ha="center", va="top", fontsize=8.8, fontweight="bold", color="#444444")
    ax.legend(loc="upper right", bbox_to_anchor=(0.995, 0.88), frameon=True, framealpha=0.95)
    ax.set_title(f"Per-Template Dispersion of Output-Order Effect Within Fixed Rules — {bundle['tag']}")
    fig.tight_layout()
    save_fig(fig, out_dir, "fig5_per_template_dots")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["dev", "final"], default="dev")
    args = parser.parse_args()

    bundle = load_bundle(args.mode)
    out_dir = ROOT / ("figures_final" if args.mode == "final" else "figures_dev_preview")
    make_fig1_protocol(out_dir)
    make_fig2_per_rule_seeds(bundle, out_dir)
    make_fig3_shortcut_and_stages(bundle, out_dir)
    make_fig4_transfer(bundle, out_dir)
    make_fig5_per_template(bundle, out_dir)
    print(f"All 5 figures generated cleanly in {out_dir}")


if __name__ == "__main__":
    main()

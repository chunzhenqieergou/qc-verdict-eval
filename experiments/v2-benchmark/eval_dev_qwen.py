"""Evaluate existing Qwen2.5-3B-Instruct base + 2x2 order SFT adapters on dev_v2.json ONLY.

Strictly never touches test_v2_candidate.json (which remains un-evaluated pending domain review & freeze).
Generates:
  - dev_outputs/*.json
  - dev_scores.json
  - DEV_REPORT.md (including cross-boundary, same-side pos/neg, paraphrase, oracle-rule comparator,
    and §3 CI-width simulation across templates/rule).
"""

import gc
import hashlib
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "dev_outputs"
OLD_DIR = Path(r"d:\Users\Administrator\Desktop\project1\experiments\scenario-sft")

for k in ("TEMP", "TMP", "HF_HOME", "TORCH_HOME", "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR", "CUDA_CACHE_PATH"):
    d = ROOT / ".cache" / k.lower()
    d.mkdir(parents=True, exist_ok=True)
    os.environ[k] = str(d)

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig

from rules import RULES_ORDER
from scorer import (
    load_json,
    paired_template_cluster_bootstrap,
    score_run,
    simulate_ci_width,
)

BATCH_SIZE = 12
MAX_NEW_TOKENS = 96
SEEDS = (42, 43, 44)


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_base_model(base_path: Path):
    return AutoModelForCausalLM.from_pretrained(
        base_path,
        local_files_only=True,
        device_map={"": 0},
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        ),
    )


def generate_all(model, tok, cases: list[dict], system_prompt: str) -> list[dict]:
    eos = model.generation_config.eos_token_id
    eos_set = set(eos if isinstance(eos, list) else [eos])
    cfg = GenerationConfig(
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        eos_token_id=eos,
        pad_token_id=tok.pad_token_id,
        use_cache=True,
    )
    rows = []
    for i in range(0, len(cases), BATCH_SIZE):
        group = cases[i : i + BATCH_SIZE]
        chats = [
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": r["question"]}]
            for r in group
        ]
        enc = tok.apply_chat_template(
            chats,
            add_generation_prompt=True,
            tokenize=True,
            padding=True,
            return_tensors="pt",
            return_dict=True,
        ).to("cuda")
        with torch.inference_mode():
            gen = model.generate(**enc, generation_config=cfg)[:, enc.input_ids.shape[1] :].tolist()
        for r, seq in zip(group, gen):
            rows.append(
                {
                    "id": r["id"],
                    "text": tok.decode(seq, skip_special_tokens=True).strip(),
                    "capped": not any(e in eos_set for e in seq),
                }
            )
    return rows


def save_json_atomic(path: Path, obj) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf8")
    tmp.replace(path)


def pct(n: int, d: int) -> str:
    return f"{100.0 * n / d:.1f}%" if d else "0.0%"


def main() -> None:
    manifest = load_json(DATA_DIR / "manifest.json")
    dev_path = DATA_DIR / "dev_v2.json"
    assert sha256_file(dev_path) == manifest["sha256"]["dev_v2.json"], "dev_v2.json hash mismatch!"
    cases = load_json(dev_path)
    expected_ids = [c["id"] for c in cases]

    base_path = Path(load_json(OLD_DIR / "environment.json")["base"])
    prompts = load_json(OLD_DIR / "order-training" / "prompts.json")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(base_path, local_files_only=True, padding_side="left")
    tok.pad_token = tok.eos_token

    # Define adapter groups -> list of (output_stem, infer_prompt_key)
    jobs = [
        ("base", None, [("base_VF", "original"), ("base_NF", "reordered")]),
    ]
    for s in SEEDS:
        jobs.append(
            (
                f"vf_{s}",
                OLD_DIR / "runs" / f"scenario-{s}" / "adapter",
                [(f"VF_train_VF_infer_s{s}", "original"), (f"VF_train_NF_infer_s{s}", "reordered")],
            )
        )
        jobs.append(
            (
                f"nf_{s}",
                OLD_DIR / "order-training" / f"scenario-{s}" / "adapter",
                [(f"NF_train_VF_infer_s{s}", "original"), (f"NF_train_NF_infer_s{s}", "reordered")],
            )
        )

    t0 = time.time()
    for tag, adapter_dir, runs in jobs:
        pending = []
        for stem, pkey in runs:
            out_file = OUT_DIR / f"{stem}.json"
            if out_file.exists():
                existing = load_json(out_file)
                assert [r["id"] for r in existing] == expected_ids
                print(f"[reuse] {stem}", flush=True)
            else:
                pending.append((stem, pkey, out_file))
        if not pending:
            continue

        model = load_base_model(base_path)
        if adapter_dir is not None:
            model = PeftModel.from_pretrained(model, adapter_dir)
        model.eval()

        for stem, pkey, out_file in pending:
            t_run = time.time()
            preds = generate_all(model, tok, cases, prompts[pkey])
            save_json_atomic(out_file, preds)
            caps = sum(p["capped"] for p in preds)
            print(f"[done] {stem} ({len(preds)} items, capped={caps}, {time.time()-t_run:.1f}s)", flush=True)

        del model
        gc.collect()
        torch.cuda.empty_cache()

    print(f"All dev_v2 generations ready in {time.time()-t0:.1f}s. Scoring...", flush=True)

    # Score all runs
    cond_specs = {
        "base_VF": ["base_VF"],
        "base_NF": ["base_NF"],
        "VF_train_VF_infer": [f"VF_train_VF_infer_s{s}" for s in SEEDS],
        "VF_train_NF_infer": [f"VF_train_NF_infer_s{s}" for s in SEEDS],
        "NF_train_VF_infer": [f"NF_train_VF_infer_s{s}" for s in SEEDS],
        "NF_train_NF_infer": [f"NF_train_NF_infer_s{s}" for s in SEEDS],
    }

    scored = {}
    for cond, stems in cond_specs.items():
        run_scores = [score_run(cases, load_json(OUT_DIR / f"{st}.json")) for st in stems]
        scored[cond] = run_scores

    # Primary comparison on dev: NF_train_NF_infer - VF_train_VF_infer
    boot_primary = paired_template_cluster_bootstrap(
        scored["NF_train_NF_infer"], scored["VF_train_VF_infer"], metric_key="cross_ok", procedure="end_to_end"
    )
    boot_same = paired_template_cluster_bootstrap(
        scored["NF_train_NF_infer"], scored["VF_train_VF_infer"], metric_key="same_ok", procedure="end_to_end"
    )
    boot_para = paired_template_cluster_bootstrap(
        scored["NF_train_NF_infer"], scored["VF_train_VF_infer"], metric_key="para_ok", procedure="end_to_end"
    )
    ci_sim = simulate_ci_width(scored["NF_train_NF_infer"], scored["VF_train_VF_infer"])

    payload = {
        "dev_sha256": manifest["sha256"]["dev_v2.json"],
        "scored": scored,
        "primary_diff_NFNF_minus_VFVF": {
            "cross_boundary": boot_primary,
            "same_side_both_correct": boot_same,
            "paraphrase_both_correct": boot_para,
        },
        "ci_width_simulation": ci_sim,
    }
    save_json_atomic(ROOT / "dev_scores.json", payload)

    # Render markdown report
    lines = [
        "# v2 开发集（`dev_v2.json`）基线评测与 CI 宽度模拟报告",
        "",
        "> **严格盲测纪律**：本报告仅评测 `dev_v2.json`（12 条规则 × 2 模板 × 2 配置 = 48 基础题组 / 192 题）。`test_v2_candidate.json`（144 题组 / 576 题）**未被触碰**，等待领域复核并冻结后再运行一次正式终测。",
        "",
        "## 1. 端到端模型在 `dev_v2` 上的综合表现（2×2 顺序格 + 底座）",
        "",
        "| 条件 | 单题正确 | 跨界双对 | 同侧双对 (正 / 负) | 同侧仅一致 | 改写双对 | 改写仅一致 | 不可解析题数 | 规则已知比较器跨界双对 |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for cond, runs in scored.items():
        ni = sum(r["end_to_end"]["n_items"] for r in runs)
        ng = sum(r["end_to_end"]["n_groups"] for r in runs)
        item_ok = sum(r["end_to_end"]["item_ok"] for r in runs)
        cross_ok = sum(r["end_to_end"]["cross_ok"] for r in runs)
        unp = sum(r["end_to_end"]["item_unparsed"] for r in runs)
        s_ok = sum(r["end_to_end"]["same"]["all"]["both_correct"] for r in runs)
        s_con = sum(r["end_to_end"]["same"]["all"]["label_consistent"] for r in runs)
        sp_ok = sum(r["end_to_end"]["same"]["pos"]["both_correct"] for r in runs)
        sp_n = sum(r["end_to_end"]["same"]["pos"]["n"] for r in runs)
        sn_ok = sum(r["end_to_end"]["same"]["neg"]["both_correct"] for r in runs)
        sn_n = sum(r["end_to_end"]["same"]["neg"]["n"] for r in runs)
        p_ok = sum(r["end_to_end"]["para"]["all"]["both_correct"] for r in runs)
        p_con = sum(r["end_to_end"]["para"]["all"]["label_consistent"] for r in runs)
        ora_cross = sum(r["oracle_rule_comparator"]["cross_ok"] for r in runs)
        lines.append(
            f"| `{cond}` | {item_ok}/{ni} ({pct(item_ok, ni)}) "
            f"| {cross_ok}/{ng} ({pct(cross_ok, ng)}) "
            f"| {s_ok}/{ng} ({pct(s_ok, ng)}) [{sp_ok}/{sp_n} / {sn_ok}/{sn_n}] "
            f"| {s_con}/{ng} ({pct(s_con, ng)}) "
            f"| {p_ok}/{ng} ({pct(p_ok, ng)}) "
            f"| {p_con}/{ng} ({pct(p_con, ng)}) "
            f"| {unp}/{ni} "
            f"| {ora_cross}/{ng} ({pct(ora_cross, ng)}) |"
        )

    lines += [
        "",
        "## 2. 互斥单题错误分解（定位数值瓶颈 vs. 比较判定瓶颈）",
        "",
        "| 条件 | 三项全对 | 双数对判错 | 双数对无结论 | 仅统计量错 | 仅阈值错 | 双数均错 | 数字缺失 | 总题数 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cond, runs in scored.items():
        b_keys = (
            "all_three_ok",
            "nums_ok_verdict_wrong",
            "nums_ok_verdict_unparsed",
            "stat_wrong_limit_ok",
            "stat_ok_limit_wrong",
            "both_nums_wrong",
            "num_missing",
        )
        vals = [sum(r["item_buckets"].get(k, 0) for r in runs) for k in b_keys]
        ni = sum(r["end_to_end"]["n_items"] for r in runs)
        lines.append(f"| `{cond}` | " + " | ".join(str(v) for v in vals) + f" | {ni} |")

    lines += [
        "",
        "## 3. 按规则族拆分（`VF_train_VF_infer` vs `NF_train_NF_infer`，含规则已知比较器净效应）",
        "",
        "| 条件 | 规则族 | 端到端跨界双对 | 比较器跨界双对 | 端到端同侧双对 | 端到端改写双对 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for cond in ("VF_train_VF_infer", "NF_train_NF_infer"):
        runs = scored[cond]
        for fam in ("sampling", "control_chart", "capability"):
            g = sum(r["end_to_end"]["by_family"][fam]["groups"] for r in runs)
            ec = sum(r["end_to_end"]["by_family"][fam]["cross_ok"] for r in runs)
            oc = sum(r["oracle_rule_comparator"]["by_family"][fam]["cross_ok"] for r in runs)
            es = sum(r["end_to_end"]["by_family"][fam]["same_ok"] for r in runs)
            ep = sum(r["end_to_end"]["by_family"][fam]["para_ok"] for r in runs)
            lines.append(
                f"| `{cond}` | {fam} | {ec}/{g} ({pct(ec, g)}) | {oc}/{g} ({pct(oc, g)}) "
                f"| {es}/{g} ({pct(es, g)}) | {ep}/{g} ({pct(ep, g)}) |"
            )

    lines += [
        "",
        "## 4. 主效应估计（`NF-NF` 减 `VF-VF`）与 §3 样本量/CI 宽度模拟",
        "",
        f"- **跨界双对差值**：`{boot_primary['point_diff_pp']:+.2f}` pp（规则内模板簇重采样 95% 区间：`[{boot_primary['ci95_template_cluster_pp'][0]:+.2f}, {boot_primary['ci95_template_cluster_pp'][1]:+.2f}]` pp；去一模板敏感性范围：`[{boot_primary['loto_range_pp'][0]:+.2f}, {boot_primary['loto_range_pp'][1]:+.2f}]` pp）",
        f"- **同侧双对差值**：`{boot_same['point_diff_pp']:+.2f}` pp（95% 区间：`[{boot_same['ci95_template_cluster_pp'][0]:+.2f}, {boot_same['ci95_template_cluster_pp'][1]:+.2f}]` pp）",
        f"- **改写双对差值**：`{boot_para['point_diff_pp']:+.2f}` pp（95% 区间：`[{boot_para['ci95_template_cluster_pp'][0]:+.2f}, {boot_para['ci95_template_cluster_pp'][1]:+.2f}]` pp）",
        "",
        f"### 不同终测模板数量下的预期 95% CI 宽度（基于 `dev_v2` 模板间配对分歧标准差 `{ci_sim['empirical_dev_template_diff_sd_pp']:.2f}` pp）",
        "",
        "| 终测规模（每规则模板数） | 模拟均值 (pp) | 预期 95% CI 总宽 (pp) | 预期半宽 (±pp) |",
        "|---|---:|---:|---:|",
    ]
    for k, v in ci_sim["simulations"].items():
        lines.append(
            f"| `{k}` | {v['sim_mean_pp']:+.2f} | {v['expected_ci95_width_pp']:.2f} | ±{v['expected_half_width_pp']:.2f} |"
        )

    lines += [
        "",
        "### 逐规则跨界双对差值（`NF-NF` 减 `VF-VF`，pp）",
        "",
        "| 规则 | 差值 (pp) | `VF-VF` 跨界对 | `NF-NF` 跨界对 | `NF-NF` 比较器跨界对 |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in RULES_ORDER:
        vf_c = sum(x["end_to_end"]["by_rule"][r]["cross_ok"] for x in scored["VF_train_VF_infer"])
        nf_c = sum(x["end_to_end"]["by_rule"][r]["cross_ok"] for x in scored["NF_train_NF_infer"])
        oc_c = sum(x["oracle_rule_comparator"]["by_rule"][r]["cross_ok"] for x in scored["NF_train_NF_infer"])
        g = sum(x["end_to_end"]["by_rule"][r]["groups"] for x in scored["NF_train_NF_infer"])
        lines.append(
            f"| `{r}` | {boot_primary['by_rule_diff_pp'][r]:+.2f} | {vf_c}/{g} | {nf_c}/{g} | {oc_c}/{g} |"
        )

    (ROOT / "DEV_REPORT.md").write_text("\n".join(lines), encoding="utf8")
    print("Wrote:", ROOT / "DEV_REPORT.md")


if __name__ == "__main__":
    main()

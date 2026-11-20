"""Week 2 Experiment: Deployable Operator-Predicting Comparator vs Fair End-to-End SFT on dev_v2.

Implements Submission Framework §4 Row 3 & Row 4 fairly:
  - Uses the EXACT 96 training questions from historical scenario-sft/data/scenario.json
    (zero leakage with dev_v2 or test_v2_candidate).
  - Adds structured operator target `，算子=<op>` to the numeric line (matching §4:
    deployable comparator predicts stat, threshold, and operator; end-to-end receives the
    exact same prompt & supervision so prompt impact is isolated fairly).
  - Trains QLoRA adapters (identical hyperparams to scenario-sft/run.py: r=16, alpha=32,
    dropout=0.05, lr=7e-5, 2 epochs = 48 steps, seeds 42/43/44) for both VF_op and NF_op.
  - Evaluates ONLY on dev_v2.json (192 items / 48 base groups). Strictly NEVER touches
    test_v2_candidate.json.
  - Compares:
      1) base_VF_op & base_NF_op (end-to-end vs deployable comparator vs oracle comparator)
      2) SFT_VF_op & SFT_NF_op (end-to-end vs deployable comparator vs oracle comparator)
      3) Prompt/supervision delta vs original VF-VF and NF-NF on dev_v2
      4) Boundary-relation audit (below / at / above) showing whether deployable comparator
         eliminates the `==` equality shortcut and strict-`>` failure.
"""

from collections import Counter
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
V2_DIR = ROOT.parent / "v2-benchmark"
OLD_DIR = Path(r"d:\Users\Administrator\Desktop\project1\experiments\scenario-sft")
sys.path.insert(0, str(V2_DIR))

for k in ("TEMP", "TMP", "HF_HOME", "TORCH_HOME", "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR", "CUDA_CACHE_PATH"):
    d = ROOT / ".cache" / k.lower()
    d.mkdir(parents=True, exist_ok=True)
    os.environ[k] = str(d)
os.environ["HF_HUB_OFFLINE"] = "1"

import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
    get_cosine_schedule_with_warmup,
    set_seed,
)

from rules import RULE_SPECS, RULES_ORDER, compare_float
from scorer import (
    load_json,
    paired_template_cluster_bootstrap,
    parse_number,
    parse_operator,
    parse_verdict,
    score_run,
)

SEEDS = (42, 43, 44)
BATCH_SIZE_EVAL = 12
MAX_NEW_TOKENS = 96

PROMPTS = {
    "VF_op": "你是民用工业质量审核员。判断题中所述结论，并用简短文字说明依据。第一行只写成立或不成立，第二行写统计量=数值，阈值=数值，算子=<=或>=或>，第三行写理由。",
    "NF_op": "你是民用工业质量审核员。判断题中所述结论，并用简短文字说明依据。第一行写统计量=数值，阈值=数值，算子=<=或>=或>，第二行只写成立或不成立，第三行写理由。",
}


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def save_json_atomic(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf8")
    tmp.replace(path)


def build_op_train_data(order: str) -> list[dict]:
    """Derive VF_op or NF_op 96-example training set from historical scenario.json."""
    orig = load_json(OLD_DIR / "data" / "scenario.json")
    rows = []
    for r in orig:
        op = RULE_SPECS[r["rule"]].op
        # Row 5 consistency check on training data
        assert compare_float(op, float(r["value"]), float(r["limit"])) == bool(r["positive"])
        lines = r["answer"].splitlines()
        assert len(lines) == 3, f"Unexpected answer lines: {r['answer']!r}"
        verdict_line, num_line, reason_line = lines[0], lines[1], lines[2]
        assert verdict_line in ("成立", "不成立")
        assert "统计量=" in num_line and "阈值=" in num_line
        num_op_line = f"{num_line}，算子={op}"
        ans = f"{verdict_line}\n{num_op_line}\n{reason_line}" if order == "VF_op" else f"{num_op_line}\n{verdict_line}\n{reason_line}"
        # Self-verify parser reads back exact verdict, stat, limit, op
        assert parse_verdict(ans) == bool(r["positive"])
        assert abs(parse_number(ans, "统计量") - float(r["value"])) < 1e-6
        assert abs(parse_number(ans, "阈值") - float(r["limit"])) < 1e-6
        assert parse_operator(ans) == op
        rows.append({**r, "op": op, "answer": ans})
    return rows


def load_base(base_path: Path):
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


def train_adapter(base_path: Path, tok, train_rows: list[dict], system_prompt: str, seed: int, out_dir: Path) -> None:
    if (out_dir / "training.json").is_file() and (out_dir / "adapter" / "adapter_config.json").is_file():
        print(f"[reuse train] {out_dir.name}", flush=True)
        return

    set_seed(seed)
    m = prepare_model_for_kbit_training(
        load_base(base_path),
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    m = get_peft_model(
        m,
        LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules="all-linear", task_type="CAUSAL_LM"),
    )
    m.config.use_cache = False
    m.train()

    encoded = []
    for r in train_rows:
        chat = [{"role": "system", "content": system_prompt}, {"role": "user", "content": r["question"]}]
        prefix = tok.apply_chat_template(chat, tokenize=True, add_generation_prompt=True, return_dict=False)
        full = tok.apply_chat_template(
            chat + [{"role": "assistant", "content": r["answer"]}], tokenize=True, return_dict=False
        )
        assert full[: len(prefix)] == prefix
        encoded.append((full, [-100] * len(prefix) + full[len(prefix) :]))

    steps = 2 * len(encoded) // 4
    opt = torch.optim.AdamW((p for p in m.parameters() if p.requires_grad), lr=7e-5, weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(opt, 2, steps)
    gen = torch.Generator().manual_seed(seed)

    log = []
    start = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    for _epoch in range(2):
        perm = torch.randperm(len(encoded), generator=gen).tolist()
        for i in range(0, len(perm), 4):
            batch = [encoded[k] for k in perm[i : i + 4]]
            width = max(len(a) for a, _ in batch)
            ids = torch.tensor([a + [tok.pad_token_id] * (width - len(a)) for a, _ in batch], device="cuda")
            mask = torch.tensor([[1] * len(a) + [0] * (width - len(a)) for a, _ in batch], device="cuda")
            labels = torch.tensor([b + [-100] * (width - len(b)) for _, b in batch], device="cuda")
            loss = m(input_ids=ids, attention_mask=mask, labels=labels).loss
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            log.append(loss.item())

    out_dir.mkdir(parents=True, exist_ok=True)
    m.save_pretrained(out_dir / "adapter")
    save_json_atomic(
        out_dir / "training.json",
        {
            "seed": seed,
            "steps": steps,
            "loss": log,
            "elapsed": time.monotonic() - start,
            "peak_gib": torch.cuda.max_memory_allocated() / 2**30,
            "input_tokens": sum(len(x) for x, _ in encoded),
            "supervised_tokens": sum(sum(v != -100 for v in y) for _, y in encoded),
        },
    )
    print(f"[trained] {out_dir.name} ({time.monotonic()-start:.1f}s, final_loss={log[-1]:.4f})", flush=True)
    del m, opt, sched
    gc.collect()
    torch.cuda.empty_cache()


def generate_dev(model, tok, cases: list[dict], system_prompt: str) -> list[dict]:
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
    for i in range(0, len(cases), BATCH_SIZE_EVAL):
        group = cases[i : i + BATCH_SIZE_EVAL]
        chats = [
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": r["question"]}]
            for r in group
        ]
        enc = tok.apply_chat_template(
            chats, add_generation_prompt=True, tokenize=True, padding=True, return_tensors="pt", return_dict=True
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


def pct(n: int, d: int) -> str:
    return f"{100.0 * n / d:.1f}%" if d else "0.0%"


def boundary_audit_compare(cases: list[dict], pred_lists: list[list[dict]], use_deployable: bool) -> dict:
    by_id = {c["id"]: c for c in cases}
    out = {}
    for rule in ("single", "first_accept", "first_reject", "weco3"):
        tot, ok, pred_t = Counter(), Counter(), Counter()
        for preds in pred_lists:
            for p in preds:
                c = by_id[p["id"]]
                if c["rule"] != rule:
                    continue
                rel = c["meta"]["boundary_relation"]
                if use_deployable:
                    v = compare_float(parse_operator(p["text"]), parse_number(p["text"], "统计量"), parse_number(p["text"], "阈值"))
                else:
                    v = parse_verdict(p["text"])
                tot[rel] += 1
                ok[rel] += int(v == c["positive"])
                pred_t[rel] += int(v is True)
        out[rule] = {rel: (ok[rel], tot[rel], pred_t[rel]) for rel in ("below", "at", "above") if tot[rel]}
    return out


def main() -> None:
    base_path = Path(load_json(OLD_DIR / "environment.json")["base"])
    v2_manifest = load_json(V2_DIR / "data" / "manifest.json")
    dev_path = V2_DIR / "data" / "dev_v2.json"
    assert sha256_file(dev_path) == v2_manifest["sha256"]["dev_v2.json"]
    dev_cases = load_json(dev_path)
    expected_ids = [c["id"] for c in dev_cases]

    # Build and freeze operator-augmented training datasets
    train_data = {order: build_op_train_data(order) for order in ("VF_op", "NF_op")}
    save_json_atomic(ROOT / "data" / "train_VF_op.json", train_data["VF_op"])
    save_json_atomic(ROOT / "data" / "train_NF_op.json", train_data["NF_op"])
    save_json_atomic(ROOT / "data" / "prompts.json", PROMPTS)

    tok_train = AutoTokenizer.from_pretrained(base_path, local_files_only=True)
    tok_train.pad_token = tok_train.eos_token

    # 1. Train 6 adapters (VF_op x 3 seeds, NF_op x 3 seeds)
    for order in ("VF_op", "NF_op"):
        for s in SEEDS:
            train_adapter(base_path, tok_train, train_data[order], PROMPTS[order], s, ROOT / "runs" / f"{order}_s{s}")

    # 2. Evaluate on dev_v2.json ONLY
    tok_eval = AutoTokenizer.from_pretrained(base_path, local_files_only=True, padding_side="left")
    tok_eval.pad_token = tok_eval.eos_token
    out_dir = ROOT / "dev_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_jobs = [
        ("base", None, [("base_VF_op", "VF_op"), ("base_NF_op", "NF_op")]),
    ]
    for order in ("VF_op", "NF_op"):
        for s in SEEDS:
            eval_jobs.append(
                (
                    f"{order}_s{s}",
                    ROOT / "runs" / f"{order}_s{s}" / "adapter",
                    [(f"SFT_{order}_s{s}", order)],
                )
            )

    for tag, adapter_dir, runs in eval_jobs:
        pending = []
        for stem, pkey in runs:
            f = out_dir / f"{stem}.json"
            if f.exists():
                assert [r["id"] for r in load_json(f)] == expected_ids
                print(f"[reuse eval] {stem}", flush=True)
            else:
                pending.append((stem, pkey, f))
        if not pending:
            continue
        model = load_base(base_path)
        if adapter_dir is not None:
            model = PeftModel.from_pretrained(model, adapter_dir)
        model.eval()
        for stem, pkey, f in pending:
            t0 = time.time()
            preds = generate_dev(model, tok_eval, dev_cases, PROMPTS[pkey])
            save_json_atomic(f, preds)
            print(f"[eval] {stem} (capped={sum(p['capped'] for p in preds)}, {time.time()-t0:.1f}s)", flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()

    # 3. Score all new conditions + compare against prior dev_v2 baseline (without op prompt)
    prior_dev = load_json(V2_DIR / "dev_scores.json")["scored"]
    cond_stems = {
        "base_VF_op": ["base_VF_op"],
        "base_NF_op": ["base_NF_op"],
        "SFT_VF_op": [f"SFT_VF_op_s{s}" for s in SEEDS],
        "SFT_NF_op": [f"SFT_NF_op_s{s}" for s in SEEDS],
    }
    scored = {}
    raw_preds = {}
    for cond, stems in cond_stems.items():
        plist = [load_json(out_dir / f"{st}.json") for st in stems]
        raw_preds[cond] = plist
        scored[cond] = [score_run(dev_cases, ps) for ps in plist]

    # Paired bootstrap comparisons
    boot_dep_vs_e2e_nf = paired_template_cluster_bootstrap(
        [{"end_to_end": r["deployable_comparator"]} for r in scored["SFT_NF_op"]],
        scored["SFT_NF_op"],
        metric_key="cross_ok",
        procedure="end_to_end",
    )
    boot_dep_vs_e2e_vf = paired_template_cluster_bootstrap(
        [{"end_to_end": r["deployable_comparator"]} for r in scored["SFT_VF_op"]],
        scored["SFT_VF_op"],
        metric_key="cross_ok",
        procedure="end_to_end",
    )
    boot_prompt_nf = paired_template_cluster_bootstrap(
        scored["SFT_NF_op"], prior_dev["NF_train_NF_infer"], metric_key="cross_ok", procedure="end_to_end"
    )
    boot_prompt_vf = paired_template_cluster_bootstrap(
        scored["SFT_VF_op"], prior_dev["VF_train_VF_infer"], metric_key="cross_ok", procedure="end_to_end"
    )

    # Boundary audits
    b_nf_e2e = boundary_audit_compare(dev_cases, raw_preds["SFT_NF_op"], use_deployable=False)
    b_nf_dep = boundary_audit_compare(dev_cases, raw_preds["SFT_NF_op"], use_deployable=True)
    b_vf_dep = boundary_audit_compare(dev_cases, raw_preds["SFT_VF_op"], use_deployable=True)

    save_json_atomic(
        ROOT / "dev_op_scores.json",
        {
            "dev_sha256": v2_manifest["sha256"]["dev_v2.json"],
            "scored": scored,
            "bootstrap": {
                "SFT_NF_op_deployable_minus_e2e": boot_dep_vs_e2e_nf,
                "SFT_VF_op_deployable_minus_e2e": boot_dep_vs_e2e_vf,
                "prompt_effect_NF_op_minus_NF_orig_e2e": boot_prompt_nf,
                "prompt_effect_VF_op_minus_VF_orig_e2e": boot_prompt_vf,
            },
            "boundary_audits": {
                "SFT_NF_op_e2e": b_nf_e2e,
                "SFT_NF_op_deployable": b_nf_dep,
                "SFT_VF_op_deployable": b_vf_dep,
            },
        },
    )

    # Render markdown report
    lines = [
        "# 第 2 周开发集报告：可部署算子预测比较器 vs. 公平端到端对照（`dev_v2.json`）",
        "",
        "> **口径说明（方案 §4 第 3–4 行）**：所有新增模型仅使用历史 `scenario.json` 的同一 96 道训练题（在数值行追加 `，算子=<= / >= / >`），超参数与种子（42/43/44）完全不变。端到端与可部署比较器读取**完全相同的模型回答**，可部署比较器不使用任何参考规则标识（缺字段或算子一律判错）。`test_v2_candidate.json` 仍严格未触碰。",
        "",
        "## 1. 三口径主表（端到端 vs. 可部署算子比较器 vs. 规则已知比较器）",
        "",
        "| 条件 | 算子解析 / 算子正确 | 端到端跨界双对 | 可部署比较器跨界双对 | 跨界纠正 / 改错 | 规则已知比较器跨界双对 | 端到端同侧双对 | 可部署比较器同侧双对 | 端到端改写双对 | 可部署比较器改写双对 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cond, runs in scored.items():
        ni = sum(r["end_to_end"]["n_items"] for r in runs)
        ng = sum(r["end_to_end"]["n_groups"] for r in runs)
        op_p = sum(r["deployable_comparator"]["op_present"] for r in runs)
        op_ok = sum(r["deployable_comparator"]["op_correct"] for r in runs)
        ec = sum(r["end_to_end"]["cross_ok"] for r in runs)
        dc = sum(r["deployable_comparator"]["cross_ok"] for r in runs)
        oc = sum(r["oracle_rule_comparator"]["cross_ok"] for r in runs)
        fix_c = sum(r["deployable_comparator"]["vs_e2e"]["fixed_cross_pairs"] for r in runs)
        brk_c = sum(r["deployable_comparator"]["vs_e2e"]["broken_cross_pairs"] for r in runs)
        es = sum(r["end_to_end"]["same"]["all"]["both_correct"] for r in runs)
        ds = sum(r["deployable_comparator"]["same"]["all"]["both_correct"] for r in runs)
        ep = sum(r["end_to_end"]["para"]["all"]["both_correct"] for r in runs)
        dp = sum(r["deployable_comparator"]["para"]["all"]["both_correct"] for r in runs)
        lines.append(
            f"| `{cond}` | {op_p}/{ni} ({pct(op_ok, ni)}) "
            f"| {ec}/{ng} ({pct(ec, ng)}) "
            f"| {dc}/{ng} ({pct(dc, ng)}) "
            f"| +{fix_c} / -{brk_c} "
            f"| {oc}/{ng} ({pct(oc, ng)}) "
            f"| {es}/{ng} ({pct(es, ng)}) "
            f"| {ds}/{ng} ({pct(ds, ng)}) "
            f"| {ep}/{ng} ({pct(ep, ng)}) "
            f"| {dp}/{ng} ({pct(dp, ng)}) |"
        )

    lines += [
        "",
        "## 2. 配对模板簇 Bootstrap 效应估计（规则等权平均，pp）",
        "",
        f"- **新增算子提示对端到端的影响（`SFT_NF_op` 端到端 − 原 `NF-NF` 端到端）**：`{boot_prompt_nf['point_diff_pp']:+.2f}` pp（95% CI `[{boot_prompt_nf['ci95_template_cluster_pp'][0]:+.2f}, {boot_prompt_nf['ci95_template_cluster_pp'][1]:+.2f}]`）",
        f"- **新增算子提示对端到端的影响（`SFT_VF_op` 端到端 − 原 `VF-VF` 端到端）**：`{boot_prompt_vf['point_diff_pp']:+.2f}` pp（95% CI `[{boot_prompt_vf['ci95_template_cluster_pp'][0]:+.2f}, {boot_prompt_vf['ci95_template_cluster_pp'][1]:+.2f}]`）",
        f"- **可部署比较器净增益（`SFT_NF_op` 比较器 − 同输出端到端）**：`{boot_dep_vs_e2e_nf['point_diff_pp']:+.2f}` pp（95% CI `[{boot_dep_vs_e2e_nf['ci95_template_cluster_pp'][0]:+.2f}, {boot_dep_vs_e2e_nf['ci95_template_cluster_pp'][1]:+.2f}]`）",
        f"- **可部署比较器净增益（`SFT_VF_op` 比较器 − 同输出端到端）**：`{boot_dep_vs_e2e_vf['point_diff_pp']:+.2f}` pp（95% CI `[{boot_dep_vs_e2e_vf['ci95_template_cluster_pp'][0]:+.2f}, {boot_dep_vs_e2e_vf['ci95_template_cluster_pp'][1]:+.2f}]`）",
        "",
        "## 3. 按规则族拆分（`SFT_VF_op` 与 `SFT_NF_op`）",
        "",
        "| 条件 | 规则族 | 端到端跨界双对 | 可部署比较器跨界双对 | 规则已知比较器跨界双对 | 可部署比较器同侧双对 | 可部署比较器改写双对 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for cond in ("SFT_VF_op", "SFT_NF_op"):
        runs = scored[cond]
        for fam in ("sampling", "control_chart", "capability"):
            g = sum(r["end_to_end"]["by_family"][fam]["groups"] for r in runs)
            ec = sum(r["end_to_end"]["by_family"][fam]["cross_ok"] for r in runs)
            dc = sum(r["deployable_comparator"]["by_family"][fam]["cross_ok"] for r in runs)
            oc = sum(r["oracle_rule_comparator"]["by_family"][fam]["cross_ok"] for r in runs)
            ds = sum(r["deployable_comparator"]["by_family"][fam]["same_ok"] for r in runs)
            dp = sum(r["deployable_comparator"]["by_family"][fam]["para_ok"] for r in runs)
            lines.append(
                f"| `{cond}` | {fam} | {ec}/{g} ({pct(ec, g)}) | {dc}/{g} ({pct(dc, g)}) "
                f"| {oc}/{g} ({pct(oc, g)}) | {ds}/{g} ({pct(ds, g)}) | {dp}/{g} ({pct(dp, g)}) |"
            )

    lines += [
        "",
        "## 4. 关键机制检验：可部署比较器能否破除 `==` 等号捷径与 `weco3` 严格越限失误？",
        "",
        "| 规则（真算子） | 边界位置 | `SFT_NF_op` 端到端正确率 | `SFT_NF_op` 可部署比较器正确率 | `SFT_VF_op` 可部署比较器正确率 |",
        "|---|---|---:|---:|---:|",
    ]
    for rule in ("single", "first_accept", "first_reject", "weco3"):
        op = RULE_SPECS[rule].op
        for rel in ("below", "at", "above"):
            if rel not in b_nf_e2e[rule]:
                continue
            o1, n1, _ = b_nf_e2e[rule][rel]
            o2, n2, _ = b_nf_dep[rule][rel]
            o3, n3, _ = b_vf_dep[rule][rel]
            lines.append(
                f"| `{rule}` (`{op}`) | `{rel}` | {o1}/{n1} ({pct(o1, n1)}) | {o2}/{n2} ({pct(o2, n2)}) | {o3}/{n3} ({pct(o3, n3)}) |"
            )

    lines += [
        "",
        "## 5. 单题互斥错误分解（`SFT_VF_op` 与 `SFT_NF_op`）",
        "",
        "| 条件 | 三项全对 | 双数对判错 | 仅统计量错 | 仅阈值错 | 双数均错 | 数字缺失 | 总题数 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cond in ("base_VF_op", "base_NF_op", "SFT_VF_op", "SFT_NF_op"):
        runs = scored[cond]
        keys = ("all_three_ok", "nums_ok_verdict_wrong", "stat_wrong_limit_ok", "stat_ok_limit_wrong", "both_nums_wrong", "num_missing")
        vals = [sum(r["item_buckets"].get(k, 0) for r in runs) for k in keys]
        ni = sum(r["end_to_end"]["n_items"] for r in runs)
        lines.append(f"| `{cond}` | " + " | ".join(str(v) for v in vals) + f" | {ni} |")

    (ROOT / "DEV_OP_REPORT.md").write_text("\n".join(lines), encoding="utf8")
    print("Wrote:", ROOT / "DEV_OP_REPORT.md", flush=True)


if __name__ == "__main__":
    main()

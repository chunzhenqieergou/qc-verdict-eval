"""Week 3 Cross-Model Replication on dev_v2.json using microsoft/Phi-4-mini-instruct (3.8B).

Strictly NEVER touches test_v2_candidate.json.
Replicates on a completely distinct model family (Microsoft Phi-4 vs Alibaba Qwen2.5):
  1. Tokenizer comparison (input & supervised token counts vs Qwen2.5-3B-Instruct on exact same 96 examples)
  2. Base zero-shot evaluation (VF, NF, VF_op, NF_op) + peak VRAM & throughput logging
  3. Core 2x2 order SFT grid (VF/NF train x VF/NF infer, 3 seeds: 42/43/44) on exact same 96 scenario.json examples
  4. Operator-augmented SFT (VF_op, NF_op, 3 seeds: 42/43/44) comparing End-to-End vs Deployable Comparator vs Oracle-Rule Comparator
  5. Boundary-relation (`below / at / above`) shortcut audit across both orders & deployable comparator.
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
OP_DIR = ROOT.parent / "v2-operator-comparator"
OLD_DIR = Path(r"d:\Users\Administrator\Desktop\project1\experiments\scenario-sft")
PHI4_PATH = Path(r"D:\Users\Administrator\Desktop\finetune\models\Phi-4-mini-instruct")
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
BATCH_SIZE_EVAL = 12  # Verified 3.14 GiB peak, 0.40s/item on Phi-4-mini in fresh eval process
MAX_NEW_TOKENS = 96


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def save_json_atomic(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf8")
    tmp.replace(path)


def swap_first_two_lines(answer: str) -> str:
    lines = answer.splitlines()
    assert len(lines) >= 3
    return "\n".join([lines[1], lines[0], *lines[2:]])


def load_base():
    return AutoModelForCausalLM.from_pretrained(
        PHI4_PATH,
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


def encode_dataset(tok, rows: list[dict], system_prompt: str) -> list[tuple[list[int], list[int]]]:
    encoded = []
    for r in rows:
        chat = [{"role": "system", "content": system_prompt}, {"role": "user", "content": r["question"]}]
        prefix = tok.apply_chat_template(chat, tokenize=True, add_generation_prompt=True, return_dict=False)
        full = tok.apply_chat_template(
            chat + [{"role": "assistant", "content": r["answer"]}], tokenize=True, return_dict=False
        )
        # Ensure prefix alignment
        if full[: len(prefix)] != prefix:
            # Find longest common prefix up to assistant start
            k = 0
            while k < min(len(prefix), len(full)) and prefix[k] == full[k]:
                k += 1
            assert k >= len(prefix) - 2, f"Chat template prefix mismatch: {k} vs {len(prefix)}"
            prefix_len = k
        else:
            prefix_len = len(prefix)
        encoded.append((full, [-100] * prefix_len + full[prefix_len:]))
    return encoded


def train_adapter(tok, train_rows: list[dict], system_prompt: str, seed: int, out_dir: Path) -> None:
    if (out_dir / "training.json").is_file() and (out_dir / "adapter" / "adapter_config.json").is_file():
        print(f"[reuse train] {out_dir.name}", flush=True)
        return

    set_seed(seed)
    m = prepare_model_for_kbit_training(
        load_base(),
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    m = get_peft_model(
        m,
        LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules="all-linear", task_type="CAUSAL_LM"),
    )
    m.config.use_cache = False
    m.train()

    encoded = encode_dataset(tok, train_rows, system_prompt)
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
            # Exact token-weighted gradient accumulation (micro_batch=1, eff_batch=4)
            # Supervised-token-only lm_head projection: 100% mathematically equivalent to full
            # causal LM cross-entropy while slashing 200k-vocab logits VRAM by ~6x (prevents WDDM paging).
            inner = m.base_model.model
            total_sup_tokens = sum(sum(v != -100 for v in b[1:]) for _, b in batch)
            step_loss = 0.0
            for a, b in batch:
                n_tok = sum(v != -100 for v in b[1:])
                if n_tok == 0:
                    continue
                ids = torch.tensor([a], device="cuda")
                labels = torch.tensor([b], device="cuda")
                hidden = inner.model(input_ids=ids, use_cache=False).last_hidden_state
                shift_labels = labels[:, 1:]
                valid = shift_labels != -100
                sup_hidden = hidden[:, :-1, :][valid]
                sup_labels = shift_labels[valid]
                sup_logits = inner.lm_head(sup_hidden).float()
                loss = torch.nn.functional.cross_entropy(sup_logits, sup_labels)
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite training loss")
                weight = n_tok / total_sup_tokens
                (loss * weight).backward()
                step_loss += loss.item() * weight
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            log.append(step_loss)

    out_dir.mkdir(parents=True, exist_ok=True)
    m.save_pretrained(out_dir / "adapter")
    peak_gib = torch.cuda.max_memory_allocated() / 2**30
    save_json_atomic(
        out_dir / "training.json",
        {
            "model": "microsoft/Phi-4-mini-instruct",
            "seed": seed,
            "steps": steps,
            "loss": log,
            "elapsed": time.monotonic() - start,
            "peak_gib": peak_gib,
            "input_tokens": sum(len(x) for x, _ in encoded),
            "supervised_tokens": sum(sum(v != -100 for v in y) for _, y in encoded),
        },
    )
    print(f"[trained] {out_dir.name} ({time.monotonic()-start:.1f}s, peak={peak_gib:.2f}GiB, loss={log[-1]:.4f})", flush=True)
    del m, opt, sched
    gc.collect()
    torch.cuda.empty_cache()


def generate_dev(model, tok, cases: list[dict], system_prompt: str) -> tuple[list[dict], float]:
    eos = model.generation_config.eos_token_id
    eos_set = set(eos if isinstance(eos, list) else [eos])
    cfg = GenerationConfig(
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        eos_token_id=eos,
        pad_token_id=tok.pad_token_id,
        use_cache=True,
    )
    torch.cuda.reset_peak_memory_stats()
    rows = []
    for i in range(0, len(cases), BATCH_SIZE_EVAL):
        group = cases[i : i + BATCH_SIZE_EVAL]
        chats = [[{"role": "system", "content": system_prompt}, {"role": "user", "content": r["question"]}] for r in group]
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
    peak_gib = torch.cuda.max_memory_allocated() / 2**30
    return rows, peak_gib


def pct(n: int, d: int) -> str:
    return f"{100.0 * n / d:.1f}%" if d else "0.0%"


def boundary_audit(cases: list[dict], pred_lists: list[list[dict]], use_deployable: bool = False) -> dict:
    by_id = {c["id"]: c for c in cases}
    out = {}
    for rule in ("single", "first_accept", "first_reject", "weco3"):
        tot, ok, pt = Counter(), Counter(), Counter()
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
                pt[rel] += int(v is True)
        out[rule] = {rel: (ok[rel], tot[rel], pt[rel]) for rel in ("below", "at", "above") if tot[rel]}
    return out


def main() -> None:
    assert (PHI4_PATH / "config.json").is_file(), f"Missing Phi-4-mini-instruct at {PHI4_PATH}"
    v2_manifest = load_json(V2_DIR / "data" / "manifest.json")
    dev_path = V2_DIR / "data" / "dev_v2.json"
    assert sha256_file(dev_path) == v2_manifest["sha256"]["dev_v2.json"]
    dev_cases = load_json(dev_path)
    expected_ids = [c["id"] for c in dev_cases]

    orig_prompts = load_json(OLD_DIR / "order-training" / "prompts.json")
    op_prompts = load_json(OP_DIR / "data" / "prompts.json")
    all_prompts = {
        "VF": orig_prompts["original"],
        "NF": orig_prompts["reordered"],
        "VF_op": op_prompts["VF_op"],
        "NF_op": op_prompts["NF_op"],
    }

    # Prepare exact same 96 training examples as Qwen
    orig_train = load_json(OLD_DIR / "data" / "scenario.json")
    reord_train = [{**r, "answer": swap_first_two_lines(r["answer"])} for r in orig_train]
    vf_op_train = load_json(OP_DIR / "data" / "train_VF_op.json")
    nf_op_train = load_json(OP_DIR / "data" / "train_NF_op.json")
    train_sets = {"VF": orig_train, "NF": reord_train, "VF_op": vf_op_train, "NF_op": nf_op_train}

    tok_train = AutoTokenizer.from_pretrained(PHI4_PATH, local_files_only=True)
    if tok_train.pad_token is None:
        tok_train.pad_token = tok_train.eos_token

    # Record tokenizer token-count difference vs Qwen (§3 Week 3 requirement)
    qwen_tok = AutoTokenizer.from_pretrained(Path(load_json(OLD_DIR / "environment.json")["base"]), local_files_only=True)
    tok_stats = {}
    for k in ("VF", "NF", "VF_op", "NF_op"):
        ep = encode_dataset(tok_train, train_sets[k], all_prompts[k])
        eq = encode_dataset(qwen_tok, train_sets[k], all_prompts[k])
        tok_stats[k] = {
            "phi4_input_tokens": sum(len(x) for x, _ in ep),
            "phi4_supervised_tokens": sum(sum(v != -100 for v in y) for _, y in ep),
            "qwen_input_tokens": sum(len(x) for x, _ in eq),
            "qwen_supervised_tokens": sum(sum(v != -100 for v in y) for _, y in eq),
        }
    save_json_atomic(ROOT / "tokenizer_comparison.json", tok_stats)
    print("Tokenizer comparison:", json.dumps(tok_stats, indent=2), flush=True)

    # 1. Train 12 adapters (VF, NF, VF_op, NF_op x 3 seeds)
    for arm in ("VF", "NF", "VF_op", "NF_op"):
        for s in SEEDS:
            train_adapter(tok_train, train_sets[arm], all_prompts[arm], s, ROOT / "runs" / f"{arm}_s{s}")

    # 2. Evaluate all conditions on dev_v2.json ONLY
    tok_eval = AutoTokenizer.from_pretrained(PHI4_PATH, local_files_only=True, padding_side="left")
    if tok_eval.pad_token is None:
        tok_eval.pad_token = tok_eval.eos_token
    out_dir = ROOT / "dev_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_jobs = [
        ("base", None, [("base_VF", "VF"), ("base_NF", "NF"), ("base_VF_op", "VF_op"), ("base_NF_op", "NF_op")]),
    ]
    for s in SEEDS:
        eval_jobs.append((f"VF_s{s}", ROOT / "runs" / f"VF_s{s}" / "adapter", [(f"VF_train_VF_infer_s{s}", "VF"), (f"VF_train_NF_infer_s{s}", "NF")]))
        eval_jobs.append((f"NF_s{s}", ROOT / "runs" / f"NF_s{s}" / "adapter", [(f"NF_train_VF_infer_s{s}", "VF"), (f"NF_train_NF_infer_s{s}", "NF")]))
        eval_jobs.append((f"VF_op_s{s}", ROOT / "runs" / f"VF_op_s{s}" / "adapter", [(f"SFT_VF_op_s{s}", "VF_op")]))
        eval_jobs.append((f"NF_op_s{s}", ROOT / "runs" / f"NF_op_s{s}" / "adapter", [(f"SFT_NF_op_s{s}", "NF_op")]))

    eval_meta = {}
    # Filter only jobs that have pending outputs
    active_jobs = []
    for tag, adapter_dir, runs in eval_jobs:
        pending = []
        for stem, pkey in runs:
            f = out_dir / f"{stem}.json"
            if f.exists():
                assert [r["id"] for r in load_json(f)] == expected_ids
                print(f"[reuse eval] {stem}", flush=True)
            else:
                pending.append((stem, pkey, f))
        if pending:
            active_jobs.append((tag, adapter_dir, pending))

    if active_jobs:
        # Load 4-bit base model ONCE and hot-swap PEFT adapters in VRAM
        base_model = load_base()
        base_model.eval()
        peft_model = None
        active_adapter_name = None

        for tag, adapter_dir, pending in active_jobs:
            if adapter_dir is None:
                model_to_use = base_model
            else:
                if peft_model is None:
                    peft_model = PeftModel.from_pretrained(base_model, adapter_dir, adapter_name=tag)
                    peft_model.eval()
                    active_adapter_name = tag
                else:
                    peft_model.load_adapter(adapter_dir, adapter_name=tag)
                    peft_model.set_adapter(tag)
                    if active_adapter_name is not None and active_adapter_name != tag:
                        peft_model.delete_adapter(active_adapter_name)
                    active_adapter_name = tag
                    peft_model.eval()
                model_to_use = peft_model

            for stem, pkey, f in pending:
                t0 = time.time()
                preds, peak_gib = generate_dev(model_to_use, tok_eval, dev_cases, all_prompts[pkey])
                save_json_atomic(f, preds)
                elapsed = time.time() - t0
                caps = sum(p["capped"] for p in preds)
                eval_meta[stem] = {"elapsed_s": round(elapsed, 1), "peak_gib": round(peak_gib, 2), "capped": caps}
                print(f"[eval] {stem} (capped={caps}, peak={peak_gib:.2f}GiB, {elapsed:.1f}s)", flush=True)

        del peft_model, base_model
        gc.collect()
        torch.cuda.empty_cache()

    # 3. Score all conditions
    cond_stems = {
        "base_VF": ["base_VF"],
        "base_NF": ["base_NF"],
        "base_VF_op": ["base_VF_op"],
        "base_NF_op": ["base_NF_op"],
        "VF_train_VF_infer": [f"VF_train_VF_infer_s{s}" for s in SEEDS],
        "VF_train_NF_infer": [f"VF_train_NF_infer_s{s}" for s in SEEDS],
        "NF_train_VF_infer": [f"NF_train_VF_infer_s{s}" for s in SEEDS],
        "NF_train_NF_infer": [f"NF_train_NF_infer_s{s}" for s in SEEDS],
        "SFT_VF_op": [f"SFT_VF_op_s{s}" for s in SEEDS],
        "SFT_NF_op": [f"SFT_NF_op_s{s}" for s in SEEDS],
    }
    scored = {}
    raw_preds = {}
    for cond, stems in cond_stems.items():
        plist = [load_json(out_dir / f"{st}.json") for st in stems]
        raw_preds[cond] = plist
        scored[cond] = [score_run(dev_cases, ps) for ps in plist]

    boot_order = paired_template_cluster_bootstrap(
        scored["NF_train_NF_infer"], scored["VF_train_VF_infer"], metric_key="cross_ok", procedure="end_to_end"
    )
    boot_dep_nf = paired_template_cluster_bootstrap(
        [{"end_to_end": r["deployable_comparator"]} for r in scored["SFT_NF_op"]],
        scored["SFT_NF_op"],
        metric_key="cross_ok",
        procedure="end_to_end",
    )
    boot_dep_vf = paired_template_cluster_bootstrap(
        [{"end_to_end": r["deployable_comparator"]} for r in scored["SFT_VF_op"]],
        scored["SFT_VF_op"],
        metric_key="cross_ok",
        procedure="end_to_end",
    )

    b_nf = boundary_audit(dev_cases, raw_preds["NF_train_NF_infer"], use_deployable=False)
    b_nf_op_e2e = boundary_audit(dev_cases, raw_preds["SFT_NF_op"], use_deployable=False)
    b_nf_op_dep = boundary_audit(dev_cases, raw_preds["SFT_NF_op"], use_deployable=True)
    b_vf_op_dep = boundary_audit(dev_cases, raw_preds["SFT_VF_op"], use_deployable=True)

    save_json_atomic(
        ROOT / "phi4_dev_scores.json",
        {
            "model": "microsoft/Phi-4-mini-instruct",
            "dev_sha256": v2_manifest["sha256"]["dev_v2.json"],
            "tokenizer_comparison": tok_stats,
            "eval_meta": eval_meta,
            "scored": scored,
            "bootstrap": {
                "NFNF_minus_VFVF_e2e": boot_order,
                "SFT_NF_op_deployable_minus_e2e": boot_dep_nf,
                "SFT_VF_op_deployable_minus_e2e": boot_dep_vf,
            },
            "boundary_audits": {
                "NF_NF_e2e": b_nf,
                "SFT_NF_op_e2e": b_nf_op_e2e,
                "SFT_NF_op_deployable": b_nf_op_dep,
                "SFT_VF_op_deployable": b_vf_op_dep,
            },
        },
    )

    # Render markdown report
    lines = [
        "# 第 3 周跨模型复现开发集报告：`microsoft/Phi-4-mini-instruct` (3.8B) on `dev_v2.json`",
        "",
        "> **严格盲测纪律**：本报告仅评测 `dev_v2.json`（48 基础题组 / 192 题），`test_v2_candidate.json` 仍严格未触碰。超参数、96 条训练样本与三个随机种子（42/43/44）与 `Qwen2.5-3B-Instruct` 完全对齐。",
        "",
        "## 1. 分词器差异（同一 96 条训练样本，每轮总 token）",
        "",
        "| 训练条件 | Phi-4-mini 输入 token | Qwen2.5-3B 输入 token | Phi-4-mini 监督答案 token | Qwen2.5-3B 监督答案 token |",
        "|---|---:|---:|---:|---:|",
    ]
    for k in ("VF", "NF", "VF_op", "NF_op"):
        ts = tok_stats[k]
        lines.append(
            f"| `{k}` | {ts['phi4_input_tokens']} | {ts['qwen_input_tokens']} | {ts['phi4_supervised_tokens']} | {ts['qwen_supervised_tokens']} |"
        )

    lines += [
        "",
        "## 2. Phi-4-mini-instruct 2×2 顺序格 + 算子比较器主表（`dev_v2.json`）",
        "",
        "| 条件 | 算子正确 | 端到端跨界双对 | 可部署比较器跨界 | 规则已知比较器跨界 | 端到端同侧双对 | 端到端改写双对 | 不可解析题数 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cond, runs in scored.items():
        ni = sum(r["end_to_end"]["n_items"] for r in runs)
        ng = sum(r["end_to_end"]["n_groups"] for r in runs)
        op_ok = sum(r["deployable_comparator"]["op_correct"] for r in runs)
        ec = sum(r["end_to_end"]["cross_ok"] for r in runs)
        dc = sum(r["deployable_comparator"]["cross_ok"] for r in runs)
        oc = sum(r["oracle_rule_comparator"]["cross_ok"] for r in runs)
        es = sum(r["end_to_end"]["same"]["all"]["both_correct"] for r in runs)
        ep = sum(r["end_to_end"]["para"]["all"]["both_correct"] for r in runs)
        unp = sum(r["end_to_end"]["item_unparsed"] for r in runs)
        op_cell = f"{op_ok}/{ni} ({pct(op_ok, ni)})" if "op" in cond else "—"
        dc_cell = f"{dc}/{ng} ({pct(dc, ng)})" if "op" in cond else "—"
        lines.append(
            f"| `{cond}` | {op_cell} | {ec}/{ng} ({pct(ec, ng)}) | {dc_cell} "
            f"| {oc}/{ng} ({pct(oc, ng)}) | {es}/{ng} ({pct(es, ng)}) | {ep}/{ng} ({pct(ep, ng)}) | {unp}/{ni} |"
        )

    lines += [
        "",
        "## 3. 配对模板簇 Bootstrap 主效应（pp）",
        "",
        f"- **顺序效应（`NF-NF` − `VF-VF`，端到端跨界双对）**：`{boot_order['point_diff_pp']:+.2f}` pp（95% CI `[{boot_order['ci95_template_cluster_pp'][0]:+.2f}, {boot_order['ci95_template_cluster_pp'][1]:+.2f}]`）",
        f"- **可部署比较器净增益（`SFT_NF_op` 比较器 − 同输出端到端）**：`{boot_dep_nf['point_diff_pp']:+.2f}` pp（95% CI `[{boot_dep_nf['ci95_template_cluster_pp'][0]:+.2f}, {boot_dep_nf['ci95_template_cluster_pp'][1]:+.2f}]`）",
        f"- **可部署比较器净增益（`SFT_VF_op` 比较器 − 同输出端到端）**：`{boot_dep_vf['point_diff_pp']:+.2f}` pp（95% CI `[{boot_dep_vf['ci95_template_cluster_pp'][0]:+.2f}, {boot_dep_vf['ci95_template_cluster_pp'][1]:+.2f}]`）",
        "",
        "## 4. 边界位置捷径复现检验（`below / at / above`）",
        "",
        "| 规则（真算子） | 边界位置 | `NF-NF` 端到端 | `SFT_NF_op` 端到端 | `SFT_NF_op` 可部署比较器 | `SFT_VF_op` 可部署比较器 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for rule in ("single", "first_accept", "first_reject", "weco3"):
        op = RULE_SPECS[rule].op
        for rel in ("below", "at", "above"):
            if rel not in b_nf[rule]:
                continue
            o0, n0, _ = b_nf[rule][rel]
            o1, n1, _ = b_nf_op_e2e[rule][rel]
            o2, n2, _ = b_nf_op_dep[rule][rel]
            o3, n3, _ = b_vf_op_dep[rule][rel]
            lines.append(
                f"| `{rule}` (`{op}`) | `{rel}` | {o0}/{n0} ({pct(o0, n0)}) | {o1}/{n1} ({pct(o1, n1)}) "
                f"| {o2}/{n2} ({pct(o2, n2)}) | {o3}/{n3} ({pct(o3, n3)}) |"
            )

    (ROOT / "PHI4_DEV_REPORT.md").write_text("\n".join(lines), encoding="utf8")
    print("Wrote:", ROOT / "PHI4_DEV_REPORT.md", flush=True)


if __name__ == "__main__":
    main()

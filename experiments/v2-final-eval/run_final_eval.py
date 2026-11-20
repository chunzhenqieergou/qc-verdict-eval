"""One-shot frozen blind evaluation runner for v2 final test (`frozen_test_v2.json`, 144 base groups / 576 items).

Refuses to execute unless `freeze_manifest.json` exists and EVERY SHA-256 hash
(frozen test set + all 24 adapter weights + scorer/rules code) passes verification.

Usage:
  & "D:\\Users\\Administrator\\Desktop\\finetune\\maintenance-demo-env\\Scripts\\python.exe" run_final_eval.py --model all
  (or --model qwen / --model phi4 / --score-only)
"""

import argparse
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
sys.path.insert(0, str(V2_DIR))

for k in ("TEMP", "TMP", "HF_HOME", "TORCH_HOME", "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR", "CUDA_CACHE_PATH"):
    d = ROOT / ".cache" / k.lower()
    d.mkdir(parents=True, exist_ok=True)
    os.environ[k] = str(d)
os.environ["HF_HUB_OFFLINE"] = "1"

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig

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

MODEL_PATHS = {
    "qwen": Path(r"D:\Users\Administrator\Desktop\finetune\models\Qwen2.5-3B-Instruct"),
    "phi4": Path(r"D:\Users\Administrator\Desktop\finetune\models\Phi-4-mini-instruct"),
}

MODEL_DISPLAY = {
    "qwen": "Qwen2.5-3B-Instruct (3.1B)",
    "phi4": "Phi-4-mini-instruct (3.8B)",
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json_atomic(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf8")
    tmp.replace(path)


def verify_freeze() -> tuple[dict, list[dict]]:
    manifest_path = ROOT / "freeze_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing {manifest_path}! Complete REVIEW_SHEET.md review and run freeze_test.py first."
        )
    manifest = load_json(manifest_path)
    test_path = ROOT / manifest["dataset"]["file"]
    assert sha256_file(test_path) == manifest["dataset"]["sha256"], "frozen_test_v2.json SHA-256 mismatch!"
    for fname, expected_sha in manifest["code_sha256"].items():
        assert sha256_file(V2_DIR / fname) == expected_sha, f"Code drift detected in {fname}!"
    for key, info in manifest["adapters_sha256"].items():
        p = Path(info["path"])
        assert p.is_file() and sha256_file(p) == info["sha256"], f"Adapter drift detected in {key}!"
    cases = load_json(test_path)
    assert len(cases) == 576
    return manifest, cases


def load_base(model_path: Path):
    return AutoModelForCausalLM.from_pretrained(
        model_path,
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


def generate_all(model, tok, cases: list[dict], system_prompt: str) -> tuple[list[dict], float]:
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
    return rows, torch.cuda.max_memory_allocated() / 2**30


def run_model_inference(model_key: str, manifest: dict, cases: list[dict], prompts: dict[str, str]) -> None:
    model_path = MODEL_PATHS[model_key]
    expected_ids = [c["id"] for c in cases]
    out_dir = ROOT / "outputs" / model_key
    out_dir.mkdir(parents=True, exist_ok=True)

    def adapter_dir(subkey: str) -> Path:
        return Path(manifest["adapters_sha256"][f"{model_key}/{subkey}"]["path"]).parent

    eval_jobs = [
        ("base", None, [("base_VF", "VF"), ("base_NF", "NF"), ("base_VF_op", "VF_op"), ("base_NF_op", "NF_op")]),
    ]
    for s in SEEDS:
        eval_jobs.append((f"VF_s{s}", adapter_dir(f"VF_s{s}"), [(f"VF_train_VF_infer_s{s}", "VF"), (f"VF_train_NF_infer_s{s}", "NF")]))
        eval_jobs.append((f"NF_s{s}", adapter_dir(f"NF_s{s}"), [(f"NF_train_VF_infer_s{s}", "VF"), (f"NF_train_NF_infer_s{s}", "NF")]))
        eval_jobs.append((f"VF_op_s{s}", adapter_dir(f"VF_op_s{s}"), [(f"SFT_VF_op_s{s}", "VF_op")]))
        eval_jobs.append((f"NF_op_s{s}", adapter_dir(f"NF_op_s{s}"), [(f"SFT_NF_op_s{s}", "NF_op")]))

    active_jobs = []
    for tag, adir, runs in eval_jobs:
        pending = []
        for stem, pkey in runs:
            f = out_dir / f"{stem}.json"
            if f.exists():
                assert [r["id"] for r in load_json(f)] == expected_ids
                print(f"[reuse] {model_key}/{stem}", flush=True)
            else:
                pending.append((stem, pkey, f))
        if pending:
            active_jobs.append((tag, adir, pending))

    if not active_jobs:
        print(f"All outputs already complete for {model_key}!", flush=True)
        return

    tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base_model = load_base(model_path)
    base_model.eval()
    peft_model = None
    active_name = None

    for tag, adir, pending in active_jobs:
        if adir is None:
            model_to_use = base_model
        else:
            if peft_model is None:
                peft_model = PeftModel.from_pretrained(base_model, adir, adapter_name=tag)
                peft_model.eval()
                active_name = tag
            else:
                peft_model.load_adapter(adir, adapter_name=tag)
                peft_model.set_adapter(tag)
                if active_name is not None and active_name != tag:
                    peft_model.delete_adapter(active_name)
                active_name = tag
                peft_model.eval()
            model_to_use = peft_model

        for stem, pkey, f in pending:
            t0 = time.time()
            preds, peak = generate_all(model_to_use, tok, cases, prompts[pkey])
            save_json_atomic(f, preds)
            caps = sum(p["capped"] for p in preds)
            print(f"[done] {model_key}/{stem} (576 items, capped={caps}, peak={peak:.2f}GiB, {time.time()-t0:.1f}s)", flush=True)

    del peft_model, base_model
    gc.collect()
    torch.cuda.empty_cache()


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
        out[rule] = {rel: {"ok": ok[rel], "n": tot[rel], "pred_true": pt[rel]} for rel in ("below", "at", "above") if tot[rel]}
    return out


def operator_confusion(cases: list[dict], pred_lists: list[list[dict]]) -> dict:
    by_id = {c["id"]: c for c in cases}
    out = {}
    for rule in RULES_ORDER:
        cnt = Counter()
        for preds in pred_lists:
            for p in preds:
                c = by_id[p["id"]]
                if c["rule"] == rule:
                    cnt[parse_operator(p["text"]) or "MISSING"] += 1
        out[rule] = dict(cnt)
    return out


def score_and_report(manifest: dict, cases: list[dict], models_to_score: list[str]) -> None:
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

    payload = {"freeze_manifest": manifest, "models": {}}

    for mk in models_to_score:
        out_dir = ROOT / "outputs" / mk
        if not (out_dir / "SFT_NF_op_s44.json").is_file():
            print(f"[skip scoring] {mk} outputs incomplete")
            continue
        scored = {}
        raw_preds = {}
        for cond, stems in cond_stems.items():
            plist = [load_json(out_dir / f"{st}.json") for st in stems]
            raw_preds[cond] = plist
            scored[cond] = [score_run(cases, ps) for ps in plist]

        boots = {
            "order_NFNF_minus_VFVF_cross": paired_template_cluster_bootstrap(
                scored["NF_train_NF_infer"], scored["VF_train_VF_infer"], metric_key="cross_ok", procedure="end_to_end"
            ),
            "order_NFNF_minus_VFVF_same": paired_template_cluster_bootstrap(
                scored["NF_train_NF_infer"], scored["VF_train_VF_infer"], metric_key="same_ok", procedure="end_to_end"
            ),
            "order_NFNF_minus_VFVF_para": paired_template_cluster_bootstrap(
                scored["NF_train_NF_infer"], scored["VF_train_VF_infer"], metric_key="para_ok", procedure="end_to_end"
            ),
            "dep_minus_e2e_VF_op_cross": paired_template_cluster_bootstrap(
                [{"end_to_end": r["deployable_comparator"]} for r in scored["SFT_VF_op"]],
                scored["SFT_VF_op"],
                metric_key="cross_ok",
                procedure="end_to_end",
            ),
            "dep_minus_e2e_NF_op_cross": paired_template_cluster_bootstrap(
                [{"end_to_end": r["deployable_comparator"]} for r in scored["SFT_NF_op"]],
                scored["SFT_NF_op"],
                metric_key="cross_ok",
                procedure="end_to_end",
            ),
            "prompt_effect_NF_op_minus_NF_e2e": paired_template_cluster_bootstrap(
                scored["SFT_NF_op"], scored["NF_train_NF_infer"], metric_key="cross_ok", procedure="end_to_end"
            ),
            "prompt_effect_VF_op_minus_VF_e2e": paired_template_cluster_bootstrap(
                scored["SFT_VF_op"], scored["VF_train_VF_infer"], metric_key="cross_ok", procedure="end_to_end"
            ),
        }

        payload["models"][mk] = {
            "display_name": MODEL_DISPLAY[mk],
            "scored": scored,
            "bootstrap": boots,
            "boundary_audits": {
                "NF_NF_e2e": boundary_audit(cases, raw_preds["NF_train_NF_infer"], False),
                "SFT_NF_op_e2e": boundary_audit(cases, raw_preds["SFT_NF_op"], False),
                "SFT_NF_op_deployable": boundary_audit(cases, raw_preds["SFT_NF_op"], True),
                "SFT_VF_op_deployable": boundary_audit(cases, raw_preds["SFT_VF_op"], True),
            },
            "operator_confusion": {
                "SFT_VF_op": operator_confusion(cases, raw_preds["SFT_VF_op"]),
                "SFT_NF_op": operator_confusion(cases, raw_preds["SFT_NF_op"]),
            },
        }

    save_json_atomic(ROOT / "final_scores.json", payload)

    # Render publication-ready FINAL_REPORT.md
    lines = [
        "# v2 冻结终测正式结果报告（12 规则 × 4 模板 × 3 配置 = 144 基础题组 / 576 题）",
        "",
        f"- **冻结时间 (UTC)**：`{manifest['frozen_at_utc']}`",
        f"- **终测集 SHA-256**：`{manifest['dataset']['sha256']}`",
        f"- **领域复核记录**：复核人 `{manifest['review_record']['reviewer']}`，实审规则 `{manifest['review_record']['reviewed_rules']}/12`，实审模板 `{manifest['review_record']['reviewed_templates']}/48`，纠错记录：`{manifest['review_record']['corrections']}`",
        "",
    ]

    for mk, mdata in payload["models"].items():
        scored = mdata["scored"]
        boots = mdata["bootstrap"]
        lines += [
            f"## 模型：{mdata['display_name']}",
            "",
            "### 1) 主结果表（端到端 + 可部署比较器 + 规则已知比较器，含逐种子范围）",
            "",
            "| 条件 | 跨界双对（端到端） | 逐种子跨界% | 同侧双对 (正/负) | 同侧仅一致 | 改写双对 | 不可解析题 | 算子正确 | 可部署比较器跨界 (纠/错) | 规则已知比较器跨界 |",
            "|---|---:|---|---|---:|---:|---:|---:|---:|---:|",
        ]
        for cond, runs in scored.items():
            ni = sum(r["end_to_end"]["n_items"] for r in runs)
            ng = sum(r["end_to_end"]["n_groups"] for r in runs)
            ec = sum(r["end_to_end"]["cross_ok"] for r in runs)
            seed_pcts = ", ".join(f"{100*r['end_to_end']['cross_ok']/r['end_to_end']['n_groups']:.1f}" for r in runs)
            s_ok = sum(r["end_to_end"]["same"]["all"]["both_correct"] for r in runs)
            s_con = sum(r["end_to_end"]["same"]["all"]["label_consistent"] for r in runs)
            sp = f"{sum(r['end_to_end']['same']['pos']['both_correct'] for r in runs)}/{sum(r['end_to_end']['same']['pos']['n'] for r in runs)}"
            sn = f"{sum(r['end_to_end']['same']['neg']['both_correct'] for r in runs)}/{sum(r['end_to_end']['same']['neg']['n'] for r in runs)}"
            p_ok = sum(r["end_to_end"]["para"]["all"]["both_correct"] for r in runs)
            unp = sum(r["end_to_end"]["item_unparsed"] for r in runs)
            oc = sum(r["oracle_rule_comparator"]["cross_ok"] for r in runs)
            if "op" in cond:
                op_ok = sum(r["deployable_comparator"]["op_correct"] for r in runs)
                dc = sum(r["deployable_comparator"]["cross_ok"] for r in runs)
                fix_c = sum(r["deployable_comparator"]["vs_e2e"]["fixed_cross_pairs"] for r in runs)
                brk_c = sum(r["deployable_comparator"]["vs_e2e"]["broken_cross_pairs"] for r in runs)
                op_str = f"{pct(op_ok, ni)}"
                dc_str = f"{dc}/{ng} ({pct(dc, ng)}, +{fix_c}/-{brk_c})"
            else:
                op_str = "—"
                dc_str = "—"
            lines.append(
                f"| `{cond}` | {ec}/{ng} ({pct(ec, ng)}) | {seed_pcts} | {pct(s_ok, ng)} ({sp} / {sn}) "
                f"| {pct(s_con, ng)} | {pct(p_ok, ng)} | {unp}/{ni} | {op_str} | {dc_str} | {oc}/{ng} ({pct(oc, ng)}) |"
            )

        lines += [
            "",
            "### 2) 预指定配对模板簇 Bootstrap 主效应与去一模板（LOTO）敏感性（pp）",
            "",
            "| 对比项 | 点估计 (pp) | 95% 模板簇 CI (pp) | LOTO 范围 (pp) |",
            "|---|---:|---|---|",
        ]
        for bname, b in boots.items():
            ci = b["ci95_template_cluster_pp"]
            loto = b["loto_range_pp"]
            loto_str = f"[{loto[0]:+.2f}, {loto[1]:+.2f}]" if loto else "—"
            lines.append(f"| `{bname}` | {b['point_diff_pp']:+.2f} | [{ci[0]:+.2f}, {ci[1]:+.2f}] | {loto_str} |")

        lines += [
            "",
            "### 3) 跨界成对错误分类 + 单题互斥错误分解",
            "",
            "| 条件 | 双对 | 双肯定 | 双否定 | 反向翻转 | 含不可解析 | 三项全对 | 双数对判错 | 仅统计量错 | 仅阈值错 | 双数均错 | 数字缺失 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for cond in ("base_VF", "base_NF", "VF_train_VF_infer", "NF_train_NF_infer", "SFT_VF_op", "SFT_NF_op"):
            runs = scored[cond]
            pt = [sum(r["end_to_end"]["cross_tax"].get(k, 0) for r in runs) for k in ("both_correct", "both_true", "both_false", "reversed_wrong", "any_unparseable")]
            ib = [sum(r["item_buckets"].get(k, 0) for r in runs) for k in ("all_three_ok", "nums_ok_verdict_wrong", "stat_wrong_limit_ok", "stat_ok_limit_wrong", "both_nums_wrong", "num_missing")]
            lines.append(f"| `{cond}` | " + " | ".join(str(x) for x in pt + ib) + " |")

        lines += [
            "",
            "### 4) 按规则族拆分（三阶段瓶颈：`SFT_VF_op` 与 `SFT_NF_op`，含可部署比较器同侧/改写）",
            "",
            "| 条件 | 规则族 | 端到端跨界 | 可部署比较器跨界 | 规则已知比较器跨界 | 可部署同侧双对 | 可部署改写双对 |",
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

        ba = mdata["boundary_audits"]
        lines += [
            "",
            "### 5) 边界位置捷径审计（`below / at / above`）",
            "",
            "| 规则（真算子） | 边界位置 | `NF-NF` 端到端 | `SFT_NF_op` 端到端 | `SFT_NF_op` 可部署比较器 | `SFT_VF_op` 可部署比较器 |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for rule in ("single", "first_accept", "first_reject", "weco3"):
            op = RULE_SPECS[rule].op
            for rel in ("below", "at", "above"):
                if rel not in ba["NF_NF_e2e"][rule]:
                    continue
                vals = []
                for bk in ("NF_NF_e2e", "SFT_NF_op_e2e", "SFT_NF_op_deployable", "SFT_VF_op_deployable"):
                    d = ba[bk][rule][rel]
                    vals.append(f"{d['ok']}/{d['n']} ({pct(d['ok'], d['n'])})")
                lines.append(f"| `{rule}` (`{op}`) | `{rel}` | " + " | ".join(vals) + " |")
        lines.append("")

    (ROOT / "FINAL_REPORT.md").write_text("\n".join(lines), encoding="utf8")
    print("Wrote:", ROOT / "FINAL_REPORT.md", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one-shot blind final evaluation on frozen_test_v2.json.")
    parser.add_argument("--model", choices=["qwen", "phi4", "all"], default="all")
    parser.add_argument("--score-only", action="store_true")
    args = parser.parse_args()

    manifest, cases = verify_freeze()
    orig_prompts = load_json(OLD_DIR / "order-training" / "prompts.json")
    op_prompts = load_json(OP_DIR / "data" / "prompts.json")
    prompts = {
        "VF": orig_prompts["original"],
        "NF": orig_prompts["reordered"],
        "VF_op": op_prompts["VF_op"],
        "NF_op": op_prompts["NF_op"],
    }

    models = ["qwen", "phi4"] if args.model == "all" else [args.model]
    if not args.score_only:
        for mk in models:
            run_model_inference(mk, manifest, cases, prompts)
    score_and_report(manifest, cases, ["qwen", "phi4"])


if __name__ == "__main__":
    main()

"""Evaluate all frozen Qwen2.5-3B and Phi-4-mini conditions (including VF_op/NF_op deployable comparator)
on the 4 real inspection document cases from `experiments/report-sampling/cases.json`.
"""

import gc
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
V2_DIR = ROOT.parent / "v2-benchmark"
OP_DIR = ROOT.parent / "v2-operator-comparator"
OLD_DIR = ROOT.parent / "scenario-sft"
REAL_CASES_PATH = ROOT.parent / "report-sampling" / "cases.json"
sys.path.insert(0, str(V2_DIR))

for k in ("TEMP", "TMP", "HF_HOME", "TORCH_HOME", "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR", "CUDA_CACHE_PATH"):
    d = ROOT / ".cache" / k.lower()
    d.mkdir(parents=True, exist_ok=True)
    os.environ[k] = str(d)
os.environ["HF_HUB_OFFLINE"] = "1"

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig

from rules import compare_float
from scorer import load_json, parse_number, parse_operator, parse_verdict

SEEDS = (42, 43, 44)
MODEL_PATHS = {
    "qwen": Path(r"D:\Users\Administrator\Desktop\finetune\models\Qwen2.5-3B-Instruct"),
    "phi4": Path(r"D:\Users\Administrator\Desktop\finetune\models\Phi-4-mini-instruct"),
}


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


def run_batch(model, tok, cases: list[dict], system_prompt: str) -> list[dict]:
    eos = model.generation_config.eos_token_id
    cfg = GenerationConfig(
        max_new_tokens=96,
        do_sample=False,
        eos_token_id=eos,
        pad_token_id=tok.pad_token_id,
        use_cache=True,
    )
    chats = [[{"role": "system", "content": system_prompt}, {"role": "user", "content": r["question"]}] for r in cases]
    enc = tok.apply_chat_template(
        chats, add_generation_prompt=True, tokenize=True, padding=True, return_tensors="pt", return_dict=True
    ).to("cuda")
    with torch.inference_mode():
        gen = model.generate(**enc, generation_config=cfg)[:, enc.input_ids.shape[1] :].tolist()
    out = []
    for r, seq in zip(cases, gen):
        txt = tok.decode(seq, skip_special_tokens=True).strip()
        vm = parse_verdict(txt)
        st = parse_number(txt, "统计量")
        lm = parse_number(txt, "阈值")
        op = parse_operator(txt)
        v_dep = compare_float(op, st, lm) if op is not None else None
        v_ora = compare_float("<=", st, lm)
        out.append(
            {
                "id": r["id"],
                "ref_positive": r["positive"],
                "ref_x": r["x"],
                "ref_ac": r["ac"],
                "text": txt,
                "verdict_e2e": vm,
                "stat": st,
                "limit": lm,
                "op": op,
                "verdict_dep": v_dep,
                "verdict_ora": v_ora,
                "e2e_ok": vm == r["positive"],
                "dep_ok": v_dep == r["positive"],
                "ora_ok": v_ora == r["positive"],
            }
        )
    return out


def main() -> None:
    manifest = load_json(ROOT / "freeze_manifest.json")
    cases = load_json(REAL_CASES_PATH)
    orig_prompts = load_json(OLD_DIR / "order-training" / "prompts.json")
    op_prompts = load_json(OP_DIR / "data" / "prompts.json")
    prompts = {
        "VF": orig_prompts["original"],
        "NF": orig_prompts["reordered"],
        "VF_op": op_prompts["VF_op"],
        "NF_op": op_prompts["NF_op"],
    }

    results = {}
    for mk in ("qwen", "phi4"):
        mpath = MODEL_PATHS[mk]
        tok = AutoTokenizer.from_pretrained(mpath, local_files_only=True, padding_side="left")
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        base = load_base(mpath)
        base.eval()
        peft_m = None
        active = None
        mres = {}

        def adir(sub: str) -> Path:
            return Path(manifest["adapters_sha256"][f"{mk}/{sub}"]["path"]).parent

        jobs = [
            ("base", None, [("base_VF", "VF"), ("base_NF", "NF"), ("base_VF_op", "VF_op"), ("base_NF_op", "NF_op")]),
        ]
        for s in SEEDS:
            jobs.append((f"VF_s{s}", adir(f"VF_s{s}"), [(f"VF_VF_s{s}", "VF")]))
            jobs.append((f"NF_s{s}", adir(f"NF_s{s}"), [(f"NF_NF_s{s}", "NF")]))
            jobs.append((f"VF_op_s{s}", adir(f"VF_op_s{s}"), [(f"SFT_VF_op_s{s}", "VF_op")]))
            jobs.append((f"NF_op_s{s}", adir(f"NF_op_s{s}"), [(f"SFT_NF_op_s{s}", "NF_op")]))

        for tag, path, runs in jobs:
            if path is None:
                model = base
            else:
                if peft_m is None:
                    peft_m = PeftModel.from_pretrained(base, path, adapter_name=tag)
                    peft_m.eval()
                    active = tag
                else:
                    peft_m.load_adapter(path, adapter_name=tag)
                    peft_m.set_adapter(tag)
                    if active and active != tag:
                        peft_m.delete_adapter(active)
                    active = tag
                    peft_m.eval()
                model = peft_m
            for stem, pkey in runs:
                mres[stem] = run_batch(model, tok, cases, prompts[pkey])

        results[mk] = mres
        del peft_m, base
        gc.collect()
        torch.cuda.empty_cache()

    (ROOT / "real_docs_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf8")

    # Markdown summary
    lines = [
        "# 四份真实质检文档案例跨模型与可部署比较器评测结果",
        "",
        "| 模型 | 条件（3种子合计或底座） | `3268288` (0 vs 0, at, T) | `3234769` (1 vs 0, above, F) | `vicc-cleaner` (4 vs 7, below, T) | `vicc-motor` (0 vs 21, below, T) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    cids = ["3268288", "3234769", "vicc-cleaner", "vicc-motor"]
    for mk in ("qwen", "phi4"):
        mres = results[mk]
        groups = [
            ("base_VF (E2E)", ["base_VF"], "e2e_ok"),
            ("base_NF (E2E)", ["base_NF"], "e2e_ok"),
            ("VF_VF (E2E)", [f"VF_VF_s{s}" for s in SEEDS], "e2e_ok"),
            ("NF_NF (E2E)", [f"NF_NF_s{s}" for s in SEEDS], "e2e_ok"),
            ("SFT_VF_op (E2E)", [f"SFT_VF_op_s{s}" for s in SEEDS], "e2e_ok"),
            ("SFT_VF_op (Deployable)", [f"SFT_VF_op_s{s}" for s in SEEDS], "dep_ok"),
            ("SFT_NF_op (E2E)", [f"SFT_NF_op_s{s}" for s in SEEDS], "e2e_ok"),
            ("SFT_NF_op (Deployable)", [f"SFT_NF_op_s{s}" for s in SEEDS], "dep_ok"),
        ]
        for label, stems, key in groups:
            cells = []
            for cid in cids:
                hits = sum(next(x[key] for x in mres[st] if x["id"] == cid) for st in stems)
                cells.append(f"{hits}/{len(stems)}")
            lines.append(f"| `{mk}` | {label} | " + " | ".join(cells) + " |")
    (ROOT / "REAL_DOCS_REPORT.md").write_text("\n".join(lines), encoding="utf8")
    print("Wrote:", ROOT / "REAL_DOCS_REPORT.md")


if __name__ == "__main__":
    main()

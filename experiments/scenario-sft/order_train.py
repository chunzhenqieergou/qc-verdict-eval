from contextlib import contextmanager
import gc
import hashlib
import json

import run
import torch
from data import DATA, ROOT, SYSTEM, save

ORIG_SYSTEM = run.SYSTEM
ORIG_READ = run.read

from order_probe import PROBE_SYSTEM

ARM = "scenario"
SEEDS = [42, 43, 44]
OUT_ROOT = ROOT / "order-training"
CONDITIONS = ("original", "reordered")
EVAL_BUDGET = 96


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf8")).hexdigest()


def sha256_file(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def swap_first_two_lines(answer: str) -> str:
    """Swap line 1 (conclusion) and line 2 (numeric value) in a training answer."""
    lines = answer.splitlines()
    if len(lines) < 3 or not all(line.strip() for line in lines[:3]):
        raise ValueError(f"Expected at least 3 non-empty lines in answer, got: {answer!r}")
    return "\n".join([lines[1], lines[0], *lines[2:]])


def has_adapter_weights(adapter_dir) -> bool:
    return (
        adapter_dir.is_dir()
        and (adapter_dir / "adapter_config.json").is_file()
        and any(p.is_file() and p.name.startswith("adapter_model") for p in adapter_dir.iterdir())
    )


def verify_original_runs() -> None:
    if ORIG_SYSTEM != SYSTEM:
        raise RuntimeError("run.SYSTEM differs from data.SYSTEM at startup")
    if not isinstance(PROBE_SYSTEM, str) or not PROBE_SYSTEM.strip() or PROBE_SYSTEM == ORIG_SYSTEM:
        raise ValueError("Invalid order_probe.PROBE_SYSTEM")
    if not run.BASE.exists():
        raise FileNotFoundError(f"Missing base model directory: {run.BASE}")
    for split in (ARM, "near", "far"):
        split_path = DATA / f"{split}.json"
        if not split_path.is_file():
            raise FileNotFoundError(f"Missing dataset split: {split_path}")
    for seed in SEEDS:
        orig_dir = ROOT / "runs" / f"{ARM}-{seed}"
        adapter_dir = orig_dir / "adapter"
        train_log = orig_dir / "training.json"
        if not has_adapter_weights(adapter_dir):
            raise FileNotFoundError(f"Missing original adapter weights: {adapter_dir}")
        if not train_log.is_file():
            raise FileNotFoundError(f"Missing original training record: {train_log}")
        record = json.loads(train_log.read_text(encoding="utf8"))
        if not isinstance(record, dict) or not isinstance(record.get("loss"), list):
            raise ValueError(f"Invalid original training record: {train_log}")


def is_complete_training(out, expected_log_len: int) -> bool:
    train_log = out / "training.json"
    if not train_log.is_file() or not has_adapter_weights(out / "adapter"):
        return False
    try:
        record = json.loads(train_log.read_text(encoding="utf8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        isinstance(record, dict)
        and "steps" in record
        and isinstance(record.get("loss"), list)
        and len(record["loss"]) == expected_log_len
    )


def is_complete_predictions(path, expected_ids: list) -> bool:
    if not path.is_file():
        return False
    try:
        rows = json.loads(path.read_text(encoding="utf8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        isinstance(rows, list)
        and len(rows) == len(expected_ids)
        and [r.get("id") for r in rows if isinstance(r, dict)] == expected_ids
        and all("text" in r and "capped" in r for r in rows)
    )


@contextmanager
def patched_run(system_prompt: str, reader):
    """Temporarily inject run.SYSTEM and run.read, then restore originals."""
    saved_system, saved_read = run.SYSTEM, run.read
    run.SYSTEM = system_prompt
    run.read = reader
    try:
        yield
    finally:
        run.SYSTEM = saved_system
        run.read = saved_read


def freeze_or_verify_json(path, payload) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf8"))
        if existing != payload:
            raise ValueError(f"Existing frozen file differs from expected content: {path}")
        return
    save(path, payload)


def clear_cuda() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    verify_original_runs()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    orig_train = ORIG_READ(ARM)
    near_cases = ORIG_READ("near")
    far_cases = ORIG_READ("far")
    expected_eval_ids = [r["id"] for r in near_cases + far_cases]
    reordered_train = [{**r, "answer": swap_first_two_lines(r["answer"])} for r in orig_train]
    expected_log_len = 2 * ((len(reordered_train) + 3) // 4)

    frozen_train_path = OUT_ROOT / "reordered_scenario.json"
    prompts_path = OUT_ROOT / "prompts.json"
    manifest_path = OUT_ROOT / "manifest.json"

    freeze_or_verify_json(frozen_train_path, reordered_train)
    freeze_or_verify_json(prompts_path, {"original": ORIG_SYSTEM, "reordered": PROBE_SYSTEM})

    manifest = {
        "purpose": "paired_mechanistic_diagnosis",
        "note": "在同一套已使用测试题（near + far）上所做的配对机理诊断（paired mechanistic diagnosis on the same previously used near+far test set）。",
        "same_used_test_set": True,
        "arm": ARM,
        "seeds": SEEDS,
        "fresh_from_base": True,
        "loads_historical_adapter": False,
        "base_model": str(run.BASE),
        "paired_original_runs": [f"runs/{ARM}-{seed}" for seed in SEEDS],
        "eval_splits": ["near", "far"],
        "eval_counts": {"near": len(near_cases), "far": len(far_cases), "total": len(expected_eval_ids)},
        "eval_budget": EVAL_BUDGET,
        "eval_batch_size": 12,
        "conditions": list(CONDITIONS),
        "frozen_train_file": frozen_train_path.name,
        "prompts_file": prompts_path.name,
        "sha256": {
            "scenario": sha256_file(DATA / f"{ARM}.json"),
            "reordered_scenario": sha256_file(frozen_train_path),
            "near": sha256_file(DATA / "near.json"),
            "far": sha256_file(DATA / "far.json"),
            "original_system": sha256_text(ORIG_SYSTEM),
            "reordered_system": sha256_text(PROBE_SYSTEM),
        },
    }
    if not manifest_path.exists() or json.loads(manifest_path.read_text(encoding="utf8")) != manifest:
        save(manifest_path, manifest)

    def reordered_read(split):
        if split == ARM:
            return [dict(r) for r in reordered_train]
        return ORIG_READ(split)

    for seed in SEEDS:
        out = OUT_ROOT / f"{ARM}-{seed}"
        if is_complete_training(out, expected_log_len):
            print("Reused training", out.name, flush=True)
        else:
            with patched_run(PROBE_SYSTEM, reordered_read):
                run.train(ARM, seed, out)
            clear_cuda()

        for condition, prompt in (("original", ORIG_SYSTEM), ("reordered", PROBE_SYSTEM)):
            cond_out = out / condition
            pred_path = cond_out / "predictions.json"
            if is_complete_predictions(pred_path, expected_eval_ids):
                print("Reused evaluation", f"{out.name}/{condition}", flush=True)
                continue
            with patched_run(prompt, ORIG_READ):
                run.evaluate(cond_out, out / "adapter", budget=EVAL_BUDGET)
            clear_cuda()


if __name__ == "__main__":
    main()

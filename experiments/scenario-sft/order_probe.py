"""Inference-only order probe swapping numeric-field and verdict lines in the system prompt."""
import gc
import json

from run import ROOT, SEEDS, base, read, tok
from data import SYSTEM, save
from peft import PeftModel
import torch
from transformers import GenerationConfig

OLD_ORDER = "第一行只写成立或不成立，第二行写统计量=数值，阈值=数值，第三行写理由。"
NEW_ORDER = "第一行写统计量=数值，阈值=数值，第二行只写成立或不成立，第三行写理由。"
assert OLD_ORDER in SYSTEM, f"Unexpected SYSTEM prompt: {SYSTEM}"
PROBE_SYSTEM = SYSTEM.replace(OLD_ORDER, NEW_ORDER, 1)

BATCH_SIZE = 12
MAX_NEW_TOKENS = 96


def probe_messages(row):
    """Build chat messages with the reordered system prompt."""
    return [
        {"role": "system", "content": PROBE_SYSTEM},
        {"role": "user", "content": row["question"]},
    ]


def is_reusable(path, expected_ids):
    """Return True if an existing predictions file matches the frozen question IDs."""
    if not path.exists():
        return False
    rows = json.loads(path.read_text(encoding="utf8"))
    if not isinstance(rows, list):
        raise ValueError(f"Expected a JSON list in {path}")
    if [row.get("id") for row in rows] != expected_ids:
        raise ValueError(f"Question ID mismatch in {path}")
    if not all({"id", "text", "capped"}.issubset(row) for row in rows):
        raise ValueError(f"Missing required keys in {path}")
    return True


def evaluate_run(name, adapter, cases, expected_ids):
    """Run greedy evaluation under the reordered system prompt and save predictions."""
    out_file = ROOT / "order-probe" / name / "predictions.json"
    if is_reusable(out_file, expected_ids):
        print("Reused", name, len(expected_ids), flush=True)
        return
    if adapter is not None and not adapter.exists():
        raise FileNotFoundError(f"Missing adapter directory: {adapter}")

    model = base()
    try:
        if adapter is not None:
            model = PeftModel.from_pretrained(model, adapter)
        tok.padding_side = "left"
        model.eval()
        cfg = GenerationConfig(
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            eos_token_id=model.generation_config.eos_token_id,
            pad_token_id=tok.pad_token_id,
            use_cache=True,
        )
        ends = cfg.eos_token_id if isinstance(cfg.eos_token_id, list) else [cfg.eos_token_id]
        predictions = []
        with torch.inference_mode():
            for i in range(0, len(cases), BATCH_SIZE):
                chunk = cases[i : i + BATCH_SIZE]
                enc = tok.apply_chat_template(
                    [probe_messages(row) for row in chunk],
                    padding=True,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                ).to("cuda")
                new_ids = model.generate(**enc, generation_config=cfg)[:, enc.input_ids.shape[1] :].tolist()
                for row, ids in zip(chunk, new_ids):
                    predictions.append(
                        {
                            "id": row["id"],
                            "text": tok.decode(ids, skip_special_tokens=True).strip(),
                            "capped": not any(k in ends for k in ids),
                        }
                    )
        assert [row["id"] for row in predictions] == expected_ids
        save(out_file, predictions)
        print("Evaluated", name, len(predictions), flush=True)
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


def main():
    """Evaluate base-0 and all paraphrase/scenario adapters on frozen near+far cases."""
    cases = read("near") + read("far")
    expected_ids = [row["id"] for row in cases]
    runs = [("base-0", None)] + [
        (f"{arm}-{seed}", ROOT / "runs" / f"{arm}-{seed}" / "adapter")
        for arm in ("paraphrase", "scenario")
        for seed in SEEDS
    ]
    for name, adapter in runs:
        evaluate_run(name, adapter, cases, expected_ids)


if __name__ == "__main__":
    main()

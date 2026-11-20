"""Freeze protocol for v2 final blind test (`test_v2.json`) and all 24 trained adapters.

Run this ONCE ONLY AFTER completing domain review of:
  D:\\Users\\Administrator\\Desktop\\project1\\experiments\\v2-benchmark\\REVIEW_SHEET.md

Example usage after domain review sign-off:
  python freeze_test.py --reviewer "YourName" --reviewed-templates 48 --corrections "0 corrections; all 12 rules & 48 templates verified"
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
V2_DIR = ROOT.parent / "v2-benchmark"
QOP_DIR = ROOT.parent / "v2-operator-comparator" / "runs"
PHI_DIR = ROOT.parent / "v2-cross-model-phi4" / "runs"
OLD_DIR = Path(r"d:\Users\Administrator\Desktop\project1\experiments\scenario-sft")
sys.path.insert(0, str(V2_DIR))

from rules import RULES_ORDER, compute_ground_truth
from scorer import load_json

SEEDS = (42, 43, 44)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect_adapter_paths() -> dict[str, Path]:
    adapters: dict[str, Path] = {}
    for s in SEEDS:
        adapters[f"qwen/VF_s{s}"] = OLD_DIR / "runs" / f"scenario-{s}" / "adapter" / "adapter_model.safetensors"
        adapters[f"qwen/NF_s{s}"] = OLD_DIR / "order-training" / f"scenario-{s}" / "adapter" / "adapter_model.safetensors"
        adapters[f"qwen/VF_op_s{s}"] = QOP_DIR / f"VF_op_s{s}" / "adapter" / "adapter_model.safetensors"
        adapters[f"qwen/NF_op_s{s}"] = QOP_DIR / f"NF_op_s{s}" / "adapter" / "adapter_model.safetensors"
        adapters[f"phi4/VF_s{s}"] = PHI_DIR / f"VF_s{s}" / "adapter" / "adapter_model.safetensors"
        adapters[f"phi4/NF_s{s}"] = PHI_DIR / f"NF_s{s}" / "adapter" / "adapter_model.safetensors"
        adapters[f"phi4/VF_op_s{s}"] = PHI_DIR / f"VF_op_s{s}" / "adapter" / "adapter_model.safetensors"
        adapters[f"phi4/NF_op_s{s}"] = PHI_DIR / f"NF_op_s{s}" / "adapter" / "adapter_model.safetensors"
    return adapters


def verify_test_cases(cases: list[dict]) -> None:
    assert len(cases) == 576, f"Expected 576 items (144 base groups x 4), got {len(cases)}"
    rules_seen = {c["rule"] for c in cases}
    assert rules_seen == set(RULES_ORDER), f"Missing rules: {set(RULES_ORDER) - rules_seen}"
    # Row 5: independent exact ground-truth recalculation on every single item
    for c in cases:
        gt = compute_ground_truth(c["rule"], c["inputs"])
        assert gt["positive"] == c["positive"], f"Truth mismatch on {c['id']}"
        assert abs(gt["value"] - c["value"]) < 1e-6, f"Value mismatch on {c['id']}"
        assert abs(gt["limit"] - c["limit"]) < 1e-6, f"Limit mismatch on {c['id']}"
        assert gt["op"] == c["op"], f"Op mismatch on {c['id']}"
    # Zero overlap with dev_v2.json
    dev_q = {x["question"] for x in load_json(V2_DIR / "data" / "dev_v2.json")}
    test_q = {x["question"] for x in cases}
    assert len(test_q) == len(cases), "Duplicate questions inside test set!"
    assert test_q.isdisjoint(dev_q), "Leakage between dev_v2 and test_v2!"


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze v2 final blind test & all 24 adapter hashes.")
    parser.add_argument("--reviewer", required=True, help="Name/role of domain reviewer who checked REVIEW_SHEET.md")
    parser.add_argument("--reviewed-rules", type=int, default=12, help="Number of rules reviewed (out of 12)")
    parser.add_argument("--reviewed-templates", type=int, default=48, help="Number of test templates reviewed (out of 48)")
    parser.add_argument("--corrections", default="None", help="Summary of corrections applied during review")
    args = parser.parse_args()

    cand_path = V2_DIR / "data" / "test_v2_candidate.json"
    frozen_path = ROOT / "frozen_test_v2.json"
    manifest_path = ROOT / "freeze_manifest.json"

    cases = load_json(cand_path)
    verify_test_cases(cases)

    raw_bytes = cand_path.read_bytes()
    test_sha = hashlib.sha256(raw_bytes).hexdigest()

    if frozen_path.exists():
        existing_sha = sha256_file(frozen_path)
        if existing_sha != test_sha:
            raise RuntimeError(
                f"REFUSING TO OVERWRITE ALREADY-FROZEN TEST SET! "
                f"existing={existing_sha} vs candidate={test_sha}"
            )
        print("frozen_test_v2.json already exists with identical hash; verifying manifest...")
    else:
        ROOT.mkdir(parents=True, exist_ok=True)
        frozen_path.write_bytes(raw_bytes)

    adapters = collect_adapter_paths()
    adapter_hashes = {}
    for key, p in sorted(adapters.items()):
        if not p.is_file():
            raise FileNotFoundError(f"Missing trained adapter weight: {key} -> {p}")
        adapter_hashes[key] = {"path": str(p), "sha256": sha256_file(p)}

    code_hashes = {
        name: sha256_file(V2_DIR / name)
        for name in ("rules.py", "templates.py", "build_dataset.py", "scorer.py")
    }

    manifest = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "FROZEN_FOR_SINGLE_BLIND_RUN",
        "review_record": {
            "reviewer": args.reviewer,
            "reviewed_rules": args.reviewed_rules,
            "reviewed_templates": args.reviewed_templates,
            "corrections": args.corrections,
            "review_sheet_sha256": sha256_file(V2_DIR / "REVIEW_SHEET.md"),
        },
        "dataset": {
            "file": frozen_path.name,
            "n_base_groups": 144,
            "n_items": len(cases),
            "sha256": test_sha,
        },
        "code_sha256": code_hashes,
        "adapters_sha256": adapter_hashes,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf8")
    print(f"FROZEN SUCCESSFULLY! test_v2 sha256={test_sha[:16]}... ({len(adapter_hashes)} adapters locked)")
    print("Manifest written to:", manifest_path)


if __name__ == "__main__":
    main()

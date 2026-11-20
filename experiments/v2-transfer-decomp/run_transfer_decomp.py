"""Week 4 Transfer Decomposition (2x2 Factorial):
    {old_wording, new_wording} x {old_num, new_num}
across all 12 rules (2 pairs/rule = 24 pairs / 48 items per cell, 192 items total).

Strictly NEVER touches test_v2_candidate.json.
Uses historical scenario-sft wording templates vs dev_v2 wording templates, crossed with
historical far.json numerical configs (j=5,6) vs dev_v2 numerical configs (t=0, c=0,1).
Evaluates frozen Qwen2.5-3B-Instruct base + VF-VF (3 seeds) + NF-NF (3 seeds) and computes
main effects of wording shift, numerical-config shift, and interaction, plus oracle-rule
comparator breakdown to isolate Stage-1 (statistic) vs Stage-2 (verdict) sensitivity.
"""

from collections import defaultdict
import gc
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
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig

from rules import RULE_SPECS, RULES_ORDER, compute_ground_truth
from scorer import load_json, parse_number, parse_verdict
from templates import DEV_PARTS, TEMPLATES

SEEDS = (42, 43, 44)
BATCH_SIZE = 12
MAX_NEW_TOKENS = 96

OLD_SETTINGS = [
    "输送机托辊", "泵站垫圈", "冷库门铰链", "分拣机导轮", "包装机滚筒", "纺织机衬套",
    "水表壳体", "家具滑轨", "纸机轴套", "印刷机隔圈", "饮料线导轨",
]


def save_json_atomic(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf8")
    tmp.replace(path)


def render_old_wording(rule: str, part: str, inp: dict) -> str:
    """Exact historical far.json wording template from scenario-sft/data.py, parameterized by inputs."""
    if rule == "single":
        desc = f"一次计数抽样的接收数为{inp['c']}，抽取{inp['n']}件，发现不合格品{inp['x']}件。"
        claim = "该批可按方案接收"
    elif rule == "first_accept":
        desc = f"二次抽样方案首样接收数为{inp['c1']}、拒收数为{inp['r1']}。首样{inp['n1']}件中不合格品有{inp['x1']}件。"
        claim = "首样结束即可接收"
    elif rule == "first_reject":
        desc = f"二次抽样首样接收数为{inp['c1']}、拒收数为{inp['r1']}。首样{inp['n1']}件中有{inp['x1']}件不合格品。"
        claim = "首样结束即可拒收"
    elif rule == "second_accept":
        desc = f"已按方案进入第二次抽样。首样有{inp['x1']}件不合格品，次样有{inp['x2']}件不合格品，累计接收数为{inp['c2']}。"
        claim = "完成两次抽样后可接收该批"
    elif rule == "weco3":
        desc = f"本期监测点相对中心线的标准化偏差为{inp['z']}。"
        claim = "该点触发上侧三西格玛越限"
    elif rule in ("weco2", "weco1", "weco8"):
        window, thresh, level = {"weco2": (3, 2, 2), "weco1": (5, 4, 1), "weco8": (8, 8, 0)}[rule]
        desc = f"最近{window}点相对中心线的标准化偏差依次为{inp['seq']}。统计量取高于+{level}的点数。"
        claim = f"触发上侧WECO的{window}点中至少{thresh}点高于+{level}规则"
    elif rule == "cp":
        desc = f"过程稳定且近似正态，已知标准差为{inp['sd']}，规格限为{inp['lo']}至{inp['hi']}。"
        claim = f"CP达到合同门槛{inp['target']}"
    elif rule == "cpu":
        desc = f"过程稳定且近似正态，已知标准差为{inp['sd']}，均值为{inp['mean']}，上规格限为{inp['hi']}。"
        claim = f"CPU达到合同门槛{inp['target']}"
    elif rule == "cpl":
        desc = f"过程稳定且近似正态，已知标准差为{inp['sd']}，均值为{inp['mean']}，下规格限为{inp['lo']}。"
        claim = f"CPL达到合同门槛{inp['target']}"
    elif rule == "cpk":
        desc = f"过程稳定且近似正态，已知标准差为{inp['sd']}，均值为{inp['mean']}，规格限为{inp['lo']}至{inp['hi']}。"
        claim = f"CPK达到合同门槛{inp['target']}"
    else:
        raise ValueError(rule)
    return f"下面是抽查人员抄录的数据。{part}质量记录。现有数据：{desc}审核员据此写下“{claim}”。这句话成立吗？"


def make_old_num_pairs() -> dict[str, list[tuple[dict, dict]]]:
    """Reproduce exact numerical configs from scenario-sft/data.py far split (j=5 and j=6) for all 12 rules."""
    out = {}
    for rule in RULES_ORDER:
        pairs = []
        for j in (5, 6):
            c = 2 + j
            pair_in = []
            for positive in (True, False):
                offset = int(not positive)
                if rule == "single":
                    inp = {"n": 80, "c": c, "x": c + offset}
                elif rule == "first_accept":
                    inp = {"n1": 80, "c1": c, "r1": c + 3, "x1": c + offset}
                elif rule == "first_reject":
                    inp = {"n1": 80, "c1": c - 2, "r1": c, "x1": c - offset}
                elif rule == "second_accept":
                    val = c + offset
                    inp = {"n1": 80, "n2": 80, "x1": 1, "x2": val - 1, "c2": c}
                elif rule == "weco3":
                    z = 4 + j / 10 if positive else 2 - j / 10
                    inp = {"z": f"+{z:g}"}
                elif rule in ("weco2", "weco1", "weco8"):
                    window, thresh, level = {"weco2": (3, 2, 2), "weco1": (5, 4, 1), "weco8": (8, 8, 0)}[rule]
                    val = thresh - offset
                    seq = [level + 0.2 + (j + i) % 7 / 10 for i in range(window)]
                    for i in range(val, window):
                        seq[i] = level - 0.2 - (j + i) % 3 / 10
                    seq = seq[j % window :] + seq[: j % window]
                    inp = {"seq": "、".join(f"{x:+.1f}" for x in seq)}
                else:
                    sigma = 1 + j
                    mean = 100
                    val = 1.5 if positive else 0.5
                    if rule == "cp":
                        lo, hi = mean - 3 * sigma, mean - 3 * sigma + 6 * sigma * val
                        inp = {"lo": f"{lo:g}", "hi": f"{hi:g}", "sd": f"{sigma:g}", "target": "1"}
                    elif rule == "cpu":
                        inp = {"mean": "100", "hi": f"{100 + 3 * sigma * val:g}", "sd": f"{sigma:g}", "target": "1"}
                    elif rule == "cpl":
                        inp = {"mean": "100", "lo": f"{100 - 3 * sigma * val:g}", "sd": f"{sigma:g}", "target": "1"}
                    else:
                        inp = {
                            "mean": "100",
                            "lo": f"{100 - 6 * sigma:g}",
                            "hi": f"{100 + 3 * sigma * val:g}",
                            "sd": f"{sigma:g}",
                            "target": "1",
                        }
                gt = compute_ground_truth(rule, inp)
                assert gt["positive"] == positive, (rule, j, positive, inp, gt)
                pair_in.append(inp)
            pairs.append((pair_in[0], pair_in[1]))
        out[rule] = pairs
    return out


def make_new_num_pairs() -> dict[str, list[tuple[dict, dict]]]:
    """Extract 2 cross-boundary pairs per rule from dev_v2.json (template 0, configs 0 & 1)."""
    dev = load_json(V2_DIR / "data" / "dev_v2.json")
    by_gid = defaultdict(dict)
    for x in dev:
        by_gid[x["group_id"]][x["role"]] = x
    out = {}
    for rule in RULES_ORDER:
        pairs = []
        for c_idx in (0, 1):
            gid = f"dev_{rule}_t0_c{c_idx}"
            b, cr = by_gid[gid]["base"], by_gid[gid]["cross"]
            pos_in = b["inputs"] if b["positive"] else cr["inputs"]
            neg_in = cr["inputs"] if b["positive"] else b["inputs"]
            pairs.append((pos_in, neg_in))
        out[rule] = pairs
    return out


def build_2x2_dataset() -> list[dict]:
    old_nums = make_old_num_pairs()
    new_nums = make_new_num_pairs()
    rows = []
    for r_idx, rule in enumerate(RULES_ORDER):
        for num_tag, num_map in (("old_num", old_nums), ("new_num", new_nums)):
            for p_idx, (pos_in, neg_in) in enumerate(num_map[rule]):
                for word_tag in ("old_word", "new_word"):
                    cell = f"{word_tag}__{num_tag}"
                    pair_id = f"{cell}__{rule}__p{p_idx}"
                    part = (
                        OLD_SETTINGS[(5 + p_idx + r_idx) % len(OLD_SETTINGS)]
                        if word_tag == "old_word"
                        else DEV_PARTS[(r_idx * 7 + p_idx) % len(DEV_PARTS)]
                    )
                    tmpl = TEMPLATES["dev"][rule][p_idx]["primary"]
                    for positive, inp in ((True, pos_in), (False, neg_in)):
                        q = render_old_wording(rule, part, inp) if word_tag == "old_word" else tmpl.format(part=part, **inp)
                        gt = compute_ground_truth(rule, inp)
                        assert gt["positive"] == positive
                        rows.append(
                            {
                                "id": f"{pair_id}__{int(positive)}",
                                "cell": cell,
                                "wording": word_tag,
                                "num_cfg": num_tag,
                                "pair_id": pair_id,
                                "rule": rule,
                                "family": RULE_SPECS[rule].family,
                                "op": gt["op"],
                                "question": q,
                                "inputs": inp,
                                "value": gt["value"],
                                "limit": gt["limit"],
                                "positive": positive,
                            }
                        )
    assert len(rows) == 192
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
    return rows


def eval_cells(cases: list[dict], pred_lists: list[list[dict]]) -> dict:
    from rules import compare_float

    by_cell = defaultdict(lambda: defaultdict(int))
    by_cell_fam = defaultdict(lambda: defaultdict(int))
    for preds in pred_lists:
        pmap = {p["id"]: p["text"] for p in preds}
        pairs = defaultdict(list)
        for c in cases:
            text = pmap[c["id"]]
            vm = parse_verdict(text)
            st = parse_number(text, "统计量")
            lm = parse_number(text, "阈值")
            vc = compare_float(c["op"], st, lm)
            num_ok = (st is not None and abs(st - c["value"]) < 1e-6) and (lm is not None and abs(lm - c["limit"]) < 1e-6)
            pairs[(c["cell"], c["family"], c["pair_id"])].append((vm == c["positive"], vc == c["positive"], num_ok))
            for store in (by_cell[c["cell"]], by_cell_fam[f"{c['cell']}|{c['family']}"]):
                store["items"] += 1
                store["item_ok"] += int(vm == c["positive"])
                store["num_ok"] += int(num_ok)
        for (cell, fam, _pid), tup in pairs.items():
            assert len(tup) == 2
            e2e_pair = int(all(t[0] for t in tup))
            ora_pair = int(all(t[1] for t in tup))
            for store in (by_cell[cell], by_cell_fam[f"{cell}|{fam}"]):
                store["pairs"] += 1
                store["e2e_pair_ok"] += e2e_pair
                store["ora_pair_ok"] += ora_pair
    return {"by_cell": {k: dict(v) for k, v in sorted(by_cell.items())}, "by_cell_fam": {k: dict(v) for k, v in sorted(by_cell_fam.items())}}


def pct(n: int, d: int) -> str:
    return f"{100.0 * n / d:.1f}%" if d else "0.0%"


def main() -> None:
    cases = build_2x2_dataset()
    save_json_atomic(ROOT / "cases_2x2.json", cases)
    expected_ids = [c["id"] for c in cases]

    base_path = Path(load_json(OLD_DIR / "environment.json")["base"])
    prompts = load_json(OLD_DIR / "order-training" / "prompts.json")
    tok = AutoTokenizer.from_pretrained(base_path, local_files_only=True, padding_side="left")
    tok.pad_token = tok.eos_token

    out_dir = ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs = [("base", None, [("base_VF", "original"), ("base_NF", "reordered")])]
    for s in SEEDS:
        jobs.append((f"vf_{s}", OLD_DIR / "runs" / f"scenario-{s}" / "adapter", [(f"VF_VF_s{s}", "original")]))
        jobs.append((f"nf_{s}", OLD_DIR / "order-training" / f"scenario-{s}" / "adapter", [(f"NF_NF_s{s}", "reordered")]))

    for tag, adapter_dir, runs in jobs:
        pending = []
        for stem, pkey in runs:
            f = out_dir / f"{stem}.json"
            if f.exists():
                assert [r["id"] for r in load_json(f)] == expected_ids
                print(f"[reuse] {stem}", flush=True)
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
            preds = generate_all(model, tok, cases, prompts[pkey])
            save_json_atomic(f, preds)
            print(f"[done] {stem} (capped={sum(p['capped'] for p in preds)}, {time.time()-t0:.1f}s)", flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()

    cond_stems = {
        "base_VF": ["base_VF"],
        "base_NF": ["base_NF"],
        "VF_VF": [f"VF_VF_s{s}" for s in SEEDS],
        "NF_NF": [f"NF_NF_s{s}" for s in SEEDS],
    }
    res = {c: eval_cells(cases, [load_json(out_dir / f"{st}.json") for st in sts]) for c, sts in cond_stems.items()}
    save_json_atomic(ROOT / "transfer_scores.json", res)

    cells_order = ["old_word__old_num", "new_word__old_num", "old_word__new_num", "new_word__new_num"]
    cell_cn = {
        "old_word__old_num": "旧措辞 × 原数值",
        "new_word__old_num": "新措辞 × 原数值",
        "old_word__new_num": "旧措辞 × 新数值",
        "new_word__new_num": "新措辞 × 新数值",
    }

    lines = [
        "# 2×2 迁移分解实验报告（旧/新措辞 × 原/新数值配置）",
        "",
        "> 全部四格共享 12 条规则（每格 24 对 / 48 题，共 192 题），模型为已冻结 Qwen2.5-3B-Instruct 适配器（3 种子合计 72 对/格）。`test_v2_candidate.json` 未被触碰。",
        "",
        "## 1. 四格总表（端到端成对全对 / 双数均对 / 规则已知比较器成对全对）",
        "",
        "| 条件 | 格子 | 端到端成对全对 | 双数均对（题） | 比较器成对全对 |",
        "|---|---|---:|---:|---:|",
    ]
    for cond in ("base_VF", "base_NF", "VF_VF", "NF_NF"):
        for cell in cells_order:
            s = res[cond]["by_cell"][cell]
            lines.append(
                f"| `{cond}` | {cell_cn[cell]} | {s['e2e_pair_ok']}/{s['pairs']} ({pct(s['e2e_pair_ok'], s['pairs'])}) "
                f"| {s['num_ok']}/{s['items']} ({pct(s['num_ok'], s['items'])}) "
                f"| {s['ora_pair_ok']}/{s['pairs']} ({pct(s['ora_pair_ok'], s['pairs'])}) |"
            )

    # Main effect summary for NF_NF
    nf = res["NF_NF"]["by_cell"]
    p_oo = 100 * nf["old_word__old_num"]["e2e_pair_ok"] / nf["old_word__old_num"]["pairs"]
    p_no = 100 * nf["new_word__old_num"]["e2e_pair_ok"] / nf["new_word__old_num"]["pairs"]
    p_on = 100 * nf["old_word__new_num"]["e2e_pair_ok"] / nf["old_word__new_num"]["pairs"]
    p_nn = 100 * nf["new_word__new_num"]["e2e_pair_ok"] / nf["new_word__new_num"]["pairs"]
    word_eff = 0.5 * ((p_no - p_oo) + (p_nn - p_on))
    num_eff = 0.5 * ((p_on - p_oo) + (p_nn - p_no))

    lines += [
        "",
        "## 2. `NF_NF` 迁移主效应分解（端到端成对正确率，pp）",
        "",
        f"- **基准（旧措辞 × 原数值）**：`{p_oo:.1f}%`",
        f"- **仅换措辞（新措辞 × 原数值）**：`{p_no:.1f}%`（差值 `{p_no - p_oo:+.1f}` pp）",
        f"- **仅换数值配置（旧措辞 × 新数值）**：`{p_on:.1f}%`（差值 `{p_on - p_oo:+.1f}` pp）",
        f"- **两者同换（新措辞 × 新数值）**：`{p_nn:.1f}%`（总差值 `{p_nn - p_oo:+.1f}` pp）",
        f"- **措辞主效应均值**：`{word_eff:+.1f}` pp；**数值配置主效应均值**：`{num_eff:+.1f}` pp",
        "",
        "## 3. 按规则族拆解（`NF_NF` 端到端成对全对 / 比较器成对全对）",
        "",
        "| 规则族 | 旧措辞×原数值 | 新措辞×原数值 | 旧措辞×新数值 | 新措辞×新数值 |",
        "|---|---:|---:|---:|---:|",
    ]
    for fam in ("sampling", "control_chart", "capability"):
        row = []
        for cell in cells_order:
            s = res["NF_NF"]["by_cell_fam"][f"{cell}|{fam}"]
            row.append(f"{s['e2e_pair_ok']}/{s['pairs']} ({pct(s['e2e_pair_ok'], s['pairs'])}) / 比较器 {pct(s['ora_pair_ok'], s['pairs'])}")
        lines.append(f"| {fam} | " + " | ".join(row) + " |")

    (ROOT / "TRANSFER_REPORT.md").write_text("\n".join(lines), encoding="utf8")
    print("Wrote:", ROOT / "TRANSFER_REPORT.md", flush=True)


if __name__ == "__main__":
    main()

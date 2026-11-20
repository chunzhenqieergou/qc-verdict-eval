"""Deterministic matched examples derived from public NIST quality-control rules."""
from pathlib import Path
import hashlib
import json
import random

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
SYSTEM = "你是民用工业质量审核员。判断题中所述结论，并用简短文字说明依据。第一行只写成立或不成立，第二行写统计量=数值，阈值=数值，第三行写理由。"
RULES = ["single", "first_accept", "first_reject", "second_accept", "weco3", "weco2", "weco1", "weco8", "cp", "cpu", "cpl", "cpk"]
SOURCES = ["https://www.itl.nist.gov/div898/handbook/pmc/section2/pmc23.htm",
           "https://www.itl.nist.gov/div898/handbook/pmc/section2/pmc251.htm",
           "https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc32.htm",
           "https://www.itl.nist.gov/div898/handbook/pmc/section1/pmc16.htm"]
SETTINGS = ["输送机托辊", "泵站垫圈", "冷库门铰链", "分拣机导轮", "包装机滚筒", "纺织机衬套", "水表壳体", "家具滑轨", "纸机轴套", "印刷机隔圈", "饮料线导轨"]
ASK = ["审核记录如下。", "请复核这份检验单。", "班组提交一项判断。", "交班时需要核实一条结论。", "请根据记录给审核意见。", "下面是抽查人员抄录的数据。"]

def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf8")
    temp.replace(path)

def case(rule, j, positive):
    # j identifies an underlying case. Each matched pair differs in one observation only.
    offset = int(not positive)
    c = 2 + j
    if rule in RULES[:4]:
        if rule == "single":
            value = c + offset
            desc = f"一次计数抽样的接收数为{c}，抽取80件，发现不合格品{value}件。"
            claim, why = "该批可按方案接收", "一次抽样的不合格品数须小于或等于接收数"
        elif rule == "first_accept":
            value = c + offset
            desc = f"二次抽样方案首样接收数为{c}、拒收数为{c+3}。首样80件中不合格品有{value}件。"
            claim, why = "首样结束即可接收", "首样直接接收要求不合格品数不超过首样接收数"
        elif rule == "first_reject":
            value = c - offset
            desc = f"二次抽样首样接收数为{c-2}、拒收数为{c}。首样80件中有{value}件不合格品。"
            claim, why = "首样结束即可拒收", "首样直接拒收要求不合格品数达到或超过拒收数"
        else:
            value = c + offset
            desc = f"已按方案进入第二次抽样。首样有1件不合格品，次样有{value-1}件不合格品，累计接收数为{c}。"
            claim, why = "完成两次抽样后可接收该批", "两次抽样的不合格品数之和须不超过累计接收数"
        return desc, claim, value, c, why
    if rule == "weco3":
        value = 4 + j / 10 if positive else 2 - j / 10
        return f"本期监测点相对中心线的标准化偏差为+{value:g}。", "该点触发上侧三西格玛越限", value, 3, "上侧单点越限要求标准化偏差大于3"
    if rule in ["weco2", "weco1", "weco8"]:
        window, threshold, level = {"weco2":(3,2,2), "weco1":(5,4,1), "weco8":(8,8,0)}[rule]
        value = threshold - offset
        seq = [level + 0.2 + (j+i)%7/10 for i in range(window)]
        for i in range(value, window):
            seq[i] = level - 0.2 - (j+i)%3/10
        seq = seq[j%window:] + seq[:j%window]
        numbers = "、".join(f"{x:+.1f}" for x in seq)
        desc = f"最近{window}点相对中心线的标准化偏差依次为{numbers}。统计量取高于+{level}的点数。"
        return desc, f"触发上侧WECO的{window}点中至少{threshold}点高于+{level}规则", value, threshold, f"窗口内高于指定界线的点数至少应有{threshold}个"
    sigma = 1 + j
    mean = 100
    value = 1.5 if positive else 0.5
    if rule == "cp":
        lo, hi = mean - 3*sigma, mean - 3*sigma + 6*sigma*value
        facts, why = f"规格限为{lo:g}至{hi:g}", "Cp等于规格宽度除以六倍标准差"
    elif rule == "cpu":
        facts, why = f"均值为100，上规格限为{100+3*sigma*value:g}", "Cpu等于上规格限与均值之差除以三倍标准差"
    elif rule == "cpl":
        facts, why = f"均值为100，下规格限为{100-3*sigma*value:g}", "Cpl等于均值与下规格限之差除以三倍标准差"
    else:
        facts, why = f"均值为100，规格限为{100-6*sigma:g}至{100+3*sigma*value:g}", "Cpk取上下单侧能力指数中的较小值"
    return f"过程稳定且近似正态，已知标准差为{sigma}，{facts}。", f"{rule.upper()}达到合同门槛1", value, 1, why

def rows(split):
    result = []
    for ri, rule in enumerate(RULES):
        settings = range(4) if split in ("paraphrase","scenario") else ([0] if split=="near" else range(5,10))
        for k, scene in enumerate(settings):
            j = 0 if split=="paraphrase" else scene
            setting = SETTINGS[(j+ri)%len(SETTINGS)]
            prefix = ASK[k if split in ("paraphrase","scenario") else 4 + int(split=="far")]
            for positive in (False, True):
                desc, claim, val, limit, why = case(rule,j,positive)
                # Lexical variants paired across both training arms.
                variants = [f"{desc}有人认为：{claim}。请判断。",f"待核实结论为“{claim}”。记录写明：{desc}",f"{desc}由这些记录能否得出“{claim}”？",f"审核意见写着“{claim}”。证据如下：{desc}"]
                tail = variants[k%4] if split in ("paraphrase","scenario") else f"现有数据：{desc}审核员据此写下“{claim}”。这句话成立吗？"
                q = prefix+setting+"质量记录。"+tail
                answer = ("成立" if positive else "不成立")+f"\n统计量={val:g}，阈值={limit:g}\n"+why+"。"
                result.append(dict(id=f"{split}-{rule}-{k}-{int(positive)}", rule=rule, pair=f"{rule}-{k}",split=split,question=q,answer=answer,positive=positive,value=val,limit=limit,source=SOURCES[0 if ri==0 else 1 if ri<4 else 2 if ri<8 else 3]))
    return result

def freeze():
    parts = {s:rows(s) for s in ["paraphrase","scenario","near","far"]}
    trainq = {r["question"] for s in ("paraphrase","scenario") for r in parts[s]}
    assert all(r["question"] not in trainq for s in ("near","far") for r in parts[s])
    manifest = {}
    for s, records in parts.items():
        assert len(set(r["question"] for r in records)) == len(records)
        path = DATA / (s+".json")
        if path.exists():
            assert json.loads(path.read_text(encoding="utf8")) == records
        else:
            save(path,records)
        manifest[s] = {"n":len(records),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
    save(DATA/"manifest.json",manifest)

if __name__=="__main__":
    freeze()


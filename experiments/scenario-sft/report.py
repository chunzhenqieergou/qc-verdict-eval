from collections import Counter
from statistics import mean
from score import ROOT,DATA,load,grade,summarize,save

scores = load(ROOT/"scores.json")
runs = scores["runs"]
cases = load(DATA/"near.json")+load(DATA/"far.json")
base_short = load(ROOT/"runs/base-0/predictions.json")
base_long = load(ROOT/"sensitivity/base-256/predictions.json")
assert [c["id"] for c in cases]==[r["id"] for r in base_long]
long_far=summarize([grade(c,r) for c,r in zip(cases,base_long) if c["split"]=="far"])
same_untruncated=sum(s["text"]==l["text"] for s,l in zip(base_short,base_long) if not s["capped"])
previously_complete=sum(not s["capped"] for s in base_short)
save(ROOT/"sensitivity/base-256/scores.json",dict(far=long_far,original_uncapped=previously_complete,uncapped_text_identical=same_untruncated))
def pc(x):
    return f"{100*x:.1f}%"
rows=[]
for name,r in runs.items():
    vals=r["far"]
    rows.append("| "+" | ".join([name,pc(r["near"]["decision"]),pc(vals["decision"]),pc(vals["pair"]),pc(vals["coverage"]),pc(vals["all_three"]),str(vals["cap"])])+" |")
deltas=[]
for k,n in zip(["decision","pair","coverage","all_three"],["单题判断","成对判断","三项覆盖","三项全对"]):
    d=scores["far_contrasts"][k]
    lo,hi=d["rule_cluster_ci"]
    deltas.append(f"| {n} | {100*d['difference']:+.2f} | [{100*lo:+.2f}, {100*hi:+.2f}] |")
base_dir=ROOT/"runs"
def bias(name):
    ps=load(base_dir/name/"predictions.json")[24:]
    return sum(r["text"].splitlines()[0].strip()=="成立" for r in ps)
audit=load(ROOT/"audit-ratings.json")
keys=load(ROOT/"audit-key.json")
assert len(keys)==len(audit)==36
counts={n:sum(r["correct_decision_numbers_and_reason"] for r,k in zip(audit,keys) if k["run"]==n) for n in ["base-0","paraphrase-42","scenario-42"]}
body="""# 同义改写与多场景微调对照实验

在 120 道新场景质量判断题上，多场景组比同义改写组多答对了一部分统计量和阈值。两组的成对判断正确率均为 1.7%。现有结果支持结构化数值作答改善，关于新场景下作出正确判断的收益，估计区间跨零。

## 设计与实际执行

从公开底座 Qwen2.5-3B-Instruct 分别新建六个 4-bit NF4 QLoRA 适配器。两组各 96 题，正反各半，三个配对种子，每次 48 个优化步，保留第二轮末权重。改写组每轮 14,608 个总输入 token，多场景组 14,610 个，参与损失的答案 token 分别为 2,848 与 2,860。全部训练与推理在本地 RTX 5060 Ti 完成。

语料由规则程序生成，覆盖民用工件抽样验收、过程控制图和过程能力指数。两组对同一组规则使用相同的四种问法。多场景组同时改变工件和观测值。控制图与能力指数题的部分阈值或答案在新场景仍会重复。新场景与训练共享 12 项规则和出题模板，部分数值超出训练范围。24 道近场景题测已有基础案例换问法，120 道新场景题构成 60 对。每对只改变决定结论的观测量。训练前写入题目、参考答案及哈希。该轮按固定方案用末轮权重，无验证集调参。6 个适配器和 1 个底座各回答 144 题，主评测共 1,008 份回答。

## 分模型结果

三项指第一行判断、统计量和阈值。评分器逐项解析，缺失或无法解析记零。判断正确率只看结论。成对正确要求一正一反两题均答对。三项全对另需两个数值正确，自由文本理由另行抽核。

| 模型或适配器 | 近场景单题 | 新场景单题 | 新场景成对 | 新场景三项覆盖 | 新场景三项全对 | 新场景触顶数 |
|---|---:|---:|---:|---:|---:|---:|
"""+ "\n".join(rows)+"""

## 多场景组减改写组

先在同规则、同种子内配对，再平均三个种子，对 12 条规则作 10,000 次簇 bootstrap。单位均为百分点。规则来自三个相关的知识类别，这些区间只描述本题库内的变动。多指标区间未经多重比较校正，种子数也限制了对训练随机性的估计。

| 新场景指标 | 平均差值 | 95% 规则簇区间 |
|---|---:|---:|
"""+ "\n".join(deltas)+f"""

## 错误复核

- 120 道新场景题中有 60 道结论为“成立”。改写组三个种子分别预测“成立” {bias("paraphrase-42")}、{bias("paraphrase-43")}、{bias("paraphrase-44")} 次，多场景组分别预测 {bias("scenario-42")}、{bias("scenario-43")}、{bias("scenario-44")} 次。固定回答一种结论即可达到 50% 单题正确率，成对得分为零。生成顺序要求模型先给结论，顺序的影响尚未单独检验。
- 一条首样接收案例中，不合格品数与接收数均为 8。两个种子 42 的适配器都给出了 8 和 8，并正确复述“不超过接收数”，第一行却写成“不成立”。同一道题同时暴露出字段覆盖与实际判断的差别。
- 事后抽取每条规则一题，对底座和两个种子 42 适配器的 36 份回答打乱并隐藏身份，由本次助手按结论、数值和理由均正确的标准复核。底座 {counts['base-0']}/12，改写 {counts['paraphrase-42']}/12，多场景 {counts['scenario-42']}/12。每条规则只抽一题，评阅者可见参考答案，缺少领域人员独立双评。
- 96-token 主评测中，底座在新场景的 42/120 题触顶，适配器均为零。以相同顺序和批次将底座重跑至 256 token，新场景判断为 {pc(long_far['decision'])}、成对为 {pc(long_far['pair'])}、覆盖为 {pc(long_far['coverage'])}，仍触顶 {long_far['cap']}/120。原来正常结束的 {previously_complete} 份回答中，扩预算后有 {same_untruncated} 份文本完全一致。首行结论可从全部主评测回答中解析，严格数值字段更受表达格式影响。
- 本次正式对照的范围是合成的民用工业质量判断。尚缺独立来源的维修工单、开放式现场问答、第二款底座和外部专家复核。多场景组的数值与工件共同变化，观察到的差异不能进一步归因到某一种变化。

## 复现与文件

本目录保存 [原始数据及哈希](data/manifest.json)、[实验协议](protocol.md)、[生成规则](data.py)、[训练和推理](run.py)、[主结果](scores.json)、[逐题评分](item-scores.json)、[全部适配器与输出](runs/)、[匿名抽查材料](audit-packet.json) 和 [运行环境](environment.json)。用项目既有 Python GPU 环境依次运行 run.py、score.py。附加预算检查使用 base_budget.py。

规则来源为 NIST/SEMATECH 工程统计手册。场景、合同门槛和观测值由本实验设定。[一次抽样](https://www.itl.nist.gov/div898/handbook/pmc/section2/pmc23.htm) · [二次抽样](https://www.itl.nist.gov/div898/handbook/pmc/section2/pmc251.htm) · [控制图](https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc32.htm) · [过程能力](https://www.itl.nist.gov/div898/handbook/pmc/section1/pmc16.htm)
"""
(ROOT/"RESULTS.md").write_text(body,encoding="utf8")

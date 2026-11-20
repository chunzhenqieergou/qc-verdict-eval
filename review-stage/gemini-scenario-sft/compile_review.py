from pathlib import Path
import hashlib
import json
from statistics import mean
here=Path(__file__).resolve().parent
root=here.parents[1]/"experiments/scenario-sft"
def load(p):
    return json.loads(p.read_text(encoding="utf-8-sig"))
v=load(here/"verified-scores.json")
g=v["groups"]
official=load(root/"order-comparison.json")
sens=load(here/"delimiter-sensitivity.json")
for name,x in g.items():
    run,condition=name.split("/")
    if run.startswith("trained-"):
        m=official["order_training"]["seeds"][run.removeprefix("trained-")][condition]
    else:
        m=official["models"][run]["original" if condition=="original" else "order_probe"]
    assert round(60*m["pair"])==x["both"],(name,m,x)
    assert round(120*m["decision"])==x["correct"],(name,m,x)
for name, split in load(root/"data/manifest.json").items():
    assert hashlib.sha256((root/"data"/(name+".json")).read_bytes()).hexdigest()==split["sha256"]
seeds=(42,43,44)
comparisons=[
 ("原训练、原提示","scenario-{}/original","runs/scenario-{}/predictions.json"),
 ("原训练、新提示","scenario-{}/reordered","order-probe/scenario-{}/predictions.json"),
 ("新训练、原提示","trained-scenario-{}/original","order-training/scenario-{}/original/predictions.json"),
 ("新训练、新提示","trained-scenario-{}/reordered","order-training/scenario-{}/reordered/predictions.json")]
table=[]
for label,pattern,path in comparisons:
    ms=[g[pattern.format(s)] for s in seeds]
    loose=[sens[path.format(s)] for s in seeds]
    table.append("| "+" | ".join([label, f"{100*mean(x['correct'] for x in ms)/120:.1f}%",f"{100*mean(x['both'] for x in ms)/60:.1f}%",f"{100*mean(x['pair'] for x in loose)/60:.1f}%", "、".join(str(x["both"]) for x in ms),"、".join(str(x["predicted_positive"]) for x in ms),"、".join(str(x["unparsed"]) for x in ms)])+" |")
effect=v["paired_changes"]["matched_retraining"]
ci=effect["ci"]
old=[g[f"scenario-{s}/original"] for s in seeds]
new=[g[f"trained-scenario-{s}/reordered"] for s in seeds]
avg_pair=mean(x["both"] for x in new)/60
decision="达到本轮继续验证的门槛。优先冻结独立来源或重写模板的测试集，检验能否复现。" if avg_pair>=.15 and ci[0]>0 else "未达到本轮继续归因于顺序的门槛。优先检查训练题上是否也有判断偏置，并在限定数值范围和模板的独立题上检验。"
report=f"""# Gemini 审查与答案顺序对照

Gemini 4 Hydrogen (XHigh) 通过 Antigravity 完成首轮方法审查，编写了推理、训练和评分脚本。Codex 审核代码，在本机 GPU 执行并用独立脚本从原始答案复算。主审查原文见 [Gemini 意见](GEMINI_REVIEW.md)。原始审查对象为合成民用工业质量判断题的训练、数据与评分材料。

## 首轮审查及核验

旧多场景训练条件中，三个种子共有 235 次把统计量和阈值都写对，其中 99 次给错结论。三个种子回答的是同一批 120 题。按真实极性拆开，数值都对时正例判断为 51/118，负例为 85/117。12 项规则中有 7 项在给定正负标签后训练目标数值与阈值恒定，其余 5 项随案例改变。这个划分从数据逐项计算。工件与观测值同时变化，使单一因素的贡献仍待验证。

## 实际补做的对照

两个提示的差异仅在前两行先给判断还是先给数值，第三行均为理由。首先对底座和六个原适配器只换提示，共 1,008 次推理。随后从原底座重新训练三个多场景适配器，每次 96 题、48 步、种子 42/43/44，将训练提示和答案的前两行一并互换，再用两种提示各测 144 题，新增 864 份回答。以下主分析限于 120 道新场景题（60 对）。底座在原提示与新提示的成对率均为 4/60。

| 多场景训练条件 | 单题正确率 | 成对正确率 | 兼容标点的成对率 | 三个种子各答对几对（满分60） | 各种子预测成立数（满分120） | 各种子无法解析的结论数 |
|---|---:|---:|---:|---|---|---|
"""+ "\n".join(table)+f"""

成对正确要求条件改变前后两题均答对。新训新测减原训原测的差值为 **{effect['percentage_points']:+.2f} 个百分点**，按 12 条规则配对重复抽样的 95% 区间为 [{ci[0]:+.2f}, {ci[1]:+.2f}]。该区间汇总三个固定种子，没有量化对其他模型、训练种子或其他题库的外推。

在新训新测条件下，模型共 {sum(x['numeric_correct'] for x in new)} 次把两个数字写对，其中 {sum(x['contradictions'] for x in new)} 次给出明确但错误的判断。原训原测对应 235 次与 99 次。严格评分要求独立行给出明确结论，补充评分也接受分号等标点分隔的明确标签。两套统计均保留原始输出。

按训练目标的数值变化作事后切分。五条数值随案例变化的规则上，三个种子合计从 3/75 对升到 53/75 对，其中累计抽样接收仍只有 1/15 对。七条给定极性后目标数值固定的规则从 0/105 对升到 10/105 对。这些分母包含同一题对在不同种子下的重复作答。分层结果提示单看总体均值会掩盖很大的规则差异。

## 下一步决定

{decision} 本轮属于在已使用测试集上提出并执行的诊断，使用单一底座、同一规则族的模板题。投稿前仍需未参与开发的题目和领域人员独立核对。多样性问题可通过分别改变工件和数值构造交叉对照。

[预先记录的诊断方案](ORDER_PROTOCOL.md) · [独立复算](verified-scores.json) · [数值与极性分层](numerical-strata.json) · [标点敏感性复算](delimiter-sensitivity.json)
"""
(here/"REVIEW_RESULTS.md").write_text(report,encoding="utf8")
print(decision)


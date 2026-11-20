# When Do Verdicts Follow the Numbers? A Paired Evaluation of a Fine-Tuned Language Model on Chinese Quality-Control Claims

## Abstract

A quality-control model can report eight nonconforming units, give the acceptance limit as eight, and still reject the claim that the lot meets the criterion. We study how often a model changes its verdict correctly when one observation crosses a decision boundary. We evaluate Qwen2.5-3B-Instruct on pairs of Chinese inspection questions, presenting each member separately and requiring both verdicts to be correct. Across three fine-tuning seeds, varying training scenarios raised the share of responses with a correct statistic, threshold, and verdict from 23.3% to 37.8% on held-out scenarios from a familiar template family. Both training conditions solved 1.7% of pairs. Placing the numerical fields before the verdict during both training and inference raised paired accuracy from 1.7% to 35.0% on those development cases. On 48 independently worded pairs covering six familiar rules, the corresponding rates were 6.9% and 11.1% (difference +4.17 percentage points, descriptive rule-resampling interval −5.56 to +17.36). The predominant failure on these pairs was to give the same verdict on both sides of the boundary. The results distinguish an improvement in structured answers from an improvement in sensitivity to the evidence used to judge a claim.

**Keywords** — behavioral evaluation, numerical claim verification, contrast sets, supervised fine-tuning, quality control

## 1. Introduction

A two-stage inspection plan allows a lot to pass the first inspection when the observed count is at most eight. In one of our cases the count was exactly eight. A fine-tuned model reported both eights and correctly restated the inclusive rule. Its first line nevertheless said that the acceptance claim was false. An application reading the verdict and one reading the numerical fields would draw different conclusions from the same response.

A single correct verdict is also ambiguous. On a balanced set of acceptance claims, an assistant that endorses every claim earns 50% accuracy. Consider asking the same question twice, in separate contexts, once with eight nonconforming units and once with nine. Endorsing both claims gives one correct answer and fails to distinguish the two lots. Written quality-control rules let us measure that distinction while also checking the evidence the model reports.

We examine two questions about small-data supervised adaptation. **RQ1.** When the training budget is held approximately fixed, does varying the underlying cases improve both three-field correctness and the fraction of decision boundaries the model handles correctly? **RQ2.** How does placing the numerical fields before the verdict during training and inference affect paired accuracy, and how much of the observed difference recurs on independently written cases for familiar rules?

We apply contrast-set evaluation to Chinese claims about acceptance sampling, control-chart signals, and process capability. For each response we can check the statistic and threshold against a rule-derived reference. For each pair we can also distinguish a correct reversal, a reversal in the wrong direction, and a repeated verdict. We first compare training conditions on a common generator, then test an output-order choice on questions written after the adapters were fixed. The latter test changes the empirical conclusion one would draw from development performance alone.

## 2. Related Work

### 2.1 Behavioral tests of whether predictions follow evidence

CheckList organizes NLP evaluation around expected behaviors under controlled input changes [1]. Contrast sets make the evaluation local by editing an input and checking the prediction against the revised reference [2]. Gupta et al. brought this question directly to evidence-based reasoning. Their probes for tabular inference test whether predictions respond to relevant cells and remain consistent under controlled variations [3]. Aarnes and Setty's NumPert is especially close to our task. It perturbs numerical claims and their evidence, including edits that flip the reference label, and measures the resulting failures of veracity prediction [4]. These studies establish the rationale for editing a quantity and reassessing a claim. An inspection rule adds two observable quantities to that test. At an acceptance boundary, for example, we can determine whether the model reported the correct count and the correct limit even on a response whose verdict was wrong.

Controlled edits can also supply supervision. Kaushik et al. asked annotators to revise documents so that a target label changed while other content was preserved, and studied training on original and revised examples [5]. In our training comparison, both arms already contain cases on both sides of each boundary. What changes is how often a fixed budget presents a new underlying case. The paired test asks whether that change improves decisions as well as the numerical fields learned from supervision.

### 2.2 Numerical inference and verification of manufacturing claims

DROP evaluates discrete operations over quantities mentioned in passages [6]. EQUATE studies the further step from a numerical relation to a textual entailment judgment [7]. Mahendra et al. distinguish number comparison and normalization from arithmetic in a numeracy-focused inference benchmark [8]. This distinction matters for quality claims. Single sampling can require only a comparison of two stated counts, whereas a process-capability claim can require computing an index and comparing it with a specified target. Both yield a binary verdict, but the same wrong verdict can follow from different errors. GSM-Symbolic further shows that changing values within a mathematical problem template can change measured performance [9]. We therefore distinguish tests on new instances of a familiar generator from tests on cases that vary both the wording and the numerical configuration.

Manufacturing evaluation also spans different stages of inspection. FORGE tests multimodal models on workpiece verification, surface inspection, and assembly verification [10]. Our text task begins after an observation has been recorded. A model receives a count, a sequence of chart points, or specification data and must judge a written claim. This choice makes the reference statistic and verdict separately checkable. It limits the inference we can draw about the larger inspection process, which also depends on the quality of the measurements.

### 2.3 Output order and explicit decision procedures

Kumar et al. directly compare reasoning-first and answer-first field orders in structured outputs. They show that the order of generated fields can materially change task accuracy [11]. Our numerical fields have a narrower interpretation than a free-text rationale. Their values can be checked independently, and swapping their position with the verdict leaves the requested reason last in both formats. We cross the order used for fine-tuning with the one requested at inference, then evaluate the two matched configurations on a separately authored set. This tests how far a development-set ordering effect travels and whether correct fields accompany a correct response to an evidence change.

An alternative decision procedure makes the calculation executable. TabVer, for example, integrates arithmetic expressions into deterministic proofs for tabular fact verification [12]. Such systems highlight a distinction that matters when interpreting a generated answer. Checking the numbers in a response can expose an inconsistency with a known rule. It does not establish which calculation the model performed internally. Our measurements concern the fields and verdicts the model actually generated.

## 3. Task and evaluation protocol

### 3.1 From a written claim to a reversal pair

Each Chinese question specifies a setting, observations, and a claim. The reference comprises a statistic, a threshold, and a binary judgment on the claim. In single sampling the claim that a lot meets the stated acceptance criterion is true when the nonconforming count is at most the acceptance number. For a control-chart window the statistic can itself be a count of observations above a limit. For a capability claim the statistic is a computed index. A true verdict always endorses the claim as written. For a claim that a signal has been triggered, endorsing it indicates an alarm.

For each pair we fix the claim and rule and change one decision-relevant observation so that the reference verdict reverses. We present the two members independently. For illustration, a first-sample acceptance claim with an allowance of eight is true at eight and false at nine. Both have the same reference threshold. The equality case checks whether the model respects the inclusive comparison.

### 3.2 Splits and their role in the study

| Set | Composition | What the comparison measures |
|---|---|---|
| Training, per condition | 96 examples, 12 rules, balanced verdicts | Wording variation over repeated cases or variation in wording, workpieces, and observations |
| Near scenarios | 24 questions | Familiar underlying cases with a different question form |
| Development reversal set | 120 questions, 60 pairs, 12 rules | New scenarios within the original generator family. These items were reused to investigate answer order. |
| Independently worded reversal set | 96 questions, 48 pairs, 6 previously seen rules | Transfer to new formulations and numerical configurations after adapter freeze |

The two training sets used the same four question forms. The first repeated the same underlying positive and negative case for each rule. The second varied workpieces and observations together. Their per-epoch totals were 14,608 and 14,610 sequence tokens, including 2,848 and 2,860 supervised answer tokens. Some thresholds and reference index values recur in the development set.

For the subsequent evaluation, a separate Gemini session received six rule specifications without the old questions or results. It supplied four templates per rule. Instantiating each template with two numerical configurations and both verdicts yielded eight pairs per rule. Inputs, reference decisions, and adapter hashes were fixed before evaluation. The six rules cover single sampling, one control-chart signal, and four related capability indices. This split tests new formulations of known rules. The question author was machine-assisted and the set has not undergone independent expert validation.

### 3.3 Outcomes and units of analysis

Single-item accuracy credits a correct verdict. Three-field correctness requires the correct verdict, statistic, and threshold in the same response. Paired accuracy is the fraction of pairs for which **both** verdicts are correct. A model that changes its answer in the wrong direction fails the pair. A model that repeats either label also fails, even though it answers one member correctly. We count unparseable verdicts as failures and report them separately where they affect interpretation.

The original verdict-first training comparison scored the first line. For the order experiments the scorer accepts an unambiguous standalone verdict in either position. Numerical scoring parses the labeled statistic and threshold after Unicode normalization. Free-text reasons are outside the three-field score. We report equal-weight means across matched seeds. For differences between configurations, the descriptive 95% intervals resample rules with replacement 10,000 times, keeping the same cases and seeds together across conditions. Six related or partly related rules provide limited information about uncertainty across new rule families.

## 4. Models and experimental design

We adapted Qwen2.5-3B-Instruct [13] with QLoRA [14], using four-bit NF4 quantization and LoRA on all linear layers (rank 16, scaling parameter 32, dropout 0.05). Each adapter was trained for two epochs on one RTX 5060 Ti. We used AdamW with learning rate 0.00007, zero weight decay, batches of four, and 48 optimization steps. The loss masked the prompt and padding. We used the final checkpoint and repeated each condition with seeds 42, 43, and 44. Greedy decoding used batches of 12 and a 96-new-token limit.

For RQ1, both training arms requested the verdict first, followed by a line containing the statistic and threshold, then a reason. For RQ2, we trained a fresh scenario adapter per seed with the first two lines exchanged. On the development set we crossed both training orders with both inference instructions. On the independently worded set we tested the two matched training–inference configurations and the unadapted model under each instruction. The matched comparison changes both training and inference order at once.

## 5. Results

### 5.1 Varying the training cases improves three-field correctness

| Training condition | Correct verdict, 120 items | All three fields correct, 120 items | Both verdicts correct, 60 pairs |
|---|---:|---:|---:|
| Unadapted | 53.3% | 9.2% | 6.7% |
| Paraphrases | 50.0% | 23.3% | 1.7% |
| Varied scenarios | 50.8% | 37.8% | 1.7% |

On the development reversal set, scenario training raised three-field correctness by 14.44 percentage points relative to paraphrasing (descriptive interval +8.33 to +20.56). The single-verdict difference was +0.83 points (−0.56 to +2.50) and the paired difference was 0.00 points (−2.22 to +2.78). The seeds expose a behavior hidden by the means. Out of 120 balanced items, the paraphrase adapters endorsed the claim 0, 84, and 12 times. The scenario adapters did so 0, 118, and 1 times. On the 24 near-scenario items, single-verdict accuracy was 62.5% for the unadapted checkpoint, 55.6% for paraphrases, and 54.2% for varied scenarios.

### 5.2 The development gain from field order shrinks on independently written cases

| Training order | Requested order | Correct development pairs | Unparseable verdicts / items |
|---|---|---:|---:|
| Unadapted | Verdict first | 4/60 (6.7%) | 0/120 |
| Unadapted | Numbers first | 4/60 (6.7%) | 19/120 |
| Verdict first | Verdict first | 3/180 (1.7%) | 0/360 |
| Verdict first | Numbers first | 7/180 (3.9%) | 105/360 |
| Numbers first | Verdict first | 14/180 (7.8%) | 0/360 |
| Numbers first | Numbers first | 63/180 (35.0%) | 2/360 |

Each adapter row evaluates the same 60 pairs at three seeds. The largest gain occurred when the numbers came first at both stages. Merely changing the inference instruction on a verdict-first adapter also affected format adherence. Two of the three seeds produced 54 and 51 unparseable verdicts, respectively.

| Model and matched instruction | Correct independent items | Correct independent pairs | Unparseable verdicts / items |
|---|---:|---:|---:|
| Unadapted, verdict first | 50/96 (52.1%) | 2/48 (4.2%) | 0/96 |
| Unadapted, numbers first | 42/96 (43.8%) | 6/48 (12.5%) | 25/96 |
| Fine-tuned, verdict first | 153/288 (53.1%) | 10/144 (6.9%) | 0/288 |
| Fine-tuned, numbers first | 155/288 (53.8%) | 16/144 (11.1%) | 0/288 |

On the 48 independently worded pairs the gain between matched adapters was +4.17 points (descriptive interval −5.56 to +17.36). Each adapted row repeats the same 48 pairs across three seeds. The unadapted numbers-first model answered 42 of its 71 parseable items correctly and solved 6 of the 48 pairs overall. None of the independent adapter outputs reached the token limit.

Rule composition changed between the two sets. Restricting development results post hoc to the six overlapping rules gave 1/90 versus 20/90 correct pair evaluations for verdict-first and numbers-first, respectively. The corresponding independent counts were 10/144 and 16/144. Within the independent set, the change was concentrated in single sampling (5/24 to 13/24). The control-chart rule remained at 0/24, and the four capability indices together went from 5/96 to 3/96. Both wording and numerical configurations changed, so the design cannot isolate the contribution of either shift.

### 5.3 Most failed pairs receive the same verdict twice

We classified the archived independent predictions by the ordered pair of verdicts on the true and false claims. This post hoc breakdown covers all adapted responses, each of which yielded a parseable verdict.

| Matched adapter order | Both correct | Endorsed both | Rejected both | Both wrong | Total pair evaluations |
|---|---:|---:|---:|---:|---:|
| Verdict first | 10 | 104 | 29 | 1 | 144 |
| Numbers first | 16 | 97 | 26 | 5 | 144 |

The verdict-first adapters repeated a label in 133/144 pair evaluations, and the numbers-first adapters did so in 123/144. The latter changed their answer on 21 pairs but reversed in the correct direction on only 16. Counting any change as sensitivity would credit the other five.

A separate item-level audit found 73/288 responses with both numerical fields correct under each matched configuration. Within those separately selected subsets, 18/73 verdict-first responses and 12/73 numbers-first responses gave an explicitly wrong verdict. The equality case in the introduction illustrates this disagreement. The two subsets can contain different items, so their conditional error rates are descriptive.

## 6. Discussion and limitations

The training intervention improved a quantity that matters to a form-filling application. It increased the share of answers in which three reported fields were jointly correct. Yet all three scenario adapters exhibited a strong preference for one label, and the preferred label varied by seed. Evaluating the two sides of each decision boundary uncovered that pattern. In the independent evaluation the model usually repeated a verdict, even after adaptation to generate numbers first.

The generator also permits simpler policies than applying each rule. For example, in the development generator the positive single-sampling case lies exactly at the acceptance number and the negative case lies above it. On this slice, testing equality alone would succeed. Some capability examples likewise reuse the same reference index values across scenarios. Paired success on these instances establishes correct behavior for the sampled contrasts. Distinguishing general rule application from such shortcuts requires new cases below, at, and above the threshold and label-preserving changes to the evidence.

Output order has a clear consequence for what the evaluator observes. A model can put a correct number before a wrong verdict, and a cross-order prompt can make verdicts difficult to parse. On the independent set, even the unadapted numbers-first checkpoint slightly exceeded the matched numbers-first adapters on the paired measure. The development result supports treating order as an experimental variable. The independent evidence is too limited to recommend one order as a reliable intervention.

Four document-based examples illustrate the scope of the finding. We restated the minor-nonconformity criterion from two public report copies and two supplier examples in Chinese (Appendix A). All eight model configurations accepted the sampling case with zero observed against zero allowed. In a one-unit full inspection, all eight reported one against zero and still accepted the criterion. In the two supplier examples all three verdict-first adapters accepted counts within the allowance, and all three numbers-first adapters rejected them despite giving the correct numbers. There is one sampled lot, one full inspection, and two illustrative documents. These observations cannot estimate the error rate on operational records.

The study covers one checkpoint, one training budget, three seeds, and synthetic Chinese questions about familiar rules. Several values were chosen to probe a boundary and await engineering validation. Four of the six independent rules concern related capability indices. Repeating the same cases across seeds improves visibility into training variation but adds no independent cases. Human validation of the questions, replication across model families, and direct comparison with an executable rule applied to model-extracted inputs would strengthen different parts of the argument.

## 7. Conclusion

On familiar templates, varying the training scenarios improved structured answers, and moving numerical fields before the verdict substantially improved paired decisions. On independently written questions, the matched order difference was smaller and uncertain. The most frequent failure was concrete: a model endorsed both claims or rejected both after the evidence crossed the boundary. Reporting numerical correctness, both verdicts in each pair, and performance on fresh formulations makes that behavior visible.

## Reproducibility and tool use

The local research archive contains generators, protocols, frozen questions and references, training scripts, adapter hashes, predictions, and scoring code. Its experiment directories are `experiments/scenario-sft`, `experiments/independent-blind`, and `experiments/report-sampling`. Public release and archival identifiers are pending. Gemini assisted with independent question writing and protocol review. Codex assisted with implementation, analysis, and manuscript preparation.

## References

1. Ribeiro, M. T., Wu, T., Guestrin, C., and Singh, S. (2020). Beyond Accuracy: Behavioral Testing of NLP Models with CheckList. ACL. [Paper](https://aclanthology.org/2020.acl-main.442/)
2. Gardner, M., et al. (2020). Evaluating Models’ Local Decision Boundaries via Contrast Sets. Findings of EMNLP. [Paper](https://aclanthology.org/2020.findings-emnlp.117/)
3. Gupta, V., et al. (2022). Is My Model Using the Right Evidence? Systematic Probes for Examining Evidence-Based Tabular Reasoning. TACL. [Paper](https://aclanthology.org/2022.tacl-1.38/)
4. Aarnes, P. R., and Setty, V. (2025). NumPert: Numerical Perturbations to Probe Language Models for Veracity Prediction. IJCNLP-AACL Student Research Workshop. [Paper](https://aclanthology.org/2025.ijcnlp-srw.8/)
5. Kaushik, D., Hovy, E., and Lipton, Z. C. (2020). Learning the Difference that Makes a Difference with Counterfactually-Augmented Data. ICLR. [Paper](https://arxiv.org/abs/1909.12434)
6. Dua, D., et al. (2019). DROP: A Reading Comprehension Benchmark Requiring Discrete Reasoning Over Paragraphs. NAACL. [Paper](https://aclanthology.org/N19-1246/)
7. Ravichander, A., Naik, A., Rose, C., and Hovy, E. (2019). EQUATE: A Benchmark Evaluation Framework for Quantitative Reasoning in Natural Language Inference. CoNLL. [Paper](https://aclanthology.org/K19-1033/)
8. Mahendra, R., Spina, D., Cavedon, L., and Verspoor, K. (2025). Evaluating Numeracy of Language Models as a Natural Language Inference Task. Findings of NAACL. [Paper](https://aclanthology.org/2025.findings-naacl.467/)
9. Mirzadeh, I., et al. (2025). GSM-Symbolic: Understanding the Limitations of Mathematical Reasoning in Large Language Models. ICLR. [Paper](https://openreview.net/forum?id=AjXkRZIvjB)
10. Jian, X., et al. (2026). FORGE: Fine-grained multimodal evaluation for manufacturing quality assurance. Findings of EMNLP. [Paper](https://aclanthology.org/2026.findings-emnlp.888/)
11. Kumar, V., Kanishka, and Mandora, B. (2026). Structure Tax: How Structured Output affects LLMs Performance. IJCNLP-AACL. [Paper](https://aclanthology.org/2026.aacl-long.209/)
12. Aly, R., and Vlachos, A. (2024). TabVer: Tabular Fact Verification with Natural Logic. TACL. [Paper](https://aclanthology.org/2024.tacl-1.89/)
13. Qwen, Yang, A., et al. (2024). Qwen2.5 Technical Report. [Paper](https://arxiv.org/abs/2412.15115)
14. Dettmers, T., Pagnoni, A., Holtzman, A., and Zettlemoyer, L. (2023). QLoRA: Efficient Finetuning of Quantized LLMs. NeurIPS. [Paper](https://arxiv.org/abs/2305.14314)

## Appendix A. Inspection documents

- Report 3268288, Pilates equipment. Five of eleven units inspected, zero minor nonconforming units, acceptance number zero. [Public report copy](https://shopcdnpro.grainajz.com/category/345655/2633/01d539d97d40b33bdd67c5766d13d8d7/Inspection%20report.pdf)
- Report 3234769, CNC equipment. One-unit full inspection, one minor nonconforming unit, acceptance number zero. [Public report copy](https://www.scribd.com/document/946774180/StyleCNC-Inspection)
- VICC high-pressure cleaner example. Four against an allowance of seven. [Supplier sample report](https://www.vicc.com/wp-content/sample-report/Hardline-electric-product-inspection-sample-report.pdf)
- VICC window lift motor example. Zero against an allowance of 21. [Supplier sample report](https://www.vicc.com/wp-content/sample-report/Hardline-auto-part-inspection-sample-report.pdf)

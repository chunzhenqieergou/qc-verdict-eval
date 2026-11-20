# Does a fine-tuned language model change its quality decision when the evidence changes?

## Abstract

An industrial quality assistant can report the observed statistic and its threshold correctly while giving the wrong disposition. We studied this discrepancy in Chinese quality-control questions using Qwen2.5-3B-Instruct and four-bit adapter fine-tuning. We constructed pairs of cases in which a change in one observation reversed the correct verdict. We first compared training on paraphrases with training on varied scenarios under nearly matched token budgets. Scenario variation increased the proportion of responses with all three requested fields correct by 14.44 percentage points on 120 held-out questions. Both conditions nevertheless solved only 1.7% of pairs. We then varied the order of the numerical fields and verdict in training and inference. On familiar templates, placing the numbers first in both stages increased paired accuracy from 1.7% to 35.0%. On 48 pairs written independently after the adapters had been frozen, paired accuracy was 6.9% with the verdict first and 11.1% with the numbers first. The descriptive rule-cluster interval for this difference spanned zero. In this setting, evaluating both members of an evidence-changing pair exposed errors that remained hidden in single-question accuracy and structured-field scores.

**Keywords**: supervised fine-tuning, acceptance sampling, quality control, behavioral evaluation, contrast sets

## 1. Introduction

Consider a lot for which a sampling plan allows eight nonconforming units. A model records an observed count of eight, repeats the rule that the count must not exceed eight, and declares the acceptance claim false. This response occurred in our experiment. A field extractor would recover both numbers, and a reviewer scanning the explanation might recognize the right rule. A downstream system reading only the verdict would take the wrong action.

This example motivates two evaluation questions. When a training set contains different workpieces and observations, does improved reporting of the evidence accompany improved decisions? When the response format places the statistic before the verdict, does any gain persist on independently worded cases? We address both questions with a fixed model, matched training conditions and pairs of opposite-label items. The practical output is a reproducible way to audit small-data fine-tuning for rule-governed Chinese quality decisions.

## 2. Background

Behavioral testing of language models uses targeted inputs to isolate capabilities that aggregate accuracy can hide [1]. Contrast sets assess the local decision boundary by editing an example in a way that changes its reference label [4]. Numerical reading-comprehension benchmarks also test operations on quantities recovered from text [5]. Here the targeted behavior is a decision reversal under a change in numerical evidence. Quality control supplies an explicit reference decision and admits equality cases, which distinguish strict from inclusive inequalities. Quantized low-rank fine-tuning makes repeated training runs with matched seeds affordable [2]. We use an instruction-tuned member of the Qwen2.5 family [3]. The experiment concerns the behavior of this checkpoint under a small supervised adaptation budget.

## 3. Materials and methods

### 3.1 Cases and reference decisions

The development corpus covered 12 rules across attribute acceptance sampling, control-chart signals and process capability indices. A generator specified the observations, computed the reference fields and rendered the Chinese question. For example, in single sampling, a claim of acceptance is true when the observed count x satisfies x ≤ Ac, where Ac is the acceptance number. For a pair, we retained the rule and wording and changed the decision-relevant observation to produce opposite reference labels. Each question was presented independently. Every pair therefore contains a failure for any model that gives the same verdict twice.

The two initial training sets each contained 96 label-balanced examples. The paraphrase condition varied the wording of the same underlying cases. The scenario condition also varied workpieces and observations. Each condition used four question forms. Per epoch, total input tokens were 14,608 and 14,610, respectively, and supervised answer tokens were 2,848 and 2,860. The development evaluation included 24 near-scenario questions and 120 questions from new scenarios built with the same template family. We report the latter as 60 pairs.

### 3.2 Training and generation

We initialized adapters from Qwen2.5-3B-Instruct with four-bit NF4 quantization. LoRA applied to all linear layers with rank 16, scaling parameter 32 and dropout 0.05. Training used AdamW, learning rate 0.00007 and zero weight decay. We trained for 48 optimization steps, retained the end-of-second-epoch adapter, and repeated each condition with seeds 42, 43 and 44. A single RTX 5060 Ti ran the experiments.

The original answer format placed the verdict on the first line, the statistic and threshold on the second, and the reason on the third. In the order experiment, we exchanged the first two lines in scenario training and changed the inference instruction to request the corresponding order. We crossed training and inference formats on the development cases. In the independent evaluation, we compared the two matched training–inference configurations and ran the unadapted checkpoint with each instruction. Generation used greedy decoding, batches of 12 and a 96-new-token limit.

### 3.3 Independent questions and scoring

A separate Gemini session received six rule specifications without the old questions or results. It supplied four Chinese templates per rule. Two numerical configurations per template each yielded a positive and negative case, giving 96 questions, or 48 pairs. Inputs, reference decisions and adapter hashes were frozen before evaluation. Independence here refers to question construction. The six rules had been seen in training.

A verdict scored correct when the answer contained an unambiguous standalone verdict matching the reference. Unparseable answers counted as failures. Paired accuracy was the proportion of pairs for which both verdicts were correct. A separate three-field measure required a correct verdict, statistic and threshold in the same response. We averaged over the three matched seeds. To describe variation across the sampled rules, we resampled whole rules with replacement 10,000 times, retaining the same questions and seeds in each comparison. The resulting intervals describe sensitivity to the rule mix. The six-rule independent set contains four related capability indices.

## 4. Results

### 4.1 Scenario variation improved structured answers

On the 120 new-scenario questions, the scenario condition improved three-field correctness by 14.44 percentage points over paraphrasing (descriptive 95% rule-cluster interval 8.33 to 20.56). The corresponding single-verdict gain was 0.83 points (−0.56 to 2.50). Paired accuracy averaged 1.7% under each condition (difference 0.00 points, −2.22 to 2.78). The absolute three-field scores averaged 23.3% for paraphrasing and 37.8% for scenario training. On the 24 near-scenario questions, single-verdict scores averaged 55.6% for paraphrasing and 54.2% for scenario training and reached 62.5% for the unadapted model. Positive predictions in the paraphrase condition numbered 0, 84 and 12 by seed. One indication of the discrepancy was label preference. Across the three scenario adapters, the numbers of positive predictions among 120 balanced items were 0, 118 and 1. Thus all three often gave the same answer on opposite sides of a decision boundary.

### 4.2 Answer order interacted with template choice

| Training format | Inference format | Development pairs correct, seed mean |
|---|---|---:|
| Verdict first | Verdict first | 1.7% |
| Verdict first | Numbers first | 3.9% |
| Numbers first | Verdict first | 7.8% |
| Numbers first | Numbers first | 35.0% |

The largest development-set change occurred when training and inference both placed the numbers first. A new question set then tested whether that observation survived changes in wording and numerical configurations.

| Frozen model and instruction | Independent single-question accuracy | Independent paired accuracy |
|---|---:|---:|
| Unadapted, verdict first | 52.1% | 4.2% |
| Unadapted, numbers first | 43.8% | 12.5% |
| Fine-tuned, verdict first | 53.1% | 6.9% |
| Fine-tuned, numbers first | 53.8% | 11.1% |

Across the three adapters, the matched order change yielded 10 versus 16 correct pair evaluations out of 144 repeated pair evaluations. The difference was 4.17 percentage points (descriptive 95% interval −5.56 to 17.36). On the unadapted checkpoint the corresponding pair counts were 2/48 and 6/48. For numbers-first inference the unadapted model scored 42/96 on all items and 42/71 among those with a parseable verdict. Under the two adapted configurations, single sampling yielded 5/24 versus 13/24 successful pair evaluations, control-chart signals 0/24 versus 0/24, and capability indices 5/96 versus 3/96. These exploratory counts reuse each pair across seeds. All fine-tuned responses had a parseable verdict, and none reached the token limit. The unadapted numbers-first model had 25 unparseable verdicts, so its single-question score also reflects adherence to the output format. In a post hoc comparison restricted to the six overlapping rule categories, the development scores were 1/90 versus 20/90 and the independent scores were 10/144 versus 16/144. Restricting the categories preserves the qualitative reduction in the observed gain. The change in templates and numbers prevents attributing the reduction to wording alone.

### 4.3 What a correct numerical field concealed

In a post hoc audit of the independent responses, both fine-tuned configurations returned the correct statistic and threshold in 73 of 288 responses per condition. Of these responses, 18 gave an explicitly wrong verdict with the verdict-first format and 12 did so with the numbers-first format. These counts refer to the responses selected separately within each condition. They do not establish a causal sequence within generation.

As a qualitative check, we restated the minor-defective criterion from two publicly circulated inspection reports and two supplier examples in Chinese. In report 3268288, five of eleven units were inspected, zero minor nonconforming units were observed and the acceptance number was zero. All eight configurations accepted this criterion. In report 3234769, the entire one-unit lot was inspected. Every configuration reported the observed one and the allowed zero, then accepted the criterion. In the supplier examples, the counts were four against an allowance of seven and zero against an allowance of 21. All three verdict-first adapters accepted both cases, and all three numbers-first adapters rejected both despite reporting the right numbers. These cases illustrate the practical meaning of the scoring discrepancy. Only the former involved sampling from a larger lot. Neither the four documents nor repeated runs on them provide a field estimate of reliability.

## 5. Discussion and limitations

The order manipulation produced a large development-set gain, which might have justified recommending a numbers-first template if evaluation had stopped there. On independently written questions the two adapted configurations differed by six successful pair evaluations across three seeds. The numbers-first model also fell slightly below the unadapted model using the same instruction on paired accuracy. The data support testing answer order as a source of evaluation sensitivity. Evidence for a transferable improvement in decisions remains weak.

For an applied audit, each score corresponds to a different practical question. A correct numerical field can populate a form. A correct verdict on a balanced single item can arise from a constant-label policy. Solving a reversal pair requires the model to issue the correct verdict on both sides of the boundary. Even paired success leaves open whether the model used the intended rule, so disagreements between reasons, numbers and decisions merit inspection.

Only one checkpoint, one training budget and three fixed seeds were studied. The independent corpus tests new formulations of familiar rules, includes related indices, and was generated with machine assistance. Several task values were chosen to exercise a decision boundary without engineering validation. The public reports were available as copies, and their presence in pretraining is unknown. An independent domain expert has yet to validate the item set. The design therefore supports claims about these written tasks. Estimating shop-floor error rates would require a defined sampling frame of operational records.

## 6. Conclusion

In small-data QLoRA adaptation for Chinese quality-control questions, scenario variation improved complete structured answers while paired verdict accuracy remained at 1.7%. Placing the evidence first during training and inference increased paired accuracy on familiar templates, with a smaller and uncertain gain on independently written questions. Reporting field correctness, paired decisions and template transfer together gives a more informative account of what the adapted model learned in this setting.

## Reproducibility and tool use

The project retains the question generators, frozen protocols, reference answers, training scripts, predictions and scoring code. The development experiments are in experiments/scenario-sft, the independent evaluation in experiments/independent-blind and report-level checks in experiments/report-sampling. Public release and archival identifiers are pending. Gemini assisted with independent question writing and protocol review. Codex assisted with implementation, analysis and manuscript preparation. No independent human expert assessment is reported.

## References

1. Ribeiro MT, Wu T, Guestrin C, Singh S. Beyond Accuracy: Behavioral Testing of NLP Models with CheckList. ACL, 2020. https://doi.org/10.18653/v1/2020.acl-main.442
2. Dettmers T, Pagnoni A, Holtzman A, Zettlemoyer L. QLoRA: Efficient Finetuning of Quantized LLMs. 2023. https://arxiv.org/abs/2305.14314
3. Yang A et al. Qwen2.5 Technical Report. 2024. https://arxiv.org/abs/2412.15115

4. Gardner M et al. Evaluating Models' Local Decision Boundaries via Contrast Sets. Findings of EMNLP, 2020. https://doi.org/10.18653/v1/2020.findings-emnlp.117
5. Dua D et al. DROP: A Reading Comprehension Benchmark Requiring Discrete Reasoning Over Paragraphs. NAACL, 2019. https://doi.org/10.18653/v1/N19-1246

## Inspection documents

- Report 3268288, Pilates equipment. [Public report copy](https://shopcdnpro.grainajz.com/category/345655/2633/01d539d97d40b33bdd67c5766d13d8d7/Inspection%20report.pdf).
- Report 3234769, CNC equipment. [Public report copy](https://www.scribd.com/document/946774180/StyleCNC-Inspection).


- VICC supplier examples: [high-pressure cleaner](https://www.vicc.com/wp-content/sample-report/Hardline-electric-product-inspection-sample-report.pdf) and [window lift motor](https://www.vicc.com/wp-content/sample-report/Hardline-auto-part-inspection-sample-report.pdf).


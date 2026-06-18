# 实验报告

## 数据

2026-06-18 通过代理从 Hugging Face 加载作业指定数据集：

- `nlile/24-game`: 1362 条，当前版本全部为 `solvable=True`。
- `test-time-compute/game-of-24`: 1362 条，数字来自 `Puzzles` 字段。
- `data/train.jsonl`: 1362 条训练样本。
- `data/eval.jsonl`: solved rate 最低的 100 条可解难例。
- `data/negative.jsonl`: 0 条，因为当前数据源没有不可解负例。

两套 24 点数据覆盖的数字组合高度重合，因此 eval 记录中保留 `seen_in_train` 字段，报告指标需要说明这一限制。

## 确定性校验

- `python -m unittest discover -s tests`: 通过。
- `python -m compileall src tests`: 通过。
- oracle solver 在 100 条难例上 solved rate 为 100/100。

## GRPO 小实验

这些实验用于验证训练链路和观察早期失败模式，不作为最终充分训练结果。

| 实验 | 训练样本 | eval 样本 | step | eval reward | format reward | correctness reward | 独立 solved rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke | 8 | 4 | 1 | 0.0875 | 0.0250 | 0.0000 | 未单独评估 |
| step20 | 160 | 20 | 20 | 0.0738 | 0.0150 | 0.0000 | 0/20 |
| step20-promptv2 | 160 | 20 | 20 | 0.0963 | 0.0275 | 0.0000 | 0/20 |

## SFT Warmup

为降低纯 GRPO 的格式探索难度，额外构造了 SFT warmup 数据：

- 从 `nlile/24-game` 的 `solutions` 字段抽取参考解。
- 将 `×`、`÷`、`−` 归一化为 `*`、`/`、`-`。
- 用同一个 `validate_expression` 校验参考解，只保留合法表达式。
- 生成 `data/sft_train.jsonl` 1262 条和 `data/sft_eval.jsonl` 100 条。

SFT 实验结果：

| 实验 | 训练样本 | step | eval token accuracy | 难例 solved rate | 主要现象 |
| --- | ---: | ---: | ---: | ---: | --- |
| sft-step20 | 160 | 20 | 0.7951 | 0/20 | 学会标签格式，但算式通常不等于 24 |
| sft-full | 1262 | 158 | 0.9590 | 0/100 | 输出格式稳定，但生成的是模式化错误表达式 |
| sft-full-v2 | 1262 | 158 | 0.9296 | 0/100 | completion-only loss 触发 tokenizer mismatch 警告，独立评估输出空答案 |
| sft-strong | 1262 | 790 | 0.9644 | 0/100 | 5 epoch 后格式更稳，但贪心答案仍算错 |

## SFT 后继续 GRPO

使用 `sft-strong` checkpoint 初始化 GRPO，先跑 400 条训练样本、50 step：

| 实验 | 训练样本 | eval 样本 | step | eval reward | correctness reward | solved@1 | solved@4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| grpo-from-sft-step50 | 400 | 20 | 50 | 0.2950 | 0.0000 | 1/100 | 16/100 |
| grpo-from-sft-shaped-step50 | 400 | 20 | 50 | 0.6625 | 0.0125 | 1/100 | 17/100 |
| grpo-shaped-step300 | 1262 | 20 | 300 | 0.6931 | 0.0125 | 1/100 | 18/100 |

训练窗口中 correctness reward 曾出现非零值，说明 verifier 奖励能被模型采到。独立评估中，贪心 `solved@1` 只有 1%，但采样 4 次的 `solved@4` 达到 16%，说明模型已经能生成部分正确表达式，只是策略还不稳定。

进一步加入两项部分奖励：合法表达式奖励和接近 24 的距离奖励。50-step 小实验把 trainer eval correctness 从 0 提到 0.0125，`solved@4` 从 16/100 小幅到 17/100，说明方向有效但训练长度仍明显不足。

继续从 shaped 50-step checkpoint 训练到 300 step 后，贪心仍是 1/100，`solved@4` 小幅到 18/100；但提高采样预算后效果明显提升：`solved@16=38/100`，`solved@64=52/100`，温度 1.1 下 `solved@64=58/100`，`solved@128=73/100`。进一步提高采样多样性到 temperature 1.3、top-p 1.0 后，单轮 `solved@128` 达到 78/100。合并 7 组采样并用 verifier 选择，候选预算为每题 532 个模型样本，达到 89/100。继续只对未解题追加 adaptive sampling 后达到 100/100，总候选数为 76240，最难题候选预算为 8212。当前主要瓶颈不是 verifier，而是模型单次命中率低。

## 现象分析

旧 prompt 的主要失败模式：

- answer 中包含等号和值，违反只输出表达式的要求。
- 输出多个 `<answer>`，早期校验器会抽到示例答案。
- 数字未按 multiset 精确使用一次。
- 算式合法但结果不是 24。

修正后：

- 校验器改为抽取最后一个 `<answer>`。
- prompt 明确要求 answer 只放表达式，不能放等号、结果值、示例或解释。
- sampled trainer eval 的 format/structure reward 有提升，但 correctness reward 仍为 0。
- promptv2 checkpoint 在贪心和 sampled 独立评估中 solved rate 仍为 0/20，说明 20 step 远远不够。
- SFT full checkpoint 的格式明显稳定，但没有学会精确算术搜索；在 100 条难例上 solved rate 仍为 0。
- SFT v2 的 completion-only masking 在当前 tokenizer 下不可靠，不建议作为后续默认方案。
- 从 SFT checkpoint 继续 GRPO 后，正确答案开始在采样中出现；短训练的主要问题从“完全不会”变成“命中率低、贪心不稳”。
- 部分奖励能让训练指标更连续，但贪心输出仍然弱；当前最好策略是多采样后用 verifier 选正确表达式。

## 结论

当前已经完成作业工程化链路：数据加载、prompt 构造、奖励函数、安全校验、SFT warmup、GRPO 训练、oracle sanity check、checkpoint 保存、多候选评估和实验汇总。300-step shaped GRPO 配合 best-of-128 verifier selection 在 100 条 hard eval 上达到 78/100；合并多组采样后达到 89/100；adaptive sampling 后达到 100/100，但贪心输出仍不稳定。

高质量结论需要诚实区分单次生成和 test-time compute：单纯 SFT 容易学到输出模板和局部算式模式，但没有可靠学会 24 点组合搜索；GRPO 加 verifier 能显著提高采样中的正确表达式覆盖率，adaptive sampling 则把覆盖率推到 100/100。

## 下一步

1. 若继续优化，优先减少达到 100/100 所需的候选数，而不是只增加训练 step。
2. 报告时同时给 `solved@1` 和 `solved@N`，明确区分贪心稳定性和采样可解性。
3. 后续重点是降低 adaptive sampling 的候选预算，并补充更严格的分布外评估；纯 LLM 单次生成目前仍不是瓶颈的最优解。

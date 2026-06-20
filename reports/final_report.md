# 基于强化学习的 24 点游戏求解报告

## 1. 任务目标

本作业要求使用小规模开源模型 `Qwen/Qwen2.5-1.5B-Instruct`，通过 GRPO/RLVR 训练模型求解 24 点游戏。输入为 4 个 1–13 之间的整数，模型需要输出 R1 风格结果：

```text
<think>...</think><answer>...</answer>
```

最终 `<answer>` 内只能包含表达式；表达式只能使用输入数字、`+ - * /` 和括号；每个输入数字必须恰好使用一次；计算结果必须为 24。

本任务适合 RLVR，因为最终答案可以由程序确定性验证，不需要训练额外的神经 reward model。

---

## 2. 当前效果概览

一句话总结：

> **GRPO 没有把 Qwen2.5-1.5B 训练成稳定的 1-shot 24 点求解器，但显著提升了采样候选中出现正确表达式的概率；配合 verifier 做 test-time selection 后效果很好。**

核心结果：

| 指标 | 最好结果 | 说明 |
| --- | ---: | --- |
| greedy / pass@1 | `3/100` | 来自 post-RL SFT/distill；多数 GRPO checkpoint 为 `1/100` |
| sample@16 | `45/100` | correctness-heavy GRPO step50，是当前较好的 sampler |
| sample@128 | 约 `78/100` | 保守报告口径；一个 JSONL run 计数为 `81/100` |
| base best-of-runs | `89/100` | 每题 532 个候选，verifier 选第一个正确答案 |
| adaptive best-of-runs | `100/100` | 总候选 `76,240`，最难题候选预算 `8,212` |
| Countdown bonus zero-shot sample@16 | `12/100` | 24-game checkpoint 迁移到任意 target 任务 |
| Countdown bonus GRPO sample@16 | `16/100` | 50-step target-aware Countdown GRPO 后小幅提升 |

这个结果说明：当前模型更适合作为 **verifier-guided sampler**，而不是稳定的 **greedy 1-shot solver**。

---

## 3. 代码结构

精简后的核心代码结构如下：

| 模块 | 文件 | 作用 |
| --- | --- | --- |
| Verifier / Solver | `src/twenty_four_rl/game24.py` | 抽取 `<answer>`，安全 AST 解析，数字 multiset 校验，`Fraction` 精确计算，oracle solver |
| Prompt | `src/twenty_four_rl/prompts.py` | 构造 R1 风格 prompt；支持 24 点和 Countdown target |
| Reward | `src/twenty_four_rl/rewards.py` | GRPO rule-based reward；支持 shaped / correctness-heavy reward sets |
| 数据 | `src/twenty_four_rl/data.py` | 生成 24-game 与 Countdown JSONL 数据 |
| SFT 数据 | `src/twenty_four_rl/sft_data.py` | 从参考解生成 SFT warmup 数据 |
| SFT 训练 | `src/twenty_four_rl/train_sft.py` | TRL SFTTrainer 入口 |
| GRPO 训练 | `src/twenty_four_rl/train_grpo.py` | TRL GRPOTrainer 入口 |
| 评估 | `src/twenty_four_rl/evaluate.py` | greedy、sample@N、oracle、失败原因统计 |
| 聚合 | `src/twenty_four_rl/aggregate_predictions.py` | best-of-runs / adaptive verifier selection |
| 诊断 | `src/twenty_four_rl/data_diagnostics.py` | train/eval overlap 与 negative set 统计 |
| 蒸馏工具 | `src/twenty_four_rl/build_distill_data.py` | 从 verifier-approved samples 构造 SFT 数据；实验性 |

建议最终提交时突出前 9 个模块，`build_distill_data.py` 作为 pass@1 改进实验工具即可。

---

## 4. 数据

使用作业指定数据集：

| 数据集 | 用途 | 当前实测情况 |
| --- | --- | --- |
| `nlile/24-game` | 训练与 SFT 参考解 | 1362 条，当前快照全部可解 |
| `test-time-compute/game-of-24` | hard eval | 选取 solved-rate 较低的 100 条 |
| `Jiayi-Pan/Countdown-Tasks-3to4` | bonus / OOD | 3–4 个数字，任意 target |

主要数据文件：

| 文件 | 数量 | 说明 |
| --- | ---: | --- |
| `data/train.jsonl` | 1362 | 24-game GRPO 训练样本 |
| `data/sft_train.jsonl` | 1262 | SFT warmup 训练样本 |
| `data/sft_eval.jsonl` | 100 | SFT eval 样本 |
| `data/eval.jsonl` | 100 | 24-game hard eval |
| `data/negative.jsonl` | 0 | 当前数据快照没有不可解负例 |
| `data/countdown_train.jsonl` | 1000 | Countdown bonus train 子集 |
| `data/countdown_eval.jsonl` | 100 | Countdown bonus eval 子集 |

数据限制：

- 当前 `data/eval.jsonl` 的 100 道 hard eval 与训练集数字组合全部重合，因此它是 hard eval，不是严格 OOD 泛化测试。
- 当前 `negative.jsonl` 为空，因此无法从官方数据快照评估不可解题 false-positive。
- Countdown 是 bonus/OOD 扩展，不替代主任务指标。

---

## 5. Verifier 与 Reward 设计

本实验使用程序化 verifier，而不是训练神经 reward model。

Verifier 逻辑：

1. 从最后一个 `<answer>...</answer>` 中抽取表达式。
2. 限制表达式只能包含数字、`+ - * /`、括号和空白。
3. 提取表达式中的整数 token。
4. 检查 token multiset 是否等于输入数字 multiset。
5. 用 Python AST 白名单安全解析表达式，只允许整数常量和二元四则运算。
6. 用 `Fraction` 做精确有理数计算。
7. 判断结果是否等于目标值；24-game 默认 target=24，Countdown 使用每行 `target`。

Reward 分为：

| Reward | 目的 |
| --- | --- |
| format | 鼓励 `<think>` + `<answer>` 格式 |
| structure | 鼓励 `<answer>` 内只放短表达式 |
| expression | 鼓励表达式可解析、可计算 |
| number usage | 鼓励恰好使用输入数字 |
| closeness | 鼓励计算值接近 target |
| correctness | 完全合法且等于 target 才给最高奖励 |

当前代码中 reward 已经 target-aware，因此同一套 verifier 可用于 24 点和 Countdown bonus。

---

## 6. 主任务实验结果

### 6.1 Sanity check

| 检查 | 结果 |
| --- | ---: |
| Oracle on 24-game hard eval | `100/100` |
| Empty negative set evaluation | 成功，`count=0` |
| 单元测试 | 通过 |
| `compileall` | 通过 |

### 6.2 24-game 模型效果

| 方法 | greedy / pass@1 | sample@4 | sample@16 | sample@64 | sample@128 | verifier-guided best-of |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SFT strong | `0/100` | - | - | - | - | - |
| GRPO from SFT 50-step | `1/100` | `16/100` | - | - | - | - |
| Shaped GRPO from SFT 50-step | `1/100` | `17/100` | - | - | - | - |
| Shaped GRPO step300 | `1/100` | `18/100` | `38/100` | `52–58/100` | `73–78/100` | `89/100` base；`100/100` adaptive |
| correctness-heavy GRPO step50 | `1/100` | `18/100` | `45/100` | 未系统评估 | - | - |
| post-RL SFT/distill | `3/100` | - | - | - | - | - |

最重要的结论：

> **GRPO 对 greedy/pass@1 的提升非常有限，但对 sampled candidate distribution 的提升明显。**

### 6.3 Candidate efficiency

| 设置 | solved | 总候选数 | 平均候选/题 | 最难题候选预算 | 备注 |
| --- | ---: | ---: | ---: | ---: | --- |
| base best-of-runs | `89/100` | `53,200` | `532` | `532` | 合并多组 sample 文件 |
| adaptive best-of-runs | `100/100` | `76,240` | `762.4` | `8,212` | 只对未解题追加采样 |

`100/100` 不是 greedy accuracy，而是 test-time sampling + verifier selection 的结果。

---

## 7. Pass@1 改进尝试

为了提高 1-shot，尝试了多种路线：

| 尝试 | greedy result |
| --- | ---: |
| pass1 strong GRPO step200b | `0/100` |
| pass1 8GPU GRPO step100 | `0/100` |
| pass1 8GPU GRPO step200 | `0/100` |
| pass1 8GPU GRPO step300 | `0/100` |
| correctness-heavy GRPO step50 | `1/100` |
| correctness-heavy GRPO step200 | `1/100` |
| verified-only distill 400 | `0/100` |
| mixed verified 400 top1 | `2/100` |
| mixed verified 400 top3 | `0/100` |
| full mixed verified top1 | `1/100` |
| old post-RL SFT/distill | `3/100` |

结论：当前小模型和训练策略下，1-shot 很难通过继续 GRPO/SFT/distillation 稳定拉升。模型主要学会了格式、数字使用和合法表达式，但没有稳定学会精确组合搜索。

---

## 8. Countdown bonus

作业 bonus 要求可参考 Countdown 风格任务。本项目已将 prompt、verifier、reward 和 evaluation 推广到：

```text
3-4 个数字 + 任意 target
```

数据：`Jiayi-Pan/Countdown-Tasks-3to4`。

当前小规模 bonus 结果：

| 设置 | greedy | sample@4 | sample@16 |
| --- | ---: | ---: | ---: |
| Oracle | `100/100` | - | - |
| 24-game checkpoint zero-shot Countdown | `1/100` | `4/100` | `12/100` |
| Countdown GRPO step50 | `1/100` | `2/100` | `16/100` |

结论：Countdown target-aware pipeline 已完成；24-game checkpoint 对 arbitrary target 有弱迁移；短 GRPO 对 sample@16 有小幅提升，但没有改善 greedy。

---

## 9. 定性分析

成功样例：

| 数字 | 表达式 | 结果 |
| --- | --- | --- |
| `[2, 4, 7, 7]` | `(7+7)*2-4` | 24 |
| `[2, 4, 7, 12]` | `(7+2)*4-12` | 24 |
| `[2, 2, 10, 11]` | `(11*2-10)*2` | 24 |
| `[9, 11, 12, 13]` | `(13+9)*12/11` | 24 |

失败模式：

| 输入 | 输出 | 问题 |
| --- | --- | --- |
| `[2, 3, 5, 12]` | `(12-5)*3*2` | 值为 42，不是 24 |
| `[2, 3, 5, 12]` | `(5+3)*12/2` | 值为 48，不是 24 |
| `[6, 6, 4, 4]` | `6*4` | 值为 24，但没有恰好使用四个数字 |

主要错误不是格式，而是精确算式搜索失败。

---

## 10. 复现命令

### 数据准备

```bash
PYTHONPATH=src python3 -m twenty_four_rl.data
PYTHONPATH=src python3 -m twenty_four_rl.sft_data
PYTHONPATH=src python3 -m twenty_four_rl.data \
  --dataset countdown \
  --train-out data/countdown_train.jsonl \
  --eval-out data/countdown_eval.jsonl \
  --negative-out data/countdown_negative.jsonl \
  --limit-train-rows 1000 \
  --limit-eval-rows 100
```

### 训练

```bash
PYTHONPATH=src python3 -m twenty_four_rl.train_sft \
  --config configs/sft_qwen_1_5b_strong.yaml

PYTHONPATH=src python3 -m twenty_four_rl.train_grpo \
  --config configs/grpo_qwen_1_5b_shaped_continue.yaml \
  --max-steps 300 \
  --limit-train-rows 1262 \
  --limit-eval-rows 20 \
  --output-dir outputs/grpo-qwen-24-shaped-continue
```

### 评估

```bash
PYTHONPATH=src python3 -m twenty_four_rl.evaluate \
  --input data/eval.jsonl \
  --model outputs/grpo-qwen-24-shaped-step300/checkpoint-final \
  --batch-size 4 \
  --max-new-tokens 128
```

```bash
PYTHONPATH=src python3 -m twenty_four_rl.evaluate \
  --input data/eval.jsonl \
  --model outputs/grpo-qwen-24-shaped-step300/checkpoint-final \
  --do-sample \
  --temperature 1.1 \
  --top-p 0.95 \
  --num-samples 128 \
  --batch-size 4 \
  --max-new-tokens 128
```

### Countdown bonus eval

```bash
PYTHONPATH=src python3 -m twenty_four_rl.evaluate \
  --input data/countdown_eval.jsonl \
  --oracle-solver

PYTHONPATH=src python3 -m twenty_four_rl.evaluate \
  --input data/countdown_eval.jsonl \
  --model outputs/grpo-qwen-24-correctness-heavy-step50/checkpoint-final \
  --do-sample \
  --num-samples 16 \
  --temperature 1.1 \
  --top-p 0.95
```

### 验证

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
PYTHONPATH=src python3 -m compileall src tests
```

---

## 11. 结论

本项目完成了 24 点 GRPO/RLVR 主任务和 Countdown bonus 扩展。核心技术包括 R1 prompt、确定性 verifier、target-aware reward、SFT warmup、GRPO 训练、greedy/sample@N 评估、best-of-runs 聚合和 Countdown OOD evaluation。

实验结论需要准确表述：

- SFT 和 GRPO 没有让 Qwen2.5-1.5B 成为稳定的 1-shot 24 点求解器。
- 多种 pass@1 改进尝试均未显著超过 `3/100`。
- GRPO 的主要收益是提升采样候选质量：sample@N 和 verifier-guided selection 效果明显。
- 在 test-time compute 足够高时，adaptive verifier selection 可以达到 `100/100`，但这不是 greedy 能力。
- Countdown bonus 表明该 verifier/reward 框架可以推广到 arbitrary target，但短训练结果仍然有限。

最终定位：**当前系统是一个有效的 RLVR + verifier-guided arithmetic sampler，而不是可靠的一次生成求解器。**

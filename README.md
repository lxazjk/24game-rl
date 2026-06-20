# 基于 GRPO 的 24 点游戏求解

本仓库对应课程选题“基于强化学习的 24 点游戏求解”。目标是使用可验证奖励（RLVR）训练 `Qwen/Qwen2.5-1.5B-Instruct`，让模型以：

```text
<think>...</think><answer>...</answer>
```

格式输出合法 24 点表达式。项目还实现了 Countdown bonus：3–4 个数字、任意 target 的算式求解。

## 当前效果

一句话总结：

> **GRPO 没有显著提升 greedy/pass@1，但显著提升了采样候选质量；模型更适合作为 verifier-guided sampler，而不是稳定的一次生成求解器。**

| 指标 | 最好结果 | 说明 |
| --- | ---: | --- |
| greedy/pass@1 | `3/100` | post-RL SFT/distill；多数 GRPO checkpoint 为 `1/100` |
| sample@16 | `45/100` | correctness-heavy GRPO step50 |
| sample@128 | 约 `78/100` | 保守报告口径 |
| best-of-runs | `89/100` | 多组采样 + verifier selection |
| adaptive best-of-runs | `100/100` | 总候选 `76,240`，最大单题预算 `8,212` |
| Countdown zero-shot sample@16 | `12/100` | 24-game checkpoint 迁移到 arbitrary target |
| Countdown GRPO sample@16 | `16/100` | 50-step Countdown GRPO 后小幅提升 |

注意：`100/100` 是 test-time sampling + verifier selection，不是 greedy/pass@1。

## 作业要求对齐

- 模型：`Qwen/Qwen2.5-1.5B-Instruct`。
- 方法：TRL `GRPOTrainer`，程序 verifier 作为 reward，不训练神经 reward model。
- 输入：4 个 1–13 整数，或 bonus 中 3–4 个数 + 任意 target。
- 输出：R1 风格 `<think>...</think><answer>...</answer>`。
- Verifier：检查 `<answer>`、合法字符、数字 multiset、安全 AST、精确 `Fraction` 计算、是否等于 target。
- 评估：greedy/pass@1、sample@N、best-of-runs、adaptive verifier selection、定性错误分析。
- Bonus：支持 `Jiayi-Pan/Countdown-Tasks-3to4` 的 target-aware 数据、reward 和 evaluation。

## 精简代码结构

```text
src/twenty_four_rl/
  game24.py              # verifier、safe AST、Fraction 计算、oracle solver
  prompts.py             # R1 prompt，支持 target-aware Countdown
  rewards.py             # GRPO reward functions，支持 shaped / correctness-heavy
  data.py                # 24-game 和 Countdown JSONL 数据生成
  sft_data.py            # SFT warmup 数据生成
  train_sft.py           # SFTTrainer 入口
  train_grpo.py          # GRPOTrainer 入口
  evaluate.py            # greedy/sample@N/oracle 评估
  aggregate_predictions.py  # best-of-runs/adaptive 聚合
  data_diagnostics.py       # 数据 overlap / negative 诊断
  summarize_run.py          # trainer_state 摘要
  build_distill_data.py     # 实验性：verified-sample distillation 数据构造
```

建议主报告只突出前 10 个文件；`build_distill_data.py` 作为 pass@1 改进实验工具即可。

## 推荐保留的配置

```text
configs/
  sft_qwen_1_5b.yaml
  sft_qwen_1_5b_strong.yaml
  grpo_qwen_1_5b.yaml
  grpo_qwen_1_5b_from_sft.yaml
  grpo_qwen_1_5b_shaped_continue.yaml
  grpo_qwen_1_5b_countdown_from_24.yaml
```

其他 pass@1、correctness-heavy、verified-distill 配置建议作为实验附录或移到 `configs/experiments/`，避免主目录显得混乱。

## 数据准备

主任务：

```bash
PYTHONPATH=src python3 -m twenty_four_rl.data \
  --train-out data/train.jsonl \
  --eval-out data/eval.jsonl \
  --negative-out data/negative.jsonl

PYTHONPATH=src python3 -m twenty_four_rl.sft_data \
  --train-out data/sft_train.jsonl \
  --eval-out data/sft_eval.jsonl
```

Countdown bonus：

```bash
PYTHONPATH=src python3 -m twenty_four_rl.data \
  --dataset countdown \
  --train-out data/countdown_train.jsonl \
  --eval-out data/countdown_eval.jsonl \
  --negative-out data/countdown_negative.jsonl \
  --limit-train-rows 1000 \
  --limit-eval-rows 100
```

当前数据限制：

- `data/eval.jsonl` 是 low solved-rate hard eval，但数字组合与 train 100% 重合；报告中不声称严格 OOD 泛化。
- 当前 `negative.jsonl` 为空，因为当前 Hugging Face 数据快照没有 `solvable=False` 样本。

## 训练

SFT warmup：

```bash
PYTHONPATH=src python3 -m twenty_four_rl.train_sft \
  --config configs/sft_qwen_1_5b_strong.yaml
```

GRPO 主实验：

```bash
PYTHONPATH=src python3 -m twenty_four_rl.train_grpo \
  --config configs/grpo_qwen_1_5b_shaped_continue.yaml \
  --max-steps 300 \
  --limit-train-rows 1262 \
  --limit-eval-rows 20 \
  --output-dir outputs/grpo-qwen-24-shaped-continue
```

Countdown bonus 短训：

```bash
PYTHONPATH=src python3 -m twenty_four_rl.train_grpo \
  --config configs/grpo_qwen_1_5b_countdown_from_24.yaml \
  --max-steps 50 \
  --limit-train-rows 1000 \
  --limit-eval-rows 100 \
  --output-dir outputs/grpo-qwen-countdown-from-24-step50
```

## 评估

Oracle sanity check：

```bash
PYTHONPATH=src python3 -m twenty_four_rl.evaluate \
  --input data/eval.jsonl \
  --oracle-solver \
  --summary-output reports/oracle_eval_summary.json
```

Greedy：

```bash
PYTHONPATH=src python3 -m twenty_four_rl.evaluate \
  --input data/eval.jsonl \
  --model outputs/grpo-qwen-24-shaped-step300/checkpoint-final \
  --summary-output reports/grpo_shaped_step300_eval_summary.json \
  --batch-size 4 \
  --max-new-tokens 128
```

Sample@N：

```bash
PYTHONPATH=src python3 -m twenty_four_rl.evaluate \
  --input data/eval.jsonl \
  --model outputs/grpo-qwen-24-shaped-step300/checkpoint-final \
  --do-sample \
  --temperature 1.1 \
  --top-p 0.95 \
  --num-samples 128 \
  --summary-output reports/grpo_shaped_step300_sample128_summary.json \
  --batch-size 4 \
  --max-new-tokens 128
```

Countdown bonus：

```bash
PYTHONPATH=src python3 -m twenty_four_rl.evaluate \
  --input data/countdown_eval.jsonl \
  --oracle-solver \
  --summary-output reports/countdown_oracle_eval_summary.json

PYTHONPATH=src python3 -m twenty_four_rl.evaluate \
  --input data/countdown_eval.jsonl \
  --model outputs/grpo-qwen-24-correctness-heavy-step50/checkpoint-final \
  --do-sample \
  --num-samples 16 \
  --temperature 1.1 \
  --top-p 0.95 \
  --summary-output reports/countdown_from_24_sample16_summary.json
```

## 验证

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
PYTHONPATH=src python3 -m compileall src tests
```

## 结果文件建议

建议保留：

```text
reports/final_report.md
reports/progress_log.md
reports/data_diagnostics_summary.json
reports/oracle_eval_summary.json
reports/negative_oracle_eval_summary.json
reports/grpo_shaped_step300_summary.json
reports/grpo_shaped_step300_best_of_runs_summary.json
reports/grpo_shaped_step300_adaptive_best_of_runs_summary.json
reports/countdown_oracle_eval_summary.json
reports/countdown_from_24_sample16_summary.json
reports/countdown_grpo_step50_sample16_summary.json
```

建议归档或不提交：

```text
reports/*predictions.jsonl
reports/*stdout.log
reports/*.pid
outputs/
data/verified_sft_*.jsonl
data/mixed_sft_verified_*.jsonl
```

这些文件用于实验追踪，但会让提交显得很乱；最终报告引用 summary 即可。

## 最终结论

本项目完成了 24 点 GRPO/RLVR 主任务和 Countdown bonus。实验表明：

- SFT/GRPO 未能让 `Qwen2.5-1.5B` 成为稳定的 1-shot 24 点求解器；最好 greedy 只有 `3/100`。
- GRPO 明显提高了采样候选质量，配合 verifier selection 可以从 sample@N 获得高成功率。
- best-of-runs 和 adaptive sampling 能分别达到 `89/100` 和 `100/100`，但代价是较高 test-time compute。
- Countdown bonus 证明了 verifier/reward/evaluate 可以推广到 arbitrary target，但短训结果仍然有限。

因此，最终定位是：**一个有效的 RLVR + verifier-guided arithmetic sampler，而不是可靠的一次生成求解器。**

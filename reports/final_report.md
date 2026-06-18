# 基于强化学习的 24 点游戏求解报告

## 1. 任务与目标

本作业要求使用 `Qwen/Qwen2.5-1.5B-Instruct` 作为小规模开源语言模型，通过 GRPO/RLVR 训练模型求解 24 点。输入为 4 个 1-13 之间的整数，模型需要在 `<think>...</think><answer>...</answer>` 格式中输出一个四则表达式；最终 `<answer>` 内只能包含表达式，且每个输入数字必须恰好使用一次，计算结果为 24。

这个任务适合可验证奖励：表达式是否合法、数字是否用对、结果是否等于 24，都可以由程序确定性判断，不需要人工偏好标注或额外奖励模型。

## 2. 数据

本仓库只使用作业文档指定的数据集：

| 数据集 | 用途 | 当前实测 |
| --- | --- | --- |
| `nlile/24-game` | 训练集与 SFT 参考解 | 1362 条，当前版本全部 `solvable=True` |
| `test-time-compute/game-of-24` | hard eval | 1362 条，从 `Puzzles` 字段解析数字 |

生成文件：

| 文件 | 数量 | 说明 |
| --- | ---: | --- |
| `data/train.jsonl` | 1362 | GRPO 训练样本 |
| `data/sft_train.jsonl` | 1262 | 从参考解构造的 SFT 训练样本 |
| `data/sft_eval.jsonl` | 100 | SFT held-out 样本 |
| `data/eval.jsonl` | 100 | solved rate 最低的可解 hard eval |
| `data/negative.jsonl` | 0 | 当前数据版本没有不可解负例 |

两套 24 点数据的数字组合高度重合，因此 `data/eval.jsonl` 中保留 `seen_in_train` 字段，评估结论需要说明这个限制。

## 3. 方法

实现分为四层：

1. `game24.py`：抽取 `<answer>`、校验字符集合、校验数字 multiset、用 AST 和 `Fraction` 精确计算表达式。
2. `rewards.py`：为 GRPO 提供格式奖励、结构奖励和正确性奖励。
3. `train_sft.py`：先用参考解做 SFT warmup，降低模型学习输出格式的难度。
4. `train_grpo.py`：从 base model 或 SFT checkpoint 启动 GRPO，用程序奖励强化正确表达式。

评估使用 `evaluate.py`。除贪心 `solved@1` 外，新增 `--num-samples` 支持 `solved@N`：每题采样 N 个答案，只要有一个通过 verifier，就计为成功。

## 4. 实验设置

主要实验配置：

| 实验 | 初始模型 | 训练数据 | 训练规模 |
| --- | --- | --- | --- |
| SFT strong | `Qwen/Qwen2.5-1.5B-Instruct` | `data/sft_train.jsonl` | 5 epoch |
| GRPO from SFT | `outputs/sft-qwen-24-strong/checkpoint-final` | 前 400 条 `data/train.jsonl` | 50 step |
| Shaped GRPO from SFT | `outputs/sft-qwen-24-strong/checkpoint-final` | 前 400 条 `data/train.jsonl` | 50 step |
| Shaped GRPO continue | shaped 50-step checkpoint | 1262 条 `data/train.jsonl` | 300 step |

为避免本地模型目录过大，配置默认不保存中间 checkpoint，只在训练结束保存 `checkpoint-final`。当前 `outputs/` 只保留两个最终 checkpoint，占用约 5.8G。

## 5. 结果

确定性校验：

| 检查 | 结果 |
| --- | --- |
| 单元测试 | 通过 |
| `compileall` | 通过 |
| oracle solver on hard eval | 100/100 |

模型结果：

| 模型/实验 | hard eval solved@1 | hard eval solved@4 | 说明 |
| --- | ---: | ---: | --- |
| SFT strong | 0/100 | 未评估 | 格式稳定，但表达式常算错 |
| GRPO from SFT 50-step | 1/100 | 16/100 | 采样中开始出现正确表达式 |
| Shaped GRPO from SFT 50-step | 1/100 | 17/100 | 部分奖励带来小幅提升 |
| Shaped GRPO 300-step | 1/100 | 18/100 | 贪心仍弱，低采样提升有限 |

更高采样预算下：

| 解码设置 | hard eval solved rate |
| --- | ---: |
| `solved@16`, temperature 0.9 | 38/100 |
| `solved@64`, temperature 0.9 | 52/100 |
| `solved@64`, temperature 1.1 | 58/100 |
| `solved@128`, temperature 1.1 | 73/100 |
| `solved@128`, temperature 1.3, top-p 1.0 | 78/100 |
| best-of-runs verifier selection, 532 candidates | 89/100 |
| adaptive best-of-runs verifier selection | 100/100 |

base best-of-runs 指标来自 7 组独立模型采样文件，逐题用同一个确定性 verifier 选择第一个正确表达式。adaptive best-of-runs 只对仍未解出的 11 题继续追加模型采样，最终总候选数为 76240，最难题候选预算为 8212。它衡量的是 test-time compute 加 verifier 的上限，不等同于贪心单次生成能力，也没有使用 oracle fallback。

GRPO-from-SFT 的 trainer summary：

| 指标 | 值 |
| --- | ---: |
| step | 300 |
| eval reward | 0.6931 |
| eval format reward mean | 0.1975 |
| eval structure reward mean | 0.0988 |
| eval expression reward mean | 0.1975 |
| eval closeness reward mean | 0.1868 |
| eval correctness reward mean | 0.0125 |

## 6. 定性案例

成功案例：

| 数字 | 输出表达式 | 结果 |
| --- | --- | ---: |
| `[2, 4, 7, 7]` | `(7+7)*2-4` | 24 |
| `[2, 4, 7, 12]` | `(7+2)*4-12` | 24 |
| `[2, 2, 10, 11]` | `(11*2-10)*2` | 24 |
| `[9, 11, 12, 13]` | `(13+9)*12/11` | 24 |

失败案例：

| 数字 | 输出表达式 | 问题 |
| --- | --- | --- |
| `[2, 3, 5, 12]` | `(12-5)*3*2` | 结果为 42 |
| `[2, 3, 5, 12]` | `(5+3)*12/2` | 结果为 48 |

现象上，SFT 能明显改善标签格式，但不足以学会组合搜索。GRPO 从 SFT checkpoint 起步后，采样答案里开始出现正确表达式，说明 verifier 奖励有效；加入部分奖励和继续训练后，best-of-N 指标明显提升，但贪心输出仍然不稳定。

## 7. 复现命令

准备数据：

```bash
export http_proxy=http://proxy.example.com:8118
export https_proxy=http://proxy.example.com:8118
export no_proxy=.example.com

PYTHONPATH=src python3 -m twenty_four_rl.data
PYTHONPATH=src python3 -m twenty_four_rl.sft_data
```

训练当前最好 checkpoint 的后续实验：

```bash
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=src python3 -m twenty_four_rl.train_sft \
  --config configs/sft_qwen_1_5b_strong.yaml

CUDA_VISIBLE_DEVICES=2 PYTHONPATH=src python3 -m twenty_four_rl.train_grpo \
  --config configs/grpo_qwen_1_5b_shaped_continue.yaml \
  --max-steps 300 \
  --limit-train-rows 1262 \
  --limit-eval-rows 20 \
  --output-dir outputs/grpo-qwen-24-shaped-continue
```

评估：

```bash
PYTHONPATH=src python3 -m twenty_four_rl.evaluate \
  --input data/eval.jsonl \
  --model outputs/grpo-qwen-24-shaped-step300/checkpoint-final \
  --output reports/grpo_shaped_step300_eval_predictions.jsonl \
  --batch-size 4 \
  --max-new-tokens 128

PYTHONPATH=src python3 -m twenty_four_rl.evaluate \
  --input data/eval.jsonl \
  --model outputs/grpo-qwen-24-shaped-step300/checkpoint-final \
  --output reports/grpo_shaped_step300_sample128_t11_eval_predictions.jsonl \
  --batch-size 4 \
  --max-new-tokens 128 \
  --do-sample \
  --temperature 1.1 \
  --top-p 0.95 \
  --num-samples 128
```

验证：

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
PYTHONPATH=src python3 -m compileall src tests
```

## 8. 结论与后续

本仓库已经完成作业所需的工程链路：数据处理、R1 prompt、可验证奖励、安全表达式校验、SFT warmup、GRPO 训练、oracle sanity check、模型评估、`solved@N` 评估、过程记录和背景流程总结。

当前结果说明方法可行：纯贪心仍弱，但 shaped GRPO 加 verifier-guided best-of-128 在 hard eval 上达到 78/100；合并多组采样后达到 89/100；adaptive sampling 加 verifier selection 达到 100/100。后续重点应转向降低候选预算、提升贪心稳定性和补充更严格的分布外评估。

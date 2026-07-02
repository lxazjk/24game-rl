# 24-game-rl

用 GRPO 训练 `Qwen2.5-1.5B-Instruct` 求解 24 点。奖励来自一个确定性的程序 verifier（RLVR），不训练神经奖励模型。模型按 R1 风格输出：

```text
<think>推理</think><answer>3*8</answer>
```

除主任务外，同一套 verifier / reward / 评估也支持 Countdown（3–4 个数、任意目标值）。

## 结果

在 100 道难题（人类 solved rate 最低的一批）上评测：

| 指标 | 结果 | 备注 |
| --- | ---: | --- |
| greedy / pass@1 | 3/100 | 多数 GRPO checkpoint 只有 1/100 |
| sample@16 | 45/100 | correctness-heavy GRPO |
| sample@128 | ~78/100 | |
| best-of-runs | 89/100 | 多组采样 + verifier 筛选 |
| adaptive best-of-runs | 100/100 | 总候选 76,240，单题最多 8,212 |
| Countdown sample@16 | 12→16/100 | zero-shot 迁移 → 50 步 GRPO |

结论：GRPO 几乎没有提升一次生成（greedy）的正确率，但显著提升了采样候选的质量。因此这个模型的合理用法是**采样 + verifier 筛选**，而不是当作稳定的一次求解器。`100/100` 是 test-time sampling 的结果，不是 pass@1。

## 安装

```bash
pip install -e .
```

editable 安装后 `core`、`data`、`train`、`eval` 直接可导入，命令无需再设 `PYTHONPATH`。主要依赖：`transformers`、`trl`、`datasets`、`torch`。

## 用法

准备数据：

```bash
# 24 点
python -m data.build \
  --train-out data/train.jsonl --eval-out data/eval.jsonl --negative-out data/negative.jsonl
python -m data.sft \
  --train-out data/sft_train.jsonl --eval-out data/sft_eval.jsonl

# Countdown
python -m data.build --dataset countdown \
  --train-out data/countdown_train.jsonl --eval-out data/countdown_eval.jsonl \
  --negative-out data/countdown_negative.jsonl --limit-train-rows 1000 --limit-eval-rows 100
```

训练（先可选 SFT warmup，再 GRPO）：

```bash
python -m train.sft \
  --config configs/sft_qwen_1_5b_strong.yaml

python -m train.grpo \
  --config configs/grpo_qwen_1_5b_shaped_continue.yaml \
  --output-dir outputs/grpo-shaped
```

评估：

```bash
# oracle sanity check（确认题目可解）
python -m eval.evaluate --input data/eval.jsonl --oracle-solver

# greedy / pass@1
python -m eval.evaluate \
  --input data/eval.jsonl --model outputs/grpo-shaped/checkpoint-final

# sample@N
python -m eval.evaluate \
  --input data/eval.jsonl --model outputs/grpo-shaped/checkpoint-final \
  --do-sample --num-samples 128 --temperature 1.1 --top-p 0.95
```

`eval.aggregate` 把多次采样的预测合并，做 best-of-runs / adaptive 筛选。

测试：

```bash
python -m unittest discover -s tests
```

## 目录结构

```text
src/
  core/
    game24.py       verifier + oracle solver（safe AST，Fraction 精确计算）
    prompts.py      R1 prompt，支持任意 target
    rewards.py      GRPO reward，含 shaped / correctness-heavy 两套
  data/
    build.py        生成 24 点与 Countdown 数据
    sft.py          生成 SFT warmup 数据
    diagnostics.py  train/eval overlap 与 negative set 统计
  train/
    grpo.py         TRL GRPOTrainer 入口
    sft.py          TRL SFTTrainer 入口
  eval/
    evaluate.py     greedy / sample@N / oracle 评估
    aggregate.py    best-of-runs / adaptive 聚合
configs/            SFT 与 GRPO 的 YAML 配置
reports/            final_report.md 与关键评估结果
tests/              verifier / reward / 评估的单元测试
```

## 局限

- eval 集是 solved rate 最低的难例，但数字组合与训练集重合，因此不声称严格的 OOD 泛化。
- 当前数据源没有 `solvable=False` 样本，`negative.jsonl` 为空。
- 高成功率依赖较大的 test-time compute（大量采样）。

详细实验记录见 [`reports/final_report.md`](reports/final_report.md)。

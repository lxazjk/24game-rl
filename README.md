# 24-game-rl

使用 GRPO 训练 `Qwen2.5-1.5B-Instruct` 求解 24 点。奖励由一个确定性的程序 verifier 给出（RLVR），不使用神经奖励模型。模型按 R1 风格输出：

```text
<think>推理过程</think><answer>(3+3)*(8/2)</answer>
```

verifier 从 `<answer>` 中提取算式，依次校验：字符是否仅含 `+ - * / ( )` 与数字、是否恰好用到题目给定的数字（multiset 比较）、能否经安全 AST 求值，以及结果是否等于目标值（用 `Fraction` 精确计算，避免浮点误差）。同一套 verifier、reward 与评估流程也适用于 Countdown 任务（3–4 个数、任意目标值）。

## 结果

评测集为人类通过率最低的 100 道题。

| 指标 | 结果 | 说明 |
| --- | ---: | --- |
| greedy / pass@1 | 3/100 | 多数 GRPO checkpoint 为 1/100 |
| sample@16 | 45/100 | correctness-heavy GRPO |
| sample@128 | ≈78/100 | |
| best-of-runs | 89/100 | 多组采样后由 verifier 筛选 |
| adaptive best-of-runs | 100/100 | 总候选 76,240，单题最大预算 8,212 |
| Countdown sample@16 | 12→16/100 | zero-shot 迁移，再经 50 步 GRPO |

GRPO 未能显著提升一次生成（greedy / pass@1）的正确率，但明显提升了采样候选的质量：在多次采样后由 verifier 选择合法解，成功率可大幅提高。因此该模型更适合作为 verifier 引导的采样器，而非稳定的单次求解器。表中 100/100 为 test-time 反复采样加筛选的结果，不等同于 pass@1。

## 安装

```bash
pip install -e .
```

## 用法

生成数据：

```bash
python -m data.build --train-out data/train.jsonl --eval-out data/eval.jsonl --negative-out data/negative.jsonl
python -m data.sft   --train-out data/sft_train.jsonl --eval-out data/sft_eval.jsonl

# Countdown
python -m data.build --dataset countdown \
  --train-out data/countdown_train.jsonl --eval-out data/countdown_eval.jsonl \
  --negative-out data/countdown_negative.jsonl --limit-train-rows 1000 --limit-eval-rows 100
```

训练（可选先做 SFT warmup，再进行 GRPO）：

```bash
python -m train.sft  --config configs/sft_qwen_1_5b_strong.yaml
python -m train.grpo --config configs/grpo_qwen_1_5b_shaped_continue.yaml --output-dir outputs/grpo-shaped
```

评估。先用 oracle 确认题目可解，再评测模型的 greedy 与 sample@N：

```bash
python -m eval.evaluate --input data/eval.jsonl --oracle-solver
python -m eval.evaluate --input data/eval.jsonl --model outputs/grpo-shaped/checkpoint-final
python -m eval.evaluate --input data/eval.jsonl --model outputs/grpo-shaped/checkpoint-final \
  --do-sample --num-samples 128 --temperature 1.1 --top-p 0.95
```

`eval.aggregate` 将多次采样的预测合并，执行 best-of-runs 与按难度分配预算的 adaptive 筛选。运行测试：`python -m unittest discover -s tests`。

## 目录结构

```text
src/
  core/
    game24.py       verifier 与 oracle solver（安全 AST、Fraction 精确计算）
    prompts.py      R1 prompt，支持任意目标值
    rewards.py      GRPO 奖励函数，含 shaped 与 correctness-heavy 两套
  data/
    build.py        生成 24 点与 Countdown 数据
    sft.py          生成 SFT warmup 数据
    diagnostics.py  统计 train/eval 重叠与 negative set
  train/
    grpo.py         TRL GRPOTrainer 入口
    sft.py          TRL SFTTrainer 入口
  eval/
    evaluate.py     greedy / sample@N / oracle 评估
    aggregate.py    best-of-runs / adaptive 聚合
configs/            SFT 与 GRPO 的 YAML 配置
reports/            final_report.md 与关键评估结果
tests/              verifier、reward 与评估的单元测试
```

`game24.py` 是核心：训练时它提供奖励，评估时它作为筛选候选的裁判。

## 局限

- 评测集选取人类通过率最低的 100 道题，但其数字组合在训练集中出现过，因此不能据此声称 OOD 泛化。
- 所用数据源不含无解样本，`negative.jsonl` 为空。
- 较高的成功率依赖于 test-time 的大量采样，并非模型单次生成的水平。

完整实验记录见 [`reports/final_report.md`](reports/final_report.md)。

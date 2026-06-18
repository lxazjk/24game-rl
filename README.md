# 基于 GRPO 的 24 点游戏求解

本仓库对应课程选题“基于强化学习的 24 点游戏求解”。目标是用可验证奖励（RLVR）训练
`Qwen/Qwen2.5-1.5B-Instruct`，让模型以
`<think>...</think><answer>...</answer>` 格式输出 24 点表达式。

## 作业要求对齐

- 任务：给定 4 个 1-13 的整数，输出只使用这些数字一次且结果为 24 的算式。
- 方法：使用 TRL `GRPOTrainer` 实现 GRPO，奖励由程序判定，无需奖励模型。
- 训练集：优先使用 `nlile/24-game` 中 `solvable=True` 的样本。
- 测试集：使用 `test-time-compute/game-of-24` 的低 solved-rate 难例，并标记是否出现在训练组合中。
- 负例检查：若当前数据版本提供 `solvable=False` 样本，则用于观察模型是否瞎编。
- 输出：训练配置、奖励曲线/准确率记录、定量评估、定性案例与过程总结。

## 仓库结构

```text
src/twenty_four_rl/
  game24.py          # 24 点表达式抽取、校验、穷举求解器
  rewards.py         # GRPO 可验证奖励函数
  data.py            # Hugging Face 数据集加载与 prompt 构造
  prompts.py         # R1 风格提示词模板
  train_grpo.py      # TRL GRPO 训练入口
  evaluate.py        # 模型/规则求解器评估入口
configs/
  grpo_qwen_1_5b.yaml
tests/
  test_game24.py
docs/
  background_and_workflow.md
  github_publish.md
reports/
  final_report.md
  progress_log.md
```

## 环境准备

建议使用 Python 3.10+ 和 CUDA GPU 环境。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 本地确定性验证

```bash
python -m unittest discover -s tests
python -m compileall src tests
```

## 数据准备

脚本通过 Hugging Face `datasets` 加载作业指定数据集。为避免评估口径混乱，不使用本地生成数据替代。2026-06-18 实测 `nlile/24-game` 和 `test-time-compute/game-of-24` 都是 1362 条可解样本，且基本覆盖同一批 4 数字组合；脚本会在测试记录中写入 `seen_in_train`，并默认导出 solved rate 最低的 100 条可解难例。若后续数据版本恢复 `solvable=False` 负例，脚本会自动写入 `data/negative.jsonl`。

```bash
python -m twenty_four_rl.data \
  --train-out data/train.jsonl \
  --eval-out data/eval.jsonl \
  --negative-out data/negative.jsonl
```

## GRPO 训练

如果直接 GRPO 输出格式不稳定，可以先做 SFT warmup：

```bash
python -m twenty_four_rl.sft_data \
  --train-out data/sft_train.jsonl \
  --eval-out data/sft_eval.jsonl

python -m twenty_four_rl.train_sft \
  --config configs/sft_qwen_1_5b.yaml
```

```bash
python -m twenty_four_rl.train_grpo \
  --config configs/grpo_qwen_1_5b.yaml
```

默认配置偏保守，便于先跑通。正式实验建议记录：

- `reward/format`：是否遵循 `<think>` 和 `<answer>` 标签。
- `reward/correctness`：表达式是否合法且等于 24。
- `eval/solved_rate`：分布内和分布外测试集成功率。
- `eval/false_positive_rate`：不可解样本上输出伪解的比例。

配置默认不保存中间 checkpoint，只在训练结束保存 `checkpoint-final`，避免本地目录被模型文件占满。

快速烟测可以限制步数和样本数：

```bash
python -m twenty_four_rl.train_grpo \
  --config configs/grpo_qwen_1_5b.yaml \
  --max-steps 1 \
  --limit-train-rows 8 \
  --limit-eval-rows 4 \
  --output-dir outputs/grpo-qwen-24-smoke
```

训练后可以汇总 checkpoint 指标：

```bash
python -m twenty_four_rl.summarize_run \
  outputs/grpo-qwen-24-smoke/checkpoint-1/trainer_state.json \
  --output reports/smoke_summary.json
```

## 评估

先用规则求解器验证数据和评估管线：

```bash
python -m twenty_four_rl.evaluate --input data/eval.jsonl --oracle-solver
```

评估训练后的模型：

```bash
python -m twenty_four_rl.evaluate \
  --input data/eval.jsonl \
  --model outputs/grpo-qwen-24/checkpoint-final \
  --output reports/eval_predictions.jsonl
```

采样评估 solved@N：

```bash
python -m twenty_four_rl.evaluate \
  --input data/eval.jsonl \
  --model outputs/grpo-qwen-24/checkpoint-final \
  --output reports/eval_sample4_predictions.jsonl \
  --do-sample \
  --num-samples 4 \
  --temperature 0.9 \
  --top-p 0.95
```

## 目前状态

已完成可复现工程骨架、确定性 24 点校验器、奖励函数、训练与评估入口、作业背景与流程文档。
已通过 Hugging Face 下载指定数据集并生成 JSONL；当前数据版本没有不可解负例。已完成 SFT warmup、GRPO-from-SFT 50-step 实验和 solved@N 评估。当前最好短实验结果：100 条 hard eval 上 `solved@1=1/100`，`solved@4=16/100`。

完整作业报告见 `reports/final_report.md`，发布 GitHub 的说明见 `docs/github_publish.md`。

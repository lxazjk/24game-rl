# 作业交付检查清单

## 要求对照

| 作业要求 | 仓库证据 | 状态 |
| --- | --- | --- |
| 使用小规模开源模型 `Qwen/Qwen2.5-1.5B-Instruct` | `configs/sft_qwen_1_5b.yaml`、`configs/sft_qwen_1_5b_strong.yaml`、`configs/grpo_qwen_1_5b.yaml` | 已完成 |
| 使用 GRPO/RLVR | `src/twenty_four_rl/train_grpo.py` 使用 TRL `GRPOTrainer`，`src/twenty_four_rl/rewards.py` 提供 rule-based rewards | 已完成 |
| R1 风格 `<think>...</think><answer>...</answer>` 输出 | `src/twenty_four_rl/prompts.py`、`src/twenty_four_rl/game24.py`、`tests/test_game24.py` | 已完成 |
| 校验每个数字恰好使用一次 | `validate_expression` 使用 multiset 检查数字 | 已完成 |
| 校验表达式只使用四则运算和括号 | `game24.py` 使用字符白名单和 AST 节点白名单 | 已完成 |
| 校验结果等于 24 | `game24.py` 使用 `Fraction` 精确计算 | 已完成 |
| 使用 `nlile/24-game` 训练 | `src/twenty_four_rl/data.py` 和 `src/twenty_four_rl/sft_data.py` | 已完成 |
| 使用 `test-time-compute/game-of-24` 测试 | `src/twenty_four_rl/data.py` 从 `Puzzles` 字段解析 hard eval | 已完成 |
| 不使用替代或生成数据 | 数据脚本只调用指定 Hugging Face 数据集；`README.md` 和 `reports/final_report.md` 已说明 | 已完成 |
| 负例检查 | 当前数据版本没有 `solvable=False`，`data/negative.jsonl` 为 0 条；报告已说明 | 受数据源限制 |
| 监控 reward/正确率 | `reports/*summary.json`、`reports/experiment_report.md`、`reports/final_report.md` | 已完成 |
| 定量分析 | `reports/final_report.md` 汇总 SFT、GRPO、`solved@1`、`solved@4` | 已完成 |
| 定性分析 | `reports/final_report.md` 包含成功/失败案例 | 已完成 |
| 背景知识和完整流程 | `docs/background_and_workflow.md` | 已完成 |
| 定期阶段总结 | `reports/progress_log.md` | 已完成 |
| GitHub repo | 本地 git repo 已完成；远端创建缺少 `origin`、`gh` 或 token | 外部阻塞 |

## 当前提交

```text
2d6d1aa Add final assignment report
3b4aa6c Add SFT-started GRPO evaluation
f26f2dd Add SFT warmup and experiment reports
4ae27b4 Build GRPO 24-game assignment repo
```

## 当前实验结论

| 项目 | 结果 |
| --- | --- |
| Oracle on hard eval | 100/100 |
| SFT strong hard eval | 0/100 |
| GRPO-from-SFT 50-step `solved@1` | 1/100 |
| GRPO-from-SFT 50-step `solved@4` | 16/100 |
| Shaped GRPO-from-SFT 50-step `solved@4` | 17/100 |

短训练已经证明工程链路和 verifier reward 可运行，但模型还没有稳定学会 24 点。后续若继续训练，应从 shaped reward 版本继续，并保持 `save_strategy: "no"`，避免本地 checkpoint 过多。

## GitHub 发布状态

当前环境检查结果：

- 没有 `origin` remote。
- 没有 `gh` 命令。
- 没有 `GH_TOKEN` 或 `GITHUB_TOKEN`。

因此无法直接创建 GitHub 远端仓库。获得远端 URL 后运行：

```bash
git remote add origin <github-repo-url>
git push -u origin main
```

# 阶段总结

## 2026-06-18

### 已确认的作业要求

- 任务是用强化学习训练小规模大语言模型求解 24 点。
- 指定模型为 `Qwen/Qwen2.5-1.5B-Instruct`。
- 指定训练范式为 GRPO/RLVR。
- 输出格式要求为 `<think>...</think><answer>...</answer>`。
- 训练集建议使用 `nlile/24-game` 中 `solvable=True` 的 1262 条样本。
- 测试集建议使用 `test-time-compute/game-of-24` 中不重叠且较难的样本。
- 文档提到 `nlile/24-game` 中 `solvable=False` 的约 100 条样本可用于检查模型是否瞎编。

### 本阶段完成

- 从 docx 作业说明中提取并整理了任务要求。
- 创建了 Python 包结构和 README。
- 实现了安全的 24 点表达式抽取、语法校验、数字 multiset 校验和精确分数计算。
- 实现了穷举规则求解器，可用于评估管线 sanity check。
- 实现了 GRPO 奖励函数，包括格式奖励、结构奖励和正确性奖励。
- 实现了 Hugging Face 数据准备脚本、TRL GRPO 训练入口和评估入口。
- 发现当前环境无法直连 Hugging Face；按作业口径保留指定数据集加载，不使用替代数据。
- 根据真实数据字段修正解析逻辑：`test-time-compute/game-of-24` 的数字来自 `Puzzles` 字段。
- 发现两套 24 点数据组合高度重合，因此评估集改为 solved rate 最低的 100 条可解难例，并记录 `seen_in_train`。
- 通过代理成功下载 Hugging Face 数据集：当前 `nlile/24-game` 与 `test-time-compute/game-of-24` 均为 1362 条可解样本，oracle 检查不可解数量为 0。
- 已生成 `data/train.jsonl`、`data/eval.jsonl`、`data/negative.jsonl`；其中负例文件为空，原因是当前数据源没有不可解样本。
- 已用规则求解器评估 `data/eval.jsonl`，100 条难例全部可由 oracle 解出，评估管线通过 sanity check。
- 已完成 Qwen2.5-1.5B + GRPO 1-step 烟测：8 条训练样本、4 条 eval 样本，训练入口、奖励函数和 checkpoint 保存流程均可运行。
- 烟测 eval 指标：`eval_reward=0.0875`、`eval_rewards/correctness_reward_func/mean=0`、`eval_rewards/format_reward_func/mean=0.025`，说明未训练模型在极小步数下尚未学会正确输出，符合预期。
- 增加 `summarize_run.py`，可从 `trainer_state.json` 自动提取关键指标用于阶段总结。
- 完成两轮 20-step 小实验：旧 prompt 与 promptv2。promptv2 的 eval reward 从 0.0738 提升到 0.0963，但 correctness reward 仍为 0。
- 独立生成评估显示两个 20-step checkpoint 在前 20 条难例上 solved rate 都是 0/20。
- 根据预测失败样例修正校验器：当输出多个 `<answer>` 时抽取最后一个答案；并强化 prompt，要求 answer 只包含表达式。
- 新增 `reports/experiment_report.md`，记录数据现状、短训练结果、失败模式和后续方案。
- 新增 SFT warmup 数据与训练入口：`sft_data.py`、`train_sft.py`、`configs/sft_qwen_1_5b.yaml`。
- SFT full 1 epoch 在 held-out SFT eval 上 token accuracy 约 0.959，但在 100 条难例上 solved rate 仍为 0/100；说明模型学会格式，但没有可靠学会组合搜索。
- SFT v2 尝试 `completion_only_loss=True`，但当前 tokenizer 触发大量 prompt/completion mismatch 警告，独立评估输出空答案，不作为默认方案。
- 编写了背景知识与完整流程文档。

### 待完成

- 启动完整 GRPO 训练。
- 从 SFT full checkpoint 继续 GRPO，并增加 solved@N 多候选评估。
- 保存 reward/accuracy 曲线和 checkpoint。
- 完成分布内、分布外、负例三类评估。
- 补充最终实验结果和案例分析。

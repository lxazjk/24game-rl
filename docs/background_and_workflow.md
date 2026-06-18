# 背景知识与完整流程

## 1. 任务背景

24 点游戏要求给定 4 个整数，每个数必须且只能使用一次，通过 `+ - * /` 和括号构造出值为 24 的表达式。这个任务适合 RLVR，因为答案是否合法可以由程序确定性验证，不需要人工偏好标注或额外奖励模型。

本作业的关键不是让模型背答案，而是让模型在小规模可验证任务上学习“先推理、再给出可校验答案”的格式和行为。文档指定的骨干模型是 `Qwen/Qwen2.5-1.5B-Instruct`，训练算法是 GRPO，输出模板参考 DeepSeek-R1 的 `<think>...</think><answer>...</answer>`。

## 2. GRPO 与 RLVR 要点

GRPO（Group Relative Policy Optimization）在同一个 prompt 下采样多条 completion，用组内相对奖励估计优势，减少对单独 value model 的依赖。对 24 点来说，奖励可以完全由规则函数给出：

- 格式奖励：是否包含 `<think>` 和 `<answer>`。
- 结构奖励：最终答案是否简洁、是否只把算式放在 answer 中。
- 正确性奖励：表达式是否只使用输入数字一次，且计算结果是否等于 24。

这种奖励设计能把“可验证正确”作为主要优化目标，同时用小的 shaping reward 降低早期训练的探索难度。

## 3. 数据流程

训练数据优先使用 `nlile/24-game`：

1. 加载全部样本。
2. 解析每条记录中的 4 个数字。
3. 取 `solvable=True` 作为训练样本。
4. 如果当前数据版本存在 `solvable=False`，取这些样本作为负例检查集，用于观察模型是否会在不可解输入上编造答案。

测试数据使用 `test-time-compute/game-of-24`：

1. 加载官方 24 点测试样本。
2. 解析 `Puzzles` 字段中的 4 个数字。
3. 标记测试组合是否出现在训练组合中。
4. 默认取 solved rate 最低的 100 条可解难例作为重点评估集。

本仓库的 `twenty_four_rl.data` 负责生成 `data/train.jsonl`、`data/eval.jsonl` 和 `data/negative.jsonl`。脚本只使用作业指定数据集，避免本地生成数据改变实验口径。2026-06-18 实测两套 24 点数据均为 1362 条可解样本，未提供不可解负例，且数字组合高度重合。因此评估文件会显式保存 `seen_in_train`，报告中需要说明这一点。

## 4. 奖励校验流程

表达式校验由 `twenty_four_rl.game24.validate_expression` 完成：

1. 从模型输出中抽取 `<answer>` 内的表达式。
2. 检查字符集合，只允许数字、四则运算符、括号和空白。
3. 用正则抽取数字，并用 multiset 判断是否和输入数字完全一致。
4. 用 Python AST 解析表达式，只允许整数常量和四则二元运算。
5. 用 `Fraction` 精确计算，最后用 `1e-6` 容差判定是否等于 24。

这能避免 `eval` 带来的安全问题，也能防止模型通过函数调用、变量、重复数字等方式绕过规则。

## 5. 训练流程

推荐实验步骤：

1. 创建 Python/CUDA 环境并安装 `requirements.txt`。
2. 运行 `python -m twenty_four_rl.data` 准备 JSONL 数据。
3. 先运行 `python -m twenty_four_rl.evaluate --oracle-solver` 验证评估管线。
4. 用 `configs/grpo_qwen_1_5b.yaml` 启动 GRPO 训练。
5. 监控 format reward、correctness reward、KL、completion length、训练 loss。
6. 每隔固定 step 保存 checkpoint，并在 eval/negative 集上计算 solved rate 和 false positive rate。
7. 选取若干成功/失败案例做定性分析，检查是否存在格式错、数字复用、算式错误、除零、不可解样本瞎编等问题。

## 6. 结果分析模板

最终报告建议包含：

- 数据统计：训练、测试、负例样本数量和去重策略。
- 训练设置：模型、batch size、采样数、学习率、epoch、KL 系数。
- 曲线：平均总奖励、正确性奖励、格式奖励、eval solved rate。
- 指标：分布内成功率、分布外成功率、负例误报率。
- 案例：成功样例、失败样例、不可解样例。
- 反思：奖励是否稀疏、模型是否学会格式、是否出现投机输出。

## 7. 当前环境限制

当前工作区已经通过代理下载 Hugging Face 指定数据集，并生成训练/难例评估 JSONL。尚未实际启动 Qwen2.5-1.5B 的完整 GRPO 训练，因此最终实验曲线和模型指标仍需要训练后补充。

短训练实验显示，纯 GRPO 从 instruct 模型直接起步时会遇到明显稀疏奖励问题：模型容易输出解释、等式或空答案，而不是纯表达式。SFT warmup 可以改善格式，但单纯 SFT 仍不能可靠学会组合搜索。更稳妥的完整流程是先用参考解做格式监督 warmup，再从 warmup checkpoint 切换到 GRPO，用 verifier 奖励强化正确性，并报告 solved@1 与 solved@N。

# GitHub 发布说明

当前本地仓库已初始化并提交，分支为 `main`。发布到 GitHub 还需要一个远端仓库地址，或在本机安装并登录 `gh`。

## 已完成

- 本地 git repo 已创建。
- 源码、配置、报告和测试已提交。
- `data/`、`outputs/` 已在 `.gitignore` 中忽略，模型 checkpoint 不会被推到 GitHub。

## 方式一：已有远端仓库

```bash
git remote add origin <github-repo-url>
git push -u origin main
```

## 方式二：使用 GitHub CLI

```bash
gh auth login
gh repo create <repo-name> --private --source=. --remote=origin --push
```

## 当前阻塞点

当前环境没有检测到 `gh`、`GH_TOKEN`、`GITHUB_TOKEN` 或已有 `origin` remote，因此无法在本机直接创建 GitHub 远端仓库。

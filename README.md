# ECS171: Label Randomization, Memorization and Generalization

UCI Dry Bean 七分类实验，比较 Logistic Regression、Random Forest、MLP。
当前实施以 `../11_Proposal_and_Master_Plan/` 的固定方案为准。

## 目录

| 位置 | 用途 |
|---|---|
| `src/core.py` | 去重/划分、标签扰动、指标、模型和缓存校验 |
| `src/pipeline.py` | 分阶段执行、调参、冻结、主训练和最终留出评估 |
| `src/visuals.py` | 仅训练集 EDA、真实结果图表与报告 |
| `config/` | 当前协议、45 次矩阵、获选参数与冻结证据 |
| `data/raw/` | 原始 ZIP 和 XLSX，不改写 |
| `data/processed/` | 去重后的数据、固定 split、原始 row_id 与 manifest |
| `data/noise/` | 15 份 seed×p 扰动标签，三个模型共用 |
| `artifacts/tuning/` | 当前十候选训练日志、模型和逐行预测 |
| `artifacts/runs/` | 当前 45 次主实验的模型、日志、预测、loss trace |
| `results/` | 每次真实测试指标、15 条件汇总和逐行概率 |
| `reports/eda/` | 数据审计、训练集图表和统计 |
| `reports/experiments/` | 结果报告、对照曲线和混淆矩阵 |
| `tests/` | 科学协议、泄漏/缓存保护与独立最终验收 |
| `logs/` / `progress.md` | 安装、测试、执行、评审与决定记录 |

## 环境

Python 3.12.14，scikit-learn 1.7.2，精确版本见 `requirements.lock.txt`。
本机环境位于 `.venv`；不需要 GPU、云计算或付费 API。
BLAS 设为一个线程，RF 使用 4 个 worker；各次 fit 顺序执行。
`environment.json` 记录本机版本和计算设置；恢复阶段会核对实际依赖，防止混用环境。

新环境（需已安装 Python 3.12）可运行：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock.txt
```

在这个目录运行阶段：

```powershell
.\Run_Project.ps1 -Stage prepare
.\Run_Project.ps1 -Stage eda
.\Run_Project.ps1 -Stage smoke
.\Run_Project.ps1 -Stage tune
.\Run_Project.ps1 -Stage train
.\Run_Project.ps1 -Stage evaluate
.\Run_Project.ps1 -Stage report
.\Run_Project.ps1 -Stage verify
```

`-Stage all` 按顺序执行到报告；每一步失败即停止。已完成 fit 只在配置、环境、数据、代码和缓存内容均一致时复用。
原始 ZIP/XLSX 已包含在项目中；不需要重新下载。若复制项目到另一台机器，先安装锁定依赖，再在**新输出副本**中从 prepare 开始生成该环境的证据。不要把旧模型和新环境拟合混合。
新输出副本仅复制 `src/`、`tests/`、`Run_Project.ps1`、`pytest.ini`、依赖文件、`data/raw/` 以及 `config/protocol.json` / `main_matrix.json`。不复制旧 `environment.json`、获选/冻结文件、processed 数据、artifacts 或 results；原项目保留不动。

## 固定实验

- 去重前 13,611 行；去除 68 行后 13,543 行。train / validation / test = 8,125 / 2,709 / 2,709。
- 使用 16 原始数值特征，保留源列名 `AspectRation` 与 `roundness`。`row_id` 不进入模型。
- LR/MLP 的完整 Pipeline 内含 StandardScaler，仅 fit 训练特征；RF 不缩放。
- 干净标签 10 次调参，验证 Macro-F1 为选择依据；0.001 内按预设复杂度规则处理平分。
- 三模型 × p∈{0,.25,.5,.75,1} × seed∈{17,29,43} = 45 主 runs。
- p 是独立重新抽标签的概率；允许抽回原类，期望实际改变率为 6p/7。标签在一个 fit 内不变。
- 不在噪声条件重新调参。`FROZEN.json` 保存测试前协议；只有完整 canonical 矩阵的全部 fits 完成才能打开真实测试评估。
- 主终点为干净测试 Macro-F1；Accuracy、Balanced Accuracy、训练对两种标签的准确率和耗时为次要指标。
- 汇总为三次重复均值和样本 SD，不称为显著性或多个数据划分的稳定性。

## 缓存与历史

完成模型、训练/验证逐行预测和训练 trace 各有 SHA-256；完整 meta.json 另有 checksum，恢复会同时验证。
`artifacts/pre_review/` 与 `config/history/` 是独立审查补强之前的第一次调参记录，**不计入当前 55 次研究 fit，也不进入汇总**。
初版代码保存在 `artifacts/initial_code_version/`。当前版在修正缓存/环境/矩阵保护后重新调参并冻结。

本次范围不含可选的 18 次容量实验、实时推理网页或公开部署。这里的 HTML 是可离线阅读的结果报告。
没有提交课程材料、发送邮件、推送仓库或部署。

## 来源

- Dataset: [UCI Dry Bean](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset), DOI [10.24432/C50S4B](https://doi.org/10.24432/C50S4B), CC BY 4.0.
- Assigned paper: Zhang et al., *Understanding Deep Learning Requires Rethinking Generalization*, ICLR 2017, [arXiv:1611.03530v2](https://arxiv.org/abs/1611.03530v2).
- [scikit-learn 1.7.2 documentation](https://scikit-learn.org/1.7/).

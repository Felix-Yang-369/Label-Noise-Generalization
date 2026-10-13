# ECS171: Label Randomization, Memorization and Generalization

UCI Dry Bean 七分类实验，比较 Logistic Regression、Random Forest、MLP。
当前实施以 `../11_Proposal_and_Master_Plan/` 的固定方案为准。

## 目录

| 位置 | 用途 |
|---|---|
| `src/core.py` | 去重/划分、标签扰动、指标、模型和缓存校验 |
| `src/pipeline.py` | 分阶段执行、调参、冻结、主训练和最终留出评估 |
| `src/visuals.py` | 仅训练集 EDA、真实结果图表与报告 |
| `app/` / `Run_Demo.ps1` | 保存模型的真实推理网页；三模型 × p=0/1，seed=17 |
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

## 本地 Git 记录

本目录已建立独立 Git 仓库，分支 `codex/ecs171`。`6fcccff` 是完成环境、数据处理、EDA、10 次调参和 45 次主实验后的基线；后续提交记录真实推理网页。
提交使用既有本地 Git 身份 Felix；提交说明和项目状态明确记录 Codex 在用户指示下编写与执行，不能视为其他队员已完成贡献。
源代码、锁定依赖、原始公开数据、配置、汇总结果、图表和校验清单纳入 Git；`.venv`、生成模型、完整 split 和逐行预测留在本机，避免重复的大文件进入历史。
`reports/Experiment_Artifact_Manifest.json` 保存 397 个当前实验生成文件的基线 SHA-256。

Git 克隆包含历史结果证据，但不包含可直接启动的模型资产。复现时按上文“新输出副本”步骤仅复制必要输入，完成训练和评估后再运行网页；请勿在带旧冻结证据的克隆目录中混合一次新的实验。新副本还需复制 `app/` 与 `Run_Demo.ps1`；训练完成后可运行完整测试和网页。

## 实时推理网页

在本目录运行：

```powershell
.\Run_Demo.ps1
```

然后在浏览器访问 `http://127.0.0.1:5172`。按 Ctrl+C 停止服务；端口冲突时可指定 `-Port 5173`。
网页使用本机已完成的六个完整 Pipeline：LR/RF/MLP 的 p=0 和 p=1，统一 seed=17。启动先验证实际依赖、冻结协议、科学代码及模型校验值；缺少或损坏资产会停止启动，请在独立新副本复现完整实验。
页面提供 16 个数值输入、7 个验证集示例、三个模型和单条件／并排比较。示例为每类验证集中 row_id 最小的一条，选择与模型预测无关。编辑输入后不再把示例原类当作当前输入的真实标签。
七类概率的顺序来自保存模型的 `classes_`；预处理来自原 Pipeline。所有 HTTP 请求只做推理，不训练、不读取留出测试文件。
缺失或非数值、NaN/Inf、布尔值与过大数值会被拒绝；超出训练范围但数值有限的输入保持原值并显示提示。原始概率未经校准，不称为置信度。请求期间锁定输入和模型选择，完成或失败后恢复。
`reports/demo/` 保存真实浏览器交互、桌面截图、手机输出局部截图和慢速请求回归证据；手机全页截图受工具超时限制，已记录检查范围。`reports/Demo_Runbook.md` 给出五分钟展示安排和恢复步骤。

当前已完成本地网页；可选 18 次容量实验、备用演示视频、报告正式排版和公开部署尚待后续完成。尚未向 GitHub 推送、提交课程材料或发送邮件。

## 来源

- Dataset: [UCI Dry Bean](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset), DOI [10.24432/C50S4B](https://doi.org/10.24432/C50S4B), CC BY 4.0.
- Assigned paper: Zhang et al., *Understanding Deep Learning Requires Rethinking Generalization*, ICLR 2017, [arXiv:1611.03530v2](https://arxiv.org/abs/1611.03530v2).
- [scikit-learn 1.7.2 documentation](https://scikit-learn.org/1.7/).

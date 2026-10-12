# GraphCliff-XACs

独立方向仓库：把 **ACES-GNN（Digital Discovery 2025）公开的原子级"不常见子结构"标注（XACs）**接到 **GraphCliff（KDD 2026）**上做受控先导，检验"换监督信号来源"能否改善活性悬崖（activity cliff）预测。

- 上游 GraphCliff 快照：`d18857db0303346c0be715a56febf71da1768592`（MIT，原 `README.md` 与 `LICENSE` 原样保留）。
- XACs 标注来源：[Liu-group/XACs](https://github.com/Liu-group/XACs)（ACES-GNN 官方仓库；数据不随本仓库分发，见下）。
- 姊妹仓库（本方向的来源与更早的损失侧实验）：<https://github.com/yxxiaocainiao/GraphCliff-ACA>。

## 结论（一句话）

**go = false：在同一 30 任务基准的 4 个开发任务 × 3 个初始化种子上，解释监督没有改善验证 Cliff RMSE。**

| 宏平均（4 任务 × 3 种子，开发验证） | MSE 对照 | 解释监督 | 相对 |
|---|---:|---:|---:|
| Overall RMSE | 0.608410 | 0.611924 | 恶化 0.58% |
| Cliff RMSE | 0.653516 | 0.659776 | 恶化 0.96% |

12 个配对块中 4 块 Cliff 改善（符号检验 p=0.388），每个任务仅 1/3 种子获益，未达事前门槛（≥3/4 任务在 ≥2/3 种子改善）。逐块相对变化标准差 5.62%，远大于宏平均差异，因此结论是**无可靠改善**，而不是稳定的微小负效应。详见 [结论](docs/xacs_explain_20261012/conclusion.md)。

这轮的主要价值是**排除法**：历史损失侧实验的监督密度只有"每批 5 个候选、辅助/回归梯度范数比 0.02–0.04%"，本轮提到"每批约 38 条配对、约 980 个标注原子、梯度比 2.4–3.4%"后仍无收益，说明那条路径的无效不是监督稀疏造成的。官方 test 未参与训练/挖掘/选型，本轮未评价 test。

## 目录

| 路径 | 内容 |
|---|---|
| `experiments/xacs_explain/task_plan.md` | **冻结协议**（事前写入：批次构造、损失形式、权重选取规则、go/no-go） |
| `experiments/xacs_explain/run.py` | 实验入口：`check`（预检+冻结）/ `queue`（24 次）/ `job` / `report` / `audit` |
| `experiments/xacs_explain/diagnose.py` | 只读机制诊断：验证集内部配对的归因-活性排序与 uncommon 归因占比 |
| `tools/xacs_intake/inspect_xacs.py` | XACs 标注与本仓库数据/划分的对齐核对（只读） |
| `tools/xacs_intake/evidence.json` | 30 任务的配对数、覆盖与掩码统计 |
| `docs/xacs_intake_20261012/report.md` | 核对报告：数据同源、监督密度、退化配对、ACES 损失源码事实 |
| `docs/xacs_explain_20261012/` | 覆盖、冻结权重校准、逐块结果、机制诊断、报告与结论 |
| `artifacts/xacs_explain_20261012/` | 冻结 manifest、每轮 history、summary、验证预测 CSV、状态（**权重与日志不入库**） |
| `experiments/aca/`、`experiments/aca_local/pilot.py` | 从姊妹仓库复用的训练骨架与梯度探针；`experiments/aca/vendor/` 为 GPL-3.0 源码，许可原样保留 |

## 复现方式（需要自备数据）

1. 到 [Liu-group/XACs](https://github.com/Liu-group/XACs) 取 `Data/<task>/mcs_dict_0.9.pkl`（ACES-GNN 的标注），放到本机 `D:\XACs\Data\<task>\`（本项目的脚本按此路径读取，只读）。
2. `benchmark_data/` 来自上游快照，已随本仓库提供（MoleculeACE 的 30 个 CHEMBL 靶点，官方 train/test 成员不变）。
3. 环境：见 `docs/xacs_explain_20261012/` 报告中记录的 torch/CUDA/GPU；本项目的实验在单卡 RTX 5060 Laptop 上完成。
4. 运行：`python experiments/xacs_explain/run.py check` → `queue` → `audit`；`python experiments/xacs_explain/diagnose.py` 出机制诊断。

## 边界声明

- 损失形式来自 ACES-GNN 论文与官方代码，本仓库只做接入与受控比较，**不宣称原创**，也不宣称复现其数字（其骨干、划分与超参与本项目不同）。
- 监督只使用开发 train 内部配对；官方 test 不参与训练、挖掘、选型。所有成绩都是开发验证成绩。
- 上游 MIT 许可与 ACA vendor 的 GPL-3.0 许可原样保留；本方向不修改上游 `model.py` / `dataset.py` / `main.py` / `train.py`。

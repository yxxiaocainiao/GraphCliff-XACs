# XACs / ACES-GNN 数据可用性核对（2026-10-12）

范围：只读核对 `D:\XACs`（用户自行下载）与本仓库 `benchmark_data` 的对应关系与标注可用量。
本轮**没有训练、没有改模型/数据/历史实验、没有读取官方 test 标签用于任何训练信号**；test 成员仅用于统计数量。
证据：`tools/xacs_intake/evidence.json`（脚本 `tools/xacs_intake/inspect_xacs.py`，可复跑）。

## 结论（可用于决策）

1. **同一批数据**：`D:\XACs\Data\<task>\<task>.csv` 与 `benchmark_data\<task>.csv` 是同一份 MoleculeACE 文件的两种导出。四个开发任务逐项核对：行数相同、SMILES 集合相同、**行序完全相同**、`cliff_mol` 标志逐行相同；标签仅差常数 9.0（我们的 `y [pEC50/pKi]` = 它的 `y` + 9，即 −log10(M) 与 −log10(nM)）。
2. **标注可逐行映射**：`mcs_dict_0.9.pkl` 的键就是 CSV 的 SMILES（全数据集 3657/631/1456/3097 个键），训练用的原子级标注可以直接按 SMILES/行号接到我们现有官方划分上，不需要重新生成。
3. **监督密度比此前的三元组挖掘高一个量级**：batch=128 时，四个开发任务预期每批约 **45–55 个分子带训练内配对、110–217 条标注对**（见下表）。此前 ACA/结构门控随机 batch 只有 5 个候选、82/83 批为空——这正是那些实验辅助梯度极小（约 0.02–0.04%）的直接原因，本数据源在结构上解决了它。
4. **必须过滤**：约 1–5% 的配对只有单侧有 uncommon 原子（取代/删除型 MCS，例如 A 有 Cl、B 没有），另一侧掩码全零；这些对在"不常见部分"项上只能贡献单侧质量，需要显式规则处理（跳过或只算有掩码一侧）。
5. **不可直接复用其训练协议**：XACs 默认 `split_method='random'`（行序随机划分，非 MoleculeACE 官方划分），且 `mcs_dict` 是**全数据集**计算的。要用它做**我们**的实验，必须只取官方 train 内部配对（本轮统计即是这么算的），否则会把 test 侧活性信息带进监督。

## 标注格式（`mcs_dict_0.9.pkl`）

```
SMILES -> [ {'is_cliff_mol': bool},                      # 自身记录
            {'smiles': 配对分子, 'potency_diff': float,   # 有符号活性差
             'uncommon_atom_idx_i/j': [...], 'common_atom_idx_i/j': [...],
             'atom_mask_i/j': [0/1 ...]}, ... ]          # 逐原子掩码，长度=原子数
```
- 语义：uncommon = 该对 MCS 之外的原子（导致活性差异的取代/片段），common = MCS 覆盖的原子。
- 几何检查（每个开发任务前 100 条训练内配对）：掩码长度与 RDKit 原子数一致、取值二值、无解析失败；`uncommon_atom_idx` 与掩码一致。
- 掩码规模：训练内配对的 uncommon 原子数均值 6.4–11.3（中位 5–11）。
- 配对是无向的（A→B 与 B→A 都出现），单侧全零在全数据集上 CHEMBL234 324 对、CHEMBL244 432 对。

## 四个开发任务的可用量（官方 train 内部）

| 任务 | 训练分子 | 训练内配对数 | 有配对的锚点数 | batch128 预期锚点 | batch128 预期配对数 | 单侧全零掩码占比 | 全数据集配对数/锚点 |
|---|---:|---:|---:|---:|---:|---:|---:|
| CHEMBL234_Ki | 2924 | 3314 | 1153 | 50.5 | 145 | 3.2% | 5158 / 1604 |
| CHEMBL2047_EC50 | 503 | 434 | 178 | 45.3 | 110 | 3.0% | 700 / 239 |
| CHEMBL2147_Ki | 1162 | 1966 | 434 | 47.8 | 217 | 1.0% | 3004 / 572 |
| CHEMBL244_Ki | 2476 | 3164 | 1055 | 54.5 | 164 | 4.8% | 4866 / 1450 |

注：本仓库先导用"官方 train 内再留 10% 验证"，可用量会略低于上表（例如 CHEMBL234 是 2632 而非 2924）；实际以冻结划分后重算为准。

## 30 任务覆盖

`D:\XACs\Data` 含 30 个 CHEMBL 目录，名称与 `run.sh` 的 30 任务清单**完全一致**（含 `CHEMBL237_Ki` 与 `CHEMBL237_EC50` 两个），另有 MoleculeNet/其他数据集（BACE、BBBP、Clintox、Esol、Freesolv、Lipo、SIDER、drd2、kcnh2、kdr）不在本仓库任务内。
逐任务配对数见 `evidence.json`；最稀疏的是 CHEMBL2835_Ki（82 对）与 CHEMBL4203_Ki（80 对），与 GraphCliff 论文中这两个任务本身样本少一致。

## 参照：ACES-GNN 的监督损失（源码事实，用于后续适配）

`XACs/train.py` + `XACs/utils/utils.py` + `XACs/models/GNN.py`：
- `uncom_prior = pairwise_ranking_loss(pooled_uncom_att_diff, potency_diff)`，其中 `pairwise_ranking_loss(x,y)=max(-x*y,0).sum()`，即**铰链项要求"不常见原子上的归因差"与"活性差"同号**（无 margin）。
- `common_prior = square(common_att).sum()`，抑制共有骨架上的归因。
- `train_loss = MSE + com_w*common_prior + uncom_w*uncom_prior`；权重默认 1e-3，作者按数据集/种子用网格搜索选（`use_opt_xweight=True`）。
- 归因来自 GradCAM 类分层梯度×激活（`process_layer_gradients_and_eval`），`uncom_pool='add'`；配对由 `pack_data(..., pair_cap=8)` 打包成"锚点+配对分子"的连通大图。
- 它们的默认训练协议：gine 骨干、batch128、800 epochs、早停 100、lr 3e-4、wd 1e-4、`split_method='random'`、`num_folds` 个种子。

## 与仓库规则的相容性

- 只用官方 train 内部配对与标注 → 符合 "Mine triplets using training molecules and labels within the same task only"。
- 官方 test 成员保持原样、本轮不评价、不参与挖掘与选型。
- 新方向（解释/归因监督）超出 AGENTS.md 现有 ACA 范围，**需用户授权后**才写进 `task_plan.md` 并实现；本轮仅做只读核对与证据留存。

## 下一步（待授权，尚未执行）

- P0：把 MCS 标注接进现有开发划分（只 train 内部配对），统计过滤后每任务可用对、单向对处理规则、梯度量级预检（零权重/空掩码/形状）。
- P1：4 任务 × 3 seed 的开发验证先导，对照匹配 MSE 基线；主指标 Cliff RMSE，机制指标沿用"归因/gate 与标注 uncommon 原子的对齐度"。
- Go/No-Go 沿用本仓库既有门槛（≥3/4 任务、≥2/3 种子 Cliff 改善且 Overall 不退化）。

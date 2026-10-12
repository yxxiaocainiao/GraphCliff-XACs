# XACs/ACES 式原子级解释监督先导（2026-10-12）

用户明确授权本轮扩大范围：在 GraphCliff 上接入 XACs（ACES-GNN, Digital Discovery 2025）的原子级"不常见子结构"监督，并在既有开发协议下检验验证 Cliff RMSE 是否改善。本轮为工程先导；损失形式来自 ACES-GNN 论文/代码，不宣称原创，也不宣称复现其数字（其骨干、划分、超参与本仓库不同）。

## 数据与来源

- 标注：`D:\XACs\Data\<task>\mcs_dict_0.9.pkl`（用户自行下载的作者仓库，**只读**）。键为数据集全部 SMILES；值为 `[{'is_cliff_mol':bool}, {smiles, uncommon_atom_idx_i/j, common_atom_idx_i/j, atom_mask_i/j, potency_diff}, ...]`。
- 标签与划分：本仓库 `benchmark_data/<task>.csv` 官方 train/test 成员不变；训练只读官方 train 标签。
- 只使用**开发 train 内部**配对：锚点与配对分子都必须落在 `base.development` 的 train 行内。官方 test 不参与训练、挖掘、选型；验证集分子不进入监督配对（否则"哪些对是悬崖"用到了验证标签）。
- 符号约定已核对：`potency_diff == y_anchor − y_partner`（用本仓库标签逐对验证）。检查阶段对全部配对断言该等式（容差 1e-6）。
- 掩码：长度必须等于该分子在本仓库构图中的节点数（RDKit 原子序，`dataset_utils.smiles_to_graph` 按 `mol.GetAtoms()` 顺序建点）。单侧全零掩码的配对（取代/删除型 MCS）**保留**：此时不常见归因差等于有掩码一侧的质量，符号仍有意义。**两侧同时全零**的配对（MCS 覆盖两个分子的全部原子，无任何可归因原子）跳过并计数；四个任务开发 train 内的跳过数为 4/0/2/10，写入 manifest。

## 冻结协议

- 任务：CHEMBL234_Ki、CHEMBL2047_EC50、CHEMBL2147_Ki、CHEMBL244_Ki；初始化 seed=42/43/44；固定 split_seed=42；两臂 × 12 = 24 次。
- 两臂：`mse`（关闭辅助项）与 `explain`（MSE + w×辅助项）。除辅助项外**完全相同**：同初始化、同批次序列、同优化器与选模。
- 批次：每批 124 个普通训练分子（每轮每分子恰好一次；排列由 `RandomState(seed*1000+epoch)` 决定，与既有先导一致），再从开发 train 配对表按 `RandomState(seed*1000+epoch+7)` 无放回抽取至多 48 条配对，把不在普通批内的配对成员作为补充样本加入，整批上限 192 个分子。两臂批次 id 序列必须逐批相同（检查阶段对全部任务/种子/轮次断言）。
- 补充样本是开发 train 内的真实分子，其标签合法；两臂同样计入 MSE，因此配对分子的重复出现对两臂等价。
- 归因：GradCAM（分层梯度×激活，ReLU），取 `SAGPool` 输入前的逐原子 256 维表示；通道权重 = 该分子节点梯度的通道均值。归因项只作用于该批参与配对的分子。
- 辅助损失（与 ACES 代码同形，`XACs/train.py`、`XACs/utils/utils.py`）：
  `common_prior = Σ_nodes (att·common_mask)²`；
  `uncom_prior = Σ_pairs max(−(Σ_uncom att_anchor − Σ_uncom att_partner)·(y_anchor − y_partner), 0)`；
  辅助项 = `w × (common_prior + uncom_prior)`。作者对 com/uncom 使用同一个网格搜索出的权重，因此这里是单旋钮 w。
- **w 的选取（不看验证集/test）**：候选 {1e-4, 1e-3, 1e-2, 1e-1}，在检查阶段对 4 个任务各自 seed42 epoch1 的首批计算"加权辅助梯度范数 / 回归梯度范数"（backbone 参数，排除 `reg_head`，与既有先导口径一致），取四个任务中位数最接近 **2%** 的候选；并列取较小者。冻结进 manifest，全部 12 次 explain 用同一 w。
- 训练超参沿既有先导：H256/L3/groups4/K3/dropout0（LongPoly 内部 0.1 保留）；AdamW lr1e-4/wd1e-5/betas(.9,.95)，clip1；warmup10/cosine100/minlr1e-6；≤100 轮；patience15、min_delta1e-4；按验证 Overall MSE 选模。
- 记录：源文件与 XACs 标注文件的 SHA256、每轮两项 loss、参与配对数、掩码原子数、梯度比/夹角（首批抽样）、采样 hash、验证预测 CSV、参数量与初始化 hash。
- 完成后独立重载 checkpoint，核验预测与用 `math.fsum` 复算的 RMSE；两臂同 (task,seed) 的初始化 hash 必须一致。

## 判定与边界

扩大门槛（操作性规则，非显著性检验）：explain 相对 mse 的四任务宏平均 Cliff RMSE 至少改善 1%，Overall 恶化不超过 1%，且至少 3/4 任务在至少 2/3 种子上 Cliff 改善 → go。否则暂停当前配方，不自动调 w、不加模块、不扩 30 任务。

即使达门槛，也不自动执行完整官方 train 重训/原 test 评价：那需要另冻结预算，并披露 test 已在历史轮次被使用。当前成绩是开发验证成绩，不与论文数字并列。

失败即停：无配对、掩码长度不符、符号等式不成立、非有限值、CUDA 故障 → 立即停止队列，不自动重试或换参。

## 执行步骤

1. `check`：配对/掩码/符号/覆盖检查，零权重等价性，真实 CUDA 三步更新，checkpoint 严格重载，w 校准；冻结 manifest。
2. `queue`：24 次独立子进程顺序执行，写 status/logs/history。
3. `audit`：独立重载 24 个模型 + CSV 复算 + 配对统计，产出配对结果与 go/no-go。
4. 报告写 `docs/xacs_explain_20261012`；权重与日志写 `artifacts/xacs_explain_20261012`。

输出目录若因本机沙箱权限不可写，队列会在启动前直接报错停止，不半途写坏数据。

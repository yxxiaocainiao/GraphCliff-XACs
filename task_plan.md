# GraphCliff-XACs 计划与状态

## 方向

GraphCliff（KDD 2026）在 MoleculeACE 30 个 CHEMBL 靶点上报告 Overall/Cliff RMSE 最佳；它自己的 Limitations 只提出 cliff-aware 对比/margin 损失与 ECFP 混合特征两条下一步。本仓库走第三条：不改架构与损失形状，改**监督信号来源**——用 ACES-GNN 公开的原子级"不常见子结构"标注监督 GraphCliff 的逐原子归因。

## 阶段

- [x] P0 数据核对（`tools/xacs_intake`，只读）：确认 XACs 的 30 个 CHEMBL 目录与本仓库 30 任务清单一致，CSV 行序/SMILES/`cliff_mol` 逐行相同、标签只差常数 9.0；`mcs_dict_0.9.pkl` 提供逐对 `uncommon/common` 原子掩码与有符号活性差。报告 `docs/xacs_intake_20261012/report.md`。
- [x] P0 冻结协议 `experiments/xacs_explain/task_plan.md`：两臂 mse/explain，4 任务 × seed 42/43/44；每批 124 个普通分子 + 抽样注入配对（上限 192）；GradCAM 归因；`Σ(att·common)² + Σ max(−(Δuncom·att)(Δy),0)`；单旋钮 w 由"梯度范数比最接近 2%"事前规则冻结；判据 ≥1% Cliff 改善且 Overall 恶化 ≤1% 且 ≥3/4 任务在 ≥2/3 种子改善。
- [x] P1 预检并冻结 manifest：合成公式三项、退化对跳过、掩码长度=节点数、`potency_diff = y_anchor − y_partner` 逐对成立、两臂批次 id 序列一致、真实 CUDA 三步更新、严格重载、100 轮 × 3 种子 × 4 任务批次覆盖。冻结 w=0.01（实测梯度比 2.44%）。
- [x] P1 24 次队列 + 独立重载审计：**go = false**（Overall 恶化 0.58%、Cliff 恶化 0.96%，12 块中 4 块改善，每任务 1/3 种子获益）。
- [x] P2 机制诊断（`diagnose.py`）：两臂归因行为确实不同，但方向不一致，且验证内可比配对只有 0–54 对，属弱证据。
- [x] P2 发布：本仓库（GitHub 新仓库），小型证据入库、权重不入库。

## 当前决策

按冻结协议**暂停该配方**：不调 w、不加模块、不扩 30 任务、不评价 test。若继续，最小可行下一步是先写新协议再做的受控对照：
1. ACES 的 common 项在全体共有原子上求和，对"大骨架相同、只差取代基"的悬崖对近似"全分子压制归因"，可能连带压掉不常见原子信号（CHEMBL244 explain 臂 uncommon 归因占比为 0 与此方向一致，未证实）；
2. 或改用 GraphCliff 自带的逐原子 gate 作为监督目标，而非 GradCAM 归因。

两者都必须重新冻结协议；不得用本轮已看过的验证结果事后挑方向。

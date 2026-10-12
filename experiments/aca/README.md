# 原始 GraphCliff 上的 ACA 固定先导

本目录 ACA 适配与训练入口按 GPL-3.0 提供，完整许可见 vendor/LICENSE.upstream.txt；根目录原始 GraphCliff 的 MIT 许可保留。未改写原作者 ACALoss/_aca_loss，AST 仅加载无结构门控所需的两个定义，省去无关依赖；启用结构门控需要补齐原作者依赖和函数。

配方固定在 protocol.json。MSE 为原回归基线，MAE 为对比学习的匹配回归基线，ACA 为 MAE + 0.1 × 官方三元组项。README 示例配方和固定旧版源码的来源分别记录，不能称为论文调优复现。

在仓库根目录使用 graphcliff 环境执行：

```powershell
python experiments/aca/run.py check
python experiments/aca/run.py run --task CHEMBL234_Ki
python experiments/aca/run.py run --task CHEMBL2047_EC50
python experiments/aca/run.py audit
```

输出至 artifacts/aca_pilot_20261010，任务目录若已存在会拒绝重跑，避免覆盖；完整历史、划分、最佳权重、验证预测、环境和核验保存在其中。官方 train 内拆分，只解析 train 的活性标签。沿用数据提供的 cliff_mol 标记进行分层与报告；该标记的全数据构建方式不由本先导重新定义。

本轮重载漏取 model_state_dict 外层后修复；recover --task 仅用于恢复已完成的 MSE 历史并执行尚未开始的两组，校验终止条件和来源，不续训未完成拟合、不覆盖已有 summary。恢复记录另存 recovery.json，MSE 训练耗时无法准确恢复，记 null。

三元组满足 i、j、k 不同，|yi-yj|<0.2，|yi-yk|≥1。Ltriplet = mean([||zi-zj||1-||zi-zk||1+|yi-yk|-|yi-yj|]+)，均值分母为所有合格三元组。该实现是 hinge 截断，不是 softplus；关闭结构门控，负例不必结构相似，因此不能把全部三元组称为悬崖三元组。空候选批次退化为 MAE。

选择验证 Overall MSE 最小的 epoch，Cliff RMSE 仅记录。单种子、两个任务只能筛查可行性。原主干内部 LongPoly dropout 仍沿用默认值 0.1；顶层 dropout=0 不会关闭它。CUDA 的原框架确定性模式为 warn_only，不承诺跨平台逐位一致。

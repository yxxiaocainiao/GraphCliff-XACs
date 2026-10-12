# GraphCliff-XACs 工作边界

- 本仓库是独立方向仓库（`GraphCliff` + `XACs` 原子级解释监督），默认分支 `main`，起点是上游 GraphCliff 快照 `d18857db0303346c0be715a56febf71da1768592`。
- 姊妹仓库 <https://github.com/yxxiaocainiao/GraphCliff-ACA> 保留本方向之前的损失侧实验（ACA 迁移、cosine-margin、ECFP 组合等）；本仓库不复制那些实验，只复用其训练骨架 `experiments/aca/run.py` 与 `experiments/aca_local/pilot.py`。
- 上游 GraphCliff 的 `model.py`、`dataset.py`、`dataset_utils.py`、`main.py`、`train.py`、`LICENSE` 保持快照原样，不在本方向修改；上游作者文档改名为 `README_UPSTREAM.md` 保留，本仓库首页 `README.md` 说明本方向。任何新机制都写在新目录里。
- 数据边界：`benchmark_data/` 的官方 train/test 成员不变。监督配对只能用**训练侧**分子与标签；开发验证只用于选模；官方 test 不得进入训练、挖掘、调参或选型。历史上评估过 test 的实验必须披露，不能当作新的盲测。
- 外部只读依赖：ACES-GNN 的标注目录（本机 `D:\XACs`）。不把该数据重新分发到本仓库；仓库只保留核对结果与哈希。
- 每个实验先写冻结协议（批次、损失、权重规则、判据），预检通过后再运行；运行中失败即停，不自动重试或改参。
- 证据分层：小型证据（manifest、覆盖、逐轮 history、summary、验证预测、报告）入库；权重、日志与运行目录不入库。
- 许可：上游 MIT 原样保留；`experiments/aca/vendor/` 的 ACANet 源码为 GPL-3.0，原始许可文件随之保留。
- 不向任何上游作者仓库推送；本仓库只推送到本账号自己的 `origin`。

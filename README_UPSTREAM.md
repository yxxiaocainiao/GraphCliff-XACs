# GraphCliff

GraphCliff: Short–Long Range Gating for Modeling Critical Activity Changes Caused by Subtle Molecular Differences

Reference implementation of **GraphCliff**, a graph neural network for activity
cliff prediction. Each filter layer runs a short-range edge-aware path (GINE) and
a long-range spectral path (Chebyshev polynomial) in parallel, and combines them
with a learned per-atom gate

```
u = sigma(x1) * Long(x2) + v
```

so that a low gate suppresses long-range smoothing and lets the local signal pass
through unchanged — exactly the behaviour needed where a single-atom modification
causes a large activity change.

## Table of Contents
1. [Results](#results)
2. [Environment Setup](#environment-setup)
3. [Data](#data)
4. [Checkpoints](#checkpoints)
5. [Training GraphCliff](#training-graphcliff)
6. [Analysis](#analysis)
7. [Repository Layout](#repository-layout)

## Results

Aggregated over the 30 MoleculeACE targets (lower is better):

| Algorithm | Desc. | RMSE | RMSE_cliff | RMSE_noncliff |
|---|---|---:|---:|---:|
| **GraphCliff** | GRAPH | **0.665** | 0.757 | **0.619** |
| SVM | ECFP | 0.671 | **0.751** | 0.629 |
| GINE + PairNorm | GRAPH | 0.706 | 0.799 | 0.662 |
| GINE + NodeNorm | GRAPH | 0.717 | 0.797 | 0.680 |
| GINE + Residual | GRAPH | 0.728 | 0.811 | 0.688 |
| Chemprop | GRAPH | 0.730 | 0.816 | 0.694 |
| GCN | GRAPH | 0.914 | 0.974 | 0.892 |
| GAT | GRAPH | 0.947 | 1.005 | 0.925 |

See the paper for the full baseline list and per-target numbers.

## Environment Setup

1. **Install Miniconda or Anaconda** and ensure the GPU driver / CUDA version
   matches the one pinned in `envs/graphcliff.yaml` (CUDA 11.8, PyTorch 2.4.0).

2. **Create the environment:**
   ```bash
   conda env create -f envs/graphcliff.yaml
   conda activate graphcliff
   pip install https://data.pyg.org/whl/torch-2.4.0%2Bcu118/torch_scatter-2.1.2%2Bpt24cu118-cp310-cp310-linux_x86_64.whl
   ```

   `model.py` uses `torch.nn.RMSNorm`, which requires **PyTorch >= 2.4**. If a
   different PyTorch is picked up from your user site-packages, run with
   `PYTHONNOUSERSITE=1` or remove the shadowing install.

## Data

`benchmark_data/` holds the 30 MoleculeACE targets plus 11 additional targets,
one CSV per target:

| Column | Meaning |
|---|---|
| `smiles` | molecule SMILES |
| `exp_mean [nM]` | measured Ki / EC50 |
| `y` | regression target (standardised `-log10` potency) |
| `cliff_mol` | 1 if the molecule belongs to at least one activity cliff pair |
| `split` | official MoleculeACE `train` / `test` assignment |

`benchmark_data/raw/` keeps the unprocessed ChEMBL exports, and
`benchmark_data/metadata/MoleculeACE_metadata.csv` the target descriptions.

**Validation split.** MoleculeACE ships only `train` and `test`. Selecting a
checkpoint on `test` would leak the evaluation set, so `dataset.py` deterministically
holds out a fraction of the official training split as `valid`
(`--valid_frac`, default 0.1; `--split_seed`, default 42), stratified on
`cliff_mol` so the cliff ratio is preserved. Early stopping and checkpointing use
`valid` only; `test` is touched once, for the final report.

## Checkpoints

The trained checkpoints for all 30 targets are hosted on Zenodo rather than in
this repository: https://zenodo.org/records/21787497

```bash
wget https://zenodo.org/records/21787497/files/graphcliff_ckpt.tar.gz
tar xzf graphcliff_ckpt.tar.gz
mv 250920_final_ckpt/* ckpt/ && rmdir 250920_final_ckpt
```

Extracting them into `ckpt/` reproduces the reported numbers without retraining
and lets the analysis scripts run directly. See [`ckpt/README.md`](ckpt/README.md)
for the expected layout and the checksum.

## Training GraphCliff

Single target:

```bash
python main.py --dataset CHEMBL204_Ki --epochs 100
```

All 30 targets across your GPUs — set `NUM_GPUS` in `run.sh` first. If a
checkpoint already exists under `ckpt/<dataset>/`, it is loaded and evaluated
instead of retraining:

```bash
bash run.sh
```

Useful flags:

| Flag | Default | Meaning |
|---|---|---|
| `--valid_frac` | `0.1` | fraction of the train split held out for model selection |
| `--split_seed` | `42` | seed of the deterministic train/valid split |
| `--hidden_size` | `256` | embedding width |
| `--num_layers` | `3` | number of GraphCliff filter layers |
| `--mid_K` | `3` | Chebyshev polynomial order of the long-range path |
| `--groups` | `4` | channel groups sharing Chebyshev coefficients |
| `--patience` | `15` | early-stopping patience on validation MSE |
| `--save_model` | off | also write `final_model.pt` next to the best checkpoint |
| `-w` / `--log_wandb` | off | log to Weights & Biases |

Checkpoints land in `ckpt/<dataset>/<timestamp>/best_model.pt`.

## Analysis

The scripts under `analysis/` reproduce the analyses reported in the paper. Run
them **from the repository root**, after training (they read `ckpt/`). Outputs go
to `analysis_out/`.

### Over-smoothing (Figure 4)

Layer-wise Mean Average Distance and final-layer Dirichlet energy for GraphCliff
and the baseline stacks (GCN, GIN, GINE, GAT, and GINE with Residual / DropEdge /
NodeNorm / PairNorm). Each baseline is trained with the same recipe as `main.py`,
then measured on the test split.

```bash
python -u analysis/oversmoothing.py --dataset CHEMBL204_Ki --epochs 100
python -u analysis/oversmoothing.py --dataset CHEMBL204_Ki --models gine gine_pairnorm graphcliff
```

Writes `oversmoothing_mad_<dataset>.csv`, `oversmoothing_dirichlet_<dataset>.csv`
and a two-panel figure. Requires `matplotlib` for the figure; the CSVs are written
either way.

### Top-k gate alignment (Table 5)

For every activity cliff pair, rank atoms by their mean Layer-2 gate `sigma(x1)`,
take the `k` smallest, and measure the overlap with the atoms that structurally
differ between the two compounds (the non-MCS atoms), stratified by the split
membership of the pair.

```bash
# Phase 1 caches gates + MCS; phase 2 computes the alignment
python -u analysis/topk_gate_alignment.py --mode all --k 1 3 5

# "Random" columns of the same table (hypergeometric chance level)
python -u analysis/random_baseline.py
```

### Gate heatmap (Figure 6)

Per-atom gate values for one cliff pair, next to the ground-truth non-MCS atoms.
The defaults reproduce the published panel — the CHEMBL244 (Ki) test–test pair
differing only in a benzimidazole dihalogen (5,6-dichloro, Ki = 140 nM vs.
5,6-difluoro, Ki = 6 nM).

```bash
python analysis/gate_heatmap.py
python analysis/gate_heatmap.py --dataset CHEMBL239_EC50 --smiles1 <SMILES> --smiles2 <SMILES>
```

To pick a different pair, rank all test–test cliff pairs by how cleanly their
lowest-gate atoms coincide with the non-MCS atoms (needs the phase-1 caches
above):

```bash
python analysis/find_cliff_pair.py
```

## Repository Layout

```
.
├── main.py                       training / evaluation entry point
├── train.py                      train and evaluation loops
├── model.py                      GraphCliff: AtomEncoder, ShortGINE, LongPoly, gated filter
├── dataset.py                    GraphDataset, train/valid split, batch collation
├── dataset_utils.py              RDKit SMILES -> graph featurisation
├── cliffs.py                     MoleculeACE activity cliff definition
├── metric.py                     calc_rmse / calc_cliff_rmse
├── run.sh                        multi-GPU launcher over all 30 targets
├── envs/graphcliff.yaml          conda environment
├── benchmark_data/               MoleculeACE CSVs (+ raw/ and metadata/)
├── ckpt/                         checkpoints (downloaded from Zenodo, git-ignored)
└── analysis/
    ├── oversmoothing.py          layer-wise MAD + Dirichlet energy
    ├── topk_gate_alignment.py    gate / non-MCS atom overlap
    ├── random_baseline.py        chance level for the table above
    ├── gate_heatmap.py           per-atom gate figure for one cliff pair
    └── find_cliff_pair.py        helper to pick a pair for the heatmap
```

## License

MIT — see [LICENSE](LICENSE).

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from dataset_utils import smiles_to_graph


def split_train_valid(data: pd.DataFrame, valid_frac: float = 0.1, split_seed: int = 42):
    """Carve a validation set out of the official MoleculeACE training split.

    MoleculeACE ships only `train` / `test`. Selecting the checkpoint on `test`
    leaks the evaluation set, so we hold out a fraction of `train` as `valid`
    and use it for model selection / early stopping. The test split is never
    touched during training.

    The split is stratified on `cliff_mol` (when present) so that the cliff
    ratio of the training pool is preserved in both parts, and is fully
    determined by `split_seed`.

    Returns:
        (train_idx, valid_idx): positional indices into `data`.
    """
    train_pos = np.flatnonzero((data['split'] == 'train').to_numpy())
    if valid_frac <= 0:
        return train_pos, np.array([], dtype=int)

    if 'cliff_mol' in data.columns:
        strata = data['cliff_mol'].to_numpy()[train_pos]
    else:
        strata = np.zeros(len(train_pos), dtype=int)

    rng = np.random.RandomState(split_seed)
    train_idx, valid_idx = [], []
    for value in np.unique(strata):
        group = train_pos[strata == value]
        group = group[rng.permutation(len(group))]
        n_valid = int(round(len(group) * valid_frac))
        # Keep at least one molecule on each side of the split.
        n_valid = min(max(n_valid, 1), len(group) - 1) if len(group) > 1 else 0
        valid_idx.append(group[:n_valid])
        train_idx.append(group[n_valid:])

    train_idx = np.sort(np.concatenate(train_idx))
    valid_idx = np.sort(np.concatenate(valid_idx)) if valid_idx else np.array([], dtype=int)
    return train_idx, valid_idx


class GraphDataset(Dataset):
    def __init__(self, dataset_name: str, split: str = 'train',
                 valid_frac: float = 0.1, split_seed: int = 42):
        super().__init__()
        data = pd.read_csv(f'./benchmark_data/{dataset_name}.csv')

        if split in ('train', 'valid'):
            train_idx, valid_idx = split_train_valid(data, valid_frac, split_seed)
            rows = data.iloc[train_idx if split == 'train' else valid_idx]
        elif split == 'test':
            rows = data[data['split'] == 'test']
        else:
            raise ValueError(f"Unknown split: {split!r} (expected 'train', 'valid' or 'test')")

        smiles_list = rows.smiles.values.tolist()
        y_list = rows.y.values.tolist()

        assert smiles_list is not None, "Could not find smiles list in MoleculeACE Data object."
        self.samples = []
        for s, y in zip(smiles_list, y_list):
            try:
                graph = smiles_to_graph(s, y)
                graph.smiles = s
                self.samples.append(graph)
            except Exception as e:
                continue
        if 'cliff_mol' in data.columns:
            self.cliff_mols_test = data[data['split'] == 'test']['cliff_mol'].values.tolist()
            self.cliff_mols = rows['cliff_mol'].values.tolist()
        else:
            self.cliff_mols_test = None
            self.cliff_mols = None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]



def collate_graphs(batch):
    xs, eis, eas, ys, batch_idx = [], [], [], [], []
    node_offset = 0
    for gi, g in enumerate(batch):
        xs.append(g.x)

        if g.edge_index.numel() > 0:
            eis.append(g.edge_index + node_offset)
            eas.append(g.edge_attr)
        else:
            eis.append(torch.zeros((2,0), dtype=torch.long))
            eas.append(torch.zeros((0, batch[0].edge_attr.size(1)), dtype=torch.float32))

        ys.append([g.y])
        batch_idx.append(torch.full((g.x.size(0),), gi, dtype=torch.long))
        node_offset += g.x.size(0)

    x = torch.cat(xs, dim=0)
    edge_index = torch.cat(eis, dim=1) if eis else torch.zeros((2,0), dtype=torch.long)
    edge_attr = torch.cat(eas, dim=0) if eas else torch.zeros((0,13), dtype=torch.float32)
    y = torch.tensor(ys, dtype=torch.float32)
    batch_vec = torch.cat(batch_idx, dim=0)

    return {
        'x': x,
        'edge_index': edge_index,
        'edge_attr': edge_attr,
        'y': y,
        'batch': batch_vec
    }

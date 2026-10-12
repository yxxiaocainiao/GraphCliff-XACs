"""GPL-3.0: read-only mechanism diagnostic for the XACs explanation-supervision pilot.

Question: did the treatment arm actually change attribution behaviour, and does the
attribution/potency ordering generalise to cliff pairs that were never used for training?

Pairs are mined from the development-VALIDATION molecules only. Validation labels were
already used to select checkpoints; they are never used here for training or selection.
Official test labels are not read.
"""
import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/xacs_explain'))
spec = importlib.util.spec_from_file_location('xacs_explain_run', ROOT / 'experiments/xacs_explain/run.py')
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)
base, torch, np, pd, F = run.base, run.torch, run.np, run.pd, run.F


def valid_pairs(task, valid):
    """XACs pairs with both members inside the validation split (diagnostic only)."""
    ann = run.annotations(task, valid)
    return ann['pairs'], ann['skipped_degenerate']


def eval_attribution(model, batch):
    """Same GradCAM definition as run.attribution, but usable with the model in eval mode."""
    captured = []
    hook = model.sagpool.register_forward_pre_hook(lambda module, args: captured.append(args[0]))
    pred = model(batch['x'], batch['edge_index'], batch['edge_attr'], batch['batch'])
    activation = captured.pop()
    hook.remove()
    grads = torch.autograd.grad(pred.sum(), activation, retain_graph=False)[0]
    mol = batch['batch']
    molecules = pred.shape[0]
    weight = torch.zeros(molecules, activation.shape[1], device=activation.device)
    weight.index_add_(0, mol, grads)
    count = torch.zeros(molecules, device=activation.device).index_add_(
        0, mol, torch.ones_like(mol, dtype=activation.dtype))
    weight = weight / count.unsqueeze(-1)
    return pred, F.relu((weight[mol] * activation).sum(-1))


def per_atom_attribution(model, graphs, batch_size=64):
    """GradCAM attribution per atom for every molecule, returned as a list of tensors."""
    out = []
    model.eval()
    for start in range(0, len(graphs), batch_size):
        chunk = graphs[start:start + batch_size]
        batch = run.to_device(base.collate_graphs(chunk))
        _, att = eval_attribution(model, batch)
        nodes = torch.bincount(batch['batch'], minlength=len(chunk))
        offset = torch.cat([torch.zeros(1, dtype=torch.long, device=nodes.device),
                            nodes.cumsum(0)[:-1]])
        for index in range(len(chunk)):
            piece = att[int(offset[index]):int(offset[index]) + int(nodes[index])]
            out.append(piece.detach().cpu())
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default=str(ROOT / 'docs/xacs_explain_20261012/mechanism.json'))
    args = parser.parse_args()
    manifest = json.loads((run.OUT / 'manifest.json').read_text(encoding='utf-8'))
    rows = []
    for task in run.TASKS:
        _, valid, _, valid_graphs = base.development(task)
        pairs, skipped = valid_pairs(task, valid)
        print(task, 'validation-internal pairs', len(pairs), 'skipped', skipped, flush=True)
        for seed in run.SEEDS:
            for arm in run.ARMS:
                folder = run.OUT / task / str(seed) / arm
                if not (folder / 'best_model.pt').exists():
                    continue
                model = run.model_new(seed)
                model.load_state_dict(torch.load(folder / 'best_model.pt',
                                                 weights_only=True)['model_state_dict'], strict=True)
                attribution = per_atom_attribution(model, valid_graphs)
                hook = None
                satisfied = total = 0
                share_uncom, share_all = [], []
                for pair in pairs:
                    att_a, att_b = attribution[pair['a']], attribution[pair['b']]
                    uncom_a = float(att_a[pair['uncom_i']].sum())
                    uncom_b = float(att_b[pair['uncom_j']].sum())
                    if uncom_a + uncom_b > 0:
                        total += 1
                        satisfied += int((uncom_a - uncom_b) * pair['delta'] >= 0)
                    for att, uncom in ((att_a, pair['uncom_i']), (att_b, pair['uncom_j'])):
                        mass = float(att.sum())
                        if mass > 0:
                            share_uncom.append(float(att[uncom].sum()) / mass)
                            share_all.append(float(att.sum()))
                rows.append(dict(task=task, seed=seed, arm=arm, pairs=len(pairs),
                                 informative_pairs=total,
                                 hinge_satisfied=satisfied,
                                 hinge_satisfaction_rate=(satisfied / total) if total else None,
                                 mean_uncommon_attribution_share=float(np.mean(share_uncom)) if share_uncom else None,
                                 median_uncommon_atoms=None))
                del model
                torch.cuda.empty_cache()
    frame = pd.DataFrame(rows)
    frame.to_csv(Path(args.out).with_suffix('.csv'), index=False)
    grouped = frame.groupby(['task', 'arm']).agg(
        hinge_satisfaction_rate=('hinge_satisfaction_rate', 'mean'),
        uncommon_attribution_share=('mean_uncommon_attribution_share', 'mean')).reset_index()
    summary = dict(rows=rows, by_task_arm=grouped.to_dict(orient='records'),
                   note=('Validation-internal pairs; labels used for diagnosis only. '
                         'Attribution share is the GradCAM mass on the annotated uncommon atoms '
                         'divided by total attribution mass of that molecule.'),
                   weight=manifest['weight'])
    Path(args.out).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print(grouped.to_string(index=False))
    print('wrote', args.out)


if __name__ == '__main__':
    main()

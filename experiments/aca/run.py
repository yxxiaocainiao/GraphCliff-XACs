"""GPL-3.0 ACA pilot; upstream GraphCliff model and checkpoint files stay untouched."""
import argparse
import hashlib
import io
import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from torch.utils.data import DataLoader
from dataset import split_train_valid, collate_graphs
from dataset_utils import smiles_to_graph
from main import set_seed, WarmupCosineScheduler
from model import GraphCliffRegressor
from train import evaluate
from loss_adapter import upstream_class

OUT = ROOT / 'artifacts/aca_pilot_20261010'
TASKS = ('CHEMBL234_Ki', 'CHEMBL2047_EC50')

def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')

def state_hash(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()

def development(task):
    path = ROOT / 'benchmark_data' / (task + '.csv')
    membership = pd.read_csv(path, usecols=['split'])
    ids = np.flatnonzero(membership['split'].eq('train').to_numpy())
    keep = set((ids + 1).tolist())
    frame = pd.read_csv(path, skiprows=lambda line: line > 0 and line not in keep,
                        usecols=['smiles', 'y', 'cliff_mol', 'split', 'exp_mean [nM]'])
    assert len(frame) == len(ids) and frame['split'].eq('train').all()
    frame['row_id'] = ids
    ti, vi = split_train_valid(frame, valid_frac=.1, split_seed=42)
    train, valid = frame.iloc[ti].copy(), frame.iloc[vi].copy()
    canonical = lambda rows: {Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in rows.smiles}
    assert not canonical(train).intersection(canonical(valid)), 'canonical train/valid overlap'
    expected = -np.log10(train['exp_mean [nM]'].to_numpy())
    assert np.isfinite(train.y).all() and np.max(np.abs(expected - train.y)) < 1e-6
    assert valid.cliff_mol.sum() > 0
    graphs = lambda rows: [smiles_to_graph(s, float(y)) for s, y in zip(rows.smiles, rows.y)]
    return train, valid, graphs(train), graphs(valid)

def model_new():
    set_seed(SimpleNamespace(seed=42))
    return GraphCliffRegressor(38, 13, hidden_size=256, num_layers=3, groups=4, mid_K=3, dropout=0).cuda()

def criterion(alpha=.1):
    return upstream_class()(alpha=alpha, cliff_lower=.2, cliff_upper=1., squared=False,
                            p=1., similarity_gate=False, dev_mode=True)

def metrics(labels, preds, cliff):
    errors = (np.asarray(labels) - np.asarray(preds)) ** 2
    return dict(rmse=float(np.sqrt(errors.mean())),
                cliff_rmse=float(np.sqrt(errors[np.asarray(cliff).astype(bool)].mean())))

def check():
    y = torch.tensor([0., .1, 1.2], device='cuda')
    p = torch.tensor([.2, .3, .8], device='cuda', requires_grad=True)
    z = torch.tensor([[0., 0.], [1., 0.], [.2, 0.]], device='cuda', requires_grad=True)
    result = criterion()(y, p, z)
    assert result[5] == 2 and result[6] == 2
    # Explicit enumeration: (0,1,2) score 1.9; (1,0,2) score 1.2.
    assert torch.allclose(result[2], torch.tensor(1.55, device='cuda'))
    zero = criterion(0)(y, p, z)[0]
    mae = torch.nn.functional.l1_loss(p, y)
    assert torch.equal(zero, mae)
    assert torch.equal(torch.autograd.grad(zero, p, retain_graph=True)[0],
                       torch.autograd.grad(mae, p, retain_graph=True)[0])
    assert torch.autograd.grad(zero, z, retain_graph=True)[0].abs().sum() == 0
    assert torch.autograd.grad(result[0], z)[0].abs().sum() > 0
    empty = criterion()(torch.zeros_like(y), p, z)
    assert empty[5] == 0 and empty[2] == 0
    report = dict(synthetic='passed', tasks={})
    for task in TASKS:
        tr, va, tg, _ = development(task)
        model = model_new()
        batch = collate_graphs(tg[:32])
        captured = []
        hook = model.reg_head.register_forward_pre_hook(lambda module, args: captured.append(args[0]))
        pred = model(batch['x'].cuda(), batch['edge_index'].cuda(), batch['edge_attr'].cuda(), batch['batch'].cuda())
        loss = criterion()(batch['y'].cuda(), pred, captured.pop())
        loss[0].backward()
        assert torch.isfinite(loss[0]) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        model.eval()
        with torch.no_grad():
            hooked = model(batch['x'].cuda(), batch['edge_index'].cuda(), batch['edge_attr'].cuda(), batch['batch'].cuda())
        hook.remove()
        with torch.no_grad():
            plain = model(batch['x'].cuda(), batch['edge_index'].cuda(), batch['edge_attr'].cuda(), batch['batch'].cuda())
        assert torch.equal(hooked, plain), 'embedding hook changed predictions'
        checkpoint = io.BytesIO()
        torch.save({'model_state_dict': model.state_dict()}, checkpoint)
        checkpoint.seek(0)
        model.load_state_dict(torch.load(checkpoint, weights_only=True)['model_state_dict'])
        captured.clear()
        report['tasks'][task] = dict(train=len(tr), valid=len(va), valid_cliff=int(va.cliff_mol.sum()),
                                    candidates=loss[5], active=loss[6], loss=float(loss[0].detach()))
        del model
    OUT.mkdir(parents=True, exist_ok=True)
    save(OUT / 'preflight.json', report)
    print(json.dumps(report), flush=True)

def run(task, recover=False):
    tr, va, tg, vg = development(task)
    folder = OUT / task
    folder.mkdir(parents=True, exist_ok=recover)
    split = dict(train=tr.row_id.tolist(), valid=va.row_id.tolist(), test_labels_read=False)
    if recover:
        assert json.loads((folder/'split.json').read_text()) == split
    else:
        save(folder / 'split.json', split)
    source_files = ['model.py', 'dataset.py', 'dataset_utils.py', 'main.py', 'train.py',
                    'experiments/aca/run.py', 'experiments/aca/loss_adapter.py', 'experiments/aca/protocol.json']
    provenance = dict(
        git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        git_status=subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True),
        source_sha256={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in source_files},
        data_sha256=hashlib.sha256((ROOT/'benchmark_data'/f'{task}.csv').read_bytes()).hexdigest(),
        protocol=json.loads((ROOT/'experiments/aca/protocol.json').read_text()),
        torch=torch.__version__, cuda=torch.version.cuda, device=torch.cuda.get_device_name(),
        label_scale='y=-log10(nM), verified on training rows', pretrained_checkpoint_used=False)
    if recover:
        original = json.loads((folder/'provenance.json').read_text())
        assert original['protocol'] == provenance['protocol'] and original['data_sha256'] == provenance['data_sha256']
        for name in ('model.py', 'dataset.py', 'dataset_utils.py', 'main.py', 'train.py', 'experiments/aca/loss_adapter.py'):
            assert original['source_sha256'][name] == provenance['source_sha256'][name]
        save(folder/'recovery.json', dict(reason='fix checkpoint envelope extraction; no training or loss changes', **provenance))
    else:
        save(folder/'provenance.json', provenance)
    initial = None
    for arm in ('mse', 'mae', 'aca'):
        start = time.time()
        destination = folder / arm
        if recover and (destination/'summary.json').exists():
            result = json.loads((destination/'summary.json').read_text())
            initial = initial or result['initial_sha256']
            assert initial == result['initial_sha256']
            continue
        destination.mkdir(exist_ok=recover)
        model = model_new()
        digest = state_hash(model)
        initial = initial or digest
        assert initial == digest
        captured = []
        hook = model.reg_head.register_forward_pre_hook(lambda module, args: captured.append(args[0])) if arm == 'aca' else None
        loader = DataLoader(tg, batch_size=32, shuffle=True, collate_fn=collate_graphs,
                            generator=torch.Generator().manual_seed(42))
        valid_loader = DataLoader(vg, batch_size=32, shuffle=False, collate_fn=collate_graphs)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5, betas=(.9,.95))
        scheduler = WarmupCosineScheduler(optimizer, 10, 100, 1e-4)
        aca = criterion()
        best, stale, history = float('inf'), 0, []
        if recover and (destination/'history.json').exists():
            history = json.loads((destination/'history.json').read_text())
            best_epoch = min(history, key=lambda row: row['rmse'])['epoch']
            assert len(history) == 100 or len(history)-best_epoch >= 15, 'unfinished training cannot be recovered by this entry'
        for epoch in range(100 if history else 0, 100):
            model.train()
            reg_sum = triplet_sum = total_sum = count = candidates = active = empty = 0
            for batch in loader:
                optimizer.zero_grad(set_to_none=True)
                pred = model(batch['x'].cuda(), batch['edge_index'].cuda(), batch['edge_attr'].cuda(), batch['batch'].cuda())
                y = batch['y'].cuda()
                if arm == 'aca':
                    loss, reg, triplet, _, _, nc, nh = aca(y, pred, captured.pop())
                else:
                    loss = (torch.nn.functional.mse_loss if arm == 'mse' else torch.nn.functional.l1_loss)(pred, y)
                    reg, triplet, nc, nh = loss, loss.new_zeros(()), 0, 0
                assert torch.isfinite(loss)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                n = len(y)
                count += n
                reg_sum += float(reg.detach()) * n
                triplet_sum += float(triplet.detach()) * n
                total_sum += float(loss.detach()) * n
                candidates += nc
                active += nh
                empty += int(nc == 0)
            if arm == 'aca' and epoch == 0:
                assert candidates > 0, 'no candidate triplets in first epoch'
            labels, predictions = evaluate(model, valid_loader, 'cuda')
            captured.clear()
            score = metrics(labels, predictions, va.cliff_mol)
            val_mse = float(np.mean((np.asarray(labels) - predictions) ** 2))
            assert np.isfinite(val_mse)
            row = dict(epoch=epoch+1, reg=reg_sum/count, triplet=triplet_sum/count, loss=total_sum/count,
                       candidates=candidates, active=active, empty_batches=empty, lr=optimizer.param_groups[0]['lr'], **score)
            history.append(row)
            if val_mse < best:
                best, stale, best_epoch = val_mse, 0, epoch+1
                torch.save({'model_state_dict': model.state_dict()}, destination / 'best_model.pt')
            else:
                stale += 1
            save(destination / 'history.json', history)
            if (epoch+1) % 5 == 0 or epoch == 0:
                print(f'{task} {arm} epoch={epoch+1} rmse={score["rmse"]:.4f} cliff={score["cliff_rmse"]:.4f} stale={stale}', flush=True)
            scheduler.step(epoch)
            if stale >= 15:
                break
        if hook:
            hook.remove()
        model.load_state_dict(torch.load(destination / 'best_model.pt', weights_only=True)['model_state_dict'])
        labels, predictions = evaluate(model, valid_loader, 'cuda')
        pd.DataFrame(dict(row_id=va.row_id, label=labels, prediction=predictions, cliff=va.cliff_mol)).to_csv(destination / 'valid_predictions.csv', index=False)
        result = dict(task=task, arm=arm, initial_sha256=digest, epochs=len(history), best_epoch=best_epoch,
                      training_seconds=None if recover and arm == 'mse' else time.time()-start,
                      recovered=recover and arm == 'mse', train=len(tr), valid=len(va), valid_cliff=int(va.cliff_mol.sum()),
                      **metrics(labels, predictions, va.cliff_mol))
        save(destination / 'summary.json', result)
        print('FINISHED ' + json.dumps(result), flush=True)
        del model, optimizer
        torch.cuda.empty_cache()

def audit():
    results = []
    for task in TASKS:
        _, va, _, vg = development(task)
        loader = DataLoader(vg, batch_size=32, collate_fn=collate_graphs)
        hashes = set()
        for arm in ('mse', 'mae', 'aca'):
            path = OUT / task / arm
            result = json.loads((path/'summary.json').read_text(encoding='utf-8'))
            hashes.add(result['initial_sha256'])
            model = model_new()
            model.load_state_dict(torch.load(path/'best_model.pt', weights_only=True)['model_state_dict'])
            labels, predictions = evaluate(model, loader, 'cuda')
            stored = pd.read_csv(path/'valid_predictions.csv')
            assert stored.row_id.tolist() == va.row_id.tolist()
            assert np.max(np.abs(np.asarray(labels)-stored.label)) < 1e-6
            difference = float(np.max(np.abs(np.asarray(predictions)-stored.prediction)))
            assert difference < 1e-5
            measured = metrics(labels, predictions, va.cliff_mol)
            for key, value in measured.items():
                assert abs(value-result[key]) < 1e-5
            result['replay_max_abs'] = difference
            results.append(result)
            del model
        assert len(hashes) == 1
    save(OUT/'audit.json', results)
    print(json.dumps(results, indent=2), flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['check', 'run', 'recover', 'audit'])
    parser.add_argument('--task', choices=TASKS)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.action == 'check':
        check()
    elif args.action == 'audit':
        audit()
    else:
        assert args.task
        run(args.task, recover=args.action == 'recover')

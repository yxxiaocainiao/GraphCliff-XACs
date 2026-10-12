"""GPL-3.0: XACs/ACES-style atom-level explanation supervision on frozen GraphCliff.

Frozen protocol: experiments/xacs_explain/task_plan.md
Only development-train molecules and their XACs annotations are used; official test
members are never read for training, mining or selection.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import pickle
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/aca'))
import run as base
spec = importlib.util.spec_from_file_location('aca_local_tools', ROOT / 'experiments/aca_local/pilot.py')
local = importlib.util.module_from_spec(spec)
spec.loader.exec_module(local)
torch, np, pd = base.torch, base.np, base.pd
F = torch.nn.functional

OUT = ROOT / 'artifacts/xacs_explain_20261012'
DOC = ROOT / 'docs/xacs_explain_20261012'
XACS = Path(r'D:\XACs\Data')
TASKS = ('CHEMBL234_Ki', 'CHEMBL2047_EC50', 'CHEMBL2147_Ki', 'CHEMBL244_Ki')
SEEDS = (42, 43, 44)
ARMS = ('mse', 'explain')
ORDINARY, PAIR_BUDGET, MAX_MOL = 124, 48, 192
CANDIDATE_W = (1e-4, 1e-3, 1e-2, 1e-1)
TARGET_RATIO = .02
SOURCES = ('model.py', 'dataset.py', 'dataset_utils.py', 'main.py', 'train.py',
           'experiments/aca/run.py', 'experiments/aca_local/pilot.py',
           'experiments/xacs_explain/run.py', 'experiments/xacs_explain/task_plan.md')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def model_new(seed):
    base.set_seed(SimpleNamespace(seed=seed))
    return base.GraphCliffRegressor(38, 13, hidden_size=256, num_layers=3,
                                    groups=4, mid_K=3, dropout=0).cuda()


def to_device(batch, device='cuda'):
    return {key: (value.to(device) if torch.is_tensor(value) else value)
            for key, value in batch.items()}


def annotations(task, train):
    """XACs pairs restricted to development-train molecules, with atom index lists.

    Masks are per (anchor, partner) pair, exactly as in the authors' `pack_data`.
    """
    pkl = XACS / task / 'mcs_dict_0.9.pkl'
    assert pkl.exists(), f'missing annotation file {pkl}'
    with pkl.open('rb') as fh:
        raw = pickle.load(fh)
    pos = {s: i for i, s in enumerate(train.smiles)}
    ys = train.y.to_numpy(dtype=np.float64)
    pairs, skipped = [], 0
    for anchor, records in raw.items():
        if anchor not in pos:
            continue
        ai = pos[anchor]
        for rec in records[1:]:
            partner = rec['smiles']
            if partner not in pos:
                continue
            bi = pos[partner]
            delta = float(ys[ai] - ys[bi])
            assert abs(delta - float(rec['potency_diff'])) < 1e-6, 'potency_diff sign/scale mismatch'
            mi = np.asarray(rec['atom_mask_i'], dtype=np.uint8).ravel()
            mj = np.asarray(rec['atom_mask_j'], dtype=np.uint8).ravel()
            assert mi.size and mj.size
            if int(mi.sum()) + int(mj.sum()) == 0:
                # MCS covers every atom on both sides: nothing can be attributed
                skipped += 1
                continue
            pairs.append(dict(a=ai, b=bi, delta=delta,
                              uncom_i=np.flatnonzero(mi), com_i=np.flatnonzero(mi == 0),
                              uncom_j=np.flatnonzero(mj), com_j=np.flatnonzero(mj == 0)))
    return dict(task=task, pkl=str(pkl), pkl_sha256=digest(pkl), pairs=pairs,
                skipped_degenerate=skipped)


def batches(ann, count, seed, epoch):
    """Ordinary molecules (same sampler as the existing dev pilots) plus injected pairs."""
    rng = np.random.RandomState(seed * 1000 + epoch)
    order = rng.permutation(count)
    result = []
    for start in range(0, count, ORDINARY):
        fill = order[start:start + ORDINARY].tolist()
        slot = {row: index for index, row in enumerate(fill)}
        ids = list(fill)
        prng = np.random.RandomState(seed * 1000 + epoch + 7 + start)
        n = min(PAIR_BUDGET, len(ann['pairs']))
        chosen = prng.choice(len(ann['pairs']), size=n, replace=False) if n else np.array([], dtype=int)
        picked = []
        for index in chosen:
            pair = ann['pairs'][int(index)]
            new = [row for row in (pair['a'], pair['b']) if row not in slot]
            if len(ids) + len(new) > MAX_MOL:
                continue
            for row in new:
                slot[row] = len(ids)
                ids.append(row)
            picked.append(pair)
        result.append((torch.tensor(ids, dtype=torch.long), slot, picked))
    return result


def gather(ids, slot, picked, batch):
    """Node-index tensors for the uncommon/common attribution sums of this batch's pairs."""
    nodes = torch.bincount(batch['batch'], minlength=len(ids))
    offset = torch.cat([torch.zeros(1, dtype=torch.long, device=nodes.device), nodes.cumsum(0)[:-1]])
    device = batch['x'].device
    a_uncom, a_pair, b_uncom, b_pair, common, delta = [], [], [], [], [], []
    for index, pair in enumerate(picked):
        oa, ob = int(offset[slot[pair['a']]]), int(offset[slot[pair['b']]])
        a_uncom.append(torch.tensor(pair['uncom_i'], dtype=torch.long) + oa)
        b_uncom.append(torch.tensor(pair['uncom_j'], dtype=torch.long) + ob)
        a_pair.append(torch.full((len(pair['uncom_i']),), index, dtype=torch.long))
        b_pair.append(torch.full((len(pair['uncom_j']),), index, dtype=torch.long))
        common.append(torch.tensor(pair['com_i'], dtype=torch.long) + oa)
        common.append(torch.tensor(pair['com_j'], dtype=torch.long) + ob)
        delta.append(pair['delta'])
    cat = lambda chunks: torch.cat(chunks) if chunks else torch.zeros(0, dtype=torch.long)
    return dict(a_uncom=cat(a_uncom).to(device), a_pair=cat(a_pair).to(device),
                b_uncom=cat(b_uncom).to(device), b_pair=cat(b_pair).to(device),
                common=cat(common).to(device),
                delta=torch.tensor(delta, dtype=torch.float32, device=device),
                uncom_atoms=sum(len(p['uncom_i']) + len(p['uncom_j']) for p in picked))


def attribution(model, batch, captured):
    """GradCAM attribution over the per-atom representation entering SAGPool."""
    pred = model(batch['x'], batch['edge_index'], batch['edge_attr'], batch['batch'])
    activation = captured.pop()
    grads = torch.autograd.grad(pred.sum(), activation, retain_graph=True)[0]
    mol = batch['batch']
    molecules = pred.shape[0]
    weight = torch.zeros(molecules, activation.shape[1], device=activation.device)
    weight.index_add_(0, mol, grads)
    count = torch.zeros(molecules, device=activation.device).index_add_(
        0, mol, torch.ones_like(mol, dtype=activation.dtype))
    weight = weight / count.unsqueeze(-1)
    return pred, F.relu((weight[mol] * activation).sum(-1))


def auxiliary(att, index):
    if not index['delta'].numel():
        return att.sum() * 0., dict(pairs=0, uncom_atoms=0, active=0, common_energy=0.)
    pairs = index['delta'].numel()
    uncom_a = torch.zeros(pairs, device=att.device).index_add_(0, index['a_pair'], att[index['a_uncom']])
    uncom_b = torch.zeros(pairs, device=att.device).index_add_(0, index['b_pair'], att[index['b_uncom']])
    hinge = F.relu(-(uncom_a - uncom_b) * index['delta'])
    common_energy = att[index['common']].square().sum()
    return common_energy + hinge.sum(), dict(pairs=pairs, uncom_atoms=index['uncom_atoms'],
                                             active=int((hinge.detach() > 0).sum()),
                                             common_energy=float(common_energy.detach()))


def validate(model, graphs, frame):
    loader = base.DataLoader(graphs, batch_size=128, collate_fn=base.collate_graphs)
    labels, predictions = base.evaluate(model, loader, 'cuda')
    return labels, predictions, base.metrics(labels, predictions, frame.cliff_mol)


def calibration(task, seed=42):
    """Pre-registered rule: median gradient-norm ratio closest to TARGET_RATIO."""
    train, _, train_graphs, _ = base.development(task)
    ann = annotations(task, train)
    ids, slot, picked = batches(ann, len(train), seed, 1)[0]
    batch = to_device(base.collate_graphs([train_graphs[i] for i in ids.tolist()]))
    index = gather(ids, slot, picked, batch)
    model = model_new(seed)
    captured = []
    hook = model.sagpool.register_forward_pre_hook(
        lambda module, args: captured.append(args[0]) if module.training else None)
    pred, att = attribution(model, batch, captured)
    reg = F.mse_loss(pred, batch['y'])
    aux, stats = auxiliary(att, index)
    shared = [value for name, value in model.named_parameters() if not name.startswith('reg_head.')]
    rows = {}
    for weight in CANDIDATE_W:
        rows[weight] = local.gradient_probe(reg, weight * aux, shared)
    all_parameters = local.gradient_probe(reg, aux, list(model.parameters()))
    hook.remove()
    del model
    torch.cuda.empty_cache()
    return dict(task=task, pairs=len(picked), samples=len(ids), aux_stats=stats,
                unit_weight_probe=rows[CANDIDATE_W[-1]], all_parameter_probe=all_parameters,
                ratios={str(w): rows[w]['grad_ratio'] for w in CANDIDATE_W})


def choose_weight(calibrations):
    medians = {}
    for weight in CANDIDATE_W:
        values = [row['ratios'][str(weight)] for row in calibrations if row['ratios'][str(weight)]]
        medians[weight] = float(np.median(values)) if values else 0.
    best = min(CANDIDATE_W, key=lambda w: (abs(math.log10(max(medians[w], 1e-12) / TARGET_RATIO)), w))
    return best, medians


def check():
    assert not (OUT / 'manifest.json').exists(), 'already frozen'
    OUT.mkdir(parents=True, exist_ok=True)
    DOC.mkdir(parents=True, exist_ok=True)

    # 1. synthetic formula checks: hinge direction, degenerate pairs, zero-weight equivalence
    att = torch.tensor([1., 1., 0., 0.], device='cuda', requires_grad=True)
    index = dict(a_uncom=torch.tensor([0], device='cuda'), a_pair=torch.tensor([0], device='cuda'),
                 b_uncom=torch.tensor([1], device='cuda'), b_pair=torch.tensor([0], device='cuda'),
                 common=torch.tensor([2, 3], device='cuda'),
                 delta=torch.tensor([1.], device='cuda'), uncom_atoms=2)
    # equal uncommon mass -> no ordering violation whatever the potency sign; no common mass
    value, stats = auxiliary(att, index)
    torch.testing.assert_close(value, att.new_tensor(0.))
    torch.testing.assert_close(
        auxiliary(att, dict(index, delta=torch.tensor([-1.], device='cuda')))[0], att.new_tensor(0.))
    # anchor more potent (delta>0) with larger uncommon mass -> hinge inactive, only common energy
    att2 = torch.tensor([2., 0., 1., 1.], device='cuda', requires_grad=True)
    value, stats = auxiliary(att2, index)
    torch.testing.assert_close(value, att2.new_tensor(2.))
    assert stats['active'] == 0
    # partner more potent: the same attribution ordering must now be penalised
    value, stats = auxiliary(att2, dict(index, delta=torch.tensor([-1.], device='cuda')))
    torch.testing.assert_close(value, att2.new_tensor(4.))     # hinge 2 + common energy 1+1
    assert stats['pairs'] == 1 and stats['active'] == 1
    empty, stats = auxiliary(att2, dict(index, delta=torch.zeros(0, device='cuda')))
    assert stats['pairs'] == 0 and float(empty.detach()) == 0
    assert torch.count_nonzero(torch.autograd.grad(empty, att2)[0]) == 0

    coverage, calibrations, timings = [], [], []
    for task in TASKS:
        train, valid, train_graphs, _ = base.development(task)
        membership = pd.read_csv(ROOT / 'benchmark_data' / f'{task}.csv', usecols=['split'])
        official_train = set(np.flatnonzero(membership.split.eq('train').to_numpy()).tolist())
        official_test = np.flatnonzero(membership.split.eq('test').to_numpy()).tolist()
        assert set(train.row_id) | set(valid.row_id) == official_train
        assert not set(train.row_id).intersection(valid.row_id)
        assert not official_train.intersection(official_test)
        ann = annotations(task, train)
        assert ann['pairs'], task + ': no development-train pairs'
        for pair in ann['pairs']:
            for side, keys in (('a', ('uncom_i', 'com_i')), ('b', ('uncom_j', 'com_j'))):
                atoms = train_graphs[pair[side]].x.shape[0]
                for key in keys:
                    assert int(pair[key].max(initial=-1)) < atoms, f'{task}: {key} beyond atom count'
                left = 'uncom_i' if side == 'a' else 'uncom_j'
                right = 'com_i' if side == 'a' else 'com_j'
                assert len(pair[left]) + len(pair[right]) == atoms
        deltas = np.asarray([p['delta'] for p in ann['pairs']])
        for seed in SEEDS:
            for epoch in range(1, 101):
                for ids, slot, picked in batches(ann, len(train), seed, epoch):
                    assert ids.numel() <= MAX_MOL and len(set(ids.tolist())) == ids.numel()
                    assert picked and len(picked) <= PAIR_BUDGET, 'batch without supervised pairs'
        # both arms must see identical batches (same deterministic builder)
        sample = batches(ann, len(train), 42, 1)
        again = batches(ann, len(train), 42, 1)
        assert all(torch.equal(x[0], y[0]) for x, y in zip(sample, again))
        first_ids, first_slot, first_picked = sample[0]
        batch = to_device(base.collate_graphs([train_graphs[i] for i in first_ids.tolist()]))
        index = gather(first_ids, first_slot, first_picked, batch)
        assert index['delta'].numel() == len(first_picked)
        model = model_new(42)
        captured = []
        hook = model.sagpool.register_forward_pre_hook(
            lambda module, args: captured.append(args[0]) if module.training else None)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5, betas=(.9, .95))
        started = time.monotonic()
        probe_weight = CANDIDATE_W[2]      # fixed weight, mechanics check only
        for step in range(3):
            optimizer.zero_grad(set_to_none=True)
            pred, att = attribution(model, batch, captured)
            assert pred.shape == batch['y'].shape and torch.isfinite(att).all()
            reg = F.mse_loss(pred, batch['y'])
            aux, stats = auxiliary(att, index)
            # step 0 exercises the exact zero-weight path of the treatment term
            loss = reg + (probe_weight * aux if step else 0. * aux)
            assert torch.isfinite(loss) and stats['pairs'] == len(first_picked)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
        torch.cuda.synchronize()
        hook.remove()
        checkpoint = OUT / f'preflight_{task}.pt'
        torch.save({'model_state_dict': model.state_dict()}, checkpoint)
        clone = model_new(42)
        clone.load_state_dict(torch.load(checkpoint, weights_only=True)['model_state_dict'], strict=True)
        model.eval()
        clone.eval()
        with torch.no_grad():
            left = model(batch['x'], batch['edge_index'], batch['edge_attr'], batch['batch'])
            right = clone(batch['x'], batch['edge_index'], batch['edge_attr'], batch['batch'])
        torch.testing.assert_close(left, right, atol=1e-6, rtol=1e-5)
        timings.append(dict(task=task, seconds_per_step=(time.monotonic() - started) / 3))
        coverage.append(dict(task=task, train=train.row_id.tolist(), valid=valid.row_id.tolist(),
                             test=official_test, pairs=len(ann['pairs']),
                             skipped_degenerate_pairs=ann['skipped_degenerate'],
                             anchors=len({p['a'] for p in ann['pairs']}),
                             delta_min=float(deltas.min()), delta_max=float(deltas.max()),
                             annotation_sha256=ann['pkl_sha256'],
                             data_sha256=digest(ROOT / 'benchmark_data' / f'{task}.csv')))
        del model, clone, optimizer
        torch.cuda.empty_cache()
        calibrations.append(calibration(task))
        print('CHECK', task, len(train), len(valid), len(ann['pairs']), flush=True)

    weight, medians = choose_weight(calibrations)
    order = []
    rng = np.random.RandomState(20261012)
    for task in TASKS:
        for seed in SEEDS:
            order.extend(dict(task=task, seed=seed, arm=arm) for arm in rng.permutation(ARMS))
    base.save(OUT / 'manifest.json', dict(
        splits=coverage, order=order, timing=timings, weight=weight, calibration=calibrations,
        calibration_medians={str(k): v for k, v in medians.items()},
        check_probe_weight=CANDIDATE_W[2],
        annotation_rule='development-train internal pairs only; per-pair atom masks',
        hyper=dict(ordinary=ORDINARY, pair_budget=PAIR_BUDGET, max_molecules=MAX_MOL, epochs=100,
                   patience=15, min_delta=1e-4, lr=1e-4, weight_decay=1e-5, batch_norm='clip1'),
        source_sha256={name: digest(ROOT / name) for name in SOURCES},
        protocol=(ROOT / 'experiments/xacs_explain/task_plan.md').read_text(encoding='utf-8'),
        git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        torch=torch.__version__, cuda=torch.version.cuda, device=torch.cuda.get_device_name(),
        test_labels_read=False))
    base.save(DOC / 'coverage.json', [{k: v for k, v in item.items() if k not in ('train', 'valid', 'test')}
                                      | dict(train_count=len(item['train']), valid_count=len(item['valid']))
                                      for item in coverage])
    print('WEIGHT', weight, {str(k): round(v, 6) for k, v in medians.items()}, flush=True)


def guard():
    manifest = json.loads((OUT / 'manifest.json').read_text(encoding='utf-8'))
    for name, expected in manifest['source_sha256'].items():
        assert digest(ROOT / name) == expected, name
    for item in manifest['splits']:
        assert digest(ROOT / 'benchmark_data' / f"{item['task']}.csv") == item['data_sha256']
        assert digest(XACS / item['task'] / 'mcs_dict_0.9.pkl') == item['annotation_sha256']
    return manifest


def job(task, seed, arm):
    manifest = guard()
    train, valid, train_graphs, valid_graphs = base.development(task)
    split = next(item for item in manifest['splits'] if item['task'] == task)
    assert train.row_id.tolist() == split['train'] and valid.row_id.tolist() == split['valid']
    ann = annotations(task, train)
    weight = float(manifest['weight'])
    folder = OUT / task / str(seed) / arm
    folder.mkdir(parents=True, exist_ok=False)
    model = model_new(seed)
    initial = base.state_hash(model)
    captured = []
    hook = model.sagpool.register_forward_pre_hook(
        lambda module, args: captured.append(args[0]) if module.training else None)
    shared = [value for name, value in model.named_parameters() if not name.startswith('reg_head.')]
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5, betas=(.9, .95))
    scheduler = base.WarmupCosineScheduler(optimizer, 10, 100, 1e-4)
    best, stale, history = float('inf'), 0, []
    best_epoch, started = 0, time.monotonic()
    for epoch in range(1, 101):
        model.train()
        regsum = auxsum = samples = pairs_used = uncom_atoms = active = 0
        sampler_hash = hashlib.sha256()
        for step, (ids, slot, picked) in enumerate(batches(ann, len(train), seed, epoch)):
            sampler_hash.update(ids.numpy().tobytes())
            optimizer.zero_grad(set_to_none=True)
            batch = to_device(base.collate_graphs([train_graphs[i] for i in ids.tolist()]))
            y = batch['y']
            torch.testing.assert_close(y.view(-1), torch.tensor(
                train.y.iloc[ids.tolist()].to_numpy(), dtype=torch.float32, device='cuda'),
                rtol=0, atol=0)
            index = gather(ids, slot, picked, batch)
            pred, att = attribution(model, batch, captured)
            assert pred.shape == y.shape and torch.isfinite(att).all()
            reg = F.mse_loss(pred, y)
            aux, stats = auxiliary(att, index)
            assert stats['pairs'] == len(picked) and stats['pairs'] > 0
            loss = reg + (weight * aux if arm == 'explain' else 0. * aux)
            if step == 0:
                probe = local.gradient_probe(reg, weight * aux, shared)
            assert torch.isfinite(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            regsum += float(reg.detach()) * len(ids)
            auxsum += float(aux.detach()) * len(ids)
            samples += len(ids)
            pairs_used += stats['pairs']
            active += stats['active']
            uncom_atoms += stats['uncom_atoms']
        labels, predictions, scores = validate(model, valid_graphs, valid)
        captured.clear()
        mse = math.fsum((float(a) - float(b)) ** 2 for a, b in zip(labels, predictions)) / len(labels)
        assert math.isfinite(mse)
        history.append(dict(epoch=epoch, train_mse=regsum / samples, auxiliary=auxsum / samples,
                            samples=samples, ordinary=len(train), pairs=pairs_used,
                            active_hinges=active, uncommon_atoms=uncom_atoms,
                            first_batch_gradient=probe, sampler_sha256=sampler_hash.hexdigest(),
                            lr=optimizer.param_groups[0]['lr'], **scores))
        if mse < best - 1e-4:
            best, stale, best_epoch = mse, 0, epoch
            torch.save({'model_state_dict': model.state_dict()}, folder / 'best_model.pt')
        else:
            stale += 1
        scheduler.step(epoch)
        base.save(folder / 'history.json', history)
        base.save(folder / 'progress.json', dict(epoch=epoch, stale=stale, best_epoch=best_epoch,
                                                 phase='training'))
        if epoch == 1 or epoch % 10 == 0:
            print(task, seed, arm, epoch, scores, 'stale', stale, flush=True)
        if stale >= 15:
            break
    hook.remove()
    model.load_state_dict(torch.load(folder / 'best_model.pt', weights_only=True)['model_state_dict'],
                          strict=True)
    labels, predictions, scores = validate(model, valid_graphs, valid)
    pd.DataFrame(dict(row_id=valid.row_id, label=labels, prediction=predictions,
                      cliff=valid.cliff_mol)).to_csv(folder / 'valid_predictions.csv', index=False)
    base.save(folder / 'summary.json', dict(task=task, seed=seed, arm=arm, initial_sha256=initial,
        parameters=sum(v.numel() for v in model.parameters()), best_epoch=best_epoch,
        epochs=len(history), train_count=len(train), valid_count=len(valid), weight=weight,
        seconds=time.monotonic() - started, test_labels_read=False, **scores))
    base.save(folder / 'progress.json', dict(epoch=len(history), phase='complete'))
    print('COMPLETE', task, seed, arm, scores, flush=True)


def report(audit=False):
    manifest = guard()
    rows = []
    for item in manifest['order']:
        task, seed, arm = item['task'], item['seed'], item['arm']
        folder = OUT / task / str(seed) / arm
        if not (folder / 'summary.json').exists():
            continue
        result = json.loads((folder / 'summary.json').read_text(encoding='utf-8'))
        csv = pd.read_csv(folder / 'valid_predictions.csv')
        split = next(s for s in manifest['splits'] if s['task'] == task)
        assert csv.row_id.tolist() == split['valid']
        for key, subset in (('rmse', csv), ('cliff_rmse', csv[csv.cliff.eq(1)])):
            measured = math.sqrt(math.fsum((float(y) - float(p)) ** 2
                                           for y, p in zip(subset.label, subset.prediction)) / len(subset))
            assert abs(measured - result[key]) < 1e-9
        if audit:
            _, valid, _, valid_graphs = base.development(task)
            model = model_new(seed)
            model.load_state_dict(torch.load(folder / 'best_model.pt',
                                             weights_only=True)['model_state_dict'], strict=True)
            labels, predictions, _ = validate(model, valid_graphs, valid)
            assert np.max(np.abs(csv.label - labels)) < 1e-6
            assert np.max(np.abs(csv.prediction - predictions)) < 1e-5
            assert csv.cliff.tolist() == valid.cliff_mol.tolist()
            del model
            torch.cuda.empty_cache()
        rows.append(result)
    deltas = []
    for task in TASKS:
        for seed in SEEDS:
            matched = {item['arm']: item for item in rows if item['task'] == task and item['seed'] == seed}
            if len(matched) != 2:
                continue
            g, e = matched['mse'], matched['explain']
            assert g['initial_sha256'] == e['initial_sha256'] and g['parameters'] == e['parameters']
            histories = [json.loads((OUT / task / str(seed) / a / 'history.json').read_text(encoding='utf-8'))
                         for a in ARMS]
            assert all(x['sampler_sha256'] == y['sampler_sha256'] for x, y in zip(*histories))
            deltas.append(dict(task=task, seed=seed, mse_rmse=g['rmse'], explain_rmse=e['rmse'],
                               mse_cliff=g['cliff_rmse'], explain_cliff=e['cliff_rmse']))
    pd.DataFrame(deltas).to_csv(DOC / 'paired_results.csv', index=False)
    aggregate = dict(completed=len(rows), total=24, paired_blocks=len(deltas),
                     independent_reload_audited=audit, weight=manifest['weight'])
    if len(deltas) == 12:
        for key in ('rmse', 'cliff'):
            g = float(np.mean([d['mse_' + key] for d in deltas]))
            e = float(np.mean([d['explain_' + key] for d in deltas]))
            aggregate[key] = dict(mse=g, explain=e, relative_improvement=1. - e / g)
        supported = sum(sum(d['explain_cliff'] < d['mse_cliff'] for d in deltas if d['task'] == task) >= 2
                        for task in TASKS)
        aggregate['tasks_winning_at_least_two_seeds'] = supported
        aggregate['go'] = (aggregate['cliff']['relative_improvement'] >= .01
                           and aggregate['rmse']['relative_improvement'] >= -.01 and supported >= 3)
    base.save(DOC / 'results.json', rows)
    base.save(DOC / 'aggregate.json', aggregate)
    lines = ['# XACs/ACES explanation-supervision dev pilot', '',
             f"completed {len(rows)}/24; paired blocks {len(deltas)}/12; independent reload audit={audit}; "
             f"frozen weight w={manifest['weight']}.",
             'Supervision uses development-train pairs only; official test was not used for training, '
             'mining or selection, and was not evaluated.', '',
             '| Task | seed | MSE RMSE | explain RMSE | MSE Cliff | explain Cliff |',
             '|---|---:|---:|---:|---:|---:|']
    lines += [f"| {d['task']} | {d['seed']} | {d['mse_rmse']:.6f} | {d['explain_rmse']:.6f} | "
              f"{d['mse_cliff']:.6f} | {d['explain_cliff']:.6f} |" for d in deltas]
    lines += ['', '```json', json.dumps(aggregate, indent=2, ensure_ascii=False), '```', '',
              'Both arms share the identical per-batch id sequence; matched (task, seed) initialisation '
              'hashes agree. Validation selects the checkpoint; three seeds share one split and estimate '
              'initialisation variance, not independent datasets.']
    (DOC / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return aggregate


def queue():
    manifest = guard()
    logs = OUT / 'logs'
    logs.mkdir(exist_ok=False)
    for done, item in enumerate(manifest['order']):
        base.save(OUT / 'status.json', dict(status='running', completed=done, total=24, **item))
        with (logs / f"{item['task']}_{item['seed']}_{item['arm']}.log").open('w', encoding='utf-8') as stream:
            child = subprocess.run([sys.executable, str(Path(__file__).resolve()), 'job',
                                    '--task', item['task'], '--seed', str(item['seed']),
                                    '--arm', item['arm']], cwd=ROOT, stdout=stream,
                                   stderr=subprocess.STDOUT)
        if child.returncode:
            base.save(OUT / 'status.json', dict(status='failed', completed=done, total=24,
                                                exit_code=child.returncode, **item))
            report()
            return
        report()
        print('QUEUE', done + 1, '/24', item, flush=True)
    base.save(OUT / 'status.json', dict(status='auditing', completed=24, total=24))
    try:
        aggregate = report(audit=True)
    except Exception as error:
        base.save(OUT / 'status.json', dict(status='failed_audit', completed=24, total=24,
                                            error=repr(error)))
        raise
    base.save(OUT / 'status.json', dict(status='complete', completed=24, total=24, go=aggregate['go']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('check', 'job', 'queue', 'report', 'audit'))
    parser.add_argument('--task', choices=TASKS)
    parser.add_argument('--seed', type=int, choices=SEEDS)
    parser.add_argument('--arm', choices=ARMS)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.action == 'job':
        job(args.task, args.seed, args.arm)
    elif args.action == 'report':
        report()
    elif args.action == 'audit':
        report(audit=True)
    else:
        {'check': check, 'queue': queue}[args.action]()

"""GPL-3.0: bounded strict-negative-gate/projection test on unchanged GraphCliff."""
import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'experiments/aca'))
import run as base
import numpy as np
import pandas as pd
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from torch.utils.data import DataLoader

OUT = ROOT/'artifacts/aca_local_20261010'
ARMS = {'mse': (False, False), 'ungated_projected': (False, True),
        'gated_raw': (True, False), 'gated_projected': (True, True)}
TASK = 'CHEMBL234_Ki'

def fingerprints(train):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    bits, vectors = [], []
    for smiles in train.smiles:
        bit = gen.GetFingerprint(Chem.MolFromSmiles(smiles))
        array = np.zeros(2048, dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(bit, array)
        bits.append(bit)
        vectors.append(array)
    fp = torch.tensor(np.stack(vectors), dtype=torch.float32, device='cuda')
    intersection = fp @ fp.T
    count = fp.sum(1)
    sim = intersection / (count[:,None] + count[None,:] - intersection).clamp_min(1)
    assert torch.isfinite(sim).all()
    for i,j in ((0,0),(0,1),(1,2),(10,20),(30,40)):
        assert abs(float(sim[i,j])-DataStructs.TanimotoSimilarity(bits[i],bits[j])) < 1e-6
    return sim

def aca(gated):
    cls = base.upstream_class()
    def strict_structure_mask(fps_smiles, **kwargs):
        n = len(fps_smiles)
        assert fps_smiles.shape == (n,n)
        return (fps_smiles >= .8)[:,None,:].expand(n,n,n)
    # Fresh upstream namespace per instance; original files and label/hinge code stay untouched.
    cls.forward.__globals__['get_structure_mask'] = strict_structure_mask
    return cls(alpha=0., cliff_lower=.2, cliff_upper=1., squared=False, p=1.,
               similarity_gate=gated, gate_type='AND', dev_mode=True)

def attach_projection(model):
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(1042)
        model.add_module('aca_projector', torch.nn.Sequential(torch.nn.Linear(512,128),
                         torch.nn.ReLU(), torch.nn.Linear(128,64)).cuda())

def triad_batches(train, sim, epoch):
    y = torch.tensor(train.y.to_numpy(),dtype=torch.float32,device='cuda')
    pair_array = ((sim>=.8) & ((y[:,None]-y[None,:]).abs()>=1.)).nonzero().cpu().numpy()
    values = y.cpu().numpy()
    positives = {int(a): np.flatnonzero((np.abs(values-values[a])<.2) & (np.arange(len(values))!=a)) for a in np.unique(pair_array[:,0])}
    pairs = [(int(a),int(n)) for a,n in pair_array if len(positives[int(a)])]
    assert pairs, 'no complete training triads'
    rng = np.random.RandomState(42+epoch)
    order = rng.permutation(len(train))
    batches = []
    for start in range(0,len(order),29):
        fill = order[start:start+29].tolist()
        for attempt in range(1000):
            a,n = pairs[rng.randint(len(pairs))]
            choices = positives[a]
            p = int(choices[rng.randint(len(choices))])
            if not set((a,p,n)).intersection(fill):
                batch = torch.tensor(fill+[a,p,n],dtype=torch.long)
                assert len(set(batch.tolist())) == len(batch)
                batches.append(batch)
                break
        else:
            raise RuntimeError('failed to select a disjoint training triad')
    return batches

def batch_loss(model, pred, h, y, sim, gated, projected, criterion=None):
    reg = torch.nn.functional.mse_loss(pred,y)
    z = torch.nn.functional.normalize(model.aca_projector(h), p=2, dim=-1) if projected else h
    criterion = criterion or aca(gated)
    _, _, triplet, _, _, candidates, active = criterion(y,pred,z,fps_smiles=sim)
    return reg+.01*triplet, reg, triplet, candidates, active

def gradient_probe(reg, aux, parameters):
    gr = torch.autograd.grad(reg,parameters,retain_graph=True,allow_unused=True)
    ga = torch.autograd.grad(aux,parameters,retain_graph=True,allow_unused=True) if aux.requires_grad else [None]*len(parameters)
    r2 = sum(g.square().sum() for g in gr if g is not None)
    a2 = sum((g.square().sum() for g in ga if g is not None), reg.new_zeros(()))
    dot = sum((r.mul(a).sum() for r,a in zip(gr,ga) if r is not None and a is not None), reg.new_zeros(()))
    rn, an = float(r2.sqrt()), float(a2.sqrt())
    return dict(reg_grad_norm=rn, weighted_aux_grad_norm=an,
                grad_ratio=an/rn if rn else None, grad_cosine=float(dot)/(rn*an) if rn and an else None)

def checkpoint(model, path):
    state = model.state_dict()
    torch.save(dict(model_state_dict={k:v for k,v in state.items() if not k.startswith('aca_projector.')},
                    projection_state_dict=model.aca_projector.state_dict() if hasattr(model,'aca_projector') else {}), path)

def check():
    tr, va, tg, _ = base.development(TASK)
    sim = fingerprints(tr)
    y = torch.tensor(tr.y.to_numpy(),dtype=torch.float32,device='cuda')
    negatives = (sim>=.8) & ((y[:,None]-y[None,:]).abs()>=1.)
    negative_count = int(negatives.sum())
    assert negative_count > 0, 'no qualifying structural negatives'
    assert float(y.max()-y.min()) < 16, 'raw label margin exceeds normalized 64D L1 necessary bound'
    synthetic_y = torch.tensor([0.,.1,1.2],device='cuda')
    z = torch.tensor([[0.,0.],[1.,0.],[.2,0.]],device='cuda',requires_grad=True)
    synthetic_sim = torch.eye(3,device='cuda')
    synthetic_sim[0,2] = synthetic_sim[2,0] = .9
    result = aca(True)(synthetic_y,synthetic_y,z,fps_smiles=synthetic_sim)
    assert result[5] == 1 and torch.allclose(result[2],z.new_tensor(1.9))
    synthetic_sim[0,2] = .79
    empty = aca(True)(synthetic_y,synthetic_y,z,fps_smiles=synthetic_sim)
    assert empty[5] == 0 and empty[2] == 0
    ids_loader = DataLoader(list(range(len(tg))),batch_size=32,shuffle=True,generator=torch.Generator().manual_seed(42))
    total, empty_batches, first_ids = 0, 0, None
    for ids in ids_loader:
        ids = ids.cuda()
        sy = y[ids]
        result = aca(True)(sy,sy,torch.zeros(len(ids),2,device='cuda'),fps_smiles=sim[ids[:,None],ids[None,:]])
        total += result[5]
        empty_batches += int(result[5] == 0)
        if result[5] and first_ids is None:
            first_ids = ids.cpu().tolist()
    assert total > 0, 'no gated candidates in first shuffled epoch'
    structured_batches = triad_batches(tr,sim,0)
    structured_candidates = structured_empty = 0
    for ids in structured_batches:
        ids = ids.cuda()
        sy = y[ids]
        result = aca(True)(sy,sy,torch.zeros(len(ids),2,device='cuda'),fps_smiles=sim[ids[:,None],ids[None,:]])
        assert result[5] > 0
        structured_candidates += result[5]
        structured_empty += int(result[5]==0)
    first_ids = structured_batches[0].tolist()
    model = base.model_new()
    before_rng = torch.get_rng_state().clone()
    before_cuda = torch.cuda.get_rng_state().clone()
    initial_hash = base.state_hash(model)
    attach_projection(model)
    assert torch.equal(before_rng,torch.get_rng_state()) and torch.equal(before_cuda,torch.cuda.get_rng_state())
    shared = [p for n,p in model.named_parameters() if not n.startswith(('reg_head.','aca_projector.'))]
    captured = []
    hook = model.reg_head.register_forward_pre_hook(lambda module,args: captured.append(args[0]))
    batch = base.collate_graphs([tg[i] for i in first_ids])
    pred = model(batch['x'].cuda(),batch['edge_index'].cuda(),batch['edge_attr'].cuda(),batch['batch'].cuda())
    h = captured.pop()
    embedding = torch.nn.functional.normalize(model.aca_projector(h),p=2,dim=-1)
    assert torch.allclose(embedding.norm(dim=-1),torch.ones(len(first_ids),device='cuda'),atol=1e-6)
    ids = torch.tensor(first_ids,device='cuda')
    loss, reg, triplet, nc, nh = batch_loss(model,pred,h,batch['y'].cuda(),sim[ids[:,None],ids[None,:]],True,True)
    probe = gradient_probe(reg,.01*triplet,shared)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert model.aca_projector[0].weight.grad is not None
    hook.remove()
    OUT.mkdir(parents=True,exist_ok=True)
    checkpoint(model,OUT/'preflight_checkpoint.pt')
    plain = base.model_new()
    plain.load_state_dict(torch.load(OUT/'preflight_checkpoint.pt',weights_only=True)['model_state_dict'])
    model.eval(); plain.eval()
    with torch.no_grad():
        args = [batch[k].cuda() for k in ('x','edge_index','edge_attr','batch')]
        assert torch.equal(model(*args),plain(*args)), 'projection changed single-molecule inference'
    base.save(OUT/'preflight.json',dict(train=len(tr),valid=len(va),valid_cliff=int(va.cliff_mol.sum()),
        initial_sha256=initial_hash,qualified_directed_negative_pairs=negative_count,
        epoch1_gated_candidates=total,epoch1_empty_batches=empty_batches,batches=len(ids_loader),
        structured_candidates=structured_candidates,structured_empty_batches=structured_empty,
        structured_batches=len(structured_batches),samples_per_epoch=sum(len(ids) for ids in structured_batches),
        real_batch_candidates=nc,real_batch_active=nh,gradient_probe=probe,
        normalized_embedding_check=True,inference_checkpoint_compatible=True))
    print((OUT/'preflight.json').read_text(),flush=True)

def run(arm):
    tr, va, tg, vg = base.development(TASK)
    gated, projected = ARMS[arm]
    folder = OUT/arm
    folder.mkdir(parents=True,exist_ok=False)
    sim = fingerprints(tr)
    model = base.model_new()
    digest = base.state_hash(model)
    if projected:
        attach_projection(model)
    shared = [p for n,p in model.named_parameters() if not n.startswith(('reg_head.','aca_projector.'))]
    captured = []
    hook = model.reg_head.register_forward_pre_hook(lambda m,a: captured.append(a[0]) if model.training else None)
    valid_loader = DataLoader(vg,batch_size=32,collate_fn=base.collate_graphs)
    loss_fn = aca(gated)
    optimizer = torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-5,betas=(.9,.95))
    scheduler = base.WarmupCosineScheduler(optimizer,10,100,1e-4)
    protocol = json.loads((Path(__file__).parent/'protocol.json').read_text())
    base.save(folder/'provenance.json',dict(git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        protocol=protocol,arm=arm,initial_sha256=digest,train=tr.row_id.tolist(),valid=va.row_id.tolist(),
        source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
            ('model.py','experiments/aca/run.py','experiments/aca/loss_adapter.py','experiments/aca_local/pilot.py','experiments/aca_local/protocol.json')},
        data_sha256=hashlib.sha256((ROOT/'benchmark_data'/f'{TASK}.csv').read_bytes()).hexdigest(),
        torch=torch.__version__,cuda=torch.version.cuda,test_labels_read=False))
    best, stale, history = float('inf'),0,[]
    for epoch in range(100):
        model.train()
        count = regsum = tripsum = loss_sum = candidates = active = empty = 0
        probe = None
        epoch_batches = triad_batches(tr,sim,epoch)
        for step, ids in enumerate(epoch_batches):
            optimizer.zero_grad(set_to_none=True)
            batch = base.collate_graphs([tg[i] for i in ids.tolist()])
            pred = model(batch['x'].cuda(),batch['edge_index'].cuda(),batch['edge_attr'].cuda(),batch['batch'].cuda())
            h, y = captured.pop(),batch['y'].cuda()
            if arm == 'mse':
                reg = loss = torch.nn.functional.mse_loss(pred,y)
                triplet,nc,nh = reg.new_zeros(()),0,0
            else:
                ids = ids.cuda()
                loss,reg,triplet,nc,nh = batch_loss(model,pred,h,y,sim[ids[:,None],ids[None,:]],gated,projected,loss_fn)
                if gated:
                    assert nc > 0, 'structured batch lost its qualifying triad'
            if step == 0:
                probe = gradient_probe(reg,.01*triplet,shared)
            assert torch.isfinite(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step()
            n = len(y)
            count += n
            regsum += float(reg.detach())*n
            tripsum += float(triplet.detach())*n
            loss_sum += float(loss.detach())*n
            candidates += nc; active += nh; empty += int(nc==0)
        if arm != 'mse' and epoch == 0:
            assert candidates > 0
        labels, predictions = base.evaluate(model,valid_loader,'cuda')
        score = base.metrics(labels,predictions,va.cliff_mol)
        mse = float(np.mean((np.asarray(labels)-predictions)**2))
        assert math.isfinite(mse)
        history.append(dict(epoch=epoch+1,reg=regsum/count,triplet=tripsum/count,loss=loss_sum/count,
                            samples=count,batches=len(epoch_batches),
                            candidates=candidates,active=active,empty_batches=empty,first_batch_gradient=probe,**score))
        if mse < best:
            best,stale,best_epoch=mse,0,epoch+1
            checkpoint(model,folder/'best_model.pt')
        else:
            stale += 1
        base.save(folder/'history.json',history)
        if epoch == 0 or (epoch+1)%5 == 0:
            print(f'{arm} epoch={epoch+1} rmse={score["rmse"]:.4f} cliff={score["cliff_rmse"]:.4f} candidates={candidates} stale={stale}',flush=True)
        scheduler.step(epoch)
        if stale >= 15:
            break
    hook.remove()
    plain = base.model_new()
    plain.load_state_dict(torch.load(folder/'best_model.pt',weights_only=True)['model_state_dict'])
    labels,predictions = base.evaluate(plain,valid_loader,'cuda')
    pd.DataFrame(dict(row_id=va.row_id,label=labels,prediction=predictions,cliff=va.cliff_mol)).to_csv(folder/'valid_predictions.csv',index=False)
    summary = dict(arm=arm,initial_sha256=digest,best_epoch=best_epoch,epochs=len(history),**base.metrics(labels,predictions,va.cliff_mol))
    base.save(folder/'summary.json',summary)
    print('FINISHED '+json.dumps(summary),flush=True)

def audit():
    _,va,_,vg = base.development(TASK)
    loader = DataLoader(vg,batch_size=32,collate_fn=base.collate_graphs)
    results = []
    incomplete = []
    for arm in ARMS:
        folder = OUT/arm
        if not (folder/'summary.json').exists():
            incomplete.append(arm)
            continue
        summary = json.loads((folder/'summary.json').read_text())
        model = base.model_new()
        model.load_state_dict(torch.load(folder/'best_model.pt',weights_only=True)['model_state_dict'])
        labels,predictions = base.evaluate(model,loader,'cuda')
        csv = pd.read_csv(folder/'valid_predictions.csv')
        assert csv.row_id.tolist() == va.row_id.tolist()
        assert np.max(np.abs(csv.label-labels)) < 1e-6
        diff = float(np.max(np.abs(csv.prediction-predictions)))
        assert diff < 1e-5
        cliff = csv[csv.cliff.eq(1)]
        for subset,key in ((csv,'rmse'),(cliff,'cliff_rmse')):
            rmse = math.sqrt(math.fsum((y-p)**2 for y,p in zip(subset.label,subset.prediction))/len(subset))
            assert abs(rmse-summary[key]) < 1e-7
        summary['replay_max_abs']=diff
        results.append(summary)
    assert len({r['initial_sha256'] for r in results}) == 1
    assert not incomplete or incomplete == ['gated_raw'], 'other planned arms have not completed'
    base.save(OUT/'audit.json',results)
    base.save(OUT/'audit_status.json',dict(completed=[row['arm'] for row in results],incomplete=incomplete,
        all_planned_fits_completed=not incomplete,reason='raw gated arm stopped after two CUDA failures' if incomplete else None))
    print(json.dumps(results,indent=2),flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action',choices=['check','run','audit'])
    parser.add_argument('--arm',choices=list(ARMS))
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.action == 'check': check()
    elif args.action == 'audit': audit()
    else:
        assert args.arm
        run(args.arm)

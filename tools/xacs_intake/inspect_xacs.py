# -*- coding: utf-8 -*-
"""Read-only intake check: does the downloaded XACs/ACES-GNN release provide atom-level
explanation supervision usable with OUR MoleculeACE splits?

Scope of this script:
  * reads D:/XACs/Data/<task>/<task>.csv and mcs_dict_0.9.pkl (read-only, never written)
  * reads D:/GraphCliff-ACA/benchmark_data/<task>.csv (official train/test members)
  * writes a JSON evidence file + prints a table; no training, no model, no test labels used
    as training signal.

Naming note: we only *count* official-test membership for reporting; annotations are
summarised for the official-train subset, which is what a training run would consume.
"""

import argparse
import json
import os
import pickle
import sys
from collections import Counter

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

XACS_ROOT = r"D:\XACs\Data"
GC_BENCH = r"D:\GraphCliff-ACA\benchmark_data"

# the four tasks already used by this repo's development pilots
DEV_TASKS = ["CHEMBL234_Ki", "CHEMBL2047_EC50", "CHEMBL2147_Ki", "CHEMBL244_Ki"]


def probe(pkl_path):
    with open(pkl_path, "rb") as fh:
        d = pickle.load(fh)
    print(f"type={type(d).__name__} n_keys={len(d)}")
    k = next(iter(d))
    v = d[k]
    print(f"first key={k!r}")
    print(f"value type={type(v).__name__} len={len(v) if hasattr(v, '__len__') else 'n/a'}")
    head = v[0]
    print(f"value[0] type={type(head).__name__}")
    if isinstance(head, dict):
        print("value[0] keys=" + ", ".join(
            f"{kk}:{type(head[kk]).__name__}[{len(head[kk]) if hasattr(head[kk], '__len__') else head[kk]}]"
            for kk in list(head)[:6]))
    if len(v) > 1:
        rec = v[1]
        print(f"value[1] type={type(rec).__name__}")
        if isinstance(rec, dict):
            print("value[1] keys=" + ", ".join(
                f"{kk}({type(rec[kk]).__name__}"
                + (f",len={len(rec[kk])})" if hasattr(rec[kk], "__len__") else ")")
                for kk in list(rec)[:8]))
    return d


def load_pair_index(pkl_path):
    """Return {anchor_smiles: [partner_record, ...]} using the authors' convention
    cliff_dict[smiles][0] == self record, [1:] == cliff partners."""
    with open(pkl_path, "rb") as fh:
        d = pickle.load(fh)
    out = {}
    for smi, v in d.items():
        partners = [r for r in v[1:]] if isinstance(v, (list, tuple)) else []
        out[smi] = partners
    return out


def mask_checks(anchor_smi, rec, n_sample):
    """Validate atom_mask_i/atom_mask_j geometry against RDKit atom counts."""
    problems = []
    mask_i = np.asarray(rec["atom_mask_i"]).ravel()
    mol_i = Chem.MolFromSmiles(anchor_smi)
    if mol_i is None:
        problems.append("anchor_unparsable")
    elif len(mask_i) != mol_i.GetNumAtoms():
        problems.append(f"mask_i_len={len(mask_i)}!=atoms={mol_i.GetNumAtoms()}")
    if mask_i.size and not np.all(np.isin(mask_i, [0, 1])):
        problems.append("mask_i_not_binary")
    if mask_i.size and int(mask_i.sum()) == 0:
        problems.append("mask_i_all_zero")
    partner = rec.get("smiles")
    mask_j = np.asarray(rec["atom_mask_j"]).ravel()
    mol_j = Chem.MolFromSmiles(partner) if partner else None
    if mol_j is None:
        problems.append("partner_unparsable")
    elif len(mask_j) != mol_j.GetNumAtoms():
        problems.append(f"mask_j_len={len(mask_j)}!=atoms={mol_j.GetNumAtoms()}")
    if mask_j.size and not np.all(np.isin(mask_j, [0, 1])):
        problems.append("mask_j_not_binary")
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", default=None, help="probe one mcs pkl and exit")
    ap.add_argument("--out", default=r"D:\GraphCliff-ACA\tools\xacs_intake\evidence.json")
    ap.add_argument("--mask-sample", type=int, default=25)
    args = ap.parse_args()

    if args.probe:
        probe(args.probe)
        return

    tasks = sorted(d for d in os.listdir(XACS_ROOT)
                   if os.path.isdir(os.path.join(XACS_ROOT, d))
                   and os.path.exists(os.path.join(XACS_ROOT, d, f"{d}.csv")))
    rows, details = [], {}
    for task in tasks:
        xcsv = os.path.join(XACS_ROOT, task, f"{task}.csv")
        pkl = os.path.join(XACS_ROOT, task, "mcs_dict_0.9.pkl")
        gc_csv = os.path.join(GC_BENCH, f"{task}.csv")
        if not os.path.exists(pkl):
            rows.append(dict(task=task, note="no mcs_dict_0.9.pkl"))
            continue
        dx = pd.read_csv(xcsv)
        dg = pd.read_csv(gc_csv) if os.path.exists(gc_csv) else None
        pairs = load_pair_index(pkl)

        rec = dict(task=task, n_mol=len(dx), n_anchor_keys=len(pairs))
        # --- comparability with our benchmark_data ---
        if dg is not None:
            same_order = len(dx) == len(dg) and (dx["smiles"].values == dg["smiles"].values).all()
            rec.update(
                gc_n_mol=int(len(dg)),
                same_row_order=bool(same_order),
                smiles_set_equal=bool(set(dx["smiles"]) == set(dg["smiles"])),
                y_offset_median=float(np.median(dg["y [pEC50/pKi]"].values - dx["y"].values)),
                cliff_flag_equal=bool((dx["cliff_mol"].values == dg["cliff_mol"].values).all()),
                n_train=int((dg["split"] == "train").sum()),
                n_test=int((dg["split"] == "test").sum()),
                n_cliff_train=int(((dg["split"] == "train") & (dg["cliff_mol"] == 1)).sum()),
                n_cliff_test=int(((dg["split"] == "test") & (dg["cliff_mol"] == 1)).sum()),
            )
            train_smiles = set(dg.loc[dg["split"] == "train", "smiles"])
            test_smiles = set(dg.loc[dg["split"] == "test", "smiles"])
        else:
            train_smiles, test_smiles = set(dx["smiles"]), set()
            rec["note"] = "missing in benchmark_data"

        csv_smiles = set(dx["smiles"])
        per_anchor, per_anchor_train = [], []
        n_pairs = n_pairs_in_csv = n_pairs_train = 0
        anchors_in_train_with_pair = 0
        for anchor, partners in pairs.items():
            if anchor not in csv_smiles:
                continue
            good = [p for p in partners if isinstance(p, dict) and p.get("smiles") in csv_smiles]
            n_pairs += len(partners)
            n_pairs_in_csv += len(good)
            per_anchor.append(len(good))
            if anchor in train_smiles:
                good_tr = [p for p in good if p["smiles"] in train_smiles]
                per_anchor_train.append(len(good_tr))
                n_pairs_train += len(good_tr)
                if good_tr:
                    anchors_in_train_with_pair += 1
        rec.update(
            pairs_total=int(n_pairs),
            pairs_both_in_csv=int(n_pairs_in_csv),
            anchors_with_pair_in_csv=int(sum(1 for c in per_anchor if c > 0)),
            pairs_within_official_train=int(n_pairs_train),
            train_anchors_with_train_partner=int(anchors_in_train_with_pair),
        )
        if per_anchor:
            a = np.asarray(per_anchor)
            rec.update(pairs_per_anchor_mean=float(a.mean()), pairs_per_anchor_median=float(np.median(a)),
                       pairs_per_anchor_p90=float(np.percentile(a, 90)), pairs_per_anchor_max=int(a.max()))
        if per_anchor_train:
            a = np.asarray(per_anchor_train)
            rec.update(train_pairs_per_anchor_mean=float(a.mean()),
                       train_pairs_per_anchor_median=float(np.median(a)),
                       train_pairs_per_anchor_p90=float(np.percentile(a, 90)),
                       train_pairs_per_anchor_max=int(a.max()),
                       train_anchors_zero_pair=int(sum(1 for c in per_anchor_train if c == 0)))

        # --- mask sanity on ALL train-internal pairs (dev tasks only) ---
        if task in DEV_TASKS:
            zero_i = zero_j = both_ok = 0
            sizes_i, sizes_j, problems = [], [], Counter()
            for anchor, partners in pairs.items():
                if anchor not in train_smiles:
                    continue
                for p in partners:
                    if not isinstance(p, dict) or p.get("smiles") not in train_smiles:
                        continue
                    mi = np.asarray(p["atom_mask_i"]).ravel()
                    mj = np.asarray(p["atom_mask_j"]).ravel()
                    sizes_i.append(int(mi.sum()))
                    sizes_j.append(int(mj.sum()))
                    if int(mi.sum()) == 0:
                        zero_i += 1
                    if int(mj.sum()) == 0:
                        zero_j += 1
                    if int(mi.sum()) > 0 and int(mj.sum()) > 0:
                        both_ok += 1
                    if len(sizes_i) <= args.mask_sample * 4:
                        for msg in mask_checks(anchor, p, args.mask_sample):
                            problems[msg] += 1
            n_tr_pairs = max(len(sizes_i), 1)
            denom = max(rec.get("n_train", 0), 1)
            rec.update(
                train_pairs_mask_all_zero_i=int(zero_i),
                train_pairs_mask_all_zero_j=int(zero_j),
                train_pairs_both_masks_nonzero=int(both_ok),
                train_pairs_mask_all_zero_i_frac=round(zero_i / n_tr_pairs, 4),
                train_uncommon_atoms_mean=round(float(np.mean(sizes_i)), 2) if sizes_i else None,
                train_uncommon_atoms_median=float(np.median(sizes_i)) if sizes_i else None,
                # a random batch of 128 molecules: how many anchors carry >=1 train-internal pair,
                # and how many supervised pairs that batch would contain
                exp_anchors_with_pair_per_batch128=round(
                    128 * rec.get("train_anchors_with_train_partner", 0) / denom, 1),
                exp_pairs_per_batch128=round(128 * rec.get("pairs_within_official_train", 0) / denom, 1),
                mask_checked=len(sizes_i),
                mask_problems=dict(problems),
            )
        rows.append(rec)
        details[task] = rec

    df = pd.DataFrame(rows)
    cols = ["task", "n_mol", "pairs_total", "pairs_both_in_csv", "anchors_with_pair_in_csv",
            "pairs_within_official_train", "train_anchors_with_train_partner",
            "train_pairs_per_anchor_mean", "pairs_per_anchor_max"]
    print(df[[c for c in cols if c in df.columns]].to_string(index=False))
    print("\n[dev tasks]")
    for t in DEV_TASKS:
        r = details.get(t)
        if r:
            print(json.dumps({k: r[k] for k in r if k not in ("task",)}, ensure_ascii=False, default=str))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(dict(tasks=rows, dev_details=details,
                       dev_tasks=DEV_TASKS, xacs_root=XACS_ROOT, bench=GC_BENCH), fh,
                  ensure_ascii=False, indent=2, default=str)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    sys.exit(main())

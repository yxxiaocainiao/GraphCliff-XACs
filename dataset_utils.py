from rdkit import Chem
import torch

from dataclasses import dataclass
# ------------------------------
# RDKit Graph Featurization
# ------------------------------
_ATOM_TYPES = ["H","B","C","N","O","F","Si","P","S","Cl","Br","I","Se","Other"]
_ATOM_TO_IDX = {s: i for i, s in enumerate(_ATOM_TYPES)}

_HYBS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2
]

_BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC
]

_BOND_STEREOS = [
    Chem.rdchem.BondStereo.STEREONONE,
    Chem.rdchem.BondStereo.STEREOANY,
    Chem.rdchem.BondStereo.STEREOZ,
    Chem.rdchem.BondStereo.STEREOE,
    Chem.rdchem.BondStereo.STEREOCIS,
    Chem.rdchem.BondStereo.STEREOTRANS
]

@dataclass
class GraphSample:
    x: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    y: float
    smiles: str

def one_hot(idx: int, size: int):
    v = [0.0] * size
    if 0 <= idx < size:
        v[idx] = 1.0
    return v


def atom_features(atom: Chem.rdchem.Atom):
    sym = atom.GetSymbol()
    sym_idx = _ATOM_TO_IDX.get(sym, _ATOM_TO_IDX["Other"])
    f = []
    f += one_hot(sym_idx, len(_ATOM_TYPES))
    deg = atom.GetDegree()
    deg = deg if deg <= 5 else 6
    f += one_hot(deg, 7)
    ch = atom.GetFormalCharge()
    ch = max(-2, min(2, ch))
    f += one_hot(ch + 2, 5)
    hyb = atom.GetHybridization()
    hyb_idx = _HYBS.index(hyb) if hyb in _HYBS else len(_HYBS)
    f += one_hot(hyb_idx, len(_HYBS) + 1)
    f += [1.0 if atom.GetIsAromatic() else 0.0]
    f += [1.0 if atom.IsInRing() else 0.0]
    h = atom.GetTotalNumHs()
    h = h if h <= 3 else 3
    f += one_hot(h, 4)
    return f


def bond_features(bond: Chem.rdchem.Bond):
    if bond is None:
        return [0.0] * (len(_BOND_TYPES) + 2 + len(_BOND_STEREOS))
    f = []
    btype = bond.GetBondType()
    bt_idx = _BOND_TYPES.index(btype) if btype in _BOND_TYPES else len(_BOND_TYPES)
    f += one_hot(bt_idx, len(_BOND_TYPES) + 1)
    f += [1.0 if bond.GetIsConjugated() else 0.0]
    f += [1.0 if bond.IsInRing() else 0.0]
    st = bond.GetStereo()
    st_idx = _BOND_STEREOS.index(st) if st in _BOND_STEREOS else 0
    f += one_hot(st_idx, len(_BOND_STEREOS))
    return f

def smiles_to_graph(smiles: str, y: float):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    Chem.Kekulize(mol, clearAromaticFlags=False)
    xf = [atom_features(a) for a in mol.GetAtoms()]
    x = torch.tensor(xf, dtype=torch.float32)
    src, dst, eattr = [], [], []
    for b in mol.GetBonds():
        i = b.GetBeginAtomIdx()
        j = b.GetEndAtomIdx()
        bf = bond_features(b)
        src += [i, j]
        dst += [j, i]
        eattr += [bf, bf]
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = torch.tensor(eattr, dtype=torch.float32) if len(eattr) > 0 else \
        torch.zeros((0, len(_BOND_TYPES) + 1 + 2 + len(_BOND_STEREOS)), dtype=torch.float32)
    return GraphSample(x=x, edge_index=edge_index, edge_attr=edge_attr, y=float(y), smiles=smiles)


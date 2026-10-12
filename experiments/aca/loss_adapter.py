"""GPL-3.0: execute unmodified upstream label-only ACA definitions, not a rewrite."""
import ast
import hashlib
from pathlib import Path
from typing import Optional, List
import torch
from torch import Tensor
from torch.nn.modules.loss import _Loss

def upstream_class():
    path=Path(__file__).parent/"vendor/upstream_loss.py"
    if hashlib.sha256(path.read_bytes()).hexdigest() != 'ad7e1722383fb85693bfd821b0c5f32827f4cb2867c591865569c445c640df37':
        raise RuntimeError('pinned upstream loss checksum mismatch')
    tree=ast.parse(path.read_text(encoding="utf-8"),filename=str(path))
    wanted=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in ("ACALoss","_aca_loss")]
    if len(wanted)!=2:
        raise RuntimeError("upstream interface changed")
    namespace=dict(torch=torch,Tensor=Tensor,_Loss=_Loss,Optional=Optional,List=List)
    exec(compile(ast.Module(body=wanted,type_ignores=[]),str(path),"exec"),namespace)
    return namespace["ACALoss"]

import argparse
import os
import time
import math
import random
import wandb
import numpy as np
from rdkit import RDLogger

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.backends.cudnn as cudnn

from dataset import GraphDataset, collate_graphs
from model import GraphCliffRegressor
from train import train_one_epoch, evaluate
from metric import calc_rmse, calc_cliff_rmse


RDLogger.DisableLog('rdApp.*')
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

def set_wandb(args):
    #wandb.login(key=os.environ["WANDB_API_KEY"])
    wandb.login()
    timestamp = args.run_name
    wandb.init(
                project = args.project_name ,
                name = timestamp,
                config = args
    )
    
def set_seed(args, deterministic=True):
    seed = args.seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("high")
    
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, max_epochs, base_lr, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
        
    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr


def save_ckpt(model, path):
    os.makedirs(path.split('best_model.pt')[0], exist_ok=True)
    torch.save({'model_state_dict': model.state_dict()}, path)

def load_ckpt(model, path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict']) 
    return model

def main():
    parser = argparse.ArgumentParser(description='GraphCliff for Activity Cliff Prediction')
    parser.add_argument('--project_name', default='GraphCliff', type=str)
    parser.add_argument('--run_name', default='run', type=str)
    parser.add_argument('--log_wandb', '-w', default=None, action='store_true')
    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='CHEMBL204_Ki')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--valid_frac', type=float, default=0.1,
                        help='Fraction of the official train split held out for model selection')
    parser.add_argument('--split_seed', type=int, default=42,
                        help='Seed for the deterministic train/valid split')
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    # Model architecture
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--groups', type=int, default=4)
    parser.add_argument('--mid_K', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0)
    # Regularization and early stopping
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--min_delta', type=float, default=1e-4)
    # System arguments
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=0)
    # Model persistence
    parser.add_argument('--save_model', default=None, action='store_true')
    args = parser.parse_args()
        
    set_seed(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    args.run_name = timestamp
    if args.log_wandb: set_wandb(args)
    
    def seed_worker():
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    g = torch.Generator()
    g.manual_seed(args.seed)

    collate_fn = lambda batch: collate_graphs(batch)

    split_kwargs = dict(valid_frac=args.valid_frac, split_seed=args.split_seed)
    train_ds = GraphDataset(args.dataset, split="train", **split_kwargs)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate_fn, num_workers=args.num_workers, worker_init_fn=seed_worker, generator=g, pin_memory=True)
    valid_ds = GraphDataset(args.dataset, split="valid", **split_kwargs)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size,
                              shuffle=False, collate_fn=collate_fn, num_workers=args.num_workers, worker_init_fn=seed_worker, generator=g)
    test_ds = GraphDataset(args.dataset, split="test")
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             shuffle=False, collate_fn=collate_fn, num_workers=args.num_workers, worker_init_fn=seed_worker, generator=g)
    print(f"Split sizes — train: {len(train_ds)}, valid: {len(valid_ds)}, test: {len(test_ds)}")

    sample = train_ds[0]
    atom_in_dim = sample.x.size(1)
    edge_dim = sample.edge_attr.size(1)
    
    print("Creating model...")
    model = GraphCliffRegressor(
                atom_in_dim, edge_dim, 
                hidden_size=args.hidden_size, 
                num_layers=args.num_layers, 
                groups=args.groups, 
                mid_K=args.mid_K, 
                dropout=args.dropout
            ).to(device)
    
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params_count:,}")
    trainable_params = model.parameters()
    
    optimizer = AdamW(
        trainable_params, 
        lr=args.lr, 
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95)
    )
    
    scheduler = WarmupCosineScheduler(
        optimizer, 
        warmup_epochs=args.warmup_epochs,
        max_epochs=args.epochs,
        base_lr=args.lr,
        min_lr=args.min_lr
    )
    
    
    try: # Load the checkpoints and validation
        import glob
        args.ckpt_path = glob.glob(f'ckpt/{args.dataset}/*/best_model.pt')[0]
        args.epochs = 0
    except: # Training from scratch
        print("Starting training...")
        best_loss = float('inf')
        epochs_no_improve = 0
    
        args.ckpt_path = f'ckpt/{args.dataset}/{timestamp}/best_model.pt'
        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, device)
            # Model selection and early stopping use the held-out validation
            # split only; the test split is untouched until the final report.
            labels, preds = evaluate(model, valid_loader, device)

            current_lr = scheduler.step(epoch)

            valid_loss = F.mse_loss(torch.tensor(preds), torch.tensor(labels))
            improved = (best_loss - valid_loss) > args.min_delta

            if improved:
                best_loss = valid_loss
                epochs_no_improve = 0
                save_ckpt(model, args.ckpt_path)
                print(f"Epoch {epoch:3d}: Train Loss={train_loss:.4f}, Valid Loss={valid_loss:.4f}, LR={current_lr:.2e} [BEST SAVED]")
            else:
                epochs_no_improve += 1
                print(f"Epoch {epoch:3d}: Train Loss={train_loss:.4f}, Valid Loss={valid_loss:.4f}, LR={current_lr:.2e} (no improve: {epochs_no_improve})")

            # Early stopping
            if epochs_no_improve >= args.patience:
                print(f"Early stopping at epoch {epoch} (patience: {args.patience})")
                break
            if args.log_wandb:
                wandb.log({"train_loss": train_loss,
                           "valid_loss": valid_loss})

    print(f"\nLoading best checkpoint: {args.ckpt_path}")
    try:
        model = load_ckpt(model, args.ckpt_path, device)
    except Exception as e:
        print(f"Warning: Could not load best checkpoint ({e}), using current weights")
    
    # Final evaluation
    print("Evaluating on test set...")
    labels, preds = evaluate(model, test_loader, device)
    rmse = calc_rmse(true=labels, pred=preds)

    cliff_mask = test_ds.cliff_mols_test
    if cliff_mask is not None:
        rmse_cliff = calc_cliff_rmse(y_test_pred=preds, y_test=labels, cliff_mols_test=cliff_mask)
        print(f"Test RMSE: {rmse:.4f} | Cliff RMSE: {rmse_cliff:.4f}")
    else:
        rmse_cliff = None
        print(f"Test RMSE: {rmse:.4f}")
    print("Training completed!")

    if args.log_wandb:
        wandb.log({
                    "rmse": rmse,
                    "cliff_rmse": rmse_cliff,
        })
        wandb.finish()
    

    if args.save_model:
        final_model_path = os.path.join(os.path.dirname(args.ckpt_path), 'final_model.pt')
        torch.save({
            'model_state_dict': model.state_dict(),
            'args': args,
            'rmse': rmse,
            'rmse_cliff': rmse_cliff if cliff_mask is not None else None
        }, final_model_path)

        print(f"Final model saved: {final_model_path}")
    

if __name__ == '__main__':
    main()
    

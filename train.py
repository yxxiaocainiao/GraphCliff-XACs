import torch
import torch.nn as nn
import torch.nn.functional as F

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for orig_batch in loader:
        optimizer.zero_grad()
        y_pred = model(
                        x_f=orig_batch['x'].to(device), 
                        edge_index=orig_batch['edge_index'].to(device),
                        edge_attr=orig_batch['edge_attr'].to(device), 
                        batch=orig_batch['batch'].to(device)
        ) 
        y_true = orig_batch['y'].to(device)
        loss = F.mse_loss( y_pred, y_true, reduction='mean')

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    preds_out, labels_out = [], []
    
    with torch.no_grad():
        for orig_batch in loader:
            y_pred = model(
                            x_f=orig_batch['x'].to(device), 
                            edge_index=orig_batch['edge_index'].to(device),
                            edge_attr=orig_batch['edge_attr'].to(device), 
                            batch=orig_batch['batch'].to(device)
            ) 
            y_true = orig_batch['y'].to(device)
            
            preds_out += y_pred.detach().float().cpu().view(-1).tolist()
            labels_out += y_true.detach().float().cpu().view(-1).tolist()
            
    return labels_out, preds_out
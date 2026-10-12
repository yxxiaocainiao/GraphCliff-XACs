import torch
import torch.nn as nn
from torch_geometric.nn import SAGPooling, GINConv, SAGEConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp


def normalize_edges(num_nodes, edge_index, edge_weight):
    """Compute D^{-1/2} normalization for graph convolution"""
    if edge_index.numel() == 0:
        return edge_weight
    
    src, dst = edge_index
    deg = torch.zeros(num_nodes, device=edge_index.device, dtype=edge_weight.dtype)
    deg = deg.scatter_add_(0, src, edge_weight)
    deg = deg.clamp(min=1e-12)
    d_inv_sqrt = deg.pow(-0.5)
    norm = edge_weight * d_inv_sqrt[src] * d_inv_sqrt[dst]
    return norm

def propagate(x, edge_index, edge_weight_norm):
    """Efficient message passing with normalized edge weights"""
    if edge_index.numel() == 0:
        return torch.zeros_like(x)
    
    src, dst = edge_index
    ew = edge_weight_norm.to(x.dtype)
    msg = x[src] * ew.unsqueeze(-1)
    out = torch.zeros_like(x)
    out.index_add_(0, dst, msg)
    return out 


class AtomEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_size: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_size), 
            nn.RMSNorm(hidden_size),
            nn.ReLU(), 
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)


class ShortGINE(nn.Module):
    def __init__(self, in_dim, edge_dim, dropout=0.0):
        super().__init__()
        # Node MLP
        node_mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim, in_dim)
        )
        
        # Edge MLP  
        edge_mlp = nn.Sequential(
            nn.Linear(edge_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, in_dim)
        ) if edge_dim > 0 else None
        
        self.conv = GINConv(node_mlp, train_eps=True)
        self.edge_mlp = edge_mlp
        self.dropout = nn.Dropout(dropout)

        self.norm = nn.RMSNorm(in_dim)
        
    def forward(self, x, edge_index, edge_attr):
        residual = x
        # Pre-process edge features
        if self.edge_mlp is not None and edge_attr is not None:
            processed_edges = self.edge_mlp(edge_attr)
            
            # Add edge info to messages
            src, dst = edge_index
            messages = x[src] + processed_edges
            
            # Manual aggregation (GIN sum)
            out = torch.zeros_like(x)
            out.scatter_add_(0, dst.unsqueeze(-1).expand(-1, x.size(-1)), messages)
            
            # GIN update: (1 + eps) * x + aggregated
            eps = self.conv.eps if hasattr(self.conv, 'eps') else 0
            combined = (1 + eps) * x + out
            
            # Apply MLP
            out = self.conv.nn(combined)
        else:
            out = self.conv(x, edge_index)
        
        out = self.dropout(out) + residual
        return out
    
  

class LongPoly(nn.Module):
    def __init__(self, hidden_size, K=5, groups=4, dropout=0.1):
        super().__init__()
        assert hidden_size % groups == 0, "hidden_size must be divisible by groups"
        self.K = K
        self.groups = groups
        self.group_channels = hidden_size // groups
        
        # More efficient parameter structure
        self.cheb_coeffs = nn.Parameter(torch.empty(groups, K + 1))
        nn.init.xavier_uniform_(self.cheb_coeffs, gain=0.1)
        
        # Add learnable scaling and bias per group
        self.group_scale = nn.Parameter(torch.ones(groups))
        self.group_bias = nn.Parameter(torch.zeros(groups))
        
        # Lightweight normalization and activation
        self.norm = nn.RMSNorm(hidden_size)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Cache for computational efficiency
        self.register_buffer('_cached_edge_index', None)
        self.register_buffer('_cached_polynomials', None)
        
        
    def forward(self, x, edge_index, edge_weight_norm):
        N, H = x.shape
        
        # Early return for empty graphs
        if edge_index.numel() == 0:
            x_grouped = x.view(N, self.groups, self.group_channels)
            result = self.cheb_coeffs[:, 0].view(1, -1, 1) * x_grouped
            result = result * self.group_scale.view(1, -1, 1) + self.group_bias.view(1, -1, 1)
            return self.dropout(self.activation(self.norm(result.reshape(N, H))))
        
        # Efficient Chebyshev computation without storing all polynomials
        x_grouped = x.view(N, self.groups, self.group_channels)
        result = self.cheb_coeffs[:, 0].view(1, -1, 1) * x_grouped  # T_0 term
        
        if self.K >= 1:
            T_prev2 = x  # T_0
            T_prev1 = propagate(x, edge_index, edge_weight_norm)  # T_1
            
            # Add T_1 contribution
            T1_grouped = T_prev1.view(N, self.groups, self.group_channels)
            result += self.cheb_coeffs[:, 1].view(1, -1, 1) * T1_grouped
            
            # Compute higher order terms on-the-fly
            for k in range(2, self.K + 1):
                T_curr = 2 * propagate(T_prev1, edge_index, edge_weight_norm) - T_prev2
                T_curr_grouped = T_curr.view(N, self.groups, self.group_channels)
                result += self.cheb_coeffs[:, k].view(1, -1, 1) * T_curr_grouped
                
                # Update for next iteration
                T_prev2, T_prev1 = T_prev1, T_curr
        
        # Apply group-wise scaling and bias
        result = result * self.group_scale.view(1, -1, 1) + self.group_bias.view(1, -1, 1)
        
        # Final transformation
        output = result.reshape(N, H)
        return self.dropout(self.activation(self.norm(output)))


# ------------------------------
# Main GraphCliff Filter
# ------------------------------
class GraphCliffFilter(nn.Module):
    def __init__(self, 
                 hidden_size, 
                 edge_dim, 
                 groups=4, 
                 short_dropout=0.1, 
                 mid_K=3):
        super().__init__()
        self.groups = groups

        # Normalization layers
        self.pre_norm = nn.LayerNorm(hidden_size)
        
        # Projection layer
        self.proj = nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        nn.init.xavier_normal_(self.proj.weight, gain=1)
        nn.init.zeros_(self.proj.bias)
        
        self.short = ShortGINE(3 * hidden_size, edge_dim, short_dropout)
        self.long = LongPoly(hidden_size, K=mid_K, groups=groups)
            
        
    def forward(self, u, edge_index, edge_attr):
        h = self.pre_norm(u)
        
        z = self.proj(h)  # [N, 3H]
        
        # Short filter
        z = self.short(z, edge_index, edge_attr)
        
        # Split for different processing paths
        x2, x1, v = torch.chunk(z, 3, dim=-1)  # each [N, H]
        
        # Compute edge normalization once
        if edge_index.numel() > 0:
            edge_weight = torch.ones(edge_index.size(1), device=edge_index.device, dtype=x2.dtype)
            edge_norm = normalize_edges(u.size(0), edge_index, edge_weight)
        else:
            edge_norm = torch.tensor([], device=u.device, dtype=u.dtype)
        
        # Long filter
        mid_out = self.long(x2, edge_index, edge_norm)

        # Gating
        gate = torch.sigmoid(x1)
        y = mid_out * gate + v
            
        # Residual connection
        z_in = y + u

        return z_in


class GraphCliffEncoder(nn.Module):
    def __init__(self, 
                 hidden_size, 
                 edge_dim, 
                 num_layers=3, 
                 groups=4, 
                 mid_K=3, 
                 dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            GraphCliffFilter(
                hidden_size, edge_dim, groups, 
                short_dropout=dropout*0.5, mid_K=mid_K)
            for _ in range(num_layers)
        ])
    
    def forward(self, x, edge_index, edge_attr):
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)
        return x


class GraphCliffRegressor(nn.Module):
    def __init__(self, 
                 atom_in_dim, 
                 edge_dim, 
                 hidden_size=256, 
                 num_layers=3, 
                 groups=4, 
                 mid_K=3, 
                 dropout=0.1):
        super().__init__()
        self.atom_encoder = AtomEncoder(atom_in_dim, hidden_size, dropout)
        self.encoder = GraphCliffEncoder(
                        hidden_size, edge_dim, num_layers, groups, mid_K, dropout)

        self.sagpool = SAGPooling(in_channels=hidden_size, ratio=0.8, GNN=SAGEConv)
        reg_hidden = hidden_size *2

        # Regression head
        self.reg_head = nn.Sequential(
            nn.Linear(reg_hidden, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, x_f, edge_index, edge_attr, batch):
        # Atom encoding
        x = self.atom_encoder(x_f)
        
        # Graph encoding
        x = self.encoder(x, edge_index, edge_attr)

        x, edge_index, _, batch, _, _ = self.sagpool(x, edge_index, edge_attr, batch)
        g = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        # Regression
        y = self.reg_head(g)
        
        return y
     

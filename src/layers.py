"""SAS-GNN propagation layer and confidence-aware exit head.

Reference: "Early-Exit Graph Neural Networks" (arXiv:2505.18088). SAS-GNN is a
weight-shared, ODE-inspired message-passing operator: the *same* weight matrix
is reused at every propagation step, so depth costs no extra parameters and
memory stays flat as the budget L grows.

The weight is split into a symmetric part (diffusion -- smooths features) and an
anti-symmetric part (information-preserving -- resists over-smoothing and
over-squashing). That stability is precisely what makes intermediate
representations trustworthy enough to exit from.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree


class SASGNNLayer(MessagePassing):
    """One shared, weight-tied ODE step:

        W_sym  = 0.5 * (W + W.T)
        W_anti = 0.5 * (W - W.T)
        W_eff  = W_anti - gamma * W_sym - gamma * I
        M      = propagate(A_norm, H)
        H'     = H + epsilon * act(M @ W_eff + b)

    Call this module repeatedly -- it holds a single W for the whole stack.
    """

    def __init__(self, hidden_dim, epsilon=0.1, gamma=0.1, activation="tanh"):
        super().__init__(aggr="add")
        self.hidden_dim = hidden_dim
        self.epsilon = epsilon
        self.gamma = gamma
        self.activation = getattr(F, activation)

        self.W = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        self.bias = nn.Parameter(torch.zeros(hidden_dim))
        nn.init.xavier_uniform_(self.W)

        self.register_buffer("eye", torch.eye(hidden_dim), persistent=False)

    def effective_weight(self):
        W_sym = 0.5 * (self.W + self.W.t())
        W_anti = 0.5 * (self.W - self.W.t())
        return W_anti - self.gamma * W_sym - self.gamma * self.eye

    def forward(self, h, edge_index, edge_weight=None):
        norm, edge_index = _gcn_norm(edge_index, edge_weight, h.size(0), h.dtype, h.device)
        m = self.propagate(edge_index, x=h, norm=norm)
        return h + self.epsilon * self.activation(m @ self.effective_weight() + self.bias)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


def _gcn_norm(edge_index, edge_weight, num_nodes, dtype, device):
    """Symmetric normalisation D^-1/2 (A + I) D^-1/2 with edge weights."""
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), dtype=dtype, device=device)
    edge_index, edge_weight = add_self_loops(
        edge_index, edge_weight, fill_value=1.0, num_nodes=num_nodes
    )
    row, col = edge_index
    deg = torch.zeros(num_nodes, dtype=dtype, device=device).scatter_add_(0, row, edge_weight)
    inv_sqrt = deg.pow(-0.5)
    inv_sqrt[torch.isinf(inv_sqrt)] = 0.0
    return inv_sqrt[row] * edge_weight * inv_sqrt[col], edge_index


class ExitHead(nn.Module):
    """Task branch + halt branch attached after a propagation step.

    The halt branch emits logits over {continue, halt}; Gumbel-Softmax makes
    that discrete choice differentiable end-to-end during training and is
    straight-through (hard) at inference. The model therefore *learns* an exit
    policy per node instead of us hand-tuning a confidence threshold.
    """

    def __init__(self, hidden_dim, n_classes=2, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.task = nn.Linear(hidden_dim, n_classes)
        self.halt = nn.Linear(hidden_dim, 2)  # [continue, halt]

    def forward(self, h, tau=1.0, hard=False, training=True):
        h = self.dropout(h)
        task_logits = self.task(h)
        halt_logits = self.halt(h)
        if training:
            halt_soft = F.gumbel_softmax(halt_logits, tau=tau, hard=hard, dim=-1)
        else:
            halt_soft = F.softmax(halt_logits / tau, dim=-1)
        return task_logits, halt_logits, halt_soft[:, 1]  # P(halt)

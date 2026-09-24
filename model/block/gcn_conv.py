import torch
import torch.nn as nn


class Gcn(nn.Module):
    """
    Spatial Topological Graph Convolution operator for single-frame 2D-to-3D pose lifting.

    Computes spatial node aggregation across partitioned adjacency matrices:
        X_out = einsum('nkcv,kvw->ncw', Linear(X), A)

    Args:
        in_channels (int): Number of input feature channels.
        out_channels (int): Number of output feature channels per partition.
        kernel_size (int): Number of spatial adjacency subsets (default: 4).
        bias (bool): Whether to add a learnable bias to the linear projection.
    """
    def __init__(self, in_channels, out_channels, kernel_size=4, bias=True):
        super().__init__()
        self.kernel_size = kernel_size
        self.linear = nn.Linear(in_channels, out_channels * kernel_size, bias=bias)

    def forward(self, x, A):
        """
        Forward computation for spatial graph convolution.

        Args:
            x (torch.Tensor): Node feature tensor of shape [B, J, C_in].
            A (torch.Tensor): Adjacency tensor of shape [K, J, J].

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Aggregated node features of shape [B, J, C_out].
                - The input adjacency tensor A.
        """
        assert A.dim() == 3 and A.size(0) == self.kernel_size, (
            f"Adjacency matrix A must have shape [{self.kernel_size}, J, J], got {A.shape}"
        )

        B, J, _ = x.shape
        out_c = self.linear.out_features // self.kernel_size

        # Linear projection and spatial partition separation
        x_proj = self.linear(x).view(B, J, self.kernel_size, out_c)
        x_perm = x_proj.permute(0, 2, 3, 1)  # -> [B, K, C_out, J]

        # Graph neighborhood message passing
        out = torch.einsum('nkcv,kvw->ncw', x_perm, A)  # -> [B, C_out, J]
        return out.permute(0, 2, 1).contiguous(), A    # -> [B, J, C_out]


class Gcn_block(nn.Module):
    """
    Spatial Graph Convolutional Block with 1D node projection and residual shortcut.

    Args:
        in_channels (int): Number of input feature channels.
        out_channels (int): Number of output feature channels.
        kernel_size (int): Number of spatial adjacency subsets (default: 4).
        stride (int): Stride for the 1D node convolution (default: 1).
        dropout (float): Dropout probability for regularization (default: 0.05).
        residual (bool): Whether to include a residual shortcut (default: True).
    """
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=1, dropout=0.05, residual=True):
        super().__init__()
        self.inplace = True
        self.momentum = 0.1

        self.gcn = Gcn(in_channels, out_channels, kernel_size=kernel_size)

        self.node_proj = nn.Sequential(
            nn.BatchNorm1d(out_channels, momentum=self.momentum),
            nn.ReLU(inplace=self.inplace),
            nn.Dropout(0.05),
            nn.Conv1d(out_channels, out_channels, kernel_size=1, stride=stride, padding=0),
            nn.BatchNorm1d(out_channels, momentum=self.momentum),
            nn.Dropout(dropout, inplace=self.inplace)
        )

        self.use_res = residual
        if self.use_res:
            if (in_channels == out_channels) and (stride == 1):
                self.residual = nn.Identity()
            else:
                self.residual = nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                    nn.BatchNorm1d(out_channels, momentum=self.momentum)
                )

        self.act = nn.GELU()

    def forward(self, x, A):
        """
        Forward computation of the spatial GCN residual block.

        Args:
            x (torch.Tensor): Input feature tensor of shape [B, J, C].
            A (torch.Tensor): Adjacency tensor of shape [K, J, J].

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Output feature tensor of shape [B, J, C_out].
                - The input adjacency tensor A.
        """
        x_gcn, A = self.gcn(x, A)
        x_gcn_t = x_gcn.permute(0, 2, 1).contiguous()  # -> [B, C, J]

        res = self.residual(x.permute(0, 2, 1).contiguous()) if self.use_res else 0.0
        out = self.node_proj(x_gcn_t) + res
        out = out.permute(0, 2, 1).contiguous()        # -> [B, J, C_out]
        return self.act(out), A
import torch
import torch.nn as nn


class Mlp(nn.Module):
    """
    Standard Multi-Layer Perceptron (MLP) block for spatial feature projection.
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.1):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class KinematicHubOrthogonalFusion(nn.Module):
    """
    Kinematic-Hub Orthogonal Fusion (KHOF) Module.
    Eliminates cross-stream collinear information redundancy between the local GCN
    stream and global self-attention stream via Gram-Schmidt orthogonal projection:
        x_proj_parallel = <x_local_aligned, u_hub> * u_hub
        x_orthogonal = x_local_aligned - x_proj_parallel
    """
    def __init__(self, in_features, hidden_features, dim1=32, dim2=128, drop=0.1):
        super().__init__()
        self.dim1 = dim1
        self.dim2 = dim2

        # Cross-stream manifold alignment
        self.align_local = nn.Linear(dim1, dim2)
        self.fc_reduce = nn.Linear(dim2 * 2, hidden_features)
        self.fc_expand = nn.Linear(hidden_features, in_features)
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop)

        # Parent kinematic hub lookup indices (Human3.6M 17-joint layout)
        homomorphic_hub_tree = torch.tensor(
            [0, 0, 1, 1, 0, 4, 4, 0, 0, 8, 8, 8, 11, 11, 8, 14, 14],
            dtype=torch.long
        )
        self.register_buffer('hub_indices', homomorphic_hub_tree, persistent=False)

        # Zero-initialization for safe residual identity mapping at initialization
        nn.init.zeros_(self.fc_expand.weight)
        nn.init.zeros_(self.fc_expand.bias)

    def forward(self, x):
        # Split local and global branch representations
        x_local, x_global = torch.split(x, [self.dim1, self.dim2], dim=-1)

        aligned_local = self.align_local(x_local)
        hub_global = x_global[:, self.hub_indices, :]

        # Safe unit directional normalization of the parent hub basis
        squared_sum = torch.sum(hub_global ** 2, dim=-1, keepdim=True)
        safe_norm = torch.sqrt(squared_sum + 1e-6)
        unit_hub = hub_global / safe_norm

        # Gram-Schmidt orthogonal projection decomposition
        dot_product = torch.sum(aligned_local * unit_hub, dim=-1, keepdim=True)
        parallel_local = dot_product * unit_hub
        orthogonal_local = aligned_local - parallel_local

        # Full-rank orthogonal complementary concatenation
        x_mod = torch.cat([orthogonal_local, x_global], dim=-1)

        out = self.act(self.fc_reduce(x_mod))
        out = self.drop(out)
        out = self.fc_expand(out)
        out = self.drop(out)
        return out


class Attention(nn.Module):
    """
    Multi-Head Spatial Self-Attention (MHSA) module.
    """
    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
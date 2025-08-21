#!/usr/bin/env python3
"""
CoCoDiff Training Module
=======================
Implements the core CoCoDiff model with budget-conditioned diffusion.
Includes policy network, dynamic gates, and compute-aware loss function.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict
import time

class ResidualBlock(nn.Module):
    """Basic residual block for the toy UNet backbone."""
    
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        
        if in_ch != out_ch:
            self.skip_proj = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.skip_proj = None
            
        self.n_params = sum(p.numel() for p in self.parameters())
        self.flops_full = 2 * 32 * 32 * (3 * 3) * in_ch * out_ch

    def forward(self, x):
        residual = x
        if self.skip_proj is not None:
            residual = self.skip_proj(residual)
            
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        return F.relu(x + residual)

class BudgetPolicy(nn.Module):
    """Lightweight policy network for budget-conditioned gating."""
    
    def __init__(self, n_blocks: int, channels: List[int]):
        super().__init__()
        self.n_blocks = n_blocks
        self.channels = channels
        hidden = 64
        
        self.mlp = nn.Sequential(
            nn.Linear(8, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_blocks + sum(channels) + 1)  # g_k + m_k + r_t
        )

    def forward(self, stats: torch.Tensor):
        """
        Args:
            stats: (B, 8) features containing latent statistics
        Returns:
            logits for gates: block gates + channel masks + timestep reuse
        """
        return self.mlp(stats)

class TinyCoCoDiff(nn.Module):
    """
    CoCoDiff implementation with hierarchical dynamic sparsity.
    Supports block gates, channel masks, and timestep reuse.
    """
    
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        
        chs = [8, 16, 16, 8]  # channel sizes per block
        self.blocks = nn.ModuleList([
            ResidualBlock(3 if i == 0 else chs[i - 1], chs[i]) for i in range(4)
        ])
        self.final_conv = nn.Conv2d(chs[-1], 3, 1)
        
        self.policy = BudgetPolicy(len(self.blocks), chs)
        
        self.last_flops = 0.0
        self.block_flops = torch.tensor([b.flops_full for b in self.blocks], device=device)
        self.channel_flops = torch.tensor(chs, device=device) * 32 * 32 * 3 * 3 * 3  # rough estimate
        self.step_flops = self.block_flops.sum() + self.channel_flops.sum()
        self.full_flops = float(self.step_flops)
        
        self.to(device)

    def extract_latent_stats(self, x: torch.Tensor) -> torch.Tensor:
        """Extract cheap latent statistics for policy input."""
        x_flat = x.view(x.size(0), -1)
        
        stats = torch.zeros(x.size(0), 8, device=x.device)
        stats[:, 0] = torch.norm(x_flat, dim=1)  # L2 norm
        stats[:, 1] = torch.var(x_flat, dim=1)   # variance
        stats[:, 2] = torch.mean(x_flat, dim=1)  # mean
        stats[:, 3] = torch.std(x_flat, dim=1)   # std
        
        centered = x_flat - stats[:, 2:3]
        stats[:, 4] = torch.mean(centered**3, dim=1)  # skew proxy
        stats[:, 5] = torch.mean(centered**4, dim=1)  # kurtosis proxy
        stats[:, 6] = torch.max(x_flat, dim=1)[0]     # max
        stats[:, 7] = torch.min(x_flat, dim=1)[0]     # min
        
        return stats

    def forward_with_budget(self, x: torch.Tensor, budget: float = 1.0) -> Tuple[torch.Tensor, float]:
        """
        Forward pass with budget-conditioned gating.
        
        Args:
            x: Input tensor (B, C, H, W)
            budget: Target compute budget in [0, 1]
            
        Returns:
            output: Generated image
            actual_flops: Realized FLOP count
        """
        batch_size = x.size(0)
        
        stats = self.extract_latent_stats(x)
        
        logits = self.policy(stats)
        
        g_logits = logits[:, :len(self.blocks)]  # block gates
        m_start = len(self.blocks)
        m_end = m_start + len(self.channel_flops)
        m_logits = logits[:, m_start:m_end]      # channel masks
        r_t_logit = logits[:, -1:]               # timestep reuse
        
        thresh = 0.5
        max_iters = 10
        
        for _ in range(max_iters):
            g = (torch.sigmoid(g_logits) > thresh).float()
            m = (torch.sigmoid(m_logits) > thresh).float()
            r_t = (torch.sigmoid(r_t_logit) > thresh).float()
            
            current_flops = self._compute_flops(g, m, r_t)
            current_budget = current_flops / self.full_flops
            
            if current_budget <= budget or thresh >= 0.95:
                break
            thresh += 0.05
        
        output = self._forward_with_gates(x, g, m, r_t)
        
        self.last_flops = current_flops
        return output, current_flops

    def _forward_with_gates(self, x: torch.Tensor, g: torch.Tensor, m: torch.Tensor, r_t: torch.Tensor) -> torch.Tensor:
        """Execute forward pass with given gate configuration."""
        batch_size = x.size(0)
        
        if r_t.mean() > 0.5:  # reuse previous timestep
            return torch.tanh(x + 0.1 * torch.randn_like(x))
        
        ch_ptr = 0
        current_channels = x.size(1)  # Track current channel count
        
        for i, block in enumerate(self.blocks):
            if g[0, i] < 0.5:  # skip block
                if i == 0:  # First block changes 3->8 channels
                    x = F.interpolate(x, size=(32, 32), mode='bilinear', align_corners=False)
                    x = F.pad(x, (0, 0, 0, 0, 0, 8-3))  # Pad channels from 3 to 8
                    current_channels = 8
                continue
                
            x = block(x)
            current_channels = x.size(1)
            
            if ch_ptr < len(m[0]):
                ch_mask = m[0, ch_ptr:ch_ptr + current_channels]
                if len(ch_mask) < current_channels:
                    ch_mask = torch.cat([ch_mask, torch.ones(current_channels - len(ch_mask), device=x.device)])
                ch_mask = ch_mask[:current_channels]  # Truncate if needed
                x = x * ch_mask.view(1, -1, 1, 1)
            ch_ptr += current_channels
        
        if x.size(1) != 8:  # final_conv expects 8 input channels
            if x.size(1) == 3:  # Still original input
                x = F.pad(x, (0, 0, 0, 0, 0, 8-3))  # Pad to 8 channels
            elif x.size(1) > 8:
                x = x[:, :8]  # Truncate to 8 channels
            elif x.size(1) < 8:
                x = F.pad(x, (0, 0, 0, 0, 0, 8-x.size(1)))  # Pad to 8 channels
        
        x = self.final_conv(x)
        return torch.tanh(x)

    def _compute_flops(self, g: torch.Tensor, m: torch.Tensor, r_t: torch.Tensor) -> float:
        """Compute analytic FLOP count for given gate configuration."""
        block_cost = (g[0] * self.block_flops).sum()
        
        channel_cost = (m[0] * self.channel_flops[:len(m[0])]).sum()
        
        reuse_factor = 0.1 if r_t[0] > 0.5 else 1.0
        
        total_flops = (block_cost + channel_cost) * reuse_factor
        return float(total_flops)

    @torch.no_grad()
    def sample(self, prompt: str, budget: float = 1.0, return_flops: bool = False):
        """
        Public API for sampling with budget control.
        
        Args:
            prompt: Text prompt (unused in toy implementation)
            budget: Target compute budget [0, 1]
            return_flops: Whether to return FLOP count
            
        Returns:
            Generated image tensor, optionally with FLOP count
        """
        self.eval()
        
        x = torch.randn(1, 3, 32, 32, device=self.device)
        
        img, flops = self.forward_with_budget(x, budget)
        
        if return_flops:
            return img.cpu(), flops
        return img.cpu()

def compute_aware_loss(model_output: torch.Tensor, target: torch.Tensor, 
                      flops: float, full_flops: float, budget: float,
                      lambda_flops: float = 1e-6, mu_budget: float = 1e-3) -> torch.Tensor:
    """
    Compute-aware loss function as specified in methodology.
    
    L = L_diff + λ·FLOPs(g,m,r) + μ·|E[C|b]/C_full - b|²
    """
    l_diff = F.mse_loss(model_output, target)
    
    flop_penalty = lambda_flops * flops
    
    budget_error = abs(flops / full_flops - budget) ** 2
    budget_loss = mu_budget * budget_error
    
    total_loss = l_diff + flop_penalty + budget_loss
    
    return total_loss

if __name__ == "__main__":
    print("Testing CoCoDiff training module...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = TinyCoCoDiff(device=device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Policy parameters: {sum(p.numel() for p in model.policy.parameters()):,}")
    
    x = torch.randn(2, 3, 32, 32, device=device)
    output, flops = model.forward_with_budget(x, budget=0.5)
    print(f"Output shape: {output.shape}")
    print(f"Realized FLOPs: {flops:.0f} ({flops/model.full_flops:.2%} of full)")
    
    img = model.sample("test prompt", budget=0.3)
    print(f"Sampled image shape: {img.shape}")
    
    target = torch.randn_like(output)
    loss = compute_aware_loss(output, target, flops, model.full_flops, 0.5)
    print(f"Compute-aware loss: {loss.item():.6f}")
    
    print("Training module test completed successfully!")

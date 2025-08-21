#!/usr/bin/env python3
"""
CoCoDiff Evaluation Module
=========================
Implements evaluation metrics and experimental protocols for CoCoDiff.
Includes FID computation, Pareto analysis, and ablation studies.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Any, Optional
from enum import Enum
import time
from tqdm import tqdm

class SparsityVariant(Enum):
    """Ablation study variants for different sparsity axes."""
    FULL = "gmr"      # All gates enabled
    G_ONLY = "g"      # Block gates only
    GM = "gm"         # Block + channel gates
    GR = "gr"         # Block + timestep reuse
    MR = "mr"         # Channel + timestep reuse

@torch.no_grad()
def dummy_fid(gen_imgs: torch.Tensor, real_imgs: torch.Tensor) -> float:
    """
    Dummy FID computation using MSE for toy implementation.
    In real implementation, this would use Inception features.
    """
    min_len = min(len(gen_imgs), len(real_imgs))
    gen_subset = gen_imgs[:min_len]
    real_subset = real_imgs[:min_len]
    
    return float(F.mse_loss(gen_subset, real_subset).item())

@torch.no_grad()
def dummy_clip_score(gen_imgs: torch.Tensor, prompts: List[str]) -> float:
    """
    Dummy CLIP score computation for toy implementation.
    In real implementation, this would use CLIP model.
    """
    img_stats = gen_imgs.mean(dim=[2, 3]).std(dim=1).mean()
    prompt_diversity = len(set(prompts)) / len(prompts)
    return float(img_stats * prompt_diversity)

class CoCoDiffEvaluator:
    """Comprehensive evaluation suite for CoCoDiff experiments."""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.results = {}
        
    def run_experiment_1(self, model, real_loader, prompts: List[str], 
                        budgets: Optional[List[float]] = None, save_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Experiment 1: Quality/Compute Pareto Curve
        Demonstrates CoCoDiff dominance over compute envelope.
        """
        if budgets is None:
            budgets = [0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 1.0]
            
        print("Running Experiment 1: Pareto Curve Analysis...")
        
        real_imgs = []
        for batch_imgs, _ in real_loader:
            real_imgs.append(batch_imgs)
            if len(real_imgs) * batch_imgs.size(0) >= 1000:  # Limit for speed
                break
        real_imgs = torch.cat(real_imgs)[:1000].to(self.device)
        
        results = []
        
        for budget in tqdm(budgets, desc="Budget sweep"):
            gen_imgs = []
            flops_list = []
            
            for i, prompt in enumerate(prompts[:100]):  # Subset for speed
                img, flops = model.sample(prompt, budget=budget, return_flops=True)
                gen_imgs.append(img)
                flops_list.append(flops)
                
            gen_imgs = torch.cat(gen_imgs).to(self.device)
            avg_flops = np.mean(flops_list)
            
            fid_score = dummy_fid(gen_imgs, real_imgs)
            clip_score = dummy_clip_score(gen_imgs, prompts[:100])
            relative_flops = avg_flops / model.full_flops
            
            result = {
                'budget': budget,
                'relative_flops': relative_flops,
                'fid': fid_score,
                'clip_score': clip_score,
                'actual_flops': avg_flops
            }
            results.append(result)
            
            print(f"Budget {budget:.2f}: FLOPs={relative_flops:.3f}, FID={fid_score:.4f}, CLIP={clip_score:.4f}")
        
        self.results['experiment_1'] = results
        
        if save_path:
            self._plot_pareto_curve(results, save_path)
            
        return results
    
    def run_experiment_2(self, model, real_loader, prompts: List[str],
                        budgets: Optional[List[float]] = None, save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Experiment 2: Ablation Study on Sparsity Axes
        Tests individual contribution of block, channel, and timestep gates.
        """
        if budgets is None:
            budgets = [0.2, 0.5, 0.8]
            
        print("Running Experiment 2: Ablation Study...")
        
        real_imgs = []
        for batch_imgs, _ in real_loader:
            real_imgs.append(batch_imgs)
            if len(real_imgs) * batch_imgs.size(0) >= 500:
                break
        real_imgs = torch.cat(real_imgs)[:500].to(self.device)
        
        variant_configs = {
            SparsityVariant.FULL: (True, True, True),   # g, m, r
            SparsityVariant.G_ONLY: (True, False, False),
            SparsityVariant.GM: (True, True, False),
            SparsityVariant.GR: (True, False, True),
            SparsityVariant.MR: (False, True, True),
        }
        
        ablation_results = {}
        
        for variant, (g_on, m_on, r_on) in variant_configs.items():
            print(f"Testing variant: {variant.value}")
            
            original_forward = model.policy.forward
            
            def modified_forward(stats):
                logits = original_forward(stats)
                if not g_on:
                    logits[:, :len(model.blocks)] = -10.0  # Force OFF
                if not m_on:
                    start_idx = len(model.blocks)
                    end_idx = start_idx + len(model.channel_flops)
                    logits[:, start_idx:end_idx] = -10.0
                if not r_on:
                    logits[:, -1] = 10.0  # Force ON (no reuse)
                return logits
            
            model.policy.forward = modified_forward
            
            variant_results = []
            
            for budget in budgets:
                gen_imgs = []
                for prompt in prompts[:50]:  # Smaller subset for ablation
                    img = model.sample(prompt, budget=budget)
                    gen_imgs.append(img)
                    
                gen_imgs = torch.cat(gen_imgs).to(self.device)
                fid_score = dummy_fid(gen_imgs, real_imgs)
                variant_results.append(fid_score)
                
            ablation_results[variant.value] = variant_results
            
            model.policy.forward = original_forward
            
        self.results['experiment_2'] = ablation_results
        
        if save_path:
            self._plot_ablation_study(ablation_results, budgets, save_path)
            
        return ablation_results
    
    def run_experiment_3(self, model, prompts: List[str], save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Experiment 3: Budget Tracking Accuracy & Policy Latency
        Validates budget control precision and policy overhead.
        """
        print("Running Experiment 3: Budget Tracking Analysis...")
        
        n_samples = 256
        random_budgets = np.random.uniform(0.05, 0.95, n_samples)
        budget_errors = []
        
        dummy_prompt = prompts[0] if prompts else "test prompt"
        
        for budget in tqdm(random_budgets, desc="Budget accuracy test"):
            _, actual_flops = model.sample(dummy_prompt, budget=budget, return_flops=True)
            actual_budget = actual_flops / model.full_flops
            error = abs(actual_budget - budget)
            budget_errors.append(error)
        
        budget_errors = np.array(budget_errors)
        
        stats = torch.randn(1, 8, device=self.device)
        
        for _ in range(10):
            _ = model.policy(stats)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start_time = time.perf_counter()
        n_iters = 1000
        
        for _ in range(n_iters):
            _ = model.policy(stats)
            
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        end_time = time.perf_counter()
        avg_latency_ms = (end_time - start_time) / n_iters * 1000
        
        results = {
            'budget_errors': budget_errors,
            'mean_error': budget_errors.mean(),
            'p99_error': np.percentile(budget_errors, 99),
            'policy_latency_ms': avg_latency_ms,
            'policy_params': sum(p.numel() for p in model.policy.parameters())
        }
        
        print(f"Mean budget error: {results['mean_error']:.3%}")
        print(f"99th percentile error: {results['p99_error']:.3%}")
        print(f"Policy latency: {avg_latency_ms:.4f} ms")
        print(f"Policy parameters: {results['policy_params']:,}")
        
        self.results['experiment_3'] = results
        
        if save_path:
            self._plot_budget_accuracy(budget_errors, save_path)
            
        return results
    
    def _plot_pareto_curve(self, results: List[Dict], save_path: str):
        """Generate Pareto curve plot for Experiment 1."""
        sns.set_style("whitegrid")
        plt.figure(figsize=(6, 4))
        
        flops = [r['relative_flops'] for r in results]
        fids = [r['fid'] for r in results]
        
        plt.plot(flops, fids, 'o-', linewidth=2, markersize=6, label='CoCoDiff-Toy')
        plt.xlabel('Relative FLOPs', fontsize=12)
        plt.ylabel('Dummy FID (↓)', fontsize=12)
        plt.title('Quality/Compute Pareto Curve', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Pareto curve saved to {save_path}")
    
    def _plot_ablation_study(self, results: Dict, budgets: List[float], save_path: str):
        """Generate ablation study plot for Experiment 2."""
        sns.set_style("whitegrid")
        plt.figure(figsize=(8, 5))
        
        x_pos = np.arange(len(budgets))
        width = 0.15
        
        variants = list(results.keys())
        colors = plt.cm.get_cmap('Set2')(np.linspace(0, 1, len(variants)))
        
        for i, (variant, fid_scores) in enumerate(results.items()):
            offset = (i - len(variants)/2) * width
            plt.bar(x_pos + offset, fid_scores, width, 
                   label=variant, color=colors[i], alpha=0.8)
        
        plt.xlabel('Budget', fontsize=12)
        plt.ylabel('Dummy FID (↓)', fontsize=12)
        plt.title('Ablation Study: Sparsity Axes', fontsize=14, fontweight='bold')
        plt.xticks(x_pos, [f'{b:.1f}' for b in budgets])
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Ablation study plot saved to {save_path}")
    
    def _plot_budget_accuracy(self, errors: np.ndarray, save_path: str):
        """Generate budget accuracy histogram for Experiment 3."""
        sns.set_style("whitegrid")
        plt.figure(figsize=(6, 4))
        
        plt.hist(errors, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(errors.mean(), color='red', linestyle='--', 
                   label=f'Mean: {errors.mean():.3%}')
        plt.axvline(float(np.percentile(errors, 99)), color='orange', linestyle='--',
                   label=f'99th %ile: {np.percentile(errors, 99):.3%}')
        
        plt.xlabel('|Budget Error|', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Budget Accuracy Distribution', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Budget accuracy plot saved to {save_path}")

if __name__ == "__main__":
    print("Testing CoCoDiff evaluation module...")
    
    real_imgs = torch.randn(100, 3, 32, 32)
    gen_imgs = torch.randn(100, 3, 32, 32)
    prompts = [f"test prompt {i}" for i in range(100)]
    
    fid = dummy_fid(gen_imgs, real_imgs)
    clip_score = dummy_clip_score(gen_imgs, prompts)
    
    print(f"Dummy FID: {fid:.4f}")
    print(f"Dummy CLIP score: {clip_score:.4f}")
    
    evaluator = CoCoDiffEvaluator()
    print("Evaluator created successfully")
    
    print("Evaluation module test completed successfully!")

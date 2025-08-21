#!/usr/bin/env python3
"""
CoCoDiff Main Experiment Script
==============================
Orchestrates the complete CoCoDiff experimental pipeline.
Executes all three experiments and generates academic-quality PDF plots.
"""

import os
import sys
import time
import json
import torch
import numpy as np
from pathlib import Path

from preprocess import get_data_loaders
from train import TinyCoCoDiff, compute_aware_loss
from evaluate import CoCoDiffEvaluator

def setup_environment():
    """Setup experimental environment and reproducibility."""
    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    return device

def create_output_directories():
    """Create required output directories."""
    output_dir = Path(".research/iteration1/images")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def run_cocodiff_experiments():
    """Main experimental pipeline for CoCoDiff."""
    print("=" * 60)
    print("CoCoDiff - Compute-Conditioned Diffusion Experiments")
    print("=" * 60)
    
    device = setup_environment()
    output_dir = create_output_directories()
    
    print("\n1. Loading data and creating model...")
    
    real_loader, prompts = get_data_loaders(batch_size=64, num_workers=0)
    print(f"Loaded {len(real_loader.dataset)} real images")
    print(f"Generated {len(prompts)} synthetic prompts")
    
    model = TinyCoCoDiff(device=device)
    total_params = sum(p.numel() for p in model.parameters())
    policy_params = sum(p.numel() for p in model.policy.parameters())
    
    print(f"Model created with {total_params:,} total parameters")
    print(f"Policy network: {policy_params:,} parameters ({policy_params/1e6:.2f}M)")
    print(f"Full model FLOPs: {model.full_flops:.0f}")
    
    evaluator = CoCoDiffEvaluator(device=device)
    
    print("\n2. Running Experiment 1: Quality/Compute Pareto Curve...")
    print("-" * 50)
    
    budgets_exp1 = [0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 1.0]
    pareto_results = evaluator.run_experiment_1(
        model=model,
        real_loader=real_loader,
        prompts=prompts,
        budgets=budgets_exp1,
        save_path=str(output_dir / "pareto_curve.pdf")
    )
    
    print("\nPareto Curve Results:")
    for result in pareto_results:
        print(f"  Budget {result['budget']:.2f}: "
              f"FLOPs={result['relative_flops']:.3f}, "
              f"FID={result['fid']:.4f}, "
              f"CLIP={result['clip_score']:.4f}")
    
    print("\n3. Running Experiment 2: Ablation Study...")
    print("-" * 50)
    
    budgets_exp2 = [0.2, 0.5, 0.8]
    ablation_results = evaluator.run_experiment_2(
        model=model,
        real_loader=real_loader,
        prompts=prompts,
        budgets=budgets_exp2,
        save_path=str(output_dir / "ablation_fid.pdf")
    )
    
    print("\nAblation Study Results:")
    for variant, fid_scores in ablation_results.items():
        print(f"  {variant}: {fid_scores}")
    
    print("\n4. Running Experiment 3: Budget Tracking & Policy Latency...")
    print("-" * 50)
    
    tracking_results = evaluator.run_experiment_3(
        model=model,
        prompts=prompts,
        save_path=str(output_dir / "budget_accuracy.pdf")
    )
    
    print("\nBudget Tracking Results:")
    print(f"  Mean error: {tracking_results['mean_error']:.3%}")
    print(f"  99th percentile error: {tracking_results['p99_error']:.3%}")
    print(f"  Policy latency: {tracking_results['policy_latency_ms']:.4f} ms")
    print(f"  Policy parameters: {tracking_results['policy_params']:,}")
    
    print("\n5. Running Quick Self-Test...")
    print("-" * 50)
    
    test_prompts = ["A photo of a cat", "A landscape scene", "Abstract art"]
    test_budgets = [0.2, 0.5, 1.0]
    
    print("Testing different budget levels:")
    for budget in test_budgets:
        for i, prompt in enumerate(test_prompts[:2]):  # Limit for speed
            img, flops = model.sample(prompt, budget=budget, return_flops=True)
            relative_flops = flops / model.full_flops
            print(f"  Budget {budget:.1f}, Prompt {i+1}: "
                  f"Generated {img.shape}, FLOPs={relative_flops:.3f}")
    
    x = torch.randn(2, 3, 32, 32, device=device)
    target = torch.randn(2, 3, 32, 32, device=device)
    output, flops = model.forward_with_budget(x, budget=0.5)
    loss = compute_aware_loss(output, target, flops, model.full_flops, 0.5)
    print(f"  Compute-aware loss test: {loss.item():.6f}")
    
    print("\n6. Verifying Output Files...")
    print("-" * 50)
    
    required_files = ["pareto_curve.pdf", "ablation_fid.pdf", "budget_accuracy.pdf"]
    all_files_exist = True
    
    for filename in required_files:
        filepath = output_dir / filename
        if filepath.exists():
            size_kb = filepath.stat().st_size / 1024
            print(f"  ✓ {filename} ({size_kb:.1f} KB)")
        else:
            print(f"  ✗ {filename} - MISSING!")
            all_files_exist = False
    
    if all_files_exist:
        print("\n✓ All required PDF plots generated successfully!")
    else:
        print("\n✗ Some required files are missing!")
        return False
    
    print("\n7. Summary Statistics...")
    print("-" * 50)
    
    best_pareto = min(pareto_results, key=lambda x: x['fid'])
    print(f"Best quality point: Budget {best_pareto['budget']:.2f}, "
          f"FID {best_pareto['fid']:.4f}")
    
    efficient_points = [r for r in pareto_results if r['relative_flops'] < 0.5]
    if efficient_points:
        best_efficient = min(efficient_points, key=lambda x: x['fid'])
        print(f"Best efficient point: Budget {best_efficient['budget']:.2f}, "
              f"FID {best_efficient['fid']:.4f}, FLOPs {best_efficient['relative_flops']:.3f}")
    
    compression_ratio = 1.0 / tracking_results['mean_error'] if tracking_results['mean_error'] > 0 else float('inf')
    print(f"Budget control precision: {1/tracking_results['mean_error']:.1f}x")
    print(f"Policy overhead: {tracking_results['policy_latency_ms']:.4f} ms per call")
    
    print("\n" + "=" * 60)
    print("CoCoDiff Experiments Completed Successfully!")
    print("=" * 60)
    
    metadata = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
        'device': str(device),
        'model_params': total_params,
        'policy_params': policy_params,
        'experiments_completed': ['pareto_curve', 'ablation_study', 'budget_tracking'],
        'output_files': required_files,
        'status_enum': 'stopped'
    }
    
    with open(output_dir / 'experiment_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Experiment metadata saved to {output_dir / 'experiment_metadata.json'}")
    print(f"All plots saved to {output_dir}")
    
    return True

if __name__ == "__main__":
    try:
        success = run_cocodiff_experiments()
        if success:
            print("\n🎉 All experiments completed successfully!")
            sys.exit(0)
        else:
            print("\n❌ Some experiments failed!")
            sys.exit(1)
    except Exception as e:
        print(f"\n💥 Experiment failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

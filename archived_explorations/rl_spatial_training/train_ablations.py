#!/usr/bin/env python3
"""
Ablation study: Train MBM with different components disabled.
Each config runs for ~90 updates with 3 seeds.
"""
import subprocess
import re
import time
import numpy as np
import os

ABLATION_CONFIGS = [
    {"name": "full_mbm", "flags": ""},
    {"name": "no_plasticity", "flags": "--use_plasticity False"},
    {"name": "no_hippocampus", "flags": "--use_hippocampus False"},
    {"name": "no_cerebellum", "flags": "--use_cerebellum False"},
    {"name": "no_memory_policy", "flags": "--use_memory_policy False"},
]

SEEDS = [200, 201, 202]  # Fresh seeds for ablations
TOTAL_STEPS = 50000000  # Same as main experiments
EVAL_EVERY = 10

def run_ablation(config_name, flags, seed):
    """Run training for one ablation config and seed."""
    log_file = f"logs/ablation_{config_name}_seed{seed}.log"
    
    cmd = f"PYTHONUNBUFFERED=1 python3 train_vectorized.py --seed {seed} --total_steps {TOTAL_STEPS} --eval_every {EVAL_EVERY} {flags}"
    
    print(f"  Running: {config_name} seed={seed}")
    start = time.time()
    
    # Remove old checkpoint
    try:
        os.remove("brain_vectorized_best.pth")
    except:
        pass
    
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True,
        cwd="/home/ahmed/Downloads/Kandel",
        timeout=1800  # 30 min timeout
    )
    
    elapsed = time.time() - start
    
    # Save log
    with open(log_file, 'w') as f:
        f.write(result.stdout + result.stderr)
    
    # Parse best SR
    matches = re.findall(r'New Best SR: ([0-9.]+)', result.stdout + result.stderr)
    if matches:
        best_sr = float(matches[-1])
    else:
        # Try alternate pattern
        matches = re.findall(r'Eval SR: ([0-9.]+)', result.stdout + result.stderr)
        best_sr = float(matches[-1]) if matches else 0.0
    
    print(f"    -> SR: {best_sr:.3f} ({elapsed:.0f}s)")
    return best_sr


def main():
    os.makedirs("logs", exist_ok=True)
    
    print("=" * 60)
    print("ABLATION STUDY")
    print("=" * 60)
    print(f"Configs: {len(ABLATION_CONFIGS)}")
    print(f"Seeds per config: {len(SEEDS)}")
    print(f"Total runs: {len(ABLATION_CONFIGS) * len(SEEDS)}")
    print()
    
    results = {}
    
    for config in ABLATION_CONFIGS:
        config_name = config["name"]
        flags = config["flags"]
        
        print(f"\n{'='*60}")
        print(f"CONFIG: {config_name}")
        print(f"{'='*60}")
        
        seed_results = []
        for seed in SEEDS:
            sr = run_ablation(config_name, flags, seed)
            seed_results.append(sr)
        
        mean_sr = np.mean(seed_results)
        std_sr = np.std(seed_results)
        results[config_name] = {
            "mean": mean_sr,
            "std": std_sr,
            "seeds": seed_results
        }
        
        print(f"  {config_name}: {mean_sr:.3f} ± {std_sr:.3f}")
    
    # Summary
    print("\n" + "=" * 60)
    print("ABLATION RESULTS SUMMARY")
    print("=" * 60)
    print(f"\n{'Config':<20} {'Mean SR':>10} {'Std':>8} {'Seeds':>20}")
    print("-" * 60)
    
    for name, data in results.items():
        seeds_str = ", ".join(f"{s:.2f}" for s in data["seeds"])
        print(f"{name:<20} {data['mean']:>10.3f} {data['std']:>8.3f} [{seeds_str}]")
    
    # Compute drops from full
    if "full_mbm" in results:
        full_sr = results["full_mbm"]["mean"]
        print(f"\n{'Config':<20} {'Drop from Full':>15}")
        print("-" * 40)
        for name, data in results.items():
            if name != "full_mbm":
                drop = full_sr - data["mean"]
                print(f"{name:<20} {drop:>+15.1%}")
    
    # Save results to file
    with open("logs/ablation_results.txt", 'w') as f:
        f.write("ABLATION STUDY RESULTS\n")
        f.write("=" * 60 + "\n\n")
        for name, data in results.items():
            f.write(f"{name}: {data['mean']:.3f} ± {data['std']:.3f}\n")
            f.write(f"  Seeds: {data['seeds']}\n\n")
    
    print("\nResults saved to logs/ablation_results.txt")


if __name__ == "__main__":
    main()

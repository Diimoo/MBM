#!/usr/bin/env python3
"""
Simple multi-seed validation using existing train_vectorized.py infrastructure.
"""
import subprocess
import re
import numpy as np
import time

def run_seed(seed, steps=10000):
    """Run training for a seed and extract final SR."""
    print(f"\n{'='*60}")
    print(f"SEED {seed}")
    print(f"{'='*60}")
    
    cmd = f"python3 train_vectorized.py --seed {seed} --total_steps {steps}"
    
    start = time.time()
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True,
        cwd="/home/ahmed/Downloads/Kandel"
    )
    elapsed = time.time() - start
    
    # Parse output for SR values
    output = result.stdout + result.stderr
    
    # Find all SR values
    sr_matches = re.findall(r'Eval SR: ([0-9.]+)', output)
    
    if sr_matches:
        srs = [float(s) for s in sr_matches]
        final_sr = srs[-1]
        best_sr = max(srs)
        print(f"  Final SR: {final_sr:.3f}, Best SR: {best_sr:.3f}")
        print(f"  Elapsed: {elapsed:.1f}s")
        return {'final_sr': final_sr, 'best_sr': best_sr, 'all_srs': srs}
    else:
        print(f"  ERROR: Could not parse SR from output")
        print(f"  Output: {output[:500]}")
        return {'final_sr': 0.0, 'best_sr': 0.0, 'all_srs': []}


def main():
    seeds = [100, 101, 102, 103, 104]  # 5 seeds for quick validation
    steps = 10000  # 10k steps per seed
    
    print("="*60)
    print("MULTI-SEED VALIDATION (5 seeds, 10k steps each)")
    print("="*60)
    
    results = {}
    for seed in seeds:
        results[seed] = run_seed(seed, steps=steps)
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    final_srs = [r['final_sr'] for r in results.values()]
    best_srs = [r['best_sr'] for r in results.values()]
    
    print(f"\nFinal Success Rates:")
    for seed, result in results.items():
        status = "✅" if result['final_sr'] >= 0.70 else "⚠️" if result['final_sr'] >= 0.40 else "❌"
        print(f"  Seed {seed}: Final={result['final_sr']:.3f}, Best={result['best_sr']:.3f} {status}")
    
    print(f"\nAggregate Statistics:")
    print(f"  Mean Final SR: {np.mean(final_srs):.3f} ± {np.std(final_srs):.3f}")
    print(f"  Mean Best SR:  {np.mean(best_srs):.3f} ± {np.std(best_srs):.3f}")
    print(f"  Range: [{np.min(final_srs):.3f}, {np.max(final_srs):.3f}]")
    
    # Success criteria
    mean_sr = np.mean(final_srs)
    std_sr = np.std(final_srs)
    min_sr = np.min(final_srs)
    
    print(f"\n{'='*60}")
    print("SUCCESS CRITERIA CHECK")
    print(f"{'='*60}")
    print(f"  Mean SR > 0.60: {'✅' if mean_sr > 0.60 else '❌'} ({mean_sr:.3f})")
    print(f"  Std SR < 0.15:  {'✅' if std_sr < 0.15 else '❌'} ({std_sr:.3f})")
    print(f"  Min SR > 0.30:  {'✅' if min_sr > 0.30 else '❌'} ({min_sr:.3f})")
    
    if mean_sr > 0.60 and std_sr < 0.15 and min_sr > 0.30:
        print(f"\n🎉 ALL CRITERIA MET!")
    elif mean_sr > 0.45:
        print(f"\n⚠️ PARTIAL SUCCESS")
    else:
        print(f"\n❌ CRITERIA NOT MET")


if __name__ == "__main__":
    main()

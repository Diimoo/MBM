import json
import numpy as np
import os

def analyze_hierarchical():
    path = "experiments/hierarchical_results.json"
    if not os.path.exists(path):
        print(f"No results found at {path}")
        return

    with open(path, 'r') as f:
        data = json.load(f)
    
    # Structure: data['mbm_hier'] is a list of seeds.
    # Each seed is a list of phases (matrices).
    # Actually, looking at validate_hierarchical.py:
    # results['mbm_hier'].append(matrix)
    # where matrix is a list of phases (3 phases).
    # Each phase is a list of SRs for [5x5, 7x7, 10x10].
    
    raw_seeds = data['mbm_hier']
    n_seeds = len(raw_seeds)
    n_phases = 3
    n_tasks = 3
    
    # Shape: (n_seeds, n_phases, n_tasks)
    results = np.array(raw_seeds)
    
    print(f"=== Hierarchical MBM Analysis ({n_seeds} Seeds) ===")
    
    # 1. Average SR Matrix (Mean ± Std)
    mean_matrix = np.mean(results, axis=0)
    std_matrix = np.std(results, axis=0)
    
    print("\n--- Average Success Rate Matrix ---")
    print(f"{'Phase':<15} | {'Eval 5x5':<15} | {'Eval 7x7':<15} | {'Eval 10x10':<15}")
    print("-" * 70)
    
    tasks = ['5x5', '7x7', '10x10']
    for i in range(n_phases):
        row_str = f"After {tasks[i]:<9}"
        vals = []
        for j in range(n_tasks):
            vals.append(f"{mean_matrix[i, j]:.3f} (±{std_matrix[i, j]:.3f})")
        print(f"{row_str} | {vals[0]:<15} | {vals[1]:<15} | {vals[2]:<15}")

    # 2. Key Metrics
    # Forward Transfer: Performance on T2 after T1 vs Baseline (we don't have a direct baseline here, but we can look at T2 immediate performance)
    # Backward Transfer: Performance on T1 after T2/T3 vs Initial T1 performance
    
    t1_initial = results[:, 0, 0] # After Phase 1, Eval Task 1
    t1_final = results[:, 2, 0]   # After Phase 3, Eval Task 1
    
    consolidation = t1_final - t1_initial
    
    print("\n--- Consolidation Metrics (Task 5x5) ---")
    print(f"Initial SR (After T1): {np.mean(t1_initial):.3f} (±{np.std(t1_initial):.3f})")
    print(f"Final SR (After T3):   {np.mean(t1_final):.3f} (±{np.std(t1_final):.3f})")
    print(f"Mean Change:           {np.mean(consolidation):.3f}")
    
    # Count "Dead" Seeds (SR < 0.05 on Task 1 after Phase 1)
    dead_seeds = np.sum(t1_initial < 0.05)
    print(f"\nDead Seeds (Initial SR < 0.05): {dead_seeds}/{n_seeds}")

if __name__ == "__main__":
    analyze_hierarchical()

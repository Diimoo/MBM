import json
import numpy as np
from scipy import stats

def analyze_results(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    mbm_runs = np.array(data['mbm']) # (Seeds, Phases, Tasks)
    ppo_runs = np.array(data['ppo']) # (Seeds, Phases, Tasks)
    
    task_names = ['5x5', '7x7', '10x10']
    
    print("=== Consolidation Analysis (10 Seeds) ===")
    
    for name, runs in [("MBM", mbm_runs), ("PPO", ppo_runs)]:
        avg_matrix = np.mean(runs, axis=0)
        std_matrix = np.std(runs, axis=0)
        
        print(f"\n--- {name} Average Success Rate Matrix ---")
        print("Columns: Eval 5x5, 7x7, 10x10")
        for i, phase in enumerate(['After 5x5', 'After 7x7', 'After 10x10']):
            row_str = " | ".join([f"{avg_matrix[i, j]:.3f} (±{std_matrix[i, j]:.3f})" for j in range(3)])
            print(f"{phase}: {row_str}")
            
        # Backward Transfer (Consolidation) on Task A (5x5)
        # Compare SR on 5x5 after Phase 1 vs After Phase 3
        sr_a_init = runs[:, 0, 0]
        sr_a_final = runs[:, 2, 0]
        
        diff = sr_a_final - sr_a_init
        mean_diff = np.mean(diff)
        t_stat, p_val = stats.ttest_rel(sr_a_final, sr_a_init)
        
        print(f"\n{name} Consolidation Effect (Task A):")
        print(f"  Initial SR (After T1): {np.mean(sr_a_init):.3f}")
        print(f"  Final SR (After T3):   {np.mean(sr_a_final):.3f}")
        print(f"  Mean Improvement:      {mean_diff:+.3f}")
        print(f"  P-value:               {p_val:.4f} ({'Significant' if p_val < 0.05 else 'Not Significant'})")

if __name__ == "__main__":
    analyze_results("experiments/consolidation_results.json")

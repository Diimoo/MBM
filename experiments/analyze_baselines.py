import json
import numpy as np
from scipy import stats

def analyze_ewc(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    ewc_runs = np.array(data['ppo_ewc']) # (Seeds, Phases, Tasks)
    
    # Load previous results for comparison if they exist
    mbm_data = None
    ppo_data = None
    try:
        with open("experiments/consolidation_results.json", 'r') as f:
            cons_data = json.load(f)
            mbm_data = np.array(cons_data['mbm'])
            ppo_data = np.array(cons_data['ppo'])
    except:
        pass

    print("=== Continual Learning Baseline Comparison ===")
    
    datasets = [("PPO+EWC", ewc_runs)]
    if mbm_data is not None: datasets.append(("MBM (Single Layer)", mbm_data))
    if ppo_data is not None: datasets.append(("Vanilla PPO", ppo_data))

    for name, runs in datasets:
        avg_matrix = np.mean(runs, axis=0)
        std_matrix = np.std(runs, axis=0)
        
        print(f"\n--- {name} Average Success Rate Matrix ---")
        for i, phase in enumerate(['After T1', 'After T2', 'After T3']):
            row_str = " | ".join([f"{avg_matrix[i, j]:.3f}" for j in range(3)])
            print(f"{phase}: {row_str}")
            
        # Backward Transfer on Task A
        sr_a_init = runs[:, 0, 0]
        sr_a_final = runs[:, 2, 0]
        diff = sr_a_final - sr_a_init
        t_stat, p_val = stats.ttest_rel(sr_a_final, sr_a_init)
        
        print(f"  Task A Consolidation: {np.mean(diff):+.3f} (p={p_val:.4f})")
        
        # Forgetting on Task A (After T1 vs After T3)
        # For CL, we usually look at the drop from the peak
        peak_a = runs[:, 0, 0]
        final_a = runs[:, 2, 0]
        forgetting = peak_a - final_a
        print(f"  Task A Forgetting:    {np.mean(forgetting):.3f}")

if __name__ == "__main__":
    analyze_ewc("experiments/ewc_results.json")

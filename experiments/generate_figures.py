import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import os

def plot_validation_results(results_path="experiments/validation_results.json"):
    if not os.path.exists(results_path):
        print(f"Skipping validation plot: {results_path} not found")
        return
        
    with open(results_path, "r") as f:
        results = json.load(f)
        
    mbm_srs = results['mbm_srs']
    ppo_srs = results['ppo_srs']
    
    plt.figure(figsize=(8, 6))
    data = [mbm_srs, ppo_srs]
    labels = ['MBM (Full)', 'PPO Baseline']
    
    sns.boxplot(data=data)
    plt.xticks(range(len(labels)), labels)
    plt.ylabel('Success Rate', fontsize=12)
    plt.title('Statistical Validation: MBM vs PPO (5x5 Gridworld)', fontsize=14)
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/fig1_statistical_validation.png', dpi=300)
    print("Saved figures/fig1_statistical_validation.png")

def plot_ablation_results(results_path="experiments/ablation_results.json"):
    if not os.path.exists(results_path):
        print(f"Skipping ablation plot: {results_path} not found")
        return
        
    with open(results_path, "r") as f:
        results = json.load(f)
        
    names = list(results.keys())
    means = [results[n]['mean'] for n in names]
    stds = [results[n]['std'] for n in names]
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=names, y=means, palette='viridis')
    plt.errorbar(x=range(len(names)), y=means, yerr=stds, fmt='none', c='black', capsize=5)
    
    plt.xticks(rotation=45)
    plt.ylabel('Success Rate', fontsize=12)
    plt.title('Ablation Study: Component Importance on POMDP', fontsize=14)
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/fig2_ablation_study.png', dpi=300)
    print("Saved figures/fig2_ablation_study.png")

def plot_continual_learning(results_path="experiments/continual_learning_results.json"):
    if not os.path.exists(results_path):
        print(f"Skipping continual learning plot: {results_path} not found")
        return
        
    with open(results_path, "r") as f:
        results = json.load(f)
        
    matrix = np.array(results['matrix'])
    tasks = results['task_names']
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=tasks, yticklabels=[f"After {t}" for t in tasks],
                vmin=0, vmax=1)
    
    plt.xlabel('Evaluation Task', fontsize=12)
    plt.ylabel('Training Phase', fontsize=12)
    plt.title('Continual Learning: Forgetting Matrix', fontsize=14)
    
    plt.tight_layout()
    plt.savefig('figures/fig3_continual_learning.png', dpi=300)
    print("Saved figures/fig3_continual_learning.png")

def plot_scaling_results(results_path="experiments/scaling_results.json"):
    if not os.path.exists(results_path):
        print(f"Skipping scaling plot: {results_path} not found")
        return
        
    with open(results_path, "r") as f:
        results = json.load(f)
        
    # Only plot successful ones
    results = [r for r in results if r.get('stable', False)]
    if not results:
        return
        
    dz = [r['d_z'] for r in results]
    times = [r['time_ms'] for r in results]
    mems = [r['mem_mb'] for r in results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Time scaling
    ax1.plot(dz, times, 'o-', linewidth=2, color='tab:blue')
    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log')
    ax1.set_xlabel('Number of Neurons (d_z)', fontsize=12)
    ax1.set_ylabel('Time per Step (ms)', fontsize=12)
    ax1.set_title('Inference Speed Scaling (Sparse)', fontsize=14)
    ax1.grid(True, which="both", ls="-", alpha=0.3)
    
    # Memory scaling
    ax2.plot(dz, mems, 's-', linewidth=2, color='tab:red')
    ax2.set_xscale('log', base=2)
    ax2.set_yscale('log')
    ax2.set_xlabel('Number of Neurons (d_z)', fontsize=12)
    ax2.set_ylabel('Memory Usage (MB)', fontsize=12)
    ax2.set_title('Memory Usage Scaling (Sparse)', fontsize=14)
    ax2.grid(True, which="both", ls="-", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/fig4_scaling_results.png', dpi=300)
    print("Saved figures/fig4_scaling_results.png")

if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)
    plot_validation_results()
    plot_ablation_results()
    plot_continual_learning()
    plot_scaling_results()

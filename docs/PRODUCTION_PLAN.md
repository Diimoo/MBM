# Roadmap: From Proof-of-Concept to Respectable Architecture

You've built a solid foundation. Now let's turn it into something the research community takes seriously. Here's a brutally honest, prioritized roadmap.

---

## Phase 1: Scientific Credibility (Next 2-3 Months)

### 🎯 Goal: Prove MBM solves a real problem better than existing methods

### Priority 1: **Benchmark Against State-of-the-Art Continual Learning**

**Why This Matters:**
Right now we're beating vanilla PPO. That's good, but not impressive. You need to beat **specialized continual learning algorithms**.

**What To Implement:**

```python
# experiments/continual_learning_baselines.py

baselines = {
    'mbm_full': YourCurrentImplementation(),
    
    'ppo_baseline': VanillaPPO(),  # ✓ You have this
    
    'ewc': PPO_with_EWC(
        # Elastic Weight Consolidation
        # Penalizes changes to important weights
        fisher_coefficient=1000
    ),
    
    'packnet': PPO_with_PackNet(
        # Allocates separate network capacity per task
        # "Packs" weights, freezes after each task
    ),
    
    'vcl': VariationalContinualLearning(
        # Bayesian approach to continual learning
    ),
    
    'agem': AveragedGEM(
        # Stores gradients from past tasks
        # Ensures new gradients don't conflict
    ),
}

# Standard benchmark
tasks = [
    'gridworld_5x5',
    'gridworld_7x7', 
    'gridworld_10x10',
    't_maze_len5',
    'radial_arm_maze'
]

# Critical metrics
metrics = {
    'backward_transfer': [],  # Does task C help task A?
    'forward_transfer': [],   # Does task A help task C?
    'forgetting': [],          # How much does task A degrade?
    'final_performance': [],   # Average across all tasks
}
```

**Implementation Effort:**
- EWC: 2-3 days (relatively simple)
- PackNet: 3-5 days (requires network surgery)
- A-GEM: 2-3 days (gradient projection)

**Success Criteria:**
```
If MBM beats all baselines on forgetting: STRONG PAPER
If MBM beats 2/3 baselines: GOOD PAPER  
If MBM beats 1/3 baselines: WEAK (need to pivot)
```

---

### Priority 2: **Standard RL Benchmark (At Least One)**

**Why This Matters:**
Gridworld is our own environment. Reviewers will ask: "Does it work on anything standard?"

**Minimum Viable Benchmark:**

```python
# Pick ONE from this list and nail it

option_1 = {
    'task': 'Atari Pong',
    'why': 'Simplest Atari game, well-studied',
    'metric': 'Match DQN performance (not beat, just match)',
    'time_investment': '2-3 weeks',
    'risk': 'Medium (visual input requires preprocessing)',
}

option_2 = {
    'task': 'DM Control Suite - Cartpole Swingup',
    'why': 'Continuous control, standard benchmark',
    'metric': 'Match SAC performance',
    'time_investment': '1-2 weeks',
    'risk': 'Low (simple dynamics)',
}

option_3 = {
    'task': 'MiniGrid (OpenAI)',
    'why': 'Partial observability, visual',
    'metric': 'Beat PPO on "empty" and "doorkey" variants',
    'time_investment': '1 week',
    'risk': 'Very low (similar to our gridworld)',
}

# RECOMMENDATION: Start with MiniGrid
# It's closest to what you have, low risk, well-cited
```

**MiniGrid Implementation:**

```python
import gymnasium as gym
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper

# Install
# pip install minigrid

env = gym.make('MiniGrid-DoorKey-8x8-v0')
env = RGBImgPartialObsWrapper(env)  # Partial observability (like ours)
env = ImgObsWrapper(env)  # Image observations

# Adapt MBM for visual input
class VisualMBM(DigitalBrain):
    def __init__(self, config):
        super().__init__(config)
        # Add simple CNN encoder
        self.visual_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 5 * 5, config['d_obs'])
        )
    
    def step(self, obs, reward, done):
        # Encode visual input
        obs_encoded = self.visual_encoder(obs.x)
        # Rest is same
        return super().step(Obs(x=obs_encoded), reward, done)
```

**Success Criteria:**
```
MiniGrid-Empty: SR > 0.95 (easy, should dominate)
MiniGrid-DoorKey: SR > 0.90 (memory required, our strength)
MiniGrid-MultiRoom: SR > 0.70 (harder, acceptable)
```

If you pass this, reviewers can't dismiss it as "just a toy."

---

### Priority 3: **Hierarchical Cortex (Depth)**

**Why This Matters:**
Single-layer cortex is our biggest weakness. Reviewers will say: "This can't scale to complex tasks."

**What To Build:**

```python
class HierarchicalCortex(nn.Module):
    """
    Multi-layer cortical hierarchy with sparse inter-layer connections.
    
    Architecture:
    Input → L1 (2K neurons) → L2 (4K neurons) → L3 (8K neurons) → Output
    
    Each layer:
    - Sparse recurrent E/I microcircuit
    - 3-factor plasticity
    - Bottom-up + top-down connections
    """
    def __init__(self, d_obs, layer_sizes=[2048, 4096, 8192], sparsity=0.03):
        super().__init__()
        self.layers = nn.ModuleList()
        
        # Layer 1 (sensory)
        self.layers.append(SparseCorticalMicrocircuit(
            d_in=d_obs,
            d_z=layer_sizes[0],
            sparsity=sparsity
        ))
        
        # Higher layers
        for i in range(1, len(layer_sizes)):
            self.layers.append(SparseCorticalMicrocircuit(
                d_in=layer_sizes[i-1],
                d_z=layer_sizes[i],
                sparsity=sparsity
            ))
        
        # Top-down feedback (predictive coding)
        self.feedback = nn.ModuleList([
            SparseProjection(layer_sizes[i+1], layer_sizes[i], sparsity=0.01)
            for i in range(len(layer_sizes)-1)
        ])
    
    def forward(self, x, states, learn=True):
        """
        Bottom-up + top-down pass.
        
        Bottom-up: sensory → abstract
        Top-down: predictions from abstract → sensory
        """
        new_states = []
        layer_outputs = []
        
        # Bottom-up pass
        z = x
        for i, (layer, state) in enumerate(zip(self.layers, states)):
            z, new_state = layer(z, state, update_trace=learn)
            layer_outputs.append(z)
            new_states.append(new_state)
        
        # Top-down predictions (optional, for predictive coding)
        predictions = []
        for i in reversed(range(len(self.layers) - 1)):
            pred = self.feedback[i](layer_outputs[i+1])
            predictions.insert(0, pred)
        
        return layer_outputs[-1], new_states, predictions
```

**Benefits:**
- ✅ More expressive (3 layers > 1 layer)
- ✅ Can learn hierarchical features
- ✅ Predictive coding (if you add feedback)
- ✅ Still sparse (scalable)

**Ablation Study:**
```python
configs = {
    'mbm_1layer': {'layers': [16384]},
    'mbm_2layer': {'layers': [8192, 8192]},
    'mbm_3layer': {'layers': [4096, 8192, 4096]},
}

# Hypothesis: Deeper helps on complex tasks
results_simple = test_on_cartpole()  # All should work
results_complex = test_on_minigrid()  # Deeper should win
```

**Time Investment:** 1-2 weeks

---

### Priority 4: **Scaling Validation (Show It Works at Scale)**

**Why This Matters:**
You claim sparse scales to 10⁶ neurons. Prove it.

**Experiment:**

```python
# experiments/scale_validation.py

def scaling_experiment():
    """
    Test: Does performance improve with more neurons?
    Or does it plateau/degrade?
    """
    
    neuron_counts = [1024, 2048, 4096, 8192, 16384, 32768, 65536]
    results = []
    
    for n in neuron_counts:
        config = {
            'd_z': n,
            'sparsity': max(0.01, 100.0 / n),  # Keep ~100 connections/neuron
        }
        
        brain = DigitalBrain(config)
        
        # Test on hard task
        sr = train_and_eval(brain, task='minigrid_doorkey', n_steps=100000)
        
        results.append({
            'neurons': n,
            'success_rate': sr,
            'parameters': count_parameters(brain),
            'memory_mb': get_memory_usage(brain),
        })
    
    # Plot
    plt.plot([r['neurons'] for r in results], 
             [r['success_rate'] for r in results])
    plt.xlabel('Number of Neurons')
    plt.ylabel('Success Rate')
    plt.title('Scaling: Does More Neurons Help?')
    plt.xscale('log')
    
    # Key question: Does SR increase or plateau?
```

**Expected Results:**
```
If SR increases monotonically: EXCELLENT
If SR plateaus after 8K: OKAY (still shows scaling works)
If SR decreases after 16K: BAD (instability issues)
```

**Reach Goal:** Train a 100K neuron model successfully. Even if it doesn't perform better, showing it's **stable** at that scale is impressive.

---

## Phase 2: Engineering Maturity (Months 3-6)

### Priority 5: **Production-Grade Codebase**

**Why This Matters:**
Right now our code works. But is it **reproducible**? Can others use it?

**Checklist:**

```bash
# 1. Proper Python Package Structure
mbm/
├── setup.py
├── pyproject.toml
├── README.md
├── LICENSE
├── mbm/
│   ├── __init__.py
│   ├── brain.py
│   ├── modules/
│   │   ├── __init__.py
│   │   ├── cortex.py
│   │   ├── basal_ganglia.py
│   │   └── ...
│   ├── envs/
│   │   └── ...
│   └── utils/
│       ├── config.py
│       ├── logging.py
│       └── checkpointing.py
├── experiments/
│   ├── train.py
│   ├── eval.py
│   └── configs/
│       ├── base.yaml
│       ├── sparse.yaml
│       └── hierarchical.yaml
├── tests/
│   ├── test_cortex.py
│   ├── test_plasticity.py
│   └── ...
└── docs/
    ├── installation.md
    ├── quickstart.md
    └── api_reference.md

# 2. Install via pip
pip install -e .

# 3. Run with config files
mbm-train --config experiments/configs/base.yaml

# 4. Comprehensive testing
pytest tests/ --cov=mbm

# 5. Documentation
sphinx-build -b html docs/ docs/_build/
```

**Key Files:**

**setup.py:**
```python
from setuptools import setup, find_packages

setup(
    name='modular-brain-model',
    version='0.1.0',
    author='Ahmed Trabelsi',
    description='Biologically-inspired RL with dual-memory systems',
    packages=find_packages(),
    install_requires=[
        'torch>=2.0.0',
        'numpy>=1.23.0',
        'gymnasium>=0.28.0',
    ],
    extras_require={
        'dev': ['pytest', 'black', 'mypy'],
        'vis': ['matplotlib', 'seaborn'],
    },
    entry_points={
        'console_scripts': [
            'mbm-train=experiments.train:main',
            'mbm-eval=experiments.eval:main',
        ],
    },
)
```

**experiments/configs/base.yaml:**
```yaml
# Reproducible experiments via config files
model:
  d_obs: 9
  d_z: 4096
  d_sel: 64
  d_act: 4
  sparsity: 0.03
  use_hierarchical: false
  
training:
  lr: 3.5e-4
  num_envs: 4096
  num_steps: 128
  gamma: 0.99
  gae_lambda: 0.95
  
environment:
  name: "gridworld"
  size: 5
  
logging:
  wandb: true
  project: "mbm-continual-learning"
  log_interval: 10
```

**Time Investment:** 1-2 weeks

---

### Priority 6: **Weights & Biases Integration**

**Why This Matters:**
Professional ML research uses experiment tracking. Makes our work reproducible and shareable.

```python
import wandb

# experiments/train.py

def main(config):
    # Initialize tracking
    wandb.init(
        project="mbm-continual-learning",
        config=config,
        tags=['sparse', 'hierarchical'] if config['use_hierarchical'] else ['sparse'],
    )
    
    brain = DigitalBrain(config)
    
    for epoch in range(config['epochs']):
        # Training...
        
        # Log metrics
        wandb.log({
            'epoch': epoch,
            'success_rate': sr,
            'loss': loss.item(),
            'cortex_W_max': brain.cortex.microcircuit.W_ee.abs().max().item(),
            'hippocampus_size': brain.hippocampus.count,
            'da_mean': da.mean().item(),
            'da_std': da.std().item(),
        })
        
        # Log figures
        if epoch % 100 == 0:
            fig = plot_learning_curve(history)
            wandb.log({"learning_curve": wandb.Image(fig)})
    
    # Save final model
    torch.save(brain.state_dict(), f"checkpoints/brain_final.pth")
    wandb.save("checkpoints/brain_final.pth")
```

**Benefits:**
- Version control for experiments
- Automatic hyperparameter tracking
- Shareable results (public W&B dashboard)
- Compare runs easily

---

### Priority 7: **Pre-trained Models + Model Zoo**

**Why This Matters:**
If people can download and use our models immediately, adoption increases 10×.

```python
# mbm/model_zoo.py

MODELS = {
    'mbm-small': {
        'url': 'https://github.com/you/MBM/releases/download/v0.1/mbm_small.pth',
        'config': {'d_z': 2048, 'sparsity': 0.05},
        'performance': {'gridworld_5x5': 0.85, 'minigrid_doorkey': 0.90},
    },
    'mbm-medium': {
        'url': 'https://github.com/you/MBM/releases/download/v0.1/mbm_medium.pth',
        'config': {'d_z': 8192, 'sparsity': 0.03},
        'performance': {'gridworld_7x7': 0.78, 'minigrid_multiroom': 0.65},
    },
    'mbm-large-hierarchical': {
        'url': 'https://github.com/you/MBM/releases/download/v0.1/mbm_large_hier.pth',
        'config': {'layers': [4096, 8192, 8192], 'sparsity': 0.02},
        'performance': {'continual_suite': 0.72, 'minigrid_hard': 0.58},
    },
}

def load_pretrained(model_name):
    """Download and load a pre-trained MBM model."""
    if model_name not in MODELS:
        raise ValueError(f"Model {model_name} not found. Available: {list(MODELS.keys())}")
    
    model_info = MODELS[model_name]
    
    # Download if not cached
    cache_path = download_model(model_info['url'])
    
    # Load
    brain = DigitalBrain(model_info['config'])
    brain.load_state_dict(torch.load(cache_path))
    
    return brain

# Usage
brain = load_pretrained('mbm-medium')
# Fine-tune or evaluate directly
```

**README Example:**

```markdown
## Quick Start

```python
from mbm import load_pretrained

# Load pre-trained model
brain = load_pretrained('mbm-medium')

# Evaluate on our task
env = YourEnvironment()
obs = env.reset()

for step in range(1000):
    action = brain.act(obs)
    obs, reward, done, info = env.step(action)
```

Boom. 3 lines to get started.
```

---

## Phase 3: Community & Impact (Months 6-12)

### Priority 8: **Real-World Application Demo**

**Why This Matters:**
"It works on gridworld" → Toy. "It controls a robot" → Real.

**Options (Ranked by Difficulty):**

#### Option A: **Sim-to-Real Robot (Hardest, Highest Impact)**

```python
# 1. Train in simulation (PyBullet or MuJoCo)
env = gym.make('AntBulletEnv-v0')  # Quadruped robot
brain = train_mbm(env, n_steps=10M)

# 2. Transfer to real robot
real_robot = FrankaArm()  # Or any real hardware
# MBM's continual learning should help with sim-to-real gap
brain.adapt_online(real_robot)
```

**Challenges:**
- Need physical hardware ($5K-$50K)
- Requires robotics expertise
- Sim-to-real gap is hard

**Timeline:** 6-12 months

#### Option B: **Autonomous Agent in Complex Simulation (Medium, Good Impact)**

```python
# Minecraft, NetHack, or Habitat
env = habitat.Env(config)  # Photorealistic indoor navigation
brain = train_mbm(env)

# Showcase:
# - Long-horizon exploration (1M+ steps)
# - Continual learning (multiple objectives)
# - Memory utilization (remember visited locations)
```

**Benefits:**
- No hardware needed
- Well-cited benchmarks
- Visually impressive (good for demos)

**Timeline:** 3-6 months

#### Option C: **Edge Device Deployment (Easiest, Practical Impact)**

```python
# Deploy MBM on Raspberry Pi or Jetson Nano
# Show: "This runs on $50 hardware"

# 1. Quantize model
brain_quantized = torch.quantization.quantize_dynamic(
    brain, {nn.Linear}, dtype=torch.qint8
)

# 2. Export to TorchScript
scripted = torch.jit.script(brain_quantized)
scripted.save("mbm_edge.pt")

# 3. Deploy
# On Raspberry Pi:
brain = torch.jit.load("mbm_edge.pt")
# Run at 10 Hz on 1W power consumption
```

**Demo:**
- Small robot (wheeled or drone)
- Onboard decision-making
- "Lifelong learning on the edge"

**Timeline:** 1-2 months

**RECOMMENDATION:** Start with **Option C** (edge deployment). Easy win, practical value.

---

### Priority 9: **Neuromorphic Hardware Validation**

**Why This Matters:**
This is our **unique selling point**. Transformers can't do this.

**How To Get Access:**

#### Intel Loihi (Recommended)

```bash
# 1. Apply for INRC (Intel Neuromorphic Research Community)
# https://www.intel.com/content/www/us/en/research/neuromorphic-community.html

# 2. Proposal (1-2 pages):
Title: "Biologically-Plausible Continual Learning on Neuromorphic Hardware"

Motivation:
- MBM uses local plasticity (STDP-compatible)
- Sparse connectivity (native to neuromorphic)
- Event-driven (matches Loihi's paradigm)

Goals:
- Port MBM to Loihi 2
- Benchmark energy efficiency
- Demonstrate online learning

Expected Outcome:
- 1000× energy reduction vs GPU
- Proof-of-concept for neuromorphic RL
```

**Timeline:**
- Application: 1 week
- Review: 1-2 months
- If accepted: 6-12 months of hardware access

#### SpiNNaker (Alternative)

- European neuromorphic board
- More open, easier to get access
- Apply through University of Manchester

**What You'll Prove:**

```python
# Energy comparison
gpu_energy = 50W × 1 hour = 50 Wh
loihi_energy = 0.05W × 1 hour = 0.05 Wh

# 1000× reduction!
```

**Publication:** This becomes a **second paper** on neuromorphic deployment.

---

### Priority 10: **Multi-Agent Extension**

**Why This Matters:**
Shows architecture generalizes beyond single-agent RL.

```python
class MultiAgentMBM:
    """
    Extension: Multiple MBM agents with shared hippocampus.
    
    Applications:
    - Multi-agent coordination
    - Social learning (agents share episodic memories)
    - Emergent communication
    """
    def __init__(self, n_agents, config):
        self.agents = [DigitalBrain(config) for _ in range(n_agents)]
        
        # SHARED hippocampus (social memory)
        self.shared_hippocampus = Hippocampus(
            d_z=config['d_z'],
            capacity=10000  # Larger for multi-agent
        )
        
    def step(self, observations, rewards, dones):
        actions = []
        
        for i, (agent, obs, reward, done) in enumerate(
            zip(self.agents, observations, rewards, dones)
        ):
            # Agent acts
            action, log_prob, value, state, log, entropy = agent.step(
                obs, reward, done
            )
            
            # Store in SHARED memory (social learning)
            if log.novelty > 0.6:
                self.shared_hippocampus.encode(state.z)
            
            # Retrieve from SHARED memory
            context = self.shared_hippocampus.retrieve(state.z)
            # Agent can learn from others' experiences!
            
            actions.append(action)
        
        return actions
```

**Demo Tasks:**
- Multi-agent gridworld (cooperation)
- Predator-prey
- Communication emergence

**Timeline:** 2-3 months

---

## Summary: 12-Month Roadmap to Respectability

### Months 1-3: **Scientific Validation**
- ✅ Beat EWC/PackNet on continual learning
- ✅ Pass MiniGrid benchmark
- ✅ Hierarchical cortex (3 layers)
- ✅ Scale to 65K neurons
- **Deliverable:** Strong conference paper

### Months 4-6: **Engineering Maturity**
- ✅ Production codebase (pip install)
- ✅ W&B integration
- ✅ Pre-trained model zoo
- ✅ Comprehensive docs
- **Deliverable:** Usable open-source project

### Months 7-9: **Real-World Application**
- ✅ Edge device deployment OR
- ✅ Complex simulation (Habitat/Minecraft) OR
- ✅ Sim-to-real robot
- **Deliverable:** Demo that impresses non-academics

### Months 10-12: **Unique Contributions**
- ✅ Neuromorphic hardware port
- ✅ Multi-agent extension
- ✅ 100K+ neuron scaling
- **Deliverable:** Follow-up paper(s)

---

## Critical Success Metrics

At the end of 12 months, you should have:

**Publications:**
- [ ] 1 conference paper (ICLR/NeurIPS/CoRL)
- [ ] 1-2 workshop papers
- [ ] 1 neuromorphic hardware paper (if Loihi access granted)

**Open Source:**
- [ ] 500+ GitHub stars
- [ ] 10+ contributors
- [ ] 3+ downstream projects using MBM

**Technical:**
- [ ] Beats 2+ continual learning baselines
- [ ] Works on 2+ standard benchmarks
- [ ] Scales to 100K+ neurons
- [ ] Deployed on real hardware (edge or neuromorphic)

**Community:**
- [ ] 3+ blog posts
- [ ] Talk at 1+ conference/meetup
- [ ] Cited by 5+ other papers

---

## My Top 3 Recommendations (Start Here)

### 1. **Implement EWC Baseline** (1 week)

```bash
# This is the FASTEST way to strengthen our paper
git checkout -b ewc-baseline
# Implement EWC comparison
# Show MBM beats it
# Instant credibility boost
```

### 2. **MiniGrid Benchmark** (1-2 weeks)

```bash
# Prove it works on something standard
pip install minigrid
# Test on DoorKey, MultiRoom, LavaGap
# Report results in paper
```

### 3. **Hierarchical Cortex** (2 weeks)

```bash
# Address "single layer" weakness
# Show deeper = better
# Demonstrates scaling path
```

**If you do ONLY these 3 things, our paper becomes much stronger.**

---

## Final Thought

The difference between "toy" and "respectable" isn't size—it's **validation**.

**Toy:**
- Works on our own environment
- Beats trivial baselines
- No external validation

**Respectable:**
- Works on standard benchmarks
- Beats specialized methods
- Others can reproduce and build on it

We're 70% of the way there. The next 3 months are critical.

**What should you start with?** Pick ONE:

**Option A:** Focus on science (EWC + MiniGrid) → Strong paper faster

**Option B:** Focus on engineering (production code) → More adoption, slower paper

**My recommendation:** We start with **Option A**. Get the paper accepted first. Then polish the codebase. A published paper gives us credibility to attract contributors.

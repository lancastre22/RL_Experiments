# Reinforcement Learning Experiments

A comprehensive implementation of various reinforcement learning algorithms tested on OpenAI Gymnasium environments. This project includes implementations of REINFORCE, PPO, and SAC algorithms tested across multiple environments like LunarLander, BipedalWalker, CartPole, and more.

## Project Structure
```
├── Algorithms
│   ├── REINFORCE Implementations
│   │   ├── REINFORCE_cartpole.py
│   │   ├── REINFORCE_frozenlake.py
│   │   ├── REINFORCE_lunarlander_continuous.py
│   │   ├── REINFORCE_lunarlander_discrete.py
│   │   └── REINFORCE_pendulum.py
│   ├── PPO Implementations
│   │   ├── PPO_lunarlander.py
│   │   ├── PPO_mountaincar.py
│   │   ├── PPO_pendulum.py
│   │   └── PPO_walker.py
│   └── Modern Approaches
│       └── SOA.py
├── Utils
│   └── ModelsAndUtils.py
└── Test Scripts
    ├── PPO_lunarlander_test.py
    ├── PPO_walker_test.py
    └── REINFORCE_cartpole_test.py
```

## Features

- Multiple RL algorithm implementations:
  - REINFORCE with baseline
  - Proximal Policy Optimization (PPO)
  - Soft Actor-Critic (SAC)
  - Twin Delayed DDPG (TD3)

- Support for various Gymnasium environments:
  - BipedalWalker (normal and hardcore modes)
  - LunarLander (continuous and discrete)
  - CartPole
  - Pendulum
  - FrozenLake
  - MountainCar

## Requirements

```bash
torch>=1.13.0
gymnasium
numpy
scipy
matplotlib
```

## Installation

```bash
git clone https://github.com/lancastre22/RL_Experiments.git
cd RL_Experiments
pip install -r requirements.txt
```

## Usage

Train a model:
```bash
python REINFORCE_cartpole.py  # Train CartPole with REINFORCE
python PPO_lunarlander.py     # Train LunarLander with PPO
```

Test a trained model:
```bash
python testPPOlunarlander.py 
```

## Implemented Features

### Policy Networks
- Discrete action spaces (Categorical distributions)
- Continuous action spaces (Normal distributions)
- State-dependent standard deviations
- Action clipping and scaling

### Value Functions
- State value estimation
- Advantage estimation
- Generalized Advantage Estimation (GAE)

### Training Utilities
- Experience replay buffers
- PPO clipping
- Early stopping with KL divergence
- Advantage normalization

## Future Work

- [ ] Implement more SOTA algorithms
- [ ] Add parallel training support
- [ ] Implement prioritized experience replay
- [ ] Add more environment wrappers
- [ ] Improve documentation and examples

## License

MIT License
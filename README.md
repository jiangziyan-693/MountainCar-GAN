# MountainCar Reinforcement Learning

This project implements reinforcement learning algorithms to solve the MountainCar environment using Advantage Actor-Critic (A2C) and Proximal Policy Optimization (PPO) agents.\
>Warning:the `gym` and `ale-py` must use the specified version in `requirements.txt`

## Requirements

To install the required packages, run:
```bash
pip install -r requirements.txt
```

## Configuration

The `config.yaml` file contains the configuration for the agent and the number of episodes:
```yaml
agent: PPO # PPO, AC
episodes: 100 # Number of episodes
```

## Usage

To run the training and testing process, execute the `main.py` script:
```bash
python main.py
```

## Files

- `config.yaml`: Configuration file for the agent and number of episodes.
- `requirements.txt`: List of required Python packages.
- `main.py`: Main script to train and test the reinforcement learning agents.

## Agent Details

### Advantage Actor-Critic (A2C)

The A2C agent uses a neural network to estimate both the policy (actor) and the value function (critic). The actor network outputs a probability distribution over actions, while the critic network estimates the value of the current state.

### Proximal Policy Optimization (PPO)

The PPO agent improves upon A2C by using a clipped objective function to ensure that updates to the policy do not deviate too much from the previous policy. This helps in maintaining stability during training.

## Training and Testing

The training process involves running episodes in the environment and updating the agent's policy based on the rewards received. After training, the agent is tested to evaluate its performance.

Training rewards are plotted over episodes to visualize the learning progress.



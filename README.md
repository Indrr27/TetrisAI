# Tetris AI Project

## Overview

This project is dedicated to developing an AI agent that learns to play Tetris, a classic tile-matching puzzle game. By leveraging the Stable Baselines3 library with a Proximal Policy Optimization (PPO) model, the agent is trained to increase its score by clearing lines with efficient block placements. The project uses a custom Tetris environment adapted for reinforcement learning.

## Getting Started

### Prerequisites

- Python 3.8
- PyTorch
- Stable Baselines3
- gym-tetris
- NES-Py (for the Tetris environment)

### Installation

1. Clone the repository:

```bash
git clone https://github.com/Indrr27/TetrisAI.git
cd TetrisAI
```

2. Install the necessary packages:

```bash
pip install -r requirements.txt
```

Your `requirements.txt` should list the following dependencies:
```
torch==1.8.1
stable-baselines3[extra]==1.1.0
gym-tetris==0.2.6
nes-py
```

### File Descriptions

- `environment_setup.py`: Scripts for setting up and customizing the Tetris environment for both training and evaluation phases.
- `callbacks.py`: Implements a custom callback class for saving the model periodically during the training process.
- `main.py`: Contains the main logic for configuring, training, and evaluating the Tetris AI model.

## Training the AI

To train the AI agent with the provided settings, execute:

```bash
python main.py
```

This command kicks off the training process based on the configurations specified in `main.py`, such as the environment setup and the model parameters. The model will undergo learning through 1,000,000 timesteps, with checkpoints saved every 10,000 steps.

## Evaluating the Model

Once training is complete, you can assess the model's performance by running the latter portion of `main.py`. This part of the script loads the trained model to play Tetris, allowing you to observe the agent's gameplay decisions in real time.

# Connect4-RL

Reinforcement Learning agent for Connect Four implemented in Python and PyTorch.

## Project Overview

This project builds a reinforcement learning agent to play Connect Four.  
The agent is trained via self-play and learns to evaluate board states and choose optimal moves.

The repository includes:

- Training notebook
- Inference / play notebook
- Model implementation
- Trained model checkpoint

## Repository Structure

- `train_sample.ipynb` — Training process demonstration
- `play.ipynb` — Load trained model and play
- `agent.py` (or N0.py) — Neural network / agent implementation
- `assets/` — Game environment and UI components
- `trained_network.pth` — Trained model weights

## How to Run

1. Install dependencies:

```bash
pip install -r requirements.txt

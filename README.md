# Q-Learning for Grid World

This repository contains a Q-learning implementation for an agent to navigate through a grid world. The agent learns through exploration and updating its Q-values to find the shortest path to a terminal state.

## Overview

The environment is a grid of 11x11 cells, where each cell can have a specific reward associated with it. The agent's goal is to navigate from a starting position to the terminal state with the maximum possible reward. The agent uses Q-learning, a reinforcement learning algorithm, to update its policy and find the best path.

### Features:
- **Q-Learning Algorithm**: The agent updates its Q-values based on actions taken and rewards received.
- **Grid World**: An 11x11 grid with some obstacles and one terminal state.
- **Visualization**: Uses Pygame to display the environment and visualize the agent's exploration and learning process.
- **Shortest Path Calculation**: After training, the agent computes and displays the shortest path from a random start to the terminal state.

## Files

### 1. `q_learning.py`
This file implements the Q-learning algorithm and the grid world environment. It handles the following:
- Initializing the grid and rewards.
- Defining the Q-learning update rule.
- Generating the shortest path using the learned Q-values.

### 2. `q_learning_visualization.py`
This file extends the previous implementation by integrating Pygame to visualize the agent's training process and the shortest path.
- Displays the agent's movements and the learning process in real-time.
- Visualizes the grid, rewards, agent, and final path.

## Requirements

To run this project, you need to have the following installed:

- Python 3.x
- `numpy` for numerical computations
- `pygame` for visualization

You can install the required dependencies using:

```bash
pip install numpy pygame
```

## How to run
### Training phase:
- Run the 'qlearning.py' script to train the agent.
- The training process will update the Q-values based on the agent's actions and rewards.
### Visualization phase:
- Run the 'qlearning_visual.py' script to see the agent's movements and the learned path in real-time.
- The grid will be shown, with the agent's position being updated as it explores the environment and learns the best path.
## Code Structure
### Functions in 'qlearning.py':
- is_terminal_state(): Determines if the agent has reached the terminal state.
- get_starting_location(): Randomly generates a valid starting location for the agent.
- get_next_action(): Chooses the next action based on epsilon-greedy policy.
- get_next_location(): Calculates the next location based on the action taken.
- get_shortest_path(): Computes the shortest path from a given starting location.
### Functions in 'qlearning_visual.py':
- draw_grid(): Renders the grid environment using Pygame, highlighting the agent's current position.
- The main training loop visualizes the agent's movements and learning process.

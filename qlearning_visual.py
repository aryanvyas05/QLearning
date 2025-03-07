import numpy as np
import pygame
import time

env_rows = 11
env_columns = 11

q_values = np.zeros((env_rows,env_columns, 4))

actions = ['up', 'right', 'down', 'left']

rewards = np.full((env_rows, env_columns), -100)
rewards[0,5] = 100

aisles = {} 
aisles[1] = [i for i in range(1, 10)]
aisles[2] = [1, 7, 9]
aisles[3] = [i for i in range(1, 8)]
aisles[3].append(9)
aisles[4] = [3, 7]
aisles[5] = [i for i in range(11)]
aisles[6] = [5]
aisles[7] = [i for i in range(1, 10)]
aisles[8] = [3, 7]
aisles[9] = [i for i in range(11)]


for row_index in range(1, 10):
  for column_index in aisles[row_index]:
    rewards[row_index, column_index] = -1.
  

for row in rewards:
  print(row)

def is_terminal_state(current_row_index, current_column_index):
    if rewards[current_row_index, current_column_index] == -1:
      return False
    else:
      return True
def get_starting_location():
  current_row_index = np.random.randint(env_rows)
  current_column_index = np.random.randint(env_columns)
  while is_terminal_state(current_row_index, current_column_index):
    current_row_index = np.random.randint(env_rows)
    current_column_index = np.random.randint(env_columns)
  return current_row_index, current_column_index
  
def get_next_action(current_row_index, current_column_index, epsilon):
  if np.random.random() < epsilon:
    return np.argmax(q_values[current_row_index, current_column_index])
  else:
    return np.random.randint(4)

def get_next_location(current_row_index, current_column_index, action_index):
  new_row_index = current_row_index
  new_column_index = current_column_index
  if actions[action_index] == 'up' and current_row_index > 0:
    new_row_index -= 1
  elif actions[action_index] == 'right' and current_column_index < env_columns - 1:
    new_column_index += 1
  elif actions[action_index] == 'down' and current_row_index < env_rows - 1:
    new_row_index += 1
  elif actions[action_index] == 'left' and current_column_index > 0:
    new_column_index -= 1
  return new_row_index, new_column_index

def get_shortest_path(start_row_index, start_column_index):
  if is_terminal_state(start_row_index, start_column_index):
    return []
  else: 
    current_row_index, current_column_index = start_row_index, start_column_index
    shortest_path = []
    shortest_path.append([current_row_index, current_column_index])
    while not is_terminal_state(current_row_index, current_column_index):
      action_index = get_next_action(current_row_index, current_column_index, 1.)
      current_row_index, current_column_index = get_next_location(current_row_index, current_column_index, action_index)
      shortest_path.append([current_row_index, current_column_index])
    return shortest_path
  
epsilon = 0.9
discount_factor = 0.9
learning_rate = 0.9
episodes = 500


pygame.init()

CELL_SIZE = 50
WINDOW_SIZE = (env_columns * CELL_SIZE, env_rows * CELL_SIZE)
screen = pygame.display.set_mode(WINDOW_SIZE)
pygame.display.set_caption("Q-Learning Training")

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

def draw_grid(agent_position=None):
    screen.fill(WHITE)
    
    for row in range(env_rows):
        for col in range(env_columns):
            color = BLACK
            if rewards[row, col] == -1:
                color = WHITE
            elif rewards[row, col] == 100:
                color = GREEN
            
            pygame.draw.rect(screen, color, (col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE))
            pygame.draw.rect(screen, BLUE, (col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE), 1)

    if agent_position:
        pygame.draw.rect(screen, RED, (agent_position[1] * CELL_SIZE, agent_position[0] * CELL_SIZE, CELL_SIZE, CELL_SIZE))

    pygame.display.update()

for episode in range(episodes):
    row, col = get_starting_location()
    
    while not is_terminal_state(row, col):
        action_index = get_next_action(row, col, epsilon)
        old_row, old_col = row, col
        row, col = get_next_location(row, col, action_index)
        
        reward = rewards[row, col]
        old_q_value = q_values[old_row, old_col, action_index]
        temporal_difference = reward + (discount_factor * np.max(q_values[row, col])) - old_q_value
        q_values[old_row, old_col, action_index] = old_q_value + (learning_rate * temporal_difference)
        
        
        draw_grid((row, col))
        time.sleep(0.05)

print("Training complete!")



start_row, start_col = get_starting_location()
shortest_path = get_shortest_path(start_row, start_col)

for row, col in shortest_path:
    draw_grid((row, col))
    time.sleep(0.2)

print("Shortest path visualization complete!")
pygame.quit()


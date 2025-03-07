import numpy as np

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

for episode in range(1000):
  row_index, column_index = get_starting_location()

  while not is_terminal_state(row_index, column_index):
    action_index = get_next_action(row_index, column_index, epsilon)

    old_row_index, old_column_index = row_index, column_index
    row_index, column_index = get_next_location(row_index, column_index, action_index)
    
    reward = rewards[row_index, column_index]
    old_q_value = q_values[old_row_index, old_column_index, action_index]
    temporal_difference = reward + (discount_factor * np.max(q_values[row_index, column_index])) - old_q_value

    new_q_value = old_q_value + (learning_rate * temporal_difference)
    q_values[old_row_index, old_column_index, action_index] = new_q_value

print('Training complete!')

def is_valid_start(row, col):
    if row < 1 or row > 10 or col < 0 or col > 10:
        print("Invalid position! Enter row 1-10 and column 0-10.")
        return False
    if is_terminal_state(row, col):
        print("You cannot start at a terminal position! Pick another spot.")
        return False
    return True

while True:
    try:
        row = int(input("Enter row value (1-10): "))
        col = int(input("Enter column value (0-10): "))
        if is_valid_start(row, col):
            break
    except ValueError:
        print("Invalid input! Please enter numbers only.")

shortest_path = get_shortest_path(row, col)
final = np.full((11, 11), 0, dtype=str)

for r, c in shortest_path:
    final[r, c] = '1'
final[0, 5] = 'X'
for row in final:
    print(" ".join(row))


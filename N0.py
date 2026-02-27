import random
import torch
import torch.nn as nn

# Define the Q-network
class DQN(nn.Module):
    def __init__(self, input_shape, output_dim):
        super(DQN, self).__init__()
        
        # Store input shape as an instance variable
        self.input_shape = input_shape
        
        # Convolutional layers to process the board
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        
        # Fully connected layers
        conv_output_size = 128 * input_shape[0] * input_shape[1]  # Flattened size after conv layers
        self.fc1 = nn.Linear(conv_output_size, 256)
        self.fc2 = nn.Linear(256, output_dim)
        
    def forward(self, x):
        # Reshape input to (batch_size, 1, rows, cols) for CNN
        x = x.view(-1, 1, *self.input_shape)  # Using self.input_shape to access rows and cols
        
        # Pass through convolutional layers
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        
        # Flatten and pass through fully connected layers
        x = x.view(x.size(0), -1)  # Flatten the convolutional output
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


# Define global variables for model and device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = (6, 7)  # For a 6x7 Connect Four board
output_dim = 7     # Number of actions (columns)

# Load the trained DQN model
model_path = "trained_network.pth"
dqn_model = DQN(input_dim, output_dim).to(device)
dqn_model.load_state_dict(torch.load(model_path, map_location=device))
dqn_model.eval()  # Set to evaluation mode

def policy1(observation):
    observation = convert_state(observation.copy())
    
    # Flatten the observation and convert to tensor
    state = torch.FloatTensor(observation).unsqueeze(0).unsqueeze(0).to(device)  # Convert to tensor and add batch dimension

    # Get Q-values for all actions
    with torch.no_grad():
        q_values = dqn_model(state)

    # Identify available actions (columns that are not full)
    available_actions = [col for col in range(q_values.shape[1]) if observation[0][col] == 0]
    
    # If all columns are full, return a random action (edge case)
    if not available_actions:
        return random.choice(range(q_values.shape[1]))

    # Choose the action with the highest Q-value among available actions
    best_action = max(available_actions, key=lambda col: q_values[0, col].item())
    
    return best_action

def policy2(observation):
    # Constants
    OPPONENT = -1
    EMPTY = 0
    PLAYER = 1
    COLS = 7
    ROWS = 6
    CENTER_COL = COLS // 2
    MAX_SCORE = 100
    BLOCK_SCORE = 90

    def is_valid_move(board, col):
        """Check if the column is not full."""
        return board[0][col] == EMPTY

    def get_next_open_row(board, col):
        """Return the next available row for the given column."""
        for row in range(ROWS-1, -1, -1):
            if board[row][col] == EMPTY:
                return row
        return -1  # If column is full

    def is_winning_move(board, piece, row, col):
        """Check if placing a piece at (row, col) results in a win."""
        
        def count_consecutive(delta_row, delta_col):
            """Count consecutive pieces of the same type starting from (row, col)."""
            count = 1  # Include the current position
            # Check in the positive direction
            r, c = row + delta_row, col + delta_col
            while 0 <= r < ROWS and 0 <= c < COLS and board[r][c] == piece:
                count += 1
                r += delta_row
                c += delta_col
            # Check in the negative direction
            r, c = row - delta_row, col - delta_col
            while 0 <= r < ROWS and 0 <= c < COLS and board[r][c] == piece:
                count += 1
                r -= delta_row
                c -= delta_col
            return count

        # Directions to check: right, down, diagonals
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dr, dc in directions:
            if count_consecutive(dr, dc) >= 4:
                return True

        return False


    def get_valid_moves(board):
        """Return all valid columns where a piece can be dropped."""
        return [col for col in range(COLS) if is_valid_move(board, col)]

    def evaluate_move(board, col):
        """Evaluate the potential of a move for a given piece."""
        if not is_valid_move(board, col):
            return -MAX_SCORE  # Invalid move (full column)

        row = get_next_open_row(board, col)

        # Assign scores based on outcomes
        if is_winning_move(board, PLAYER, row, col):
            return BLOCK_SCORE  
        elif is_winning_move(board, OPPONENT, row, col):
            return MAX_SCORE  
        elif col == CENTER_COL:
            return 1  # Preference for center
        elif col == CENTER_COL - 1 or col == CENTER_COL + 1:
            return 0.5  # Slight preference for adjacent to center
        else:
            return 0  # Neutral for other moves

    def best_move(board):
        """Find the move with the highest score, breaking ties randomly."""
        valid_moves = get_valid_moves(board)
        move_scores = [(col, evaluate_move(board, col)) for col in valid_moves]

        # Find the maximum score
        max_score = max(move_scores, key=lambda x: x[1])[1]

        # Collect all moves with the highest score
        best_moves = [col for col, score in move_scores if score == max_score]

        # Randomly select one of the best moves
        return random.choice(best_moves)

    observation = convert_state(observation.copy())

    # Find the best move based on the scoring system
    valid_moves = get_valid_moves(observation)
    if not valid_moves:
        print("No valid moves available.\n")
        return 0  # Fallback to column 0 if no moves are available
    return best_move(observation)



# Convert 2 to -1 for better learning
def convert_state(state):
    state[state == 2] = -1
    return state

def flip_board(state):
    """Flip the board so that player 2 is treated as player 1 and vice versa."""
    state = state.copy()
    return state * -1
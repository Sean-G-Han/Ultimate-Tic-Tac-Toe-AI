# Run the following cell to import utilities

import numpy as np
import time

from utils import State, Action, load_data

data = load_data()
assert len(data) == 80000
for state, value in data[:3]:
    print(state)
    print(f"Value = {value}\n\n")

class StudentAgent:
    def __init__(self):
        """Instantiates your agent.
        """
    
    def choose_action(self, state: State) -> Action:
        _, best_action = self.maxValue(state, 0, 4, -np.inf, np.inf)
        return best_action
    
    def evaluate_state(self, state: State) -> float:
        if state.is_terminal():
            return state.terminal_utility() * np.inf
        #Corners = 2 points, Edges = 1 point, Center = 3 points
        local_tic_tac_toe_score = self.get_tic_tac_toe(state.local_board_status)
        sub_tic_tac_toe_score = self.sub_tic_tac_toe(state.board)
        return local_tic_tac_toe_score * 10  + sub_tic_tac_toe_score
    
    
    def sub_tic_tac_toe(self, arr: np.ndarray) -> int:
        sub_boards_score = 0
        multiplier = 1
        for i in range(3):
            for j in range(3):
                if (i == 0 or i == 2) and (j == 0 or j == 2):
                    multiplier = 2
                elif i == 1 and j == 1:
                    multiplier = 3
                sub_boards_score += self.get_tic_tac_toe(arr[i][j]) * multiplier
        return sub_boards_score
        

    def get_tic_tac_toe(self, board: np.ndarray) -> int:
        score = 0
        def score_line(line):
            count_player = np.sum(line == 1)
            count_opponent = np.sum(line == 2)
            count_empty = np.sum(line == 0)
            
            if count_player == 3:
                return 10
            elif count_opponent == 3:
                return -10
            elif count_player == 2 and count_empty == 1:
                return 5
            elif count_opponent == 2 and count_empty == 1:
                return -5
            elif count_player == 1 and count_empty == 2:
                return 1
            elif count_opponent == 1 and count_empty == 2:
                return -1
            return 0
        
        def score_pos(board: np.ndarray) -> int:
            score = 0
            if board[0, 0] == 1:
                score += 1
            elif board[0, 0] == 2:
                score -= 1
            if board[0, 2] == 1:
                score += 1
            elif board[0, 2] == 2:
                score -= 1
            if board[2, 0] == 1:
                score += 1
            elif board[2, 0] == 2:
                score -= 1
            if board[2, 2] == 1:
                score += 1
            elif board[2, 2] == 2:
                score -= 1
            if board[1, 1] == 1:
                score += 2
            elif board[1, 1] == 2:
                score -= 2
            return score

        for i in range(3):
            score += score_line(board[i, :])
            score += score_line(board[:, i])
        
        # Evaluate diagonals
        score += score_line(board.diagonal())
        score += score_line(np.fliplr(board).diagonal())
        score += score_pos(board)
        
        return score

    def maxValue(self, state: State, depth: int, max_depth: int, alpha, beta):
        if depth == max_depth or state.is_terminal():
            return self.evaluate_state(state), None
        value = -np.inf
        optimalMove = None
        for move in state.get_all_valid_actions():
            newState = state.change_state(move)
            temp, _ = self.minValue(newState, depth + 1, max_depth, alpha, beta)
            if temp > value:
                optimalMove = move
                value = temp
            alpha = max(alpha, value)
            if value >= beta:
                return value, optimalMove
        return value, optimalMove
    
    def minValue(self, state: State, depth: int, max_depth: int, alpha, beta):
        if depth == max_depth or state.is_terminal():
            return self.evaluate_state(state), None
        value = np.inf
        optimalMove = None
        for move in state.get_all_valid_actions():
            newState = state.change_state(move)
            temp,  _ = self.maxValue(newState, depth + 1, max_depth, alpha, beta)
            if temp < value:
                value = temp
                optimalMove = move
            beta = min(beta, value)
            if value <= alpha:
                return value, optimalMove
        return value, optimalMove
    
# Use this cell to test your agent in two full games against a random agent.
# The random agent will choose actions randomly among the valid actions.

class RandomStudentAgent(StudentAgent):
    def choose_action(self, state: State) -> Action:
        # If you're using an existing Player 1 agent, you may need to invert the state
        # to have it play as Player 2. Uncomment the next line to invert the state.
        # state = state.invert()

        # Choose a random valid action from the current game state
        return state.get_random_valid_action()

class UserAgent(StudentAgent):
    def __init__(self):
        """Instantiates your agent.
        """
    
    def choose_action(self, state: State) -> Action:
        valid_actions = state.get_all_valid_actions()
        print("\nAvailable moves:", valid_actions)
        
        while True:
            try:
                row = int(input("Enter row: "))
                col = int(input("Enter column: "))
                sub_row = int(input("Enter sub-row: "))
                sub_col = int(input("Enter sub-column: "))
                
                # You may need to adjust this parsing depending on how actions are represented
                user_move = Action([row, col, sub_row, sub_col])
                
                if user_move in valid_actions:
                    print(f"Chose action: {user_move}")
                    return user_move
                else:
                    print("Invalid move. Please choose a move from the available moves.")
            except Exception as e:
                print(f"Invalid input. Error: {e}. Please try again.")

class OpponentAgent(StudentAgent):
    def __init__(self):
        """Instantiates your agent.
        """
    
    def choose_action(self, state: State) -> Action:
        state = state.invert()
        _, best_action = self.maxValue(state, 0, 3, -np.inf, np.inf)
        return best_action
    
    def evaluate_state(self, state: State) -> float:
        if state.is_terminal():
            return state.terminal_utility() * np.inf
        #Corners = 2 points, Edges = 1 point, Center = 3 points
        local_tic_tac_toe_score = self.get_tic_tac_toe(state.local_board_status)
        sub_tic_tac_toe_score = self.sub_tic_tac_toe(state.board)
        return local_tic_tac_toe_score * 10  + sub_tic_tac_toe_score
    
    
    def sub_tic_tac_toe(self, arr: np.ndarray) -> int:
        sub_boards_score = 0
        multiplier = 1
        for i in range(3):
            for j in range(3):
                if (i == 0 or i == 2) and (j == 0 or j == 2):
                    multiplier = 2
                elif i == 1 and j == 1:
                    multiplier = 3
                sub_boards_score += self.get_tic_tac_toe(arr[i][j]) * multiplier
        return sub_boards_score
        

    def get_tic_tac_toe(self, board: np.ndarray) -> int:
        score = 0
        def score_line(line):
            count_player = np.sum(line == 1)
            count_opponent = np.sum(line == 2)
            count_empty = np.sum(line == 0)
            
            if count_player == 3:
                return 10
            elif count_opponent == 3:
                return -10
            elif count_player == 2 and count_empty == 1:
                return 5
            elif count_opponent == 2 and count_empty == 1:
                return -5
            elif count_player == 1 and count_empty == 2:
                return 1
            elif count_opponent == 1 and count_empty == 2:
                return -1
            return 0
        
        def score_pos(board: np.ndarray) -> int:
            score = 0
            if board[0, 0] == 1:
                score += 1
            elif board[0, 0] == 2:
                score -= 1
            if board[0, 2] == 1:
                score += 1
            elif board[0, 2] == 2:
                score -= 1
            if board[2, 0] == 1:
                score += 1
            elif board[2, 0] == 2:
                score -= 1
            if board[2, 2] == 1:
                score += 1
            elif board[2, 2] == 2:
                score -= 1
            if board[1, 1] == 1:
                score += 2
            elif board[1, 1] == 2:
                score -= 2
            return score

        for i in range(3):
            score += score_line(board[i, :])
            score += score_line(board[:, i])
        
        # Evaluate diagonals
        score += score_line(board.diagonal())
        score += score_line(np.fliplr(board).diagonal())
        score += score_pos(board)
        
        return score

    def maxValue(self, state: State, depth: int, max_depth: int, alpha, beta):
        if depth == max_depth or state.is_terminal():
            return self.evaluate_state(state), None
        value = -np.inf
        optimalMove = None
        for move in state.get_all_valid_actions():
            newState = state.change_state(move)
            temp, _ = self.minValue(newState, depth + 1, max_depth, alpha, beta)
            if temp > value:
                optimalMove = move
                value = temp
            alpha = max(alpha, value)
            if value >= beta:
                return value, optimalMove
        return value, optimalMove
    
    def minValue(self, state: State, depth: int, max_depth: int, alpha, beta):
        if depth == max_depth or state.is_terminal():
            return self.evaluate_state(state), None
        value = np.inf
        optimalMove = None
        for move in state.get_all_valid_actions():
            newState = state.change_state(move)
            temp,  _ = self.maxValue(newState, depth + 1, max_depth, alpha, beta)
            if temp < value:
                value = temp
                optimalMove = move
            beta = min(beta, value)
            if value <= alpha:
                return value, optimalMove
        return value, optimalMove

def run(your_agent: StudentAgent, opponent_agent: StudentAgent, start_num: int):
    your_agent_stats = {"timeout_count": 0, "invalid_count": 0}
    opponent_agent_stats = {"timeout_count": 0, "invalid_count": 0}
    turn_count = 0
    
    state = State(fill_num=start_num)
    
    while not state.is_terminal():
        turn_count += 1

        agent_name = "your_agent" if state.fill_num == 1 else "opponent_agent"
        agent = your_agent if state.fill_num == 1 else opponent_agent
        stats = your_agent_stats if state.fill_num == 1 else opponent_agent_stats

        start_time = time.time()
        action = agent.choose_action(state.clone())
        end_time = time.time()
        
        random_action = state.get_random_valid_action()
        if end_time - start_time > 3:
            print(f"{agent_name} timed out!")
            stats["timeout_count"] += 1
            action = random_action
        if not state.is_valid_action(action):
            print(f"{agent_name} made an invalid action!")
            stats["invalid_count"] += 1
            action = random_action
                
        state = state.change_state(action)
        #print(f"{agent_name} chose action: {action}")
        #print(state)

    print(f"== {your_agent.__class__.__name__} (1) vs {opponent_agent.__class__.__name__} (2) - First Player: {start_num} ==")
        
    if state.terminal_utility() == 1:
        print("You win!")
    elif state.terminal_utility() == 0:
        print("You lose!")
    else:
        print("Draw")

    for agent_name, stats in [("your_agent", your_agent_stats), ("opponent_agent", opponent_agent_stats)]:
        print(f"{agent_name} statistics:")
        print(f"Timeout count: {stats['timeout_count']}")
        print(f"Invalid count: {stats['invalid_count']}")
        
    print(f"Turn count: {turn_count}\n")

your_agent = lambda: StudentAgent()
opponent_agent = lambda: OpponentAgent()

run(your_agent(), opponent_agent(), 1)
run(your_agent(), opponent_agent(), 2)
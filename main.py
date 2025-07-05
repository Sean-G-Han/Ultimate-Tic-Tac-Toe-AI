import time
from typing import Callable
from utils import State, Action
import numpy as np


class StudentAgent:
    def __init__(self):
        """Instantiates your agent."""
        self.multiplier_matrix = np.array([
            [1.4, 1.0, 1.4],
            [1.0, 1.5, 1.0],
            [1.4, 1.0, 1.4]
        ])

        self.multiplier_matrix2 = np.array([
            [1.1, 1.0, 1.1],
            [1.0, 1.2, 1.0],
            [1.1, 1.0, 1.1]
        ])
    
    def choose_action(self, state: State) -> Action:
        _, best_action = self.maxValue(state, 0, 3, -np.inf, np.inf)
        return best_action
    
    def evaluate_state(self, state: State) -> float:
        if state.is_terminal():
            return state.terminal_utility() * np.inf
        local_tic_tac_toe_score = self.get_tic_tac_toe(state.local_board_status)
        sub_tic_tac_toe_score = self.sub_tic_tac_toe(state.board)
        sub_tic_tac_toe_score_line = self.sub_tic_tac_toe_line(state.board)
        return local_tic_tac_toe_score * 7 + sub_tic_tac_toe_score * 1.1 + sub_tic_tac_toe_score_line * 0.5 + self.minimize_actions(state) * 0.5
    
    def minimize_actions(self, state: State) -> int:
        x = len(state.get_all_valid_actions())
        if  x < 10:
            return 9 - x
        else:
            return -18
        
    
    def sub_tic_tac_toe_line(self, arr: np.ndarray) -> int:
        # "Rounds up" the local board then evaluates it
        # So given a sub-board, if 1 is winning we just assume they won that square
        new_board = np.zeros((3, 3), dtype=int)
        for i in range(3):
            for j in range(3):
                temp = self.get_tic_tac_toe(arr[i, j])
                if temp > 0:
                    new_board[i, j] = 1
                elif temp < 0:
                    new_board[i, j] = 2
        return self.get_tic_tac_toe(new_board)
    
    def sub_tic_tac_toe(self, arr: np.ndarray) -> int:
        sub_boards_score = np.sum(self.get_tic_tac_toe(arr[i, j]) * self.multiplier_matrix2[i, j] 
                                  for i in range(3) for j in range(3))
        return sub_boards_score
        

    def get_tic_tac_toe(self, board: np.ndarray) -> int:
        pieces_diff = np.sum(board == 1) - np.sum(board == 2)
        two_in_a_row = 0
        winner = 0
        score = 0

        def score_line(line, winner, two_in_a_row):
            if winner != 0:
                return winner, two_in_a_row
            count_player = np.sum(line == 1)
            count_opponent = np.sum(line == 2)
            count_empty = np.sum(line == 0)

            if count_player == 3:
                winner = 1
            elif count_opponent == 3:
                winner = 2
            elif count_player == 2 and count_empty == 1:
                two_in_a_row += 1
            elif count_opponent == 2 and count_empty == 1:
                two_in_a_row -= 1

            return winner, two_in_a_row

        def score_pos(board: np.ndarray) -> int:
            pos_weights = self.multiplier_matrix
            return np.sum((board == 1) * pos_weights) - np.sum((board == 2) * pos_weights)

        for i in range(3):
            winner, two_in_a_row = score_line(board[i, :], winner, two_in_a_row)  # Rows
            winner, two_in_a_row = score_line(board[:, i], winner, two_in_a_row)  # Columns

        # Evaluate diagonals
        winner, two_in_a_row = score_line(board.diagonal(), winner, two_in_a_row)
        winner, two_in_a_row = score_line(np.fliplr(board).diagonal(), winner, two_in_a_row)
        if winner == 1:
            score = 12
        elif winner == 2:
            score = -12
        elif two_in_a_row > 1:
            score = 8
        elif two_in_a_row < -1:
            score = -8
        elif two_in_a_row == 1:
            score = 6
        elif two_in_a_row == -1:
            score = -6
        else:
            score = pieces_diff # magnitude max = 4
        
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

def run(your_agent, opponent_agent, start_num: int):
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
        if end_time - start_time > 3000:
            print(f"{agent_name} timed out!")
            stats["timeout_count"] += 1
            action = random_action
        if not state.is_valid_action(action):
            print(f"{agent_name} made an invalid action!")
            stats["invalid_count"] += 1
            action = random_action
                
        state = state.change_state(action)
        print(f"{agent_name} chose action: {action}")
        print(state)

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
    return state.terminal_utility()

result_a = run(StudentAgent(), UserAgent(), 1)
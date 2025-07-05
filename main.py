# Run the following cell to import utilities


import time
from typing import Callable
from utils import State, Action
import numpy as np

import numpy as np
from typing import Callable
from utils import State, Action

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
        """Instantiates your agent."""
        self.multiplier_matrix = np.array([
            [1.55, 1.0, 1.55],
            [1.0, 1.8, 1.0],
            [1.55, 1.0, 1.55]
        ])

        self.multiplier_matrix2 = np.array([
            [1.1, 1.0, 1.1],
            [1.0, 1.4, 1.0],
            [1.1, 1.0, 1.1]
        ])
    
    def choose_action(self, state: State) -> Action:
        state = state.invert()
        _, best_action = self.maxValue(state, 0, 3, -np.inf, np.inf)
        return best_action
    
    def evaluate_state(self, state: State) -> float:
        if state.is_terminal():
            return state.terminal_utility() * np.inf
        local_tic_tac_toe_score = self.get_tic_tac_toe(state.local_board_status)
        sub_tic_tac_toe_score = self.sub_tic_tac_toe(state.board)
        sub_tic_tac_toe_score_line = self.sub_tic_tac_toe_line(state.board)
        return local_tic_tac_toe_score * 10 + sub_tic_tac_toe_score + sub_tic_tac_toe_score_line * 0.75 + self.minimize_actions(state) * 0.5
    
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
        if end_time - start_time > 3000:
            print(f"{agent_name} timed out!")
            stats["timeout_count"] += 1
            action = random_action
        if not state.is_valid_action(action):
            print(f"{agent_name} made an invalid action!")
            stats["invalid_count"] += 1
            action = random_action
                
        state = state.change_state(action)
        # print(f"{agent_name} chose action: {action}")
        # print(state)

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


class LearningAgent:
    
    def __init__(self):
        input = [0.3268932959477513, 0.21301576580756698, 0.3123438379035029, 0.2963360039398291, 1.6242569214194165, 2.8590876521368775, 1.381518990389354, 1.4918169364020524]
        self.local_mult = input[7]
        """Instantiates your agent."""

        self.increments = [input[4], input[5], input[6]]
        self.multiplier_matrix = np.array([
            [1.0+input[0], 1.0, 1.0+input[0]],
            [1.0, 1.0+input[0]+input[1], 1.0],
            [1.0+input[0], 1.0, 1.0+input[0]]
        ])

        self.multiplier_matrix2 = np.array([
            [1.0+input[2], 1.0, 1.0+input[2]],
            [1.0, 1.0+input[2]+input[3], 1.0],
            [1.0+input[2], 1.0, 1.0+input[2]]
        ])
    
    def choose_action(self, state: State) -> Action:
        _, best_action = self.maxValue(state, 0, 3, -np.inf, np.inf)
        return best_action
    
    def evaluate_state(self, state: State) -> float:
        if state.is_terminal():
            return state.terminal_utility() * np.inf
        local_tic_tac_toe_score = self.get_tic_tac_toe(state.local_board_status)
        sub_tic_tac_toe_score = self.sub_tic_tac_toe(state.board)
        return local_tic_tac_toe_score * 10 * self.local_mult + sub_tic_tac_toe_score
    
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
            score = 4 + self.increments[0] + self.increments[1] + self.increments[2]
        elif winner == 2:
            score = -4 - self.increments[0] - self.increments[1] - self.increments[2]
        elif two_in_a_row > 1:
            score = 4 + self.increments[0] + self.increments[1]
        elif two_in_a_row < -1:
            score = -4 - self.increments[0] - self.increments[1]
        elif two_in_a_row == 1:
            score = 4 + self.increments[0]
        elif two_in_a_row == -1:
            score = -4 - self.increments[0]
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
# your_agent = lambda: LearningAgent(1.1, 1.4, 1.1, 1.2, 2, 2, 4, 1)
opponent_agent = lambda: OpponentAgent()
#2.5
# opponent_agent2 = lambda: OpponentAgent(0.3268932959477513, 0.21301576580756698, 0.3123438379035029, 0.2963360039398291, 1.6242569214194165, 2.8590876521368775, 1.381518990389354, 1.4918169364020524)
#6.0
# opponent_agent3 = lambda: OpponentAgent(0.3714879318873679, 0.06402971402868746, 0.004885870695738814, 0.2881522822832295, 1.6270470469310314, 2.0052703986523603, 2.8637905596183337, 1.29811629407747)
#6.0
# opponent_agent4 = lambda: OpponentAgent(0.41037093763918897, 0.10947127924916267, 0.1300815655202221, 0.02334010340330578, 3, 1, 1, 2)
#5.5
# opponent_agent5 = lambda: OpponentAgent(0.430041839444747, 0.38267274969228104, 0.19320354908643012, 0.22733543824400615, 2.4794713100204335, 2.453526428768166, 1.8565918153533911, 0.9353264952104179)
#5.0
# result_a = run(your_agent(), opponent_agent(), 1)
# result_b = run(your_agent(), opponent_agent(), 2)

your_agent = lambda: StudentAgent()
result_a = run(your_agent(), opponent_agent(), 1)
result_b = run(your_agent(), opponent_agent(), 2)
# result_c = run(your_agent(), opponent_agent2(), 1)
# result_d = run(your_agent(), opponent_agent2(), 2)
# result_e = run(your_agent(), opponent_agent3(), 1)
# result_f = run(your_agent(), opponent_agent3(), 2)
# result_g = run(your_agent(), opponent_agent4(), 1)
# result_h = run(your_agent(), opponent_agent4(), 2)
# result_i = run(your_agent(), opponent_agent5(), 1)
# result_j = run(your_agent(), opponent_agent5(), 2)
# result = result_a + result_b + result_c + result_d + result_e + result_f + result_g + result_h + result_i + result_j
# print(f"Result: {result}")
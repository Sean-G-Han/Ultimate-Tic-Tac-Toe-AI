# Ultimate Tic Tac Toe AI (Python)
An intelligent Ultimate Tic Tac Toe game implemented in Python, featuring an AI opponent powered by the Minimax algorithm with optional heuristics and pruning for optimal performance.

## About the Game
Ultimate Tic Tac Toe is a meta version of the classic 3x3 Tic Tac Toe. Instead of playing on a single board, you play on nine Tic Tac Toe boards arranged in a 3x3 grid. The twist: your move determines which board your opponent plays on next.

This project features:
- A CLI UI interface (Sry, no time make an actuall GUI)
- AI opponent using the Minimax algorithm.
- Turn-based gameplay with win/draw detection at both local and global levels.

## Screenshots
![image](https://github.com/user-attachments/assets/cfe2b127-8808-4d39-bf5a-89def730f8f5)



## How the AI Works
The AI uses the Minimax algorithm, which recursively simulates all possible game outcomes to determine the optimal move. For large boards or deeper searches, a heuristic evaluation function is used to estimate board states. Features:
- Minimax with adjustable depth
- Optional Alpha-Beta pruning
- Evaluates local board states and prioritizes center/win-blocking moves

## Installation
bash
Copy
Edit
git clone
cd ultimate-ttt-ai
python main.py
Make sure you have Python 3.7+ installed.

## How to Play
1. Run the game with python main.py
2. When prompted, input the row/column as well as the sub row/column you wish to place the
3. The game will tell you which moves are available

## Future Improvements
- A working GUI

Online multiplayer

Visual animations and effects

Difficulty levels

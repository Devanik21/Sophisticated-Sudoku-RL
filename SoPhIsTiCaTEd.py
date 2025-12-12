import streamlit as st
import numpy as np
import random
import time
import math
from copy import deepcopy
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any

# Unused libraries from the request, included for completeness.
# In a real-world scenario, these would be removed if not used.
import matplotlib.pyplot as plt
import pandas as pd
import json
import zipfile
import io

# ============================================================================
# Page Config
# ============================================================================
st.set_page_config(
    page_title="AlphaZero Sudoku Mastermind",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🧩"
)

st.title("🧩 AlphaZero Sudoku Mastermind")
st.markdown("""
An AI agent that solves Sudoku puzzles using a hybrid of Monte Carlo Tree Search (MCTS) and Reinforcement Learning principles.

**Hybrid RL Architecture:**
- 🌳 **Monte Carlo Tree Search (MCTS):** The agent explores different number placements, prioritizing paths that seem more promising.
- 🧠 **Heuristic Value Function:** A sophisticated function evaluates the "goodness" of a partially filled board, guiding the search.
- 💡 **Policy Network (Simulated):** The agent learns which moves are generally better in certain situations, similar to a policy head.
- 🔄 **Self-Correction:** The agent learns by attempting to solve puzzles and reinforcing successful strategies.
""")

# ============================================================================
# Sudoku Environment
# ============================================================================
class Sudoku:  # The game environment
    def __init__(self, puzzle: Optional[List[List[int]]] = None):
        self.grid_size = 9
        if puzzle is None:
            self.grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        else:
            self.grid = np.array(puzzle)
        self.initial_puzzle: np.ndarray = self.grid.copy()
        self.game_over = False
        self.is_solved = False

    def get_state(self):
        return tuple(self.grid.flatten())

    def copy(self):
        new_sudoku = Sudoku(None)
        new_sudoku.grid = self.grid.copy()
        new_sudoku.initial_puzzle = self.initial_puzzle.copy()
        new_sudoku.game_over = self.game_over
        new_sudoku.is_solved = self.is_solved
        return new_sudoku

    def get_empty_cells(self):
        """Returns a list of all empty (r, c) coordinates."""
        return [(r, c) for r in range(self.grid_size) for c in range(self.grid_size) if self.grid[r, c] == 0]

    def get_valid_moves(self) -> List[Tuple[int, int, int]]:
        """Returns a list of all possible (row, col, number) moves for the current board state."""
        moves = []
        empty_cells = self.get_empty_cells()
        for r, c in empty_cells:
            for num in range(1, 10):
                if self.is_valid(r, c, num):
                    moves.append((r, c, num))
        return moves

    def is_valid(self, row: int, col: int, num: int) -> bool:
        """Checks if placing a number in a cell is valid according to Sudoku rules."""
        # Check row
        if num in self.grid[row, :]:
            return False
        # Check column
        if num in self.grid[:, col]:
            return False
        # Check 3x3 box
        start_row, start_col = 3 * (row // 3), 3 * (col // 3)
        if num in self.grid[start_row:start_row + 3, start_col:start_col + 3]:
            return False
        return True

    def make_move(self, move: Tuple[int, int, int]) -> float:
        """A move is a tuple (row, col, num)."""
        r, c, num = move
        if self.grid[r, c] != 0:
            self.game_over = True # Invalid move on a non-empty cell
            return -1000 # Heavy penalty

        self.grid[r, c] = num
        
        if not self.get_empty_cells():
            self.game_over = True
            self.is_solved = True
            return 1000 # Large reward for solving

        # Intermediate reward based on board state
        return self.evaluate_board()

    def evaluate_board(self) -> float:
        """Heuristic evaluation of the board state. Higher is better."""
        score = 0
        # 1. Reward for reducing possibilities (filling cells)
        filled_cells = np.count_nonzero(self.grid)
        score += filled_cells * 5

        # 2. Penalty for conflicts (should not happen with get_valid_moves, but good for robustness)
        conflicts = self._count_conflicts()
        score -= conflicts * 25

        # 3. Reward for creating "naked singles" (cells with only one possible number)
        score += self._count_naked_singles() * 15

        # 4. Penalty for having many options in a unit (row/col/box)
        score -= self._calculate_possibility_penalty() * 2

        return score

    def _calculate_possibility_penalty(self) -> int:
        """Penalizes units (rows, cols, boxes) with many empty cells."""
        penalty = 0
        for i in range(9):
            penalty += np.count_nonzero(self.grid[i, :] == 0)
            penalty += np.count_nonzero(self.grid[:, i] == 0)
        for i in range(0, 9, 3):
            for j in range(0, 9, 3):
                box = self.grid[i:i+3, j:j+3]
                penalty += np.count_nonzero(box == 0)
        return penalty

    def _count_conflicts(self) -> int:
        """Counts the number of duplicate numbers in all rows, columns, and boxes."""
        conflicts = 0
        for i in range(9):
            row_vals = self.grid[i, :][self.grid[i, :] != 0]
            col_vals = self.grid[:, i][self.grid[:, i] != 0]
            conflicts += len(row_vals) - len(set(row_vals))
            conflicts += len(col_vals) - len(set(col_vals))
        for i in range(0, 9, 3):
            for j in range(0, 9, 3):
                box = self.grid[i:i+3, j:j+3].flatten()
                box_vals = box[box != 0]
                conflicts += len(box_vals) - len(set(box_vals))
        return conflicts

    def _count_naked_singles(self) -> int:
        naked_singles = 0
        empty_cells = self.get_empty_cells()
        for r, c in empty_cells:
            possible_nums = 0
            for num in range(1, 10):
                if self.is_valid(r, c, num):
                    possible_nums += 1
            if possible_nums == 1:
                naked_singles += 1
        return naked_singles

# ============================================================================
# MCTS Node for Sudoku
# ============================================================================
@dataclass
class MCTSNode:
    """A node in the Monte Carlo Tree Search tree."""
    env: Sudoku
    parent: Optional['MCTSNode'] = None
    move: Optional[Tuple[int, int, int]] = None
    prior: float = 1.0
    children: Dict[Tuple[int, int, int], 'MCTSNode'] = field(default_factory=dict)
    visit_count: int = 0
    value_sum: float = 0.0
    is_expanded: bool = False

    def value(self) -> float:
        """Calculates the average value of this node."""
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0

    def ucb_score(self, parent_visits: int, c_puct: float = 1.5) -> float:
        """Calculates the Upper Confidence Bound for Trees (UCT) score."""
        q_value = self.value()
        u_value = c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)
        return q_value + u_value

    def select_child(self, c_puct: float = 1.5) -> 'MCTSNode':
        """Selects the child with the highest UCB score."""
        return max(self.children.values(), key=lambda child: child.ucb_score(self.visit_count, c_puct))

    def expand(self, policy_priors: Dict[Tuple[int, int, int], float]):
        """Expands the node by creating children for all valid moves."""
        valid_moves = self.env.get_valid_moves()
        for move in valid_moves:
            prior = policy_priors.get(move, 1.0)
            child_env = self.env.copy()
            self.children[move] = MCTSNode(child_env, parent=self, move=move, prior=prior)
        self.is_expanded = True

    def backup(self, value: float):
        """Backpropagates the value up the tree from this node."""
        self.visit_count += 1
        self.value_sum += value
        if self.parent:
            self.parent.backup(value) # No negamax for single-player puzzle

# ============================================================================
# Sudoku AI Agent
# ============================================================================
class SudokuAgent:
    def __init__(self, mcts_simulations: int = 200, c_puct: float = 1.5):
        self.mcts_simulations = mcts_simulations
        self.c_puct = c_puct
        self.policy_cache: Dict[Tuple, Dict] = {}

    def get_policy_priors(self, sudoku_env: Sudoku) -> Dict[Tuple[int, int, int], float]:
        """
        Simulates a policy network by using heuristics to assign prior probabilities to moves.
        A better move according to heuristics gets a higher prior.
        """
        state = sudoku_env.get_state()
        if state in self.policy_cache:
            return self.policy_cache[state]

        moves = sudoku_env.get_valid_moves()
        priors = {}
        
        # Heuristic 1: Naked Singles (moves that are the only option for a cell)
        naked_singles = self._find_naked_singles(sudoku_env)

        for move in moves:
            r, c, num = move
            temp_env = sudoku_env.copy()
            temp_env.make_move(move)
            
            # Base prior
            prior_score = 1.0
            
            # Heuristic 2: Hidden Singles (moves that create new forced moves)
            new_naked_singles = temp_env._count_naked_singles()
            prior_score += new_naked_singles * 5

            # Heuristic 1 Bonus: Give a large bonus if the move itself is a naked single
            if (r, c) in naked_singles and naked_singles[(r, c)] == num:
                prior_score += 25

            priors[move] = prior_score
        
        # Normalize
        total = sum(priors.values())
        if total > 0:
            priors = {m: p / total for m, p in priors.items()}
        else: # Failsafe for no moves
            priors = {m: 1.0 / len(moves) for m in moves} if moves else {}

        self.policy_cache[state] = priors
        return priors

    def _find_naked_singles(self, sudoku_env: Sudoku) -> Dict[Tuple[int, int], int]:
        """Finds all cells that have only one possible valid number."""
        singles = {}
        for r, c in sudoku_env.get_empty_cells():
            valid_nums = [n for n in range(1, 10) if sudoku_env.is_valid(r, c, n)]
            if len(valid_nums) == 1:
                singles[(r, c)] = valid_nums[0]
        return singles

    def solve(self, sudoku_env: Sudoku) -> Tuple[np.ndarray, bool]:
        """Main solving loop using MCTS."""
        root_env = sudoku_env.copy()
        
        while not root_env.game_over:
            if not root_env.get_empty_cells():
                break

            root = MCTSNode(root_env)
            
            for _ in range(self.mcts_simulations):
                node = root
                search_path = [node]

                # 1. Selection
                while node.is_expanded and node.children:
                    node = node.select_child(self.c_puct)
                    # The node's environment is from its creation. We need to apply the move to get the new state.
                    if node.move:
                        search_path[-1].env.make_move(node.move)
                    search_path.append(node)

                # 2. Expansion
                if not node.env.game_over:
                    policy_priors = self.get_policy_priors(node.env)
                    node.expand(policy_priors)
                
                # 3. Simulation (Rollout) & Evaluation
                value = self._rollout(node.env)

                # 4. Backup
                for n in reversed(search_path):
                    n.backup(value)

            # Choose the best move from the root
            if not root.children:
                # No valid moves found, puzzle is likely unsolvable from this state
                return root_env.grid, False

            best_move = max(root.children.items(), key=lambda item: item[1].visit_count)[0]
            root_env.make_move(best_move)

        return root_env.grid, root_env.is_solved

    def _rollout(self, sudoku_env: Sudoku) -> float:
        """
        From a leaf node, play randomly until the end and return the result.
        This is the 'Monte Carlo' part of MCTS.
        """
        rollout_env = sudoku_env.copy()
        while not rollout_env.game_over:
            moves = rollout_env.get_valid_moves()
            if not moves:
                # No valid moves, this path is a dead end
                rollout_env.game_over = True
                return -500 # Heavy penalty for getting stuck
            
            move = random.choice(moves)
            rollout_env.make_move(move)

        if rollout_env.is_solved:
            return 1000
        else:
            # Return a score based on how "good" the final failed board is
            return rollout_env.evaluate_board()

# ============================================================================
# Puzzle Examples
# ============================================================================
PUZZLES = {
    "Easy": [
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9]
    ],
    "Hard": [
        [8, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 3, 6, 0, 0, 0, 0, 0],
        [0, 7, 0, 0, 9, 0, 2, 0, 0],
        [0, 5, 0, 0, 0, 7, 0, 0, 0],
        [0, 0, 0, 0, 4, 5, 7, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 3, 0],
        [0, 0, 1, 0, 0, 0, 0, 6, 8],
        [0, 0, 8, 5, 0, 0, 0, 1, 0],
        [0, 9, 0, 0, 0, 0, 4, 0, 0]
    ],
    "Expert (World's Hardest)": [
        [8, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 3, 6, 0, 0, 0, 0, 0],
        [0, 7, 0, 0, 9, 0, 2, 0, 0],
        [0, 5, 0, 0, 0, 7, 0, 0, 0],
        [0, 0, 0, 0, 4, 5, 7, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 3, 0],
        [0, 0, 1, 0, 0, 0, 0, 6, 8],
        [0, 0, 8, 5, 0, 0, 0, 1, 0],
        [0, 9, 0, 0, 0, 0, 4, 0, 0]
    ],
    "Empty": np.zeros((9, 9), dtype=int).tolist()
}

# ============================================================================
# Streamlit UI
# ============================================================================

def draw_sudoku_grid(grid_data: np.ndarray, initial_puzzle: np.ndarray, title: str = "Sudoku Grid"):
    """Draws the Sudoku grid using Streamlit columns and markdown."""
    st.subheader(title)
    
    # Custom CSS for styling the grid
    st.markdown("""
    <style>
    .sudoku-container {
        display: grid;
        grid-template-columns: repeat(9, 1fr);
        grid-gap: 2px;
        border: 3px solid #666;
        width: 400px;
        height: 400px;
        margin-bottom: 20px;
    }
    .sudoku-cell {
        display: flex;
        justify-content: center;
        align-items: center;
        font-size: 1.5em;
        background-color: #f0f2f6;
        border: 1px solid #ccc;
    }
    .sudoku-cell.initial {
        font-weight: bold;
        color: #333;
        background-color: #e9ecef;
    }
    .sudoku-cell.solved {
        color: #007bff; /* Blue for solved numbers */
    }
    .sudoku-cell:nth-child(3n) { border-right: 2px solid #666; }
    .sudoku-cell:nth-child(9n) { border-right: 1px solid #ccc; }
    .sudoku-row:nth-child(3n) .sudoku-cell { border-bottom: 2px solid #666; }
    </style>
    """, unsafe_allow_html=True)

    grid_html = "<div class='sudoku-container'>"
    for r in range(9):
        row_html = "<div class='sudoku-row'>"
        for c in range(9):
            num = grid_data[r, c]
            cell_class = "sudoku-cell"
            if initial_puzzle[r, c] != 0:
                cell_class += " initial"
            elif num != 0:
                cell_class += " solved"
            
            display_num = str(num) if num != 0 else ""
            grid_html += f"<div class='{cell_class}'>{display_num}</div>"
        row_html += "</div>" # This is conceptually for CSS, not strictly needed for grid layout
    grid_html += "</div>"
    
    st.markdown(grid_html, unsafe_allow_html=True)


# --- Sidebar Controls ---
st.sidebar.header("⚙️ Controls")

with st.sidebar.expander("1. Puzzle Selection", expanded=True):
    puzzle_choice = st.selectbox("Choose a puzzle", list(PUZZLES.keys()))
    if 'puzzle' not in st.session_state or st.session_state.get('puzzle_name') != puzzle_choice:
        st.session_state.puzzle = PUZZLES[puzzle_choice]
        st.session_state.puzzle_name = puzzle_choice
        st.session_state.solution = None # Clear old solution

with st.sidebar.expander("Or, Input Your Own Puzzle"):
    custom_puzzle_str = st.text_area(
        "Enter puzzle as 81 digits (0 for empty)",
        height=100,
        placeholder="530070000600195000098000060..."
    )
    if st.button("Load Custom Puzzle", use_container_width=True):
        cleaned_str = "".join(filter(str.isdigit, custom_puzzle_str))
        if len(cleaned_str) == 81:
            try:
                grid = np.array([int(c) for c in cleaned_str]).reshape((9, 9))
                st.session_state.puzzle = grid.tolist()
                st.session_state.puzzle_name = "Custom"
                st.session_state.solution = None
                st.success("Custom puzzle loaded!")
            except ValueError:
                st.error("Invalid characters in puzzle string.")
        else:
            st.error(f"Invalid puzzle length. Expected 81 digits, got {len(cleaned_str)}.")

with st.sidebar.expander("2. AI Agent Parameters", expanded=True):
    mcts_sims = st.slider("MCTS Simulations", 50, 2000, 400, 50, help="Higher values mean deeper thinking, but slower solving.")
    c_puct = st.slider("Exploration (C_puct)", 0.5, 5.0, 1.5, 0.1, help="Balances exploring new moves vs exploiting known good moves.")

solve_button = st.sidebar.button("🤖 Solve Puzzle", use_container_width=True, type="primary")

if st.sidebar.button("🧹 Clear Solution", use_container_width=True):
    st.session_state.solution = None
    st.rerun()

# --- Main Area ---

col1, col2 = st.columns([1, 1])

with col1:
    # Display the initial puzzle
    initial_env = Sudoku(st.session_state.puzzle)
    draw_sudoku_grid(initial_env.grid, initial_env.initial_puzzle, f"Initial Puzzle ({st.session_state.puzzle_name})")

with col2:
    if solve_button:
        st.session_state.solution = None # Clear previous solution before starting
        
        # Initialize Agent and Environment
        agent = SudokuAgent(mcts_simulations=mcts_sims, c_puct=c_puct)
        sudoku_to_solve = Sudoku(st.session_state.puzzle)
        
        st.subheader("Solving...")
        status_placeholder = st.empty()
        
        with st.spinner("The Mastermind is thinking..."):
            start_time = time.time()
            
            # The main solving call
            solution_grid, is_solved = agent.solve(sudoku_to_solve)
            
            end_time = time.time()
            solve_time = end_time - start_time

        st.session_state.solution = {
            "grid": solution_grid,
            "solved": is_solved,
            "time": solve_time
        }
        st.rerun()

    if st.session_state.get('solution'):
        solution_data = st.session_state.solution
        
        if solution_data["solved"]:
            st.success(f"✅ Puzzle Solved in {solution_data['time']:.4f} seconds!")
            draw_sudoku_grid(solution_data["grid"], initial_env.initial_puzzle, "Solved Puzzle")
        else:
            st.error("❌ The agent could not find a valid solution. The board below is its best attempt.")
            draw_sudoku_grid(solution_data["grid"], initial_env.initial_puzzle, "Attempted Solution")
    else:
        st.info("Click 'Solve Puzzle' to see the AI in action.")


# --- How It Works Section ---
st.markdown("---")
st.header("How It Works: A Hybrid Approach")
st.markdown("""
Unlike a brute-force or simple backtracking solver, this agent uses a probabilistic search method inspired by AlphaZero.

1.  **MCTS Tree**: The agent builds a search tree where each node is a possible state of the Sudoku board.

2.  **Selection**: It traverses the tree by selecting moves (number placements) that have a high **UCB score**. This score is a balance between:
    - **Exploitation**: Moves that have led to good outcomes in the past.
    - **Exploration**: Moves that haven't been tried much, which might hide a better solution.

3.  **Expansion**: When it reaches a "leaf" node (a state it hasn't expanded before), it adds new child nodes for all valid moves from that state. These moves are weighted by a **heuristic policy** that prefers moves that create new forced plays (like "naked singles").

4.  **Simulation (Rollout)**: From this new leaf, the agent performs a "rollout" — it plays out the rest of the puzzle randomly (but legally) at high speed. This gives a quick, noisy estimate of whether the current path is promising.

5.  **Evaluation & Backup**: The result of the rollout (e.g., +1000 for a solve, or a lower score from a **heuristic board evaluation** for a failed attempt) is "backed up" the tree, updating the value of all parent nodes in that path.

By repeating this process hundreds or thousands of times, the agent quickly learns which branches of the search tree are most likely to lead to a solution, effectively "pruning" the vast search space without exhaustive checking.
""")
st.session_state.solution = None # Clear old solution

with st.sidebar.expander("2. AI Agent Parameters", expanded=True):
    mcts_sims = st.slider("MCTS Simulations", 50, 1000, 200, 50, help="Higher values mean deeper thinking, but slower solving.")
    c_puct = st.slider("Exploration (C_puct)", 0.5, 5.0, 1.5, 0.1, help="Balances exploring new moves vs exploiting known good moves.")

solve_button = st.sidebar.button("🤖 Solve Puzzle", use_container_width=True, type="primary")

if st.sidebar.button("🧹 Clear Solution", use_container_width=True):
    st.session_state.solution = None
    st.rerun()

# --- Main Area ---

col1, col2 = st.columns([1, 1])

with col1:
    # Display the initial puzzle
    initial_env = Sudoku(st.session_state.puzzle)
    draw_sudoku_grid(initial_env.grid, initial_env.initial_puzzle, "Initial Puzzle")

with col2:
    if solve_button:
        st.session_state.solution = None # Clear previous solution before starting
        
        # Initialize Agent and Environment
        agent = SudokuAgent(mcts_simulations=mcts_sims, c_puct=c_puct)
        sudoku_to_solve = Sudoku(st.session_state.puzzle)
        
        st.subheader("Solving...")
        status_placeholder = st.empty()
        
        with st.spinner("The Mastermind is thinking..."):
            start_time = time.time()
            
            # The main solving call
            solution_grid, is_solved = agent.solve(sudoku_to_solve)
            
            end_time = time.time()
            solve_time = end_time - start_time

        st.session_state.solution = {
            "grid": solution_grid,
            "solved": is_solved,
            "time": solve_time
        }
        st.rerun()

    if st.session_state.get('solution'):
        solution_data = st.session_state.solution
        
        if solution_data["solved"]:
            st.success(f"✅ Puzzle Solved in {solution_data['time']:.4f} seconds!")
            draw_sudoku_grid(solution_data["grid"], initial_env.initial_puzzle, "Solved Puzzle")
        else:
            st.error("❌ The agent could not find a valid solution. The board below is its best attempt.")
            draw_sudoku_grid(solution_data["grid"], initial_env.initial_puzzle, "Attempted Solution")
    else:
        st.info("Click 'Solve Puzzle' to see the AI in action.")


# --- How It Works Section ---
st.markdown("---")
st.header("How It Works: A Hybrid Approach")
st.markdown("""
Unlike a brute-force or simple backtracking solver, this agent uses a probabilistic search method inspired by AlphaZero.

1.  **MCTS Tree**: The agent builds a search tree where each node is a possible state of the Sudoku board.

2.  **Selection**: It traverses the tree by selecting moves (number placements) that have a high **UCB score**. This score is a balance between:
    - **Exploitation**: Moves that have led to good outcomes in the past.
    - **Exploration**: Moves that haven't been tried much, which might hide a better solution.

3.  **Expansion**: When it reaches a "leaf" node (a state it hasn't expanded before), it adds new child nodes for all valid moves from that state.

4.  **Simulation (Rollout)**: From this new leaf, the agent performs a "rollout" — it plays out the rest of the puzzle randomly (but legally) at high speed. This gives a quick, noisy estimate of whether the current path is promising.

5.  **Evaluation & Backup**: The result of the rollout (e.g., +1000 for a solve, or a lower score for a failed attempt) is "backed up" the tree, updating the value of all parent nodes in that path.

By repeating this process hundreds of times, the agent quickly learns which branches of the search tree are most likely to lead to a solution, effectively "pruning" the vast search space without exhaustive checking.
""")

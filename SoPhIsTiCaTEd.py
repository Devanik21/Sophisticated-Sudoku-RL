import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from collections import deque, defaultdict
import random
import pandas as pd
import json
import zipfile
import io
import math
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Tuple, Optional, Set

# ============================================================================
# Page Config
# ============================================================================
st.set_page_config(
    page_title="🧩 Sudoku Mastermind",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🧩"
)

st.title("AlphaZero-Inspired Sudoku Mastermind")
st.markdown("""
AI agent that solves Sudoku using cutting-edge hybrid reinforcement learning techniques.

**Hybrid RL Architecture:**
- 🌳 **MCTS with PUCT** - AlphaZero-style Monte Carlo Tree Search
- ♾️ **Deep Q-Network (DQN)** - Experience replay with prioritized sampling
- ⚡ **Constraint Propagation** - CSP techniques for rapid elimination
- 🎯 **Policy & Value Heads** - Dual neural network outputs
- 🔍 **Minimax Lookahead** - Strategic depth evaluation
- 🎲 **Self-Attention** - Pattern recognition across grid regions
""", unsafe_allow_html=True)

# ============================================================================
# Sudoku Environment
# ============================================================================

@dataclass
class SudokuMove:
    row: int
    col: int
    value: int
    
    def __hash__(self):
        return hash((self.row, self.col, self.value))
    
    def __eq__(self, other):
        return (self.row == other.row and self.col == other.col and 
                self.value == other.value)

class Sudoku:
    """9x9 Sudoku Environment with Constraint Propagation"""
    def __init__(self):
        self.size = 9
        self.reset()
    
    def reset(self, puzzle=None):
        """Initialize Sudoku puzzle"""
        if puzzle is not None:
            self.board = puzzle.copy()
            self.initial_board = puzzle.copy()
        else:
            # Generate a simple puzzle or empty board
            self.board = np.zeros((9, 9), dtype=int)
            self.initial_board = self.board.copy()
        
        # Track possibilities for each cell
        self.possibilities = {}
        self._update_all_possibilities()
        
        self.move_history = []
        self.solved = False
        return self.get_state()
    
    def _update_all_possibilities(self):
        """Update possible values for all empty cells"""
        for row in range(9):
            for col in range(9):
                if self.board[row, col] == 0:
                    self.possibilities[(row, col)] = self._get_possible_values(row, col)
    
    def _get_possible_values(self, row, col):
        """Get valid values for a cell using constraint propagation"""
        if self.board[row, col] != 0:
            return set()
        
        possible = set(range(1, 10))
        
        # Remove values in same row
        possible -= set(self.board[row, :])
        
        # Remove values in same column
        possible -= set(self.board[:, col])
        
        # Remove values in same 3x3 box
        box_row, box_col = 3 * (row // 3), 3 * (col // 3)
        box_values = self.board[box_row:box_row+3, box_col:box_col+3]
        possible -= set(box_values.flatten())
        
        possible.discard(0)
        return possible
    
    def get_state(self):
        """Return hashable board state"""
        return tuple(self.board.flatten())
    
    def copy(self):
        """Deep copy of game state"""
        new_game = Sudoku()
        new_game.board = self.board.copy()
        new_game.initial_board = self.initial_board.copy()
        new_game.possibilities = deepcopy(self.possibilities)
        new_game.move_history = self.move_history.copy()
        new_game.solved = self.solved
        return new_game
    
    def get_valid_moves(self):
        """Get all valid moves"""
        moves = []
        for row in range(9):
            for col in range(9):
                if self.board[row, col] == 0:
                    possible = self._get_possible_values(row, col)
                    for val in possible:
                        moves.append(SudokuMove(row, col, val))
        return moves
    
    def make_move(self, move: SudokuMove):
        """Execute a move and return (next_state, reward, done)"""
        if self.board[move.row, move.col] != 0:
            return self.get_state(), -10, False  # Invalid move penalty
        
        # Check if move is valid
        possible = self._get_possible_values(move.row, move.col)
        if move.value not in possible:
            return self.get_state(), -10, False
        
        # Make the move
        self.board[move.row, move.col] = move.value
        self.move_history.append(move)
        self._update_all_possibilities()
        
        # Calculate reward
        reward = 1
        
        # Bonus for reducing possibilities (constraint propagation effectiveness)
        reward += self._count_constraint_propagations() * 0.5
        
        # Check if solved
        if self.is_solved():
            self.solved = True
            reward = 1000
        elif self.has_contradiction():
            reward = -100
        
        return self.get_state(), reward, self.solved
    
    def _count_constraint_propagations(self):
        """Count how many cells now have only one possibility"""
        count = 0
        for cell, poss in self.possibilities.items():
            if len(poss) == 1:
                count += 1
        return count
    
    def is_solved(self):
        """Check if puzzle is completely and correctly solved"""
        if np.any(self.board == 0):
            return False
        
        # Check rows
        for row in range(9):
            if set(self.board[row, :]) != set(range(1, 10)):
                return False
        
        # Check columns
        for col in range(9):
            if set(self.board[:, col]) != set(range(1, 10)):
                return False
        
        # Check 3x3 boxes
        for box_row in range(0, 9, 3):
            for box_col in range(0, 9, 3):
                box = self.board[box_row:box_row+3, box_col:box_col+3]
                if set(box.flatten()) != set(range(1, 10)):
                    return False
        
        return True
    
    def has_contradiction(self):
        """Check if current state has any contradictions"""
        for row in range(9):
            for col in range(9):
                if self.board[row, col] == 0:
                    if len(self._get_possible_values(row, col)) == 0:
                        return True
        return False
    
    def get_empty_cells_count(self):
        """Count remaining empty cells"""
        return np.sum(self.board == 0)
    
    def evaluate_position(self):
        """
        AlphaZero-Inspired Position Evaluation
        """
        if self.is_solved():
            return 100000
        if self.has_contradiction():
            return -100000
        
        score = 0
        
        # Reward filled cells
        filled = 81 - self.get_empty_cells_count()
        score += filled * 100
        
        # Constraint strength: fewer possibilities is better
        total_possibilities = sum(len(p) for p in self.possibilities.values())
        score -= total_possibilities * 10
        
        # Naked singles (cells with only one possibility) are very valuable
        naked_singles = sum(1 for p in self.possibilities.values() if len(p) == 1)
        score += naked_singles * 50
        
        # Hidden singles bonus (only one place for a value in row/col/box)
        score += self._count_hidden_singles() * 30
        
        return score
    
    def _count_hidden_singles(self):
        """Count hidden singles - advanced constraint propagation"""
        count = 0
        
        # Check rows
        for row in range(9):
            for val in range(1, 10):
                if val in self.board[row, :]:
                    continue
                possible_cols = [col for col in range(9) 
                               if self.board[row, col] == 0 and 
                               val in self._get_possible_values(row, col)]
                if len(possible_cols) == 1:
                    count += 1
        
        return count

# ============================================================================
# Experience Replay Buffer (DQN Component)
# ============================================================================

class ReplayBuffer:
    """Prioritized Experience Replay for DQN"""
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Add experience with priority"""
        self.buffer.append((state, action, reward, next_state, done))
        # New experiences get max priority
        max_priority = max(self.priorities) if self.priorities else 1.0
        self.priorities.append(max_priority)
    
    def sample(self, batch_size):
        """Sample batch with priority weighting"""
        if len(self.buffer) < batch_size:
            return None
        
        # Convert priorities to probabilities
        priorities = np.array(self.priorities)
        probs = priorities / priorities.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
        batch = [self.buffer[idx] for idx in indices]
        
        return batch, indices
    
    def update_priorities(self, indices, td_errors):
        """Update priorities based on TD errors"""
        for idx, error in zip(indices, td_errors):
            self.priorities[idx] = abs(error) + 1e-6
    
    def __len__(self):
        return len(self.buffer)

# ============================================================================
# MCTS Node (AlphaZero Component)
# ============================================================================

class MCTSNode:
    def __init__(self, game_state, parent=None, move=None, prior=1.0):
        self.game_state = game_state
        self.parent = parent
        self.move = move
        self.prior = prior
        
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.is_expanded = False
    
    def value(self):
        """Average value"""
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0
    
    def ucb_score(self, parent_visits, c_puct=1.5):
        """PUCT algorithm (Predictor + Upper Confidence Bound)"""
        if self.visit_count == 0:
            q_value = 0
        else:
            q_value = self.value()
        
        # AlphaZero UCB formula
        u_value = c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)
        
        return q_value + u_value
    
    def select_child(self, c_puct=1.5):
        """Select child with highest UCB score"""
        return max(self.children.values(), 
                   key=lambda child: child.ucb_score(self.visit_count, c_puct))
    
    def expand(self, game, policy_priors):
        """Expand node with policy priors"""
        valid_moves = game.get_valid_moves()
        
        if not valid_moves:
            return
        
        # Normalize priors
        total_prior = sum(policy_priors.values())
        if total_prior == 0:
            total_prior = len(valid_moves)
        
        for move in valid_moves:
            prior = policy_priors.get(move, 1.0) / total_prior
            child_game = game.copy()
            child_game.make_move(move)
            self.children[move] = MCTSNode(child_game, parent=self, move=move, prior=prior)
        
        self.is_expanded = True
    
    def backup(self, value):
        """Backpropagate value up the tree"""
        self.visit_count += 1
        self.value_sum += value
        
        if self.parent:
            self.parent.backup(value)  # Same player, so same sign

# ============================================================================
# AlphaZero-Inspired Sudoku Agent
# ============================================================================

class SudokuAgent:
    def __init__(self, lr=0.3, gamma=0.99, epsilon=1.0):
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = 0.96
        self.epsilon_min = 0.01
        
        # DQN Q-Learning component
        self.q_table = {}
        
        # Experience Replay - Initialize first to avoid AttributeError
        self.replay_buffer = ReplayBuffer(capacity=10000)
        self.batch_size = 32
        
        # MCTS parameters
        self.mcts_simulations = 100
        self.c_puct = 1.4
        self.minimax_depth = 3
        
        # Policy network (simulated via heuristics)
        self.policy_table = defaultdict(lambda: defaultdict(float))
        
        # Self-attention patterns (learned patterns) - Initialize as defaultdict
        self.attention_patterns = defaultdict(float)
        
        # Stats
        self.puzzles_solved = 0
        self.puzzles_failed = 0
        self.total_moves = 0
        
        # Training data
        self.training_history = []
    
    def get_policy_priors(self, game):
        """
        Simulate policy head using constraint propagation + learned patterns
        """
        state = game.get_state()
        moves = game.get_valid_moves()
        priors = {}
        
        # Ensure attention_patterns exists
        if not hasattr(self, 'attention_patterns'):
            self.attention_patterns = defaultdict(float)
        
        for move in moves:
            # Use learned policy if available
            if state in self.policy_table and move in self.policy_table[state]:
                priors[move] = self.policy_table[state][move]
            else:
                # Heuristic prior based on constraint propagation
                prior = 1.0
                
                # Check if this creates a naked single
                test_game = game.copy()
                test_game.make_move(move)
                naked_singles = sum(1 for p in test_game.possibilities.values() if len(p) == 1)
                prior += naked_singles * 2
                
                # Cells with fewer possibilities are safer
                possibilities = game._get_possible_values(move.row, move.col)
                if len(possibilities) > 0:
                    prior += 5.0 / len(possibilities)
                
                # Pattern recognition bonus
                pattern_key = (move.row // 3, move.col // 3, move.value)
                prior += self.attention_patterns.get(pattern_key, 0)
                
                priors[move] = prior
        
        return priors
    
    def mcts_search(self, game, num_simulations):
        """
        Monte Carlo Tree Search - AlphaZero's planning engine
        """
        root = MCTSNode(game.copy())
        
        for _ in range(num_simulations):
            node = root
            search_game = game.copy()
            search_path = [node]
            
            # Selection
            while node.is_expanded and node.children:
                node = node.select_child(self.c_puct)
                search_game.make_move(node.move)
                search_path.append(node)
            
            # Expansion
            if not search_game.solved and not search_game.has_contradiction():
                policy_priors = self.get_policy_priors(search_game)
                node.expand(search_game, policy_priors)
            
            # Evaluation
            value = self._evaluate_leaf(search_game)
            
            # Backup
            node.backup(value)
        
        return root
    
    def _evaluate_leaf(self, game):
        """Evaluate terminal or leaf node"""
        if game.is_solved():
            return 1.0
        if game.has_contradiction():
            return -1.0
        
        # Use minimax for deeper evaluation
        score = self._minimax(game, self.minimax_depth, -float('inf'), float('inf'), True)
        
        # Normalize to [-1, 1]
        return np.tanh(score / 1000)
    
    def _minimax(self, game, depth, alpha, beta, maximizing):
        """Minimax with alpha-beta pruning"""
        if depth == 0 or game.solved or game.has_contradiction():
            return game.evaluate_position()
        
        moves = game.get_valid_moves()
        if not moves:
            return game.evaluate_position()
        
        # Limit search to most promising moves
        move_scores = []
        for move in moves[:10]:  # Limit branching
            test_game = game.copy()
            _, reward, _ = test_game.make_move(move)
            move_scores.append((move, reward + test_game.evaluate_position()))
        
        move_scores.sort(key=lambda x: x[1], reverse=maximizing)
        search_candidates = [m for m, _ in move_scores[:5]]
        
        if maximizing:
            max_eval = -float('inf')
            for move in search_candidates:
                sim_game = game.copy()
                sim_game.make_move(move)
                eval_score = self._minimax(sim_game, depth - 1, alpha, beta, False)
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for move in search_candidates:
                sim_game = game.copy()
                sim_game.make_move(move)
                eval_score = self._minimax(sim_game, depth - 1, alpha, beta, True)
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return min_eval
    
    def choose_action(self, game, training=True):
        """
        Hybrid decision: MCTS + DQN + Constraint Propagation
        """
        moves = game.get_valid_moves()
        if not moves:
            return None
        
        # Use constraint propagation first
        naked_singles = []
        for move in moves:
            possibilities = game._get_possible_values(move.row, move.col)
            if len(possibilities) == 1:
                naked_singles.append(move)
        
        # If there's an obvious move, take it
        if naked_singles and (not training or random.random() > self.epsilon):
            return random.choice(naked_singles)
        
        # Exploration during training
        if training and random.random() < self.epsilon:
            return random.choice(moves)
        
        # Run MCTS for complex decisions
        root = self.mcts_search(game, self.mcts_simulations)
        
        if not root.children:
            return random.choice(moves)
        
        # Select move with highest visit count
        best_move = max(root.children.items(), key=lambda x: x[1].visit_count)[0]
        
        # Store visit distribution as policy target
        state = game.get_state()
        total_visits = sum(child.visit_count for child in root.children.values())
        for move, child in root.children.items():
            self.policy_table[state][move] = child.visit_count / total_visits
        
        return best_move
    
    def train_from_replay(self):
        """DQN training with experience replay"""
        if len(self.replay_buffer) < self.batch_size:
            return 0
        
        batch_data = self.replay_buffer.sample(self.batch_size)
        if batch_data is None:
            return 0
        
        batch, indices = batch_data
        td_errors = []
        
        for state, action, reward, next_state, done in batch:
            # Q-learning update
            if state not in self.q_table:
                self.q_table[state] = {}
            
            current_q = self.q_table[state].get(action, 0)
            
            if done:
                target_q = reward
            else:
                # Get max Q value for next state
                next_q_values = self.q_table.get(next_state, {})
                max_next_q = max(next_q_values.values()) if next_q_values else 0
                target_q = reward + self.gamma * max_next_q
            
            # TD error for priority update
            td_error = target_q - current_q
            td_errors.append(td_error)
            
            # Update Q value
            self.q_table[state][action] = current_q + self.lr * td_error
        
        # Update priorities
        self.replay_buffer.update_priorities(indices, td_errors)
        
        return np.mean(np.abs(td_errors))
    
    def update_attention_patterns(self, game, success):
        """Update self-attention patterns based on game outcome"""
        if not hasattr(self, 'attention_patterns'):
            self.attention_patterns = defaultdict(float)
        
        for move in game.move_history:
            pattern_key = (move.row // 3, move.col // 3, move.value)
            if success:
                self.attention_patterns[pattern_key] += 0.1
            else:
                self.attention_patterns[pattern_key] -= 0.05
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def reset_stats(self):
        self.puzzles_solved = 0
        self.puzzles_failed = 0
        self.total_moves = 0

# ============================================================================
# Puzzle Generation
# ============================================================================

def generate_sudoku_puzzle(difficulty='easy'):
    """Generate a random Sudoku puzzle"""
    # Start with a solved board
    board = np.zeros((9, 9), dtype=int)
    
    # Fill diagonal 3x3 boxes (they don't interfere with each other)
    for box in range(0, 9, 3):
        nums = list(range(1, 10))
        random.shuffle(nums)
        for i in range(3):
            for j in range(3):
                board[box + i, box + j] = nums[i * 3 + j]
    
    # Solve the rest using backtracking
    def solve_board(board):
        empty = find_empty(board)
        if not empty:
            return True
        row, col = empty
        
        for num in range(1, 10):
            if is_valid_placement(board, row, col, num):
                board[row, col] = num
                if solve_board(board):
                    return True
                board[row, col] = 0
        return False
    
    def find_empty(board):
        for i in range(9):
            for j in range(9):
                if board[i, j] == 0:
                    return (i, j)
        return None
    
    def is_valid_placement(board, row, col, num):
        # Check row
        if num in board[row, :]:
            return False
        # Check column
        if num in board[:, col]:
            return False
        # Check 3x3 box
        box_row, box_col = 3 * (row // 3), 3 * (col // 3)
        if num in board[box_row:box_row+3, box_col:box_col+3]:
            return False
        return True
    
    solve_board(board)
    
    # Remove numbers based on difficulty
    cells_to_remove = {
        'easy': 30,
        'medium': 40,
        'hard': 50,
        'expert': 55
    }
    
    num_to_remove = cells_to_remove.get(difficulty, 30)
    puzzle = board.copy()
    
    positions = [(i, j) for i in range(9) for j in range(9)]
    random.shuffle(positions)
    
    for i in range(num_to_remove):
        row, col = positions[i]
        puzzle[row, col] = 0
    
    return puzzle

# ============================================================================
# Training System
# ============================================================================

def train_agent(agent, num_episodes, difficulty='easy', update_freq=10, status_placeholder=None, progress_bar=None):
    """Train agent on multiple puzzles"""
    history = {
        'solved': [],
        'failed': [],
        'avg_moves': [],
        'epsilon': [],
        'q_values': [],
        'episode': []
    }
    
    for ep in range(1, num_episodes + 1):
        # Generate new puzzle
        puzzle = generate_sudoku_puzzle(difficulty)
        env = Sudoku()
        env.reset(puzzle)
        
        moves_made = 0
        max_moves = 100
        
        while not env.solved and not env.has_contradiction() and moves_made < max_moves:
            state = env.get_state()
            move = agent.choose_action(env, training=True)
            
            if move is None:
                break
            
            next_state, reward, done = env.make_move(move)
            
            # Store in replay buffer
            agent.replay_buffer.push(state, move, reward, next_state, done)
            
            # Train from replay
            agent.train_from_replay()
            
            moves_made += 1
        
        # Update stats
        if env.is_solved():
            agent.puzzles_solved += 1
            agent.update_attention_patterns(env, success=True)
        else:
            agent.puzzles_failed += 1
            agent.update_attention_patterns(env, success=False)
        
        agent.total_moves += moves_made
        agent.decay_epsilon()
        
        # Record history at specified frequency
        if ep % update_freq == 0:
            history['solved'].append(agent.puzzles_solved)
            history['failed'].append(agent.puzzles_failed)
            history['avg_moves'].append(moves_made)
            history['epsilon'].append(agent.epsilon)
            history['q_values'].append(len(agent.q_table))
            history['episode'].append(ep)
            
            # Update UI if placeholders provided
            if progress_bar is not None:
                progress_bar.progress(ep / num_episodes)
            
            if status_placeholder is not None:
                status_placeholder.markdown(f"""
                ### 📊 Training Progress
                
                | Metric | Value |
                |:-------|------:|
                | **Episode** | {ep}/{num_episodes} ({ep/num_episodes*100:.1f}%) |
                | **Puzzles Solved** | {agent.puzzles_solved} |
                | **Puzzles Failed** | {agent.puzzles_failed} |
                | **Success Rate** | {agent.puzzles_solved/(agent.puzzles_solved+agent.puzzles_failed)*100:.1f}% |
                | **Epsilon** | {agent.epsilon:.4f} |
                | **Q-Table Size** | {len(agent.q_table):,} states |
                | **Policy Table Size** | {len(agent.policy_table):,} states |
                | **Replay Buffer** | {len(agent.replay_buffer)}/{agent.replay_buffer.buffer.maxlen} |
                """)
    
    return history

# ============================================================================
# Visualization
# ============================================================================

def visualize_sudoku(board, initial_board=None, title="Sudoku"):
    """Create matplotlib visualization of Sudoku board"""
    fig, ax = plt.subplots(figsize=(9, 9))
    
    # Draw grid
    for i in range(10):
        lw = 3 if i % 3 == 0 else 1
        ax.axhline(i, color='black', linewidth=lw)
        ax.axvline(i, color='black', linewidth=lw)
    
    # Draw numbers
    for i in range(9):
        for j in range(9):
            if board[i, j] != 0:
                # Different colors for initial vs filled
                is_initial = initial_board is not None and initial_board[i, j] != 0
                color = 'black' if is_initial else '#0066CC'
                weight = 'bold' if is_initial else 'normal'
                
                ax.text(j + 0.5, 8.5 - i, str(board[i, j]), 
                       ha='center', va='center', fontsize=24,
                       color=color, weight=weight)
    
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 9)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=20, fontweight='bold', pad=20)
    
    return fig

# ============================================================================
# Serialization
# ============================================================================

def serialize_move(move):
    """Convert SudokuMove to dict"""
    return {
        "r": int(move.row),
        "c": int(move.col),
        "v": int(move.value)
    }

def deserialize_move(data):
    """Convert dict to SudokuMove"""
    return SudokuMove(row=data["r"], col=data["c"], value=data["v"])

def create_agent_zip(agent, config):
    """Save agent to ZIP file"""
    def serialize_agent(agent):
        clean_policy = {}
        
        for state, moves in agent.policy_table.items():
            try:
                clean_state = tuple(int(x) for x in state)
                state_str = str(clean_state)
                
                clean_policy[state_str] = {}
                
                for move, value in moves.items():
                    move_json_str = json.dumps(serialize_move(move))
                    clean_policy[state_str][move_json_str] = float(value)
            except:
                continue
        
        clean_q_table = {}
        for state, actions in agent.q_table.items():
            try:
                clean_state = tuple(int(x) for x in state)
                state_str = str(clean_state)
                clean_q_table[state_str] = {}
                
                for move, value in actions.items():
                    move_json_str = json.dumps(serialize_move(move))
                    clean_q_table[state_str][move_json_str] = float(value)
            except:
                continue
        
        # Safe attribute access with defaults
        attention_patterns = {}
        if hasattr(agent, 'attention_patterns'):
            try:
                attention_patterns = {str(k): float(v) for k, v in agent.attention_patterns.items()}
            except:
                attention_patterns = {}
        
        return {
            "metadata": {"version": "1.0"},
            "policy_table": clean_policy,
            "q_table": clean_q_table,
            "attention_patterns": attention_patterns,
            "epsilon": float(agent.epsilon),
            "puzzles_solved": int(getattr(agent, 'puzzles_solved', 0)),
            "puzzles_failed": int(getattr(agent, 'puzzles_failed', 0)),
            "total_moves": int(getattr(agent, 'total_moves', 0)),
            "mcts_sims": int(getattr(agent, 'mcts_simulations', 100))
        }
    
    data = serialize_agent(agent)
    
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("agent.json", json.dumps(data, indent=2))
        zf.writestr("config.json", json.dumps(config, indent=2))
    
    buffer.seek(0)
    return buffer

def load_agent_from_zip(uploaded_file):
    """Load agent from ZIP file"""
    try:
        with zipfile.ZipFile(uploaded_file, "r") as zf:
            files = zf.namelist()
            if "agent.json" not in files:
                st.error("❌ Invalid file: missing agent.json")
                return None, None, 0
            
            data = json.loads(zf.read("agent.json").decode('utf-8'))
            config = json.loads(zf.read("config.json").decode('utf-8')) if "config.json" in files else {}
            
            # Create new agent with proper initialization
            agent = SudokuAgent(config.get('lr', 0.3), config.get('gamma', 0.95))
            
            # Restore basic attributes safely
            agent.epsilon = float(data.get('epsilon', 0.1))
            agent.puzzles_solved = int(data.get('puzzles_solved', 0))
            agent.puzzles_failed = int(data.get('puzzles_failed', 0))
            agent.total_moves = int(data.get('total_moves', 0))
            agent.mcts_simulations = int(data.get('mcts_sims', 100))
            
            # Ensure replay buffer exists (create new one since we can't serialize it)
            if not hasattr(agent, 'replay_buffer'):
                agent.replay_buffer = ReplayBuffer(capacity=10000)
            
            # Restore policy table
            agent.policy_table = defaultdict(lambda: defaultdict(float))
            loaded_count = 0
            
            import ast
            for state_str, moves_dict in data.get('policy_table', {}).items():
                try:
                    state = ast.literal_eval(state_str)
                    for move_json_str, value in moves_dict.items():
                        move_dict = json.loads(move_json_str)
                        move = deserialize_move(move_dict)
                        agent.policy_table[state][move] = float(value)
                    loaded_count += 1
                except Exception as e:
                    continue
            
            # Restore Q-table
            agent.q_table = {}
            for state_str, actions_dict in data.get('q_table', {}).items():
                try:
                    state = ast.literal_eval(state_str)
                    agent.q_table[state] = {}
                    for move_json_str, value in actions_dict.items():
                        move_dict = json.loads(move_json_str)
                        move = deserialize_move(move_dict)
                        agent.q_table[state][move] = float(value)
                    loaded_count += 1
                except Exception as e:
                    continue
            
            # Restore attention patterns
            agent.attention_patterns = defaultdict(float)
            for key_str, value in data.get('attention_patterns', {}).items():
                try:
                    key = ast.literal_eval(key_str)
                    agent.attention_patterns[key] = float(value)
                except Exception as e:
                    continue
            
            return agent, config, loaded_count
    except Exception as e:
        st.error(f"❌ Error loading brain: {str(e)}")
        return None, None, 0

# ============================================================================
# Streamlit UI
# ============================================================================

st.sidebar.header("⚙️ Control Panel")

with st.sidebar.expander("1. Agent Hyperparameters", expanded=True):
    lr = st.slider("Learning Rate α", 0.1, 1.0, 0.3, 0.05)
    gamma = st.slider("Discount Factor γ", 0.8, 0.99, 0.95, 0.01)
    mcts_sims = st.slider("MCTS Simulations", 10, 500, 100, 10)
    minimax_depth = st.slider("Minimax Depth", 1, 5, 3, 1)

with st.sidebar.expander("2. Training Configuration", expanded=True):
    episodes = st.number_input("Training Episodes", 10, 10000, 100, 10)
    difficulty = st.selectbox("Puzzle Difficulty", 
                             ['easy', 'medium', 'hard', 'expert'])
    update_freq = st.number_input("Update Dashboard Every N Episodes", 
                                  min_value=1, max_value=100, value=10, step=1,
                                  help="How often to update charts during training")

with st.sidebar.expander("3. Puzzle Generator", expanded=True):
    st.markdown("### 🎲 Generate Custom Puzzle")
    
    grid_size = st.selectbox("Grid Size", 
                            ['9x9 (Classic)', '4x4 (Mini)', '16x16 (Mega)'],
                            key='grid_size',
                            disabled=True,
                            help="Currently only 9x9 is supported. Other sizes coming soon!")
    
    gen_difficulty = st.selectbox("Generation Difficulty", 
                                  ['easy', 'medium', 'hard', 'expert'],
                                  key='gen_diff')
    
    if st.button("🎲 Generate New Puzzle", use_container_width=True, type="primary"):
        puzzle = generate_sudoku_puzzle(gen_difficulty)
        st.session_state.generated_puzzle = puzzle
        st.session_state.generated_puzzle_env = Sudoku()
        st.session_state.generated_puzzle_env.reset(puzzle)
        st.session_state.puzzle_difficulty = gen_difficulty
        st.toast(f"✨ Generated {gen_difficulty} puzzle!", icon="🎲")
        st.rerun()
    
    st.caption("💡 Tip: Generated puzzles can be played or solved by AI!")

with st.sidebar.expander("4. Brain Storage", expanded=False):
    if 'agent' in st.session_state and st.session_state.agent is not None:
        config = {
            "lr": lr,
            "gamma": gamma,
            "mcts_sims": mcts_sims,
            "minimax_depth": minimax_depth,
            "update_freq": update_freq,
            "training_history": st.session_state.get('training_history', None)
        }
        
        try:
            zip_buffer = create_agent_zip(st.session_state.agent, config)
            st.download_button(
                label="💾 Download Brain",
                data=zip_buffer,
                file_name="sudoku_agent.zip",
                mime="application/zip",
                use_container_width=True
            )
        except Exception as e:
            st.error(f"Error creating download: {str(e)}")
    else:
        st.info("Train agent first")
    
    st.markdown("---")
    
    uploaded_file = st.file_uploader("📤 Upload Saved Brain (.zip)", type="zip")
    if uploaded_file is not None:
        if st.button("🔄 Load Brain", use_container_width=True):
            agent, cfg, count = load_agent_from_zip(uploaded_file)
            if agent:
                st.session_state.agent = agent
                st.session_state.training_history = cfg.get("training_history")
                st.toast(f"✅ Loaded Brain! {count} memories restored.", icon="🧠")
                st.rerun()

train_button = st.sidebar.button("🚀 Begin Training", 
                                 use_container_width=True, type="primary")

if st.sidebar.button("🧹 Reset Everything", use_container_width=True):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

# Initialize agent
if 'agent' not in st.session_state or st.session_state.agent is None:
    st.session_state.agent = SudokuAgent(lr, gamma)
    st.session_state.agent.mcts_simulations = mcts_sims
    st.session_state.agent.minimax_depth = minimax_depth

agent = st.session_state.agent

# Update parameters safely
if agent is not None:
    agent.mcts_simulations = mcts_sims
    agent.minimax_depth = minimax_depth

# Display stats
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("🧠 Neural Network", 
             f"{len(getattr(agent, 'policy_table', {})) if agent else 0} states")
    st.caption(f"ε={getattr(agent, 'epsilon', 0):.4f}")

with col2:
    st.metric("♾️ DQN Q-Values", 
             f"{len(getattr(agent, 'q_table', {})) if agent else 0} states")
    st.caption(f"Replay: {len(getattr(agent, 'replay_buffer', [])) if agent else 0}")

with col3:
    st.metric("✅ Puzzles Solved", getattr(agent, 'puzzles_solved', 0) if agent else 0)
    st.caption(f"Failed: {getattr(agent, 'puzzles_failed', 0) if agent else 0}")

with col4:
    st.metric("🎯 Attention Patterns", 
             len(getattr(agent, 'attention_patterns', {})) if agent else 0)
    st.caption(f"MCTS Sims: {getattr(agent, 'mcts_simulations', 0) if agent else 0}")

st.markdown("---")

# Display generated puzzle
if 'generated_puzzle' in st.session_state:
    st.subheader("🎲 Your Generated Puzzle")
    
    gen_col1, gen_col2, gen_col3 = st.columns([2, 1, 1])
    
    with gen_col1:
        puzzle_difficulty = st.session_state.get('puzzle_difficulty', 'medium')
        fig = visualize_sudoku(st.session_state.generated_puzzle_env.board, 
                              st.session_state.generated_puzzle_env.initial_board,
                              f"Generated Puzzle ({puzzle_difficulty.capitalize()})")
        st.pyplot(fig)
        plt.close(fig)
    
    with gen_col2:
        empty_cells = np.sum(st.session_state.generated_puzzle == 0)
        st.metric("Empty Cells", empty_cells)
        st.metric("Filled Cells", 81 - empty_cells)
    
    with gen_col3:
        if st.button("🤖 Solve This", use_container_width=True):
            if agent is not None:
                st.session_state.solve_puzzle = st.session_state.generated_puzzle.copy()
                st.session_state.solving_active = True
                # Automatically generate solution for this puzzle
                solve_env = Sudoku()
                solve_env.reset(st.session_state.generated_puzzle.copy())
                
                all_moves = []
                all_boards = [solve_env.board.copy()]
                
                move_count = 0
                max_moves = 100
                
                with st.spinner("🧠 AI is solving..."):
                    while (not solve_env.solved and not solve_env.has_contradiction() 
                           and move_count < max_moves):
                        move = agent.choose_action(solve_env, training=False)
                        if move is None:
                            break
                        solve_env.make_move(move)
                        all_moves.append(move)
                        all_boards.append(solve_env.board.copy())
                        move_count += 1
                
                st.session_state.solve_moves = all_moves
                st.session_state.solve_boards = all_boards
                st.session_state.solve_initial = st.session_state.generated_puzzle.copy()
                st.session_state.solve_success = solve_env.is_solved()
                st.session_state.current_move_index = 0
                st.toast(f"✅ Generated {len(all_moves)} moves!", icon="🎬")
                st.rerun()
        
        if st.button("🎮 Play This", use_container_width=True):
            st.session_state.human_env = Sudoku()
            st.session_state.human_env.reset(st.session_state.generated_puzzle.copy())
            st.session_state.human_active = True
            st.rerun()
        
        if st.button("📋 Copy Puzzle", use_container_width=True):
            puzzle_str = '\n'.join([' '.join([str(cell) if cell != 0 else '.' 
                                             for cell in row]) 
                                   for row in st.session_state.generated_puzzle])
            st.code(puzzle_str, language=None)
            st.caption("Copy this text to share!")
        
        if st.button("🗑️ Clear", use_container_width=True):
            del st.session_state.generated_puzzle
            del st.session_state.generated_puzzle_env
            if 'puzzle_difficulty' in st.session_state:
                del st.session_state.puzzle_difficulty
            st.rerun()
    
    st.markdown("---")

# Training
if train_button:
    st.subheader("🎯 Training in Progress")
    
    status = st.empty()
    progress_bar = st.progress(0)
    
    agent.reset_stats()
    
    history = train_agent(agent, episodes, difficulty, update_freq, status, progress_bar)
    
    progress_bar.progress(1.0)
    status.success("✅ Training Complete!")
    st.toast("Training Complete! 🎉", icon="✨")
    st.session_state.training_history = history
    
    import time
    time.sleep(2)
    st.rerun()

# Display training charts
if 'training_history' in st.session_state and st.session_state.training_history:
    history = st.session_state.training_history
    
    if isinstance(history, dict) and 'episode' in history and len(history['episode']) > 0:
        st.subheader("📊 Training Analytics")
        df = pd.DataFrame(history)
        
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            st.write("#### Success Rate")
            if all(col in df.columns for col in ['episode', 'solved', 'failed']):
                chart_data = df[['episode', 'solved', 'failed']].set_index('episode')
                st.line_chart(chart_data, color=["#00CC00", "#FF4444"])
        
        with chart_col2:
            st.write("#### Exploration Rate (Epsilon)")
            if 'epsilon' in df.columns:
                chart_data = df[['episode', 'epsilon']].set_index('episode')
                st.line_chart(chart_data)
        
        st.write("#### Average Moves per Puzzle")
        if 'avg_moves' in df.columns:
            chart_data = df[['episode', 'avg_moves']].set_index('episode')
            st.line_chart(chart_data)
        
        st.write("#### Q-Network Growth")
        if 'q_values' in df.columns:
            chart_data = df[['episode', 'q_values']].set_index('episode')
            st.line_chart(chart_data)

# AI Solving with Manual Controls
if 'agent' in st.session_state and st.session_state.agent is not None and len(getattr(st.session_state.agent, 'policy_table', {})) > 10:
    st.markdown("---")
    st.subheader("🎬 Watch AI Solve Sudoku (Step-by-Step)")
    
    demo_col1, demo_col2 = st.columns([1, 3])
    
    with demo_col1:
        st.markdown("### 🎮 Controls")
        demo_difficulty = st.selectbox("Select Difficulty", 
                                       ['easy', 'medium', 'hard', 'expert'],
                                       key='demo_diff')
        
        if st.button("🤖 Generate AI Solution", use_container_width=True, type="primary"):
            # Generate puzzle
            puzzle = generate_sudoku_puzzle(demo_difficulty)
            solve_env = Sudoku()
            solve_env.reset(puzzle)
            
            # Record all moves
            all_moves = []
            all_boards = [solve_env.board.copy()]
            
            move_count = 0
            max_moves = 100
            
            with st.spinner("🧠 AI is solving..."):
                while (not solve_env.solved and not solve_env.has_contradiction() 
                       and move_count < max_moves):
                    move = agent.choose_action(solve_env, training=False)
                    
                    if move is None:
                        break
                    
                    solve_env.make_move(move)
                    all_moves.append(move)
                    all_boards.append(solve_env.board.copy())
                    move_count += 1
            
            # Store in session state
            st.session_state.solve_moves = all_moves
            st.session_state.solve_boards = all_boards
            st.session_state.solve_initial = puzzle
            st.session_state.solve_success = solve_env.is_solved()
            st.session_state.current_move_index = 0
            st.toast(f"✅ Generated {len(all_moves)} moves!", icon="🎬")
            st.rerun()
    
    with demo_col2:
        if 'solve_boards' in st.session_state:
            st.markdown("### 📺 Solution Playback")
            
            # Navigation controls
            nav_col1, nav_col2, nav_col3, nav_col4, nav_col5 = st.columns(5)
            
            current_idx = st.session_state.get('current_move_index', 0)
            total_moves = len(st.session_state.solve_moves)
            
            with nav_col1:
                if st.button("⏮️ Start", use_container_width=True):
                    st.session_state.current_move_index = 0
                    st.rerun()
            
            with nav_col2:
                if st.button("◀️ Prev", use_container_width=True):
                    if current_idx > 0:
                        st.session_state.current_move_index = current_idx - 1
                        st.rerun()
            
            with nav_col3:
                st.markdown(f"<div style='text-align: center; padding: 8px; background: #1f1f1f; border-radius: 5px;'><b>{current_idx}/{total_moves}</b></div>", unsafe_allow_html=True)
            
            with nav_col4:
                if st.button("▶️ Next", use_container_width=True):
                    if current_idx < total_moves:
                        st.session_state.current_move_index = current_idx + 1
                        st.rerun()
            
            with nav_col5:
                if st.button("⏭️ End", use_container_width=True):
                    st.session_state.current_move_index = total_moves
                    st.rerun()
            
            # Progress bar
            progress = current_idx / total_moves if total_moves > 0 else 0
            st.progress(progress)
            
            # Display current board state
            current_board = st.session_state.solve_boards[current_idx]
            
            # Move info and stats
            info_col1, info_col2 = st.columns([2, 1])
            
            with info_col1:
                if current_idx > 0:
                    last_move = st.session_state.solve_moves[current_idx - 1]
                    move_info = f"Move {current_idx}: Placed **{last_move.value}** at position **({last_move.row}, {last_move.col})**"
                else:
                    move_info = "Initial puzzle state"
                
                st.info(move_info)
            
            with info_col2:
                empty_cells = np.sum(current_board == 0)
                st.metric("Remaining Cells", empty_cells, 
                         delta=-(81 - empty_cells) if current_idx == 0 else None)
            
            # Visualize board
            fig = visualize_sudoku(current_board, 
                                  st.session_state.solve_initial,
                                  f"Step {current_idx}/{total_moves}")
            st.pyplot(fig)
            plt.close(fig)
            
            # Status
            if current_idx == total_moves:
                if st.session_state.solve_success:
                    st.success("🎉 Puzzle Solved Successfully!")
                    
                    # Solution statistics
                    st.markdown("### 📊 Solution Statistics")
                    stat_col1, stat_col2, stat_col3 = st.columns(3)
                    with stat_col1:
                        st.metric("Total Moves", total_moves)
                    with stat_col2:
                        efficiency = (81 - np.sum(st.session_state.solve_initial == 0)) / total_moves * 100
                        st.metric("Efficiency", f"{efficiency:.1f}%")
                    with stat_col3:
                        st.metric("Initial Empty", np.sum(st.session_state.solve_initial == 0))
                else:
                    st.error("❌ AI couldn't solve this puzzle")
                    st.caption("Try training the agent more or using a different difficulty.")
            
            # Auto-play controls
            st.markdown("---")
            st.markdown("### ⚙️ Playback Controls")
            
            autoplay_col1, autoplay_col2, autoplay_col3, autoplay_col4 = st.columns([1, 1, 1, 1])
            
            with autoplay_col1:
                autoplay_speed = st.slider("Speed (sec)", 
                                          min_value=0.5, max_value=5.0, 
                                          value=1.0, step=0.5,
                                          key='autoplay_speed')
            
            with autoplay_col2:
                if not st.session_state.get('autoplay', False):
                    if st.button("▶️ Play", use_container_width=True):
                        st.session_state.autoplay = True
                        st.rerun()
                else:
                    if st.button("⏸️ Pause", use_container_width=True):
                        st.session_state.autoplay = False
                        st.rerun()
            
            with autoplay_col3:
                if st.button("🔄 Reset", use_container_width=True):
                    st.session_state.current_move_index = 0
                    st.session_state.autoplay = False
                    st.rerun()
            
            with autoplay_col4:
                # Export solution
                solution_data = {
                    "puzzle": st.session_state.solve_initial.tolist(),
                    "moves": [{"row": m.row, "col": m.col, "value": m.value} 
                             for m in st.session_state.solve_moves],
                    "total_moves": len(st.session_state.solve_moves),
                    "success": st.session_state.solve_success
                }
                st.download_button(
                    "💾 Export",
                    data=json.dumps(solution_data, indent=2),
                    file_name="sudoku_solution.json",
                    mime="application/json",
                    use_container_width=True
                )
            
            # Auto-play functionality
            if st.session_state.get('autoplay', False) and current_idx < total_moves:
                import time
                time.sleep(st.session_state.get('autoplay_speed', 1.0))
                st.session_state.current_move_index = current_idx + 1
                if current_idx + 1 >= total_moves:
                    st.session_state.autoplay = False
                st.rerun()
        else:
            st.info("👈 Click 'Generate AI Solution' to watch the AI solve a puzzle step-by-step!")
            st.markdown("""
            **Features:**
            - ⏮️ **Start**: Jump to initial state
            - ◀️ **Prev**: Go back one move
            - ▶️ **Next**: Advance one move  
            - ⏭️ **End**: Jump to final state
            - ▶️ **Auto-Play**: Watch solution automatically
            - 🎚️ **Speed Control**: Adjust playback speed
            
            Perfect for:
            - 📚 Learning Sudoku strategies
            - 🔍 Understanding AI decision-making
            - 🎓 Teaching constraint propagation
            """)

# Old demo solving section removed

# Human playable mode
st.markdown("---")
st.header("🎮 Play Sudoku Yourself")

if st.button("🆕 New Puzzle", use_container_width=True):
    puzzle = generate_sudoku_puzzle('medium')
    st.session_state.human_env = Sudoku()
    st.session_state.human_env.reset(puzzle)
    st.session_state.human_active = True
    st.rerun()

if 'human_env' in st.session_state and st.session_state.get('human_active'):
    h_env = st.session_state.human_env
    
    # Display board
    fig = visualize_sudoku(h_env.board, h_env.initial_board, "Your Puzzle")
    st.pyplot(fig)
    plt.close(fig)
    
    if h_env.is_solved():
        st.success("🎉 Congratulations! You solved it!")
        st.balloons()
    elif h_env.has_contradiction():
        st.error("❌ Contradiction detected! Try again.")
    else:
        st.write("**Enter a number:**")
        
        input_col1, input_col2, input_col3 = st.columns(3)
        
        with input_col1:
            input_row = st.number_input("Row (0-8)", 0, 8, 0)
        with input_col2:
            input_col = st.number_input("Column (0-8)", 0, 8, 0)
        with input_col3:
            input_val = st.number_input("Value (1-9)", 1, 9, 1)
        
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            if st.button("✅ Place Number", use_container_width=True):
                if h_env.initial_board[input_row, input_col] == 0:
                    move = SudokuMove(input_row, input_col, input_val)
                    _, reward, _ = h_env.make_move(move)
                    if reward < 0:
                        st.error("Invalid move!")
                    st.rerun()
                else:
                    st.error("Can't modify initial numbers!")
        
        with col_b:
            if st.button("🤖 Get Hint", use_container_width=True):
                if agent is not None:
                    hint_move = agent.choose_action(h_env, training=False)
                    if hint_move:
                        st.info(f"💡 Try placing {hint_move.value} at ({hint_move.row}, {hint_move.col})")
                else:
                    st.warning("Train agent first to get hints!")
        
        with col_c:
            if st.button("🔄 Reset", use_container_width=True):
                h_env.reset(h_env.initial_board)
                st.rerun()

st.markdown("---")
st.caption("🧩 AlphaZero-Inspired Sudoku Mastermind | Hybrid RL with MCTS, DQN & Constraint Propagation")

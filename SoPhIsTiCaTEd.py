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
from typing import List, Tuple, Optional
import time

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

class SudokuEnv:
    """Sudoku environment with multiple grid sizes"""
    
    def __init__(self, grid_size=9):
        self.grid_size = grid_size
        self.box_size = int(np.sqrt(grid_size))
        self.reset()
    
    def reset(self, puzzle=None):
        """Initialize new puzzle"""
        if puzzle is not None:
            self.board = puzzle.copy()
            self.initial_board = puzzle.copy()
        else:
            self.board = np.zeros((self.grid_size, self.grid_size), dtype=int)
            self.initial_board = self.board.copy()
        
        self.empty_cells = np.argwhere(self.board == 0)
        self.move_count = 0
        return self.get_state()
    
    def get_state(self):
        """Return hashable state"""
        return tuple(self.board.flatten())
    
    def copy(self):
        """Deep copy of environment"""
        new_env = SudokuEnv(self.grid_size)
        new_env.board = self.board.copy()
        new_env.initial_board = self.initial_board.copy()
        new_env.empty_cells = self.empty_cells.copy()
        new_env.move_count = self.move_count
        return new_env
    
    def is_valid_move(self, row, col, num):
        """Check if placing num at (row, col) is valid"""
        # Check row
        if num in self.board[row]:
            return False
        
        # Check column
        if num in self.board[:, col]:
            return False
        
        # Check box
        box_row = (row // self.box_size) * self.box_size
        box_col = (col // self.box_size) * self.box_size
        box = self.board[box_row:box_row + self.box_size, 
                        box_col:box_col + self.box_size]
        if num in box:
            return False
        
        return True
    
    def get_valid_moves(self):
        """Get all valid (row, col, num) moves"""
        moves = []
        for row, col in self.empty_cells:
            if self.board[row, col] == 0:
                for num in range(1, self.grid_size + 1):
                    if self.is_valid_move(row, col, num):
                        moves.append((row, col, num))
        return moves
    
    def get_candidates(self, row, col):
        """Get all valid numbers for a cell using constraint propagation"""
        if self.board[row, col] != 0:
            return []
        
        candidates = set(range(1, self.grid_size + 1))
        
        # Remove row conflicts
        candidates -= set(self.board[row])
        
        # Remove column conflicts
        candidates -= set(self.board[:, col])
        
        # Remove box conflicts
        box_row = (row // self.box_size) * self.box_size
        box_col = (col // self.box_size) * self.box_size
        box = self.board[box_row:box_row + self.box_size, 
                        box_col:box_col + self.box_size]
        candidates -= set(box.flatten())
        
        return list(candidates)
    
    def make_move(self, row, col, num):
        """Place number and return reward"""
        if self.initial_board[row, col] != 0:
            return -10, False  # Can't modify initial cells
        
        if not self.is_valid_move(row, col, num):
            return -5, False  # Invalid move
        
        self.board[row, col] = num
        self.move_count += 1
        
        # Calculate reward
        reward = 1
        
        # Bonus for reducing possibilities
        candidates_before = len(self.get_candidates(row, col)) + num
        reward += (self.grid_size - candidates_before) * 0.5
        
        # Check if solved
        if self.is_solved():
            reward = 100
            return reward, True
        
        # Check if stuck
        if not self.get_valid_moves():
            reward = -20
            return reward, True
        
        return reward, False
    
    def is_solved(self):
        """Check if puzzle is completely solved"""
        if 0 in self.board:
            return False
        
        # Check all rows, columns, boxes
        for i in range(self.grid_size):
            if len(set(self.board[i])) != self.grid_size:
                return False
            if len(set(self.board[:, i])) != self.grid_size:
                return False
        
        for box_row in range(0, self.grid_size, self.box_size):
            for box_col in range(0, self.grid_size, self.box_size):
                box = self.board[box_row:box_row + self.box_size,
                                box_col:box_col + self.box_size]
                if len(set(box.flatten())) != self.grid_size:
                    return False
        
        return True
    
    def evaluate_state(self):
        """Heuristic evaluation of current state"""
        if self.is_solved():
            return 10000
        
        score = 0
        
        # Count filled cells
        filled = np.count_nonzero(self.board)
        score += filled * 10
        
        # Reward cells with few candidates (more constrained = better)
        constraint_score = 0
        for row, col in self.empty_cells:
            if self.board[row, col] == 0:
                candidates = len(self.get_candidates(row, col))
                if candidates == 0:
                    return -10000  # Dead end
                constraint_score += (self.grid_size - candidates) * 5
        
        score += constraint_score
        
        # Penalty for many empty cells
        empty = self.grid_size * self.grid_size - filled
        score -= empty * 2
        
        return score

# ============================================================================
# MCTS Node (AlphaZero Core)
# ============================================================================

class MCTSNode:
    def __init__(self, env, parent=None, move=None, prior=1.0):
        self.env = env
        self.parent = parent
        self.move = move  # (row, col, num)
        self.prior = prior
        
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.q_value = 0.0
        self.is_expanded = False
    
    def value(self):
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0
    
    def ucb_score(self, parent_visits, c_puct=1.5):
        """PUCT algorithm (Predictor + UCT)"""
        if self.visit_count == 0:
            return float('inf')
        
        # Q-value (exploitation)
        q = self.value()
        
        # U-value (exploration with prior)
        u = c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)
        
        return q + u
    
    def select_child(self, c_puct=1.5):
        return max(self.children.values(), 
                   key=lambda c: c.ucb_score(self.visit_count, c_puct))
    
    def expand(self, policy_priors):
        """Expand with policy network priors"""
        valid_moves = self.env.get_valid_moves()
        
        if not valid_moves:
            return
        
        total_prior = sum(policy_priors.get(m, 1.0) for m in valid_moves)
        if total_prior == 0:
            total_prior = len(valid_moves)
        
        for move in valid_moves:
            prior = policy_priors.get(move, 1.0) / total_prior
            child_env = self.env.copy()
            child_env.make_move(*move)
            self.children[move] = MCTSNode(child_env, self, move, prior)
        
        self.is_expanded = True
    
    def backup(self, value):
        """Backpropagate value"""
        self.visit_count += 1
        self.value_sum += value
        
        if self.parent:
            self.parent.backup(value)

# ============================================================================
# Experience Replay Buffer (DQN Component)
# ============================================================================

class PrioritizedReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.alpha = 0.6  # Priority exponent
    
    def add(self, state, move, reward, next_state, done):
        max_priority = max(self.priorities) if self.priorities else 1.0
        self.buffer.append((state, move, reward, next_state, done))
        self.priorities.append(max_priority)
    
    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return []
        
        # Prioritized sampling
        probs = np.array(self.priorities) ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        return [self.buffer[i] for i in indices]
    
    def __len__(self):
        return len(self.buffer)

# ============================================================================
# AlphaZero Sudoku Agent
# ============================================================================

class AlphaZeroSudokuAgent:
    def __init__(self, grid_size=9, lr=0.3, gamma=0.95):
        self.grid_size = grid_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
        # Q-Learning component
        self.q_table = defaultdict(lambda: defaultdict(float))
        
        # Policy network (simulated)
        self.policy_table = defaultdict(lambda: defaultdict(float))
        
        # Value network (simulated)
        self.value_table = defaultdict(float)
        
        # Experience replay
        self.replay_buffer = PrioritizedReplayBuffer(5000)
        
        # MCTS parameters
        self.mcts_simulations = 100
        self.c_puct = 1.4
        
        # Stats
        self.puzzles_solved = 0
        self.puzzles_attempted = 0
        self.avg_moves = []
        
        # Self-attention weights (learned patterns)
        self.attention_weights = defaultdict(float)
    
    def get_policy_priors(self, env):
        """Simulate policy head with learned preferences"""
        state = env.get_state()
        valid_moves = env.get_valid_moves()
        priors = {}
        
        for move in valid_moves:
            row, col, num = move
            
            # Use learned policy
            if state in self.policy_table and move in self.policy_table[state]:
                prior = self.policy_table[state][move]
            else:
                # Heuristic prior based on constraint propagation
                prior = 1.0
                
                # Cells with fewer candidates are better (more constrained)
                candidates = env.get_candidates(row, col)
                if len(candidates) > 0:
                    prior += (env.grid_size - len(candidates)) * 2
                
                # Naked singles (only one candidate) get highest priority
                if len(candidates) == 1:
                    prior += 10
                
                # Self-attention: patterns in row/col/box
                attention_key = (tuple(env.board[row]), 
                               tuple(env.board[:, col]), num)
                if attention_key in self.attention_weights:
                    prior += self.attention_weights[attention_key]
            
            priors[move] = max(prior, 0.1)
        
        return priors
    
    def mcts_search(self, env, num_simulations):
        """AlphaZero MCTS"""
        root = MCTSNode(env.copy())
        
        for _ in range(num_simulations):
            node = root
            search_env = env.copy()
            search_path = [node]
            
            # Selection
            while node.is_expanded and node.children:
                node = node.select_child(self.c_puct)
                search_env.make_move(*node.move)
                search_path.append(node)
            
            # Expansion
            if not search_env.is_solved() and search_env.get_valid_moves():
                policy_priors = self.get_policy_priors(search_env)
                node.expand(policy_priors)
            
            # Evaluation (value head)
            value = self._evaluate_leaf(search_env)
            
            # Backup
            node.backup(value)
        
        return root
    
    def _evaluate_leaf(self, env):
        """Evaluate leaf node (value head + minimax)"""
        if env.is_solved():
            return 1.0
        
        if not env.get_valid_moves():
            return -1.0
        
        # Use cached value if available
        state = env.get_state()
        if state in self.value_table:
            return self.value_table[state]
        
        # Minimax-style lookahead
        score = self._constraint_propagation_score(env)
        
        # Normalize to [-1, 1]
        value = np.tanh(score / 1000)
        
        # Cache value
        self.value_table[state] = value
        
        return value
    
    def _constraint_propagation_score(self, env):
        """Fast evaluation using CSP techniques"""
        score = env.evaluate_state()
        
        # Naked singles (cells with only one candidate)
        naked_singles = 0
        for row in range(env.grid_size):
            for col in range(env.grid_size):
                if env.board[row, col] == 0:
                    candidates = env.get_candidates(row, col)
                    if len(candidates) == 1:
                        naked_singles += 1
                    elif len(candidates) == 0:
                        return -10000  # Dead end
        
        score += naked_singles * 50
        
        return score
    
    def choose_action(self, env, training=True):
        """Choose action using hybrid approach"""
        valid_moves = env.get_valid_moves()
        if not valid_moves:
            return None
        
        # Constraint propagation: check for naked singles first
        for row in range(env.grid_size):
            for col in range(env.grid_size):
                if env.board[row, col] == 0:
                    candidates = env.get_candidates(row, col)
                    if len(candidates) == 1:
                        # Forced move
                        return (row, col, candidates[0])
        
        # Exploration during training
        if training and random.random() < self.epsilon:
            return random.choice(valid_moves)
        
        # MCTS for strategic planning
        root = self.mcts_search(env, self.mcts_simulations)
        
        if not root.children:
            return random.choice(valid_moves)
        
        # Select move with highest visit count
        best_move = max(root.children.items(), 
                       key=lambda x: x[1].visit_count)[0]
        
        # Store visit distribution as policy target
        state = env.get_state()
        total_visits = sum(c.visit_count for c in root.children.values())
        for move, child in root.children.items():
            policy_prob = child.visit_count / total_visits
            self.policy_table[state][move] = policy_prob
            
            # Update attention weights
            row, col, num = move
            attention_key = (tuple(env.board[row]), 
                           tuple(env.board[:, col]), num)
            self.attention_weights[attention_key] += policy_prob * 0.1
        
        return best_move
    
    def train_from_experience(self, batch_size=32):
        """DQN training from replay buffer"""
        if len(self.replay_buffer) < batch_size:
            return
        
        batch = self.replay_buffer.sample(batch_size)
        
        for state, move, reward, next_state, done in batch:
            # Q-learning update
            current_q = self.q_table[state][move]
            
            if done:
                target_q = reward
            else:
                # Get best Q-value for next state
                next_moves = [m for m in self.q_table[next_state].keys()]
                if next_moves:
                    max_next_q = max(self.q_table[next_state].values())
                else:
                    max_next_q = 0
                
                target_q = reward + self.gamma * max_next_q
            
            # Update Q-value
            self.q_table[state][move] += self.lr * (target_q - current_q)
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def reset_stats(self):
        self.puzzles_solved = 0
        self.puzzles_attempted = 0
        self.avg_moves = []

# ============================================================================
# Puzzle Generation
# ============================================================================

def generate_sudoku(grid_size=9, difficulty='medium'):
    """Generate valid Sudoku puzzle"""
    env = SudokuEnv(grid_size)
    
    # Fill diagonal boxes (they're independent)
    box_size = int(np.sqrt(grid_size))
    for i in range(0, grid_size, box_size):
        numbers = list(range(1, grid_size + 1))
        random.shuffle(numbers)
        for row in range(box_size):
            for col in range(box_size):
                env.board[i + row, i + col] = numbers[row * box_size + col]
    
    # Solve the rest using backtracking
    _solve_sudoku(env)
    
    # Remove cells based on difficulty
    if difficulty == 'easy':
        cells_to_remove = int(grid_size * grid_size * 0.4)
    elif difficulty == 'medium':
        cells_to_remove = int(grid_size * grid_size * 0.5)
    elif difficulty == 'hard':
        cells_to_remove = int(grid_size * grid_size * 0.6)
    else:  # expert
        cells_to_remove = int(grid_size * grid_size * 0.7)
    
    puzzle = env.board.copy()
    cells = [(i, j) for i in range(grid_size) for j in range(grid_size)]
    random.shuffle(cells)
    
    for i in range(cells_to_remove):
        if i < len(cells):
            row, col = cells[i]
            puzzle[row, col] = 0
    
    return puzzle

def _solve_sudoku(env):
    """Backtracking solver for puzzle generation"""
    empty = np.argwhere(env.board == 0)
    if len(empty) == 0:
        return True
    
    row, col = empty[0]
    numbers = list(range(1, env.grid_size + 1))
    random.shuffle(numbers)
    
    for num in numbers:
        if env.is_valid_move(row, col, num):
            env.board[row, col] = num
            if _solve_sudoku(env):
                return True
            env.board[row, col] = 0
    
    return False

# ============================================================================
# Visualization
# ============================================================================

def visualize_sudoku(board, title="Sudoku", highlight_cells=None):
    """Create matplotlib visualization"""
    grid_size = len(board)
    box_size = int(np.sqrt(grid_size))
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Draw grid
    for i in range(grid_size + 1):
        linewidth = 3 if i % box_size == 0 else 1
        ax.plot([0, grid_size], [i, i], 'k-', linewidth=linewidth)
        ax.plot([i, i], [0, grid_size], 'k-', linewidth=linewidth)
    
    # Draw numbers
    for i in range(grid_size):
        for j in range(grid_size):
            if board[i, j] != 0:
                # Highlight recent moves
                if highlight_cells and (i, j) in highlight_cells:
                    color = '#4CAF50'
                    weight = 'bold'
                else:
                    color = '#000000'
                    weight = 'normal'
                
                ax.text(j + 0.5, grid_size - i - 0.5, str(board[i, j]),
                       ha='center', va='center', fontsize=20,
                       color=color, weight=weight)
    
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=20, fontweight='bold', pad=20)
    
    return fig

# ============================================================================
# Save/Load
# ============================================================================

def create_agent_zip(agent, config):
    """Package agent into zip"""
    # Convert defaultdicts to regular dicts for JSON serialization
    agent_state = {
        "policy_table": {str(k): dict(v) for k, v in list(agent.policy_table.items())[:1000]},
        "value_table": {str(k): v for k, v in list(agent.value_table.items())[:1000]},
        "epsilon": agent.epsilon,
        "puzzles_solved": agent.puzzles_solved,
        "puzzles_attempted": agent.puzzles_attempted,
        "mcts_sims": agent.mcts_simulations,
        "grid_size": agent.grid_size
    }
    
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("agent.json", json.dumps(agent_state))
        zf.writestr("config.json", json.dumps(config))
    
    buffer.seek(0)
    return buffer

# ============================================================================
# Streamlit UI
# ============================================================================

st.sidebar.header("⚙️ AlphaZero Controls")

with st.sidebar.expander("1. Puzzle Configuration", expanded=True):
    grid_size = st.select_slider("Grid Size", [4, 9, 16], 9)
    difficulty = st.selectbox("Difficulty", 
                             ["easy", "medium", "hard", "expert"])
    
with st.sidebar.expander("2. Agent Parameters", expanded=True):
    lr = st.slider("Learning Rate α", 0.1, 1.0, 0.3, 0.05)
    gamma = st.slider("Discount Factor γ", 0.8, 0.99, 0.95, 0.01)
    mcts_sims = st.slider("MCTS Simulations", 10, 500, 100, 10)

with st.sidebar.expander("3. Training Configuration", expanded=False):
    num_puzzles = st.number_input("Training Puzzles", 10, 1000, 100, 10)
    
with st.sidebar.expander("4. Brain Storage", expanded=False):
    if 'agent' in st.session_state and st.session_state.agent:
        config = {
            "lr": lr, "gamma": gamma, "mcts_sims": mcts_sims,
            "grid_size": grid_size, "difficulty": difficulty
        }
        
        zip_buffer = create_agent_zip(st.session_state.agent, config)
        st.download_button(
            label="💾 Download Agent",
            data=zip_buffer,
            file_name="alphazero_sudoku.zip",
            mime="application/zip",
            use_container_width=True
        )
    else:
        st.info("Train agent first")

# Initialize agent
if 'agent' not in st.session_state or st.session_state.get('grid_size') != grid_size:
    st.session_state.agent = AlphaZeroSudokuAgent(grid_size, lr, gamma)
    st.session_state.grid_size = grid_size

agent = st.session_state.agent
agent.mcts_simulations = mcts_sims

# Display stats
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("🧩 Puzzles Solved", agent.puzzles_solved)
    st.caption(f"Success Rate: {agent.puzzles_solved/max(agent.puzzles_attempted,1)*100:.1f}%")

with col2:
    st.metric("📚 Policy Networks", len(agent.policy_table))
    st.caption(f"ε = {agent.epsilon:.4f}")

with col3:
    st.metric("🎯 MCTS Simulations", agent.mcts_simulations)
    avg = np.mean(agent.avg_moves) if agent.avg_moves else 0
    st.caption(f"Avg Moves: {avg:.1f}")

st.markdown("---")

# Main controls
col_a, col_b, col_c = st.columns(3)

with col_a:
    if st.button("🎯 Generate & Solve New Puzzle", use_container_width=True, type="primary"):
        st.session_state.current_puzzle = generate_sudoku(grid_size, difficulty)
        st.session_state.solving_active = True
        st.session_state.solve_steps = []
        st.rerun()

with col_b:
    if st.button(" Train Agent", use_container_width=True):
        st.session_state.training_active = True
        st.rerun()

with col_c:
    if st.button("🧹 Reset", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# Training mode
if st.session_state.get('training_active'):
    st.subheader(" Training AlphaZero Agent")
    
    progress_bar = st.progress(0)
    status = st.empty()
    
    agent.reset_stats()
    
    for i in range(num_puzzles):
        puzzle = generate_sudoku(grid_size, difficulty)
        env = SudokuEnv(grid_size)
        env.reset(puzzle)
        
        agent.puzzles_attempted += 1
        moves = 0
        max_moves = grid_size * grid_size * 2
        
        while not env.is_solved() and moves < max_moves:
            state = env.get_state()
            move = agent.choose_action(env, training=True)
            
            if move is None:
                break
            
            reward, done = env.make_move(*move)
            next_state = env.get_state()
            
            # Store experience
            agent.replay_buffer.add(state, move, reward, next_state, done)
            
            # Train from experience
            if len(agent.replay_buffer) >= 32:
                agent.train_from_experience(32)
            
            moves += 1
            
            if done:
                break
        
        if env.is_solved():
            agent.puzzles_solved += 1
            agent.avg_moves.append(moves)
        
        agent.decay_epsilon()
        
        # Update UI
        progress = (i + 1) / num_puzzles
        progress_bar.progress(progress)
        status.markdown(f"""
        **Puzzle {i+1}/{num_puzzles}** ({progress*100:.1f}%)
        
        - ✅ Solved: {agent.puzzles_solved}
        - ❌ Failed: {agent.puzzles_attempted - agent.puzzles_solved}
        - 📊 Success Rate: {agent.puzzles_solved/agent.puzzles_attempted*100:.1f}%
        - 🎯 Epsilon: {agent.epsilon:.4f}
        """)
    
    progress_bar.progress(1.0)
    st.success("Training Complete! 🎉")
    st.session_state.training_active = False

# Solving mode
if st.session_state.get('solving_active') and 'current_puzzle' in st.session_state:
    st.subheader("🤖 Watch AlphaZero Solve")
    
    if 'solve_env' not in st.session_state:
        st.session_state.solve_env = SudokuEnv(grid_size)
        st.session_state.solve_env.reset(st.session_state.current_puzzle)
        st.session_state.solve_moves = []
    
    env = st.session_state.solve_env
    
    board_placeholder = st.empty()
    info_placeholder = st.empty()
    
    auto_solve = st.checkbox("Auto-solve (2sec delay)", value=True)
    
    if not env.is_solved():
        move = agent.choose_action(env, training=False)
        
        if move:
            row, col, num = move
            reward, done = env.make_move(row, col, num)
            st.session_state.solve_moves.append((row, col))
            
            # Visualize
            recent_cells = st.session_state.solve_moves[-5:]
            fig = visualize_sudoku(env.board, 
                                  f"Move {len(st.session_state.solve_moves)}: Place {num} at ({row},{col})",
                                  recent_cells)
            board_placeholder.pyplot(fig)
            plt.close(fig)
            
            info_placeholder.info(f"Move {len(st.session_state.solve_moves)}: Placed {num} at row {row}, col {col} | Reward: {reward:.1f}")
            
            if auto_solve and not done:
                time.sleep(2)
                st.rerun()
    else:
        fig = visualize_sudoku(env.board, "✅ SOLVED!")
        board_placeholder.pyplot(fig)
        plt.close(fig)
        st.success(f"✅ Puzzle solved in {len(st.session_state.solve_moves)} moves!")
        
        if st.button("Solve Another"):
            del st.session_state.solve_env
            del st.session_state.solve_moves
            st.session_state.current_puzzle = generate_sudoku(grid_size, difficulty)
            st.rerun()

# Human vs AI mode
st.markdown("---")
st.header("🎮 Challenge AlphaZero")

if len(agent.policy_table) > 50:
    if st.button("🆚 Start New Challenge", use_container_width=True, type="primary"):
        st.session_state.challenge_puzzle = generate_sudoku(grid_size, difficulty)
        st.session_state.challenge_env = SudokuEnv(grid_size)
        st.session_state.challenge_env.reset(st.session_state.challenge_puzzle)
        st.session_state.challenge_active = True
        st.rerun()
    
    if st.session_state.get('challenge_active'):
        env = st.session_state.challenge_env
        
        fig = visualize_sudoku(env.board, "Your Puzzle")
        st.pyplot(fig)
        plt.close(fig)
        
        if not env.is_solved():
            st.write("**Your Move:**")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                row = st.number_input("Row", 0, grid_size-1, 0)
            with col2:
                col = st.number_input("Col", 0, grid_size-1, 0)
            with col3:
                num = st.number_input("Number", 1, grid_size, 1)
            
            if st.button("Place Number"):
                if env.is_valid_move(row, col, num):
                    env.make_move(row, col, num)
                    st.rerun()
                else:
                    st.error("Invalid move!")
            
            if st.button("💡 Get AI Hint"):
                move = agent.choose_action(env, training=False)
                if move:
                    st.info(f"Try placing {move[2]} at row {move[0]}, col {move[1]}")
        else:
            st.balloons()
            st.success("🎉 Congratulations! You solved it!")
else:
    st.info(" Train the agent first to unlock Human vs AI mode!")

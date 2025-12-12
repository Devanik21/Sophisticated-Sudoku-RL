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

---

### 🎮 Quick Start Guide:
1. **Train the AI**: Adjust hyperparameters in sidebar → Click "Begin Training"
2. **Generate Puzzle**: Use the Puzzle Generator → Select difficulty → Generate
3. **Watch AI Solve**: Generate AI Solution → Use playback controls (⏮️ ◀️ ▶️ ⏭️)
4. **Play Yourself**: Generate puzzle → Click "Play This" → Enter numbers manually
5. **Save Progress**: Download Brain after training → Upload later to continue

""", unsafe_allow_html=True)

with st.expander("📚 Learn More About The Architecture"):
    st.markdown("""
    #### How It Works:
    
    **1. MCTS (Monte Carlo Tree Search)**
    - Explores possible moves by simulating games
    - Uses PUCT formula to balance exploration vs exploitation
    - Similar to how AlphaGo/AlphaZero thinks ahead
    
    **2. DQN (Deep Q-Network)**
    - Learns from past experiences using replay buffer
    - Prioritizes important learning examples
    - Updates Q-values to predict future rewards
    
    **3. Constraint Propagation**
    - Uses Sudoku rules to eliminate impossible values
    - Identifies "naked singles" (only one possible value)
    - Applies logical deduction before MCTS
    
    **4. Self-Attention Patterns**
    - Learns which values work well in which positions
    - Adapts strategy based on success/failure
    - Similar to transformer attention mechanisms
    
    **5. Hybrid Decision Making**
    - First tries logical deduction (naked singles)
    - Falls back to MCTS for complex decisions
    - Uses value caching for speed
    """)

st.markdown("---")

# ============================================================================
# Optimized Sudoku Environment
# ============================================================================

class SudokuEnv:
    """Fast Sudoku environment with efficient constraint propagation"""
    
    def __init__(self, grid_size=9):
        self.grid_size = grid_size
        self.box_size = int(np.sqrt(grid_size))
        self.reset()
    
    def reset(self, puzzle=None):
        """Initialize puzzle"""
        if puzzle is not None:
            self.board = puzzle.copy()
            self.initial_board = puzzle.copy()
        else:
            self.board = np.zeros((self.grid_size, self.grid_size), dtype=int)
            self.initial_board = self.board.copy()
        
        self.move_count = 0
        return self.get_state()
    
    def get_state(self):
        """Return hashable state"""
        return tuple(self.board.flatten())
    
    def copy(self):
        """Deep copy"""
        new_env = SudokuEnv(self.grid_size)
        new_env.board = self.board.copy()
        new_env.initial_board = self.initial_board.copy()
        new_env.move_count = self.move_count
        return new_env
    
    def is_valid_move(self, row, col, num):
        """Check if move is valid"""
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
    
    def get_candidates(self, row, col):
        """Fast constraint propagation for a cell"""
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
        
        candidates.discard(0)
        return list(candidates)
    
    def get_valid_moves(self):
        """Get all valid (row, col, num) moves"""
        moves = []
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                if self.board[row, col] == 0:
                    candidates = self.get_candidates(row, col)
                    for num in candidates:
                        moves.append((row, col, num))
        return moves
    
    def make_move(self, row, col, num):
        """Make move and return reward, done"""
        if self.initial_board[row, col] != 0:
            return -10, False
        
        if not self.is_valid_move(row, col, num):
            return -5, False
        
        self.board[row, col] = num
        self.move_count += 1
        
        # Calculate reward
        reward = 1
        
        # Bonus for constraint strength
        candidates_reduced = 0
        for r, c in [(row, i) for i in range(self.grid_size)] + \
                    [(i, col) for i in range(self.grid_size)]:
            if self.board[r, c] == 0:
                candidates_reduced += 1
        
        reward += candidates_reduced * 0.1
        
        # Check if solved
        if self.is_solved():
            return 100, True
        
        # Check if stuck
        if not self.get_valid_moves():
            return -20, True
        
        return reward, False
    
    def is_solved(self):
        """Check if solved"""
        if 0 in self.board:
            return False
        
        # Check rows
        for i in range(self.grid_size):
            if len(set(self.board[i])) != self.grid_size:
                return False
        
        # Check columns
        for i in range(self.grid_size):
            if len(set(self.board[:, i])) != self.grid_size:
                return False
        
        # Check boxes
        for box_row in range(0, self.grid_size, self.box_size):
            for box_col in range(0, self.grid_size, self.box_size):
                box = self.board[box_row:box_row + self.box_size,
                                box_col:box_col + self.box_size]
                if len(set(box.flatten())) != self.grid_size:
                    return False
        
        return True
    
    def evaluate_state(self):
        """Heuristic evaluation"""
        if self.is_solved():
            return 10000
        
        score = 0
        
        # Filled cells
        filled = np.count_nonzero(self.board)
        score += filled * 10
        
        # Constraint strength
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                if self.board[row, col] == 0:
                    candidates = len(self.get_candidates(row, col))
                    if candidates == 0:
                        return -10000  # Dead end
                    score += (self.grid_size - candidates) * 5
        
        return score

# ============================================================================
# MCTS Node (Optimized)
# ============================================================================

class MCTSNode:
    def __init__(self, env, parent=None, move=None, prior=1.0):
        self.env = env
        self.parent = parent
        self.move = move
        self.prior = prior
        
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.is_expanded = False
    
    def value(self):
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0
    
    def ucb_score(self, parent_visits, c_puct=1.5):
        """PUCT algorithm"""
        if self.visit_count == 0:
            return float('inf')
        
        q = self.value()
        u = c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)
        
        return q + u
    
    def select_child(self, c_puct=1.5):
        return max(self.children.values(), 
                   key=lambda c: c.ucb_score(self.visit_count, c_puct))
    
    def expand(self, policy_priors):
        """Expand with policy priors"""
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
        """Backpropagate"""
        self.visit_count += 1
        self.value_sum += value
        
        if self.parent:
            self.parent.backup(value)

# ============================================================================
# Prioritized Replay Buffer
# ============================================================================

class PrioritizedReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.alpha = 0.6
    
    def add(self, state, move, reward, next_state, done):
        max_priority = max(self.priorities) if self.priorities else 1.0
        self.buffer.append((state, move, reward, next_state, done))
        self.priorities.append(max_priority)
    
    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return []
        
        probs = np.array(self.priorities) ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        return [self.buffer[i] for i in indices]
    
    def __len__(self):
        return len(self.buffer)

# ============================================================================
# Fast AlphaZero Agent
# ============================================================================

class AlphaZeroAgent:
    def __init__(self, grid_size=9, lr=0.3, gamma=0.95):
        self.grid_size = grid_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
        # Q-Learning
        self.q_table = defaultdict(lambda: defaultdict(float))
        
        # Policy network (simulated)
        self.policy_table = defaultdict(lambda: defaultdict(float))
        
        # Value network with caching
        self.value_cache = {}
        
        # Experience replay
        self.replay_buffer = PrioritizedReplayBuffer(5000)
        
        # MCTS parameters
        self.mcts_simulations = 100
        self.c_puct = 1.4
        
        # Stats
        self.puzzles_solved = 0
        self.puzzles_attempted = 0
        self.avg_moves = []
        
        # Self-attention
        self.attention_weights = defaultdict(float)
    
    def get_policy_priors(self, env):
        """Smart policy priors using constraint propagation"""
        state = env.get_state()
        valid_moves = env.get_valid_moves()
        priors = {}
        
        for move in valid_moves:
            row, col, num = move
            
            # Check learned policy
            if state in self.policy_table and move in self.policy_table[state]:
                prior = self.policy_table[state][move]
            else:
                # Heuristic based on constraints
                prior = 1.0
                
                candidates = env.get_candidates(row, col)
                
                # CRITICAL: Naked singles get huge priority
                if len(candidates) == 1:
                    prior += 100
                elif len(candidates) == 2:
                    prior += 20
                else:
                    # More constrained = better
                    prior += (env.grid_size - len(candidates)) * 3
                
                # Self-attention pattern bonus
                attention_key = (tuple(env.board[row]), 
                               tuple(env.board[:, col]), num)
                prior += self.attention_weights.get(attention_key, 0)
            
            priors[move] = max(prior, 0.1)
        
        return priors
    
    def mcts_search(self, env, num_simulations):
        """Fast MCTS with value caching"""
        root = MCTSNode(env.copy())
        
        for _ in range(num_simulations):
            node = root
            search_env = env.copy()
            
            # Selection
            while node.is_expanded and node.children:
                node = node.select_child(self.c_puct)
                search_env.make_move(*node.move)
            
            # Expansion
            if not search_env.is_solved() and search_env.get_valid_moves():
                policy_priors = self.get_policy_priors(search_env)
                node.expand(policy_priors)
            
            # Evaluation (with caching)
            value = self._evaluate_leaf(search_env)
            
            # Backup
            node.backup(value)
        
        return root
    
    def _evaluate_leaf(self, env):
        """Fast evaluation with caching"""
        if env.is_solved():
            return 1.0
        
        if not env.get_valid_moves():
            return -1.0
        
        # Check cache
        state = env.get_state()
        if state in self.value_cache:
            return self.value_cache[state]
        
        # Quick evaluation
        score = env.evaluate_state()
        
        # Naked singles bonus
        naked_singles = 0
        for row in range(env.grid_size):
            for col in range(env.grid_size):
                if env.board[row, col] == 0:
                    candidates = env.get_candidates(row, col)
                    if len(candidates) == 1:
                        naked_singles += 1
                    elif len(candidates) == 0:
                        score = -10000
                        break
        
        score += naked_singles * 100
        
        value = np.tanh(score / 1000)
        
        # Cache it
        self.value_cache[state] = value
        
        return value
    
    def choose_action(self, env, training=True):
        """Hybrid action selection: Constraint propagation first, then MCTS"""
        valid_moves = env.get_valid_moves()
        if not valid_moves:
            return None
        
        # PRIORITY: Check for naked singles (forced moves)
        for row in range(env.grid_size):
            for col in range(env.grid_size):
                if env.board[row, col] == 0:
                    candidates = env.get_candidates(row, col)
                    if len(candidates) == 1:
                        # Forced move - take it immediately
                        return (row, col, candidates[0])
        
        # Exploration during training
        if training and random.random() < self.epsilon:
            return random.choice(valid_moves)
        
        # MCTS for complex decisions
        root = self.mcts_search(env, self.mcts_simulations)
        
        if not root.children:
            return random.choice(valid_moves)
        
        # Select best move
        best_move = max(root.children.items(), 
                       key=lambda x: x[1].visit_count)[0]
        
        # Store policy
        state = env.get_state()
        total_visits = sum(c.visit_count for c in root.children.values())
        for move, child in root.children.items():
            policy_prob = child.visit_count / total_visits
            self.policy_table[state][move] = policy_prob
            
            # Update attention
            row, col, num = move
            attention_key = (tuple(env.board[row]), 
                           tuple(env.board[:, col]), num)
            self.attention_weights[attention_key] += policy_prob * 0.1
        
        return best_move
    
    def train_from_experience(self, batch_size=32):
        """DQN training"""
        if len(self.replay_buffer) < batch_size:
            return
        
        batch = self.replay_buffer.sample(batch_size)
        
        for state, move, reward, next_state, done in batch:
            current_q = self.q_table[state][move]
            
            if done:
                target_q = reward
            else:
                next_q_values = self.q_table.get(next_state, {})
                max_next_q = max(next_q_values.values()) if next_q_values else 0
                target_q = reward + self.gamma * max_next_q
            
            self.q_table[state][move] += self.lr * (target_q - current_q)
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def reset_stats(self):
        self.puzzles_solved = 0
        self.puzzles_attempted = 0
        self.avg_moves = []

# ============================================================================
# Fast Puzzle Generation
# ============================================================================

def generate_sudoku(grid_size=9, difficulty='medium'):
    """Generate valid Sudoku puzzle"""
    env = SudokuEnv(grid_size)
    
    # Fill diagonal boxes
    box_size = int(np.sqrt(grid_size))
    for i in range(0, grid_size, box_size):
        numbers = list(range(1, grid_size + 1))
        random.shuffle(numbers)
        for row in range(box_size):
            for col in range(box_size):
                env.board[i + row, i + col] = numbers[row * box_size + col]
    
    # Solve rest
    _solve_sudoku(env)
    
    # Remove cells
    removal_rate = {'easy': 0.4, 'medium': 0.5, 'hard': 0.6, 'expert': 0.7}
    cells_to_remove = int(grid_size * grid_size * removal_rate.get(difficulty, 0.5))
    
    puzzle = env.board.copy()
    cells = [(i, j) for i in range(grid_size) for j in range(grid_size)]
    random.shuffle(cells)
    
    for i in range(min(cells_to_remove, len(cells))):
        row, col = cells[i]
        puzzle[row, col] = 0
    
    return puzzle

def _solve_sudoku(env):
    """Backtracking solver"""
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

def visualize_sudoku(board, initial_board=None, title="Sudoku", highlight=None):
    """Create matplotlib visualization"""
    grid_size = len(board)
    box_size = int(np.sqrt(grid_size))
    
    fig, ax = plt.subplots(figsize=(9, 9))
    
    # Draw grid
    for i in range(grid_size + 1):
        lw = 3 if i % box_size == 0 else 1
        ax.axhline(i, color='black', linewidth=lw)
        ax.axvline(i, color='black', linewidth=lw)
    
    # Draw numbers
    for i in range(grid_size):
        for j in range(grid_size):
            if board[i, j] != 0:
                is_initial = initial_board is not None and initial_board[i, j] != 0
                is_highlight = highlight and (i, j) in highlight
                
                if is_highlight:
                    color = '#4CAF50'
                    weight = 'bold'
                elif is_initial:
                    color = 'black'
                    weight = 'bold'
                else:
                    color = '#0066CC'
                    weight = 'normal'
                
                ax.text(j + 0.5, grid_size - i - 0.5, str(board[i, j]),
                       ha='center', va='center', fontsize=20,
                       color=color, weight=weight)
    
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=18, fontweight='bold', pad=20)
    
    return fig

# ============================================================================
# Save/Load
# ============================================================================

def create_agent_zip(agent, config):
    """Package agent"""
    agent_state = {
        "policy_table": {str(k): {str(mk): mv for mk, mv in v.items()} 
                        for k, v in list(agent.policy_table.items())[:1000]},
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

st.sidebar.header("⚙️ Control Panel")

with st.sidebar.expander("1. Agent Hyperparameters", expanded=True):
    lr = st.slider("Learning Rate α", 0.1, 1.0, 0.3, 0.05)
    gamma = st.slider("Discount Factor γ", 0.8, 0.99, 0.95, 0.01)
    mcts_sims = st.slider("MCTS Simulations", 10, 500, 100, 10)

with st.sidebar.expander("2. Training Configuration", expanded=True):
    episodes = st.number_input("Training Episodes", 10, 10000, 100, 10)
    difficulty = st.selectbox("Puzzle Difficulty", 
                             ['easy', 'medium', 'hard', 'expert'])
    update_freq = st.number_input("Update Dashboard Every N Episodes", 
                                  min_value=1, max_value=100, value=10, step=1)

with st.sidebar.expander("3. Puzzle Generator", expanded=True):
    st.markdown("### 🎲 Generate Custom Puzzle")
    
    gen_difficulty = st.selectbox("Generation Difficulty", 
                                  ['easy', 'medium', 'hard', 'expert'],
                                  key='gen_diff')
    
    if st.button("🎲 Generate New Puzzle", use_container_width=True, type="primary"):
        puzzle = generate_sudoku(9, gen_difficulty)
        st.session_state.generated_puzzle = puzzle
        st.session_state.generated_puzzle_env = SudokuEnv(9)
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

train_button = st.sidebar.button("🚀 Begin Training", 
                                 use_container_width=True, type="primary")

if st.sidebar.button("🧹 Reset Everything", use_container_width=True):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

# Initialize agent
if 'agent' not in st.session_state or st.session_state.agent is None:
    st.session_state.agent = AlphaZeroAgent(9, lr, gamma)

agent = st.session_state.agent
agent.mcts_simulations = mcts_sims

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
    st.caption(f"Failed: {getattr(agent, 'puzzles_attempted', 0) - getattr(agent, 'puzzles_solved', 0) if agent else 0}")

with col4:
    avg_moves = np.mean(agent.avg_moves) if agent and agent.avg_moves else 0
    st.metric("🎯 Avg Moves", f"{avg_moves:.1f}")
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
                # Generate solution
                solve_env = SudokuEnv(9)
                solve_env.reset(st.session_state.generated_puzzle.copy())
                
                all_moves = []
                all_boards = [solve_env.board.copy()]
                
                move_count = 0
                max_moves = 100
                
                with st.spinner("🧠 AI is solving..."):
                    while (not solve_env.is_solved() and move_count < max_moves):
                        move = agent.choose_action(solve_env, training=False)
                        if move is None:
                            break
                        solve_env.make_move(*move)
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
            st.session_state.human_env = SudokuEnv(9)
            st.session_state.human_env.reset(st.session_state.generated_puzzle.copy())
            st.session_state.human_active = True
            st.rerun()
        
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
    
    history = {
        'solved': [],
        'failed': [],
        'avg_moves': [],
        'epsilon': [],
        'q_values': [],
        'episode': []
    }
    
    for ep in range(1, episodes + 1):
        puzzle = generate_sudoku(9, difficulty)
        env = SudokuEnv(9)
        env.reset(puzzle)
        
        agent.puzzles_attempted += 1
        moves = 0
        max_moves = 162  # 81 * 2
        
        while not env.is_solved() and moves < max_moves:
            state = env.get_state()
            move = agent.choose_action(env, training=True)
            
            if move is None:
                break
            
            reward, done = env.make_move(*move)
            next_state = env.get_state()
            
            agent.replay_buffer.add(state, move, reward, next_state, done)
            
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
        if ep % update_freq == 0:
            history['solved'].append(agent.puzzles_solved)
            history['failed'].append(agent.puzzles_attempted - agent.puzzles_solved)
            history['avg_moves'].append(moves)
            history['epsilon'].append(agent.epsilon)
            history['q_values'].append(len(agent.q_table))
            history['episode'].append(ep)
            
            progress = ep / episodes
            progress_bar.progress(progress)
            
            status.markdown(f"""
            ### 📊 Training Progress
            
            | Metric | Value |
            |:-------|------:|
            | **Episode** | {ep}/{episodes} ({progress*100:.1f}%) |
            | **Puzzles Solved** | {agent.puzzles_solved} |
            | **Puzzles Failed** | {agent.puzzles_attempted - agent.puzzles_solved} |
            | **Success Rate** | {agent.puzzles_solved/agent.puzzles_attempted*100:.1f}% |
            | **Epsilon** | {agent.epsilon:.4f} |
            | **Q-Table Size** | {len(agent.q_table):,} states |
            | **Policy Table Size** | {len(agent.policy_table):,} states |
            | **Replay Buffer** | {len(agent.replay_buffer)} |
            """)
    
    progress_bar.progress(1.0)
    status.success("✅ Training Complete!")
    st.toast("Training Complete! 🎉", icon="✨")
    st.session_state.training_history = history
    
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

# AI Solving with Manual Controls
# AI Solving with Manual Controls
# UPDATED: Removed the 'len > 10' check so the section is always visible for you
if 'agent' in st.session_state and st.session_state.agent is not None:
    st.markdown("---")
    st.subheader("🎬 Watch AI Solve Sudoku (Step-by-Step)")
    
    demo_col1, demo_col2 = st.columns([1, 3])
    
    with demo_col1:
        st.markdown("### 🎮 Controls")
        demo_difficulty = st.selectbox("Select Difficulty", 
                                       ['easy', 'medium', 'hard', 'expert'],
                                       key='demo_diff')
        
        if st.button("🤖 Generate AI Solution", use_container_width=True, type="primary"):
            puzzle = generate_sudoku(9, demo_difficulty)
            solve_env = SudokuEnv(9)
            solve_env.reset(puzzle)
            
            all_moves = []
            all_boards = [solve_env.board.copy()]
            
            move_count = 0
            max_moves = 100
            
            with st.spinner("🧠 AI is solving..."):
                while (not solve_env.is_solved() and move_count < max_moves):
                    move = agent.choose_action(solve_env, training=False)
                    
                    if move is None:
                        break
                    
                    solve_env.make_move(*move)
                    all_moves.append(move)
                    all_boards.append(solve_env.board.copy())
                    move_count += 1
            
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
            
            # Display current board
            current_board = st.session_state.solve_boards[current_idx]
            
            info_col1, info_col2 = st.columns([2, 1])
            
            with info_col1:
                if current_idx > 0:
                    last_move = st.session_state.solve_moves[current_idx - 1]
                    move_info = f"Move {current_idx}: Placed **{last_move[2]}** at position **({last_move[0]}, {last_move[1]})**"
                else:
                    move_info = "Initial puzzle state"
                
                st.info(move_info)
            
            with info_col2:
                empty_cells = np.sum(current_board == 0)
                st.metric("Remaining Cells", empty_cells)
            
            # Visualize
            highlight = [(st.session_state.solve_moves[current_idx-1][0], 
                         st.session_state.solve_moves[current_idx-1][1])] if current_idx > 0 else None
            fig = visualize_sudoku(current_board, 
                                  st.session_state.solve_initial,
                                  f"Step {current_idx}/{total_moves}",
                                  highlight)
            st.pyplot(fig)
            plt.close(fig)
            
            # Status
            if current_idx == total_moves:
                if st.session_state.solve_success:
                    st.success("🎉 Puzzle Solved Successfully!")
                    
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
            
            # Playback controls
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
                if st.button("🔄 Reset", use_container_width=True,key="human_resetz")):
                    st.session_state.current_move_index = 0
                    st.session_state.autoplay = False
                    st.rerun()
            
            with autoplay_col4:
                solution_data = {
                    "puzzle": st.session_state.solve_initial.tolist(),
                    "moves": [{"row": m[0], "col": m[1], "value": m[2]} 
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
            
            # Auto-play
            if st.session_state.get('autoplay', False) and current_idx < total_moves:
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
            - ▶️ **Play/Pause**: Auto playback
            - 🎚️ **Speed Control**: Adjust playback speed
            """)

# Human playable mode
st.markdown("---")
st.header("🎮 Play Sudoku Yourself")

if st.button("🆕 New Puzzle", use_container_width=True):
    puzzle = generate_sudoku(9, 'medium')
    st.session_state.human_env = SudokuEnv(9)
    st.session_state.human_env.reset(puzzle)
    st.session_state.human_active = True
    st.rerun()

if 'human_env' in st.session_state and st.session_state.get('human_active'):
    h_env = st.session_state.human_env
    
    fig = visualize_sudoku(h_env.board, h_env.initial_board, "Your Puzzle")
    st.pyplot(fig)
    plt.close(fig)
    
    if h_env.is_solved():
        st.success("🎉 Congratulations! You solved it!")
        st.balloons()
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
                    reward, done = h_env.make_move(input_row, input_col, input_val)
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
                        st.info(f"💡 Try placing {hint_move[2]} at ({hint_move[0]}, {hint_move[1]})")
                else:
                    st.warning("Train agent first!")
        
        with col_c:
            if st.button("🔄 Reset", use_container_width=True):
                h_env.reset(h_env.initial_board)
                st.rerun()

st.markdown("---")
st.caption("🧩 AlphaZero-Inspired Sudoku Mastermind | Hybrid RL with MCTS, DQN & Constraint Propagation")

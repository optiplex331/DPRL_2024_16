import math
import random
import copy
import matplotlib.pyplot as plt

class Connect4:
    def __init__(self, rows=6, cols=7, board=None):
        self.rows = rows
        self.cols = cols
        if board:
            self.board = board
        else:
            self.board = [[0 for _ in range(cols)] for _ in range(rows)]
        self.current_player = 1  # 1 for AI, -1 for Opponent

    def clone(self):
        new_game = Connect4(self.rows, self.cols)
        new_game.board = copy.deepcopy(self.board)
        new_game.current_player = self.current_player
        return new_game

    def get_legal_moves(self):
        return [c for c in range(self.cols) if self.board[0][c] == 0]

    def make_move(self, col):
        for row in reversed(range(self.rows)):
            if self.board[row][col] == 0:
                self.board[row][col] = self.current_player
                self.current_player = -self.current_player
                return True
        return False

    def check_winner(self):
        board = self.board
        rows = self.rows
        cols = self.cols

        # horizontal
        for r in range(rows):
            for c in range(cols - 3):
                if board[r][c] != 0 and board[r][c] == board[r][c+1] == board[r][c+2] == board[r][c+3]:
                    return board[r][c]

        # vertical
        for r in range(rows - 3):
            for c in range(cols):
                if board[r][c] != 0 and board[r][c] == board[r+1][c] == board[r+2][c] == board[r+3][c]:
                    return board[r][c]

        # diagonal
        for r in range(rows - 3):
            for c in range(cols - 3):
                if board[r][c] != 0 and board[r][c] == board[r+1][c+1] == board[r+2][c+2] == board[r+3][c+3]:
                    return board[r][c]

        # anti-diagonal
        for r in range(rows - 3):
            for c in range(3, cols):
                if board[r][c] != 0 and board[r][c] == board[r+1][c-1] == board[r+2][c-2] == board[r+3][c-3]:
                    return board[r][c]

        # draw check
        for row in self.board:
            if any(x == 0 for x in row):
                return None

        return 0

    def is_terminal(self):
        return self.check_winner() is not None

    def get_result(self):
        return self.check_winner()


class Node:
    def __init__(self, state: Connect4, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.untried_actions = state.get_legal_moves()
        self.visits = 0
        self.value = 0.0
        self.action = action

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def best_child(self, c=math.sqrt(2)):
        best_score = -float('inf')
        best_child_node = None
        for child in self.children:
            if child.visits == 0:
                uct = float('inf')
            else:
                Q = child.value
                N = child.visits
                N_parent = self.visits
                uct = Q / N + c * math.sqrt(math.log(N_parent) / N)
            if uct > best_score:
                best_score = uct
                best_child_node = child
        return best_child_node

    def expand(self):
        if not self.untried_actions:
            return None
        action = self.untried_actions.pop()
        next_state = self.state.clone()
        success = next_state.make_move(action)
        if not success:
            return None
        child_node = Node(next_state, parent=self, action=action)
        self.children.append(child_node)
        return child_node

    def backpropagate(self, reward):
        self.visits += 1
        self.value += reward
        if self.parent:
            self.parent.backpropagate(reward)


def simulation_policy(state: Connect4):
    while not state.is_terminal():
        moves = state.get_legal_moves()
        action = random.choice(moves)
        state.make_move(action)
    result = state.get_result()
    return 1 if result == 1 else (-1 if result == -1 else 0)


def mcts_search(root_state: Connect4, iterations=2000):
    root_node = Node(root_state)
    actions = root_state.get_legal_moves()

    # Dictionaries to track changes of N and Q/N
    visit_counts = {a: [] for a in actions}
    average_values = {a: [] for a in actions}

    for i in range(iterations):
        # Selection
        current_node = root_node
        while not current_node.state.is_terminal() and current_node.is_fully_expanded():
            current_node = current_node.best_child()

        # Expansion
        if not current_node.state.is_terminal() and current_node.untried_actions:
            new_node = current_node.expand()
            if new_node is not None:
                current_node = new_node

        # Simulation
        sim_state = current_node.state.clone()
        reward = simulation_policy(sim_state)

        # Backpropagation
        current_node.backpropagate(reward)

        # Record data
        for child in root_node.children:
            action = child.action
            visit_counts[action].append(child.visits)
            avg_val = child.value / child.visits if child.visits > 0 else 0
            average_values[action].append(avg_val)

    if not root_node.children:
        # If no children, choose a random move
        return random.choice(root_state.get_legal_moves()), visit_counts, average_values

    best_child_node = max(root_node.children, key=lambda c: c.visits)
    return best_child_node.action, visit_counts, average_values


def print_board(game: Connect4):
    print("\nCurrent Board:")
    for r in range(game.rows):
        print("  " + " ".join(f"{game.board[r][c]:2}" for c in range(game.cols)))
    print("-" * 30)


def play_game():
    # Initialize a new game
    game = Connect4()
    print("[Game Start]")
    print_board(game)

    # Store data from each AI turn for plotting after the game
    # We'll keep them in a list of tuples (visit_counts, average_values, turn_index)
    ai_turn_data = []
    turn_number = 0

    while not game.is_terminal():
        if game.current_player == 1:
            # AI's turn: run MCTS
            print("\n[Game] AI Thinking...")
            move, visit_counts, average_values = mcts_search(game.clone(), iterations=5000)
            print(f"[Game] AI chooses column: {move}")
            game.make_move(move)

            turn_number += 1
            # Store the data for this AI turn
            ai_turn_data.append((visit_counts, average_values, turn_number))

        else:
            # Opponent's turn: random move
            moves = game.get_legal_moves()
            opp_move = random.choice(moves)
            print(f"[Game] Opponent chooses column: {opp_move}")
            game.make_move(opp_move)

        print_board(game)

    result = game.get_result()
    print("\n[Result]")
    if result == 1:
        print("AI Wins!")
    elif result == -1:
        print("Opponent Wins!")
    else:
        print("It's a Draw!")

    # Now plot the data for each AI turn after the game ends
    for (visit_counts, average_values, t) in ai_turn_data:
        plt.figure(figsize=(14, 6))

        # Subplot 1: Visits over iterations
        plt.subplot(1, 2, 1)
        for action, counts in visit_counts.items():
            if counts:
                plt.plot(range(1, len(counts)+1), counts, label=f"Action {action}")
        plt.title(f"Root Children Visits Over Iterations (AI Turn {t})")
        plt.xlabel("Iterations")
        plt.ylabel("Visits (N)")
        plt.legend()

        # Subplot 2: Average values (Q/N) over iterations
        plt.subplot(1, 2, 2)
        for action, values in average_values.items():
            if values:
                plt.plot(range(1, len(values)+1), values, label=f"Action {action}")
        plt.title(f"Root Children Average Values (Q/N) Over Iterations (AI Turn {t})")
        plt.xlabel("Iterations")
        plt.ylabel("Average Value (Q/N)")
        plt.legend()

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    play_game()
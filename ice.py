# "MDPs on Ice - Assignment 5"
# Ported from Java

import random
import numpy as np
import copy
import sys

GOLD_REWARD = 100.0
PIT_REWARD = -150.0
DISCOUNT_FACTOR = 0.5
EXPLORE_PROB = 0.2 # for Q-learning
LEARNING_RATE = 0.1
ITERATIONS = 10000
MAX_MOVES = 1000
ACTIONS = 4
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3
MOVES = ['U','R','D','L']

# Fixed random number generator seed for result reproducibility --
# don't use a random number generator besides this to match sol
random.seed(5100)

# Problem class:  represents the physical space, transition probabilities, reward locations,
# and approach to use (MDP or Q) - in short, the info in the text file
class Problem:
    # Fields:
    # approach - string, "MDP" or "Q"
    # move_probs - list of doubles, probability of going 1,2,3 spaces
    # map - list of list of strings: "-" (safe, empty space), "G" (gold), "P" (pit)

    # Format looks like
    # MDP    [approach to be used]
    # 0.7 0.2 0.1   [probability of going 1, 2, 3 spaces]
    # - - - - - - P - - - -   [space-delimited map rows]
    # - - G - - - - - P - -   [G is gold, P is pit]
    #
    # You can assume the maps are rectangular, although this isn't enforced
    # by this constructor.

    # __init__ consumes stdin; don't call it after stdin is consumed or outside that context
    def __init__(self):
        self.approach = input('Reading mode...')
        print(self.approach)
        probs_string = input("Reading transition probabilities...\n")
        self.move_probs = [float(s) for s in probs_string.split()]
        self.map = []
        for line in sys.stdin:
            self.map.append(line.split())

    def solve(self, iterations):            
        if self.approach == "MDP":
            return mdp_solve(self, iterations)
        elif self.approach == "Q":
            return q_solve(self, iterations)
        return None
        
# Policy: Abstraction on the best action to perform in each state - just a 2D string list-of-lists
class Policy:
    def __init__(self, problem): # problem is a Problem
        # Signal 'no policy' by just displaying the map there
        self.best_actions = copy.deepcopy(problem.map)

    def __str__(self):
        return '\n'.join([' '.join(row) for row in self.best_actions])

# roll_steps:  helper for try_policy and q_solve -- "rolls the dice" for the ice and returns
# the new location (r,c), taking map bounds into account
# note that move is expecting a string, not an integer constant
def roll_steps(move_probs, row, col, move, rows, cols):
    displacement = 1
    total_prob = 0
    move_sample = random.random()
    for p, prob in enumerate(problem.move_probs):
        total_prob += prob
        if move_sample <= total_prob:
            displacement = p+1
            break
    # Handle "slipping" into edge of map
    new_row = row
    new_col = col
    if not isinstance(move,str):
        print("Warning: roll_steps wants str for move, got a different type")
    if move == "U":
        new_row -= displacement
        if new_row < 0:
            new_row = 0
    elif move == "R":
        new_col += displacement
        if new_col >= cols:
            new_col = cols-1
    elif move == "D":
        new_row += displacement
        if new_row >= rows:
            new_row = rows-1
    elif move == "L":
        new_col -= displacement
        if new_col < 0:
            new_col = 0
    return new_row, new_col


# try_policy:  returns avg utility per move of the policy, as measured by "iterations"
# random drops of an agent onto empty spaces, running until gold, pit, or time limit 
# MAX_MOVES is reached
def try_policy(policy, problem, iterations):
    total_utility = 0
    total_moves = 0
    for i in range(iterations):
        # Resample until we have an empty starting square
        while True:
            row = random.randrange(0,len(problem.map))
            col = random.randrange(0,len(problem.map[0]))
            if problem.map[row][col] == "-":
                break
        for moves in range(MAX_MOVES):
            total_moves += 1
            policy_rec = policy.best_actions[row][col]
            # Take the move - roll to see how far we go, bump into map edges as necessary
            row, col = roll_steps(problem.move_probs, row, col, policy_rec, len(problem.map), len(problem.map[0]))
            if problem.map[row][col] == "G":
                total_utility += GOLD_REWARD
                break
            if problem.map[row][col] == "P":
                total_utility -= PIT_REWARD
                break
    return total_utility / total_moves

# mdp_solve:  use [iterations] iterations of the Bellman equations over the whole map in [problem]
# and return the policy of what action to take in each square
def mdp_solve(problem, iterations):
    policy = Policy(problem)
    num_rows = problem.map.__len__()
    num_cols = problem.map[0].__len__()
    utilities = copy.deepcopy(problem.map)
    rewards = copy.deepcopy(problem.map)
    for i in range(0, num_rows):
        for j in range(0, num_cols):
            if problem.map[i][j] == 'G':
                utilities[i][j] = GOLD_REWARD
                rewards[i][j] = GOLD_REWARD
            elif problem.map[i][j] == 'P':
                utilities[i][j] = PIT_REWARD
                rewards[i][j] = PIT_REWARD
            else:
                utilities[i][j] = 0.0
                rewards[i][j] = 0.0
    for iter in range(0, iterations):
        for i in range(0, utilities.__len__()):
            for j in range(0, utilities[0].__len__()):
                if utilities[i][j] == GOLD_REWARD:
                    policy.best_actions[i][j] = "G"
                elif utilities[i][j] == PIT_REWARD:
                    policy.best_actions[i][j] = "P"
                else:
                    expected_vals = [-1, -1, -1, -1]
                    for a in range(0, MOVES.__len__()):
                        action_val = 0
                        for b in range(0, problem.move_probs.__len__()):
                            move_spaces = b + 1
                            neighbor_r, neighbor_c = move(MOVES[a], move_spaces, i, j, rewards)
                            neighbor_utlity = utilities[neighbor_r][neighbor_c]
                            action_val += problem.move_probs[b] * neighbor_utlity
                        expected_vals[a] = action_val
                    direction = 0
                    max_val = expected_vals[0]
                    for m in range(1, expected_vals.__len__()):
                        if expected_vals[m] > max_val:
                            direction = m
                            max_val = expected_vals[m]
                    utilities[i][j] = rewards[i][j] + DISCOUNT_FACTOR * max_val
                    policy.best_actions[i][j] = direction.__str__()
    return policy

def q_solve(problem, iterations):
    policy = Policy(problem)
    num_rows = problem.map.__len__()
    num_cols = problem.map[0].__len__()
    utilities = [copy.deepcopy(MOVES)]
    for i in range(0, utilities.__len__()):
        utilities[i].append(copy.deepcopy(problem.map))
    rewards = copy.deepcopy(problem.map)
    for i in range(0, num_rows):
        for j in range(0, num_cols):
            if problem.map[i][j] == 'G':
                rewards[i][j] = GOLD_REWARD
            elif problem.map[i][j] == 'P':
                rewards[i][j] = PIT_REWARD
            else:
                rewards[i][j] = 0.0

    for iter in range(0, iterations):
        i = random.randrange(0,len(problem.map))
        j = random.randrange(0,len(problem.map[0]))

        while rewards[i][j] != GOLD_REWARD and rewards[i][j] != PIT_REWARD:
            if random.random() < EXPLORE_PROB:
                direction = random.randrange(ACTIONS)
                move_spaces = move_prob(problem)
                move_r, move_c = move(MOVES[direction], move_spaces, i, j, rewards)
                q_val = new_q(rewards, utilities, i, j, move_r, move_c, MOVES[direction])
                utilities[MOVES[direction]][i][j] = q_val
                i = move_r
                j = move_c
            else:
                best_action = 0
                best_q = -999999
                for a in range(0, ACTIONS):
                    q_val = utilities[a][i][j]
                    if q_val > best_q:
                        best_action = a
                        best_q = q_val
                spaces = move_prob(problem)
                new_r, new_c = move(best_action, spaces, i, j, rewards)
                q_val = new_q(rewards, utilities, i, j, new_r, new_c, best_action)
                utilities[best_action][i][j] = q_val
                i = new_r
                j = new_c
        for a in range(0, ACTIONS):
            utilities[a][i][j] = rewards[i][j]
    
    for row in range(0, utilities[0].__len__()):
        for col in range(0, utilities[0][0].__len__()):
            if rewards[row][col] == GOLD_REWARD:
                policy.best_actions[row][col] = "G"
            elif rewards[row][col] == PIT_REWARD:
                policy.best_actions[row][col] = "P"
            else:
                best_action = 0
                best_q = -999999
                for a in range(0, ACTIONS):
                    for move_space in range(0, problem.move_probs.__len__()):
                        new_r, new_c = move(a, move_space, row, col, rewards)
                        q_val = new_q(rewards, utilities, row, col, new_r, new_c, a)
                if q_val > best_q:
                    best_action = a
                    best_q = q_val

            policy.best_actions[row][col] = best_action.__str__()
    return policy

def new_q(rewards, utilities, r, c, new_r, new_c, move):
    max_util = -99999999
    for a in range(0, ACTIONS):
        utility = utilities[a][new_r][new_c]
        if utility > max_util:
            max_util = utility
    return utilities[move][r][c] + LEARNING_RATE * (rewards[r][c] + DISCOUNT_FACTOR * max_util - utilities[move][r][c])

def move_prob(prob):
    rand_prob = random.random()
    prob_sum = 0
    for p in range(0, prob.move_probs.__len__()):
        prob_sum += prob.move_probs[p]
        if rand_prob <= prob_sum:
            return p + 1

def move(action, spaces, r, c, rewards):
    temp_r = r
    temp_c = c
    while True:
        if action == 'U':
            temp_r -= spaces
        elif action == 'R':
            temp_c += spaces
        elif action == 'D':
            temp_r += spaces
        elif action == 'L':
            temp_c -= spaces
        else:
            print ("Illegal Move")
            return None
        if in_bounds(rewards, temp_r, temp_c):
            break
        else:
            temp_r = r
            temp_c = c
            spaces -= 1
        
    return temp_r, temp_c

def in_bounds(rewards, r, c):
    return r >= 0 and c >= 0 and r < rewards.__len__() and c < rewards[0].__len__()

# Main:  read the problem from stdin, print the policy and the utility over a test run
if __name__ == "__main__":
    problem = Problem()
    policy = problem.solve(ITERATIONS)
    print(policy)
    print("Calculating average utility...")
    print("Average utility per move: {utility:.2f}".format(utility = try_policy(policy, problem, ITERATIONS)))
        

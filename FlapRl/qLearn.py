import csv
import ast
import random
import numpy as np
import matplotlib.pyplot as plt

class qLearning:

    def __init__(self, isTrainable):
        self.isTrainable = isTrainable
        self.reward = {0: 10, 1: -1000}  # Reward function, focus on not dying and using best approach strategies
        self.discountFactor = 0.95
        self.alpha = 0.9
        self.alpha_decay = 0.00003 # Reach α=0.75 in 5000 episodes
        #self.alpha_decay = 0.00016 # Reach α=0.1 in 5000 episodes
        self.epsilon = 0.1 # Chance to explore 0.1 initially
        self.epsilon_decay = 0.000035 # Reach 0 in 3000 episodes
        self.episode = 0
        self.previousAction = 0
        self.previousState = (0, 0, 0, 0)
        self.moves = []
        self.scores = []
        self.maxScores = []
        self.maxScore = 0
        self.qTable = self.load_table()
        random.seed()

    def load_table(self):
        load = True
        qTable = {}
        if load:
            with open("qTable.csv", newline='') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    qTable[ast.literal_eval(row['A'])] = ast.literal_eval(row['B'])
        else:
            state = (0, 0, 0, 0)
            qTable[state] = [0, 0, 1]
        return qTable

    def save_qvalues(self):
        w = csv.writer(open("qTable.csv", "w"))
        w.writerow((['A', 'B']))
        for key, val in self.qTable.items():
            w.writerow([key, val])

    def end_episode(self, score):
        self.episode += 1
        self.scores.append(score)
        self.maxScore = max(score, self.max_score)
        if self.isTrainable:
            history = list(reversed(self.moves))
            for move in history:
                state, action, new_state = move
                self.qTable[state][action] = (1 - self.alpha) * (self.q_values[state][action]) + \
                                               self.alpha * (self.reward[0] + self.discountFactor *
                                                             max(self.qTable[new_state][0:2]))
            self.moves = []

    def get_state(self, x, y, vel, pipe):
        # Set the state from the game coordinates using the following map:
        #   - x0  : x distance of bird to next lowest pipe
        #   - y0  : y distance of bird to next lowest pipe
        #   - vel : y-velocity of bird (free falling)
        #   - y1  : y distance of bird to second next lowest pipe

        # Get pipe coordinates
        pipe0, pipe1 = pipe[0], pipe[1]
        # Pipe has passed
        if x - pipe[0]["x"] >= 50:
            pipe0 = pipe[1]
            if len(pipe) > 2:
                pipe1 = pipe[2]
        x0 = pipe0["x"] - x
        y0 = pipe0["y"] - y
        if -50 < x0 <= 0:
            y1 = pipe1["y"] - y
        else:
            y1 = 0
        # Evaluate player position compared to pipe
        if x0 < -40:
            x0 = int(x0)
        elif x0 < 140:
            x0 = int(x0) - (int(x0) % 10)
        else:
            x0 = int(x0) - (int(x0) % 70)

        if -180 < y0 < 180:
            y0 = int(y0) - (int(y0) % 10)
        else:

            y0 = int(y0) - (int(y0) % 60)
        if -180 < y1 < 180:
            y1 = int(y1) - (int(y1) % 10)
        else:
            y1 = int(y1) - (int(y1) % 60)
        state = (int(x0), int(y0), int(vel), int(y1))
        if self.qTable.get(state) is None:
            self.qTable[state] = [0, 0, 0]
        return state

    def update_qvalues(self, score):
        # Update Q values at the end of session using the history of moves while keeping the score
        self.episode += 1
        self.scores.append(score)
        self.maxScore = max(score, self.maxScore)

        if self.isTrainable:
            history = list(reversed(self.moves))
            # Flag if the bird died in the top pipe, don't flap if this is the case
            high_death_flag = True if int(history[0][2][1]) > 120 else False
            t, last_flap = 0, True
            for move in history:
                t += 1
                state, action, new_state = move
                self.qTable[state][2] += 1  # number of times this state has been seen
                curr_reward = self.reward[0]
                # Set reward
                if t <= 2:
                    # Penalise last 2 states before dying
                    curr_reward = self.reward[1]
                    if action:  # flapped
                        last_flap = False
                elif (last_flap or high_death_flag) and action:
                    # Penalise flapping
                    curr_reward = self.reward[1]
                    last_flap = False
                    high_death_flag = False
                self.qTable[state][action] = (1 - self.alpha) * (self.qTable[state][action]) + \
                                             self.alpha * (curr_reward + self.discountFactor *
                                                           max(self.qTable[new_state][0:2]))
            # Decay value alpha for convergence
            #if self.alpha > 0.1:
            #    self.alpha = max(self.alpha - self.alpha_decay, 0.1)
            #if self.epsilon > 0:
            #    self.epsilon = max(self.epsilon - self.epsilon_decay, 0)
            self.moves = []  # clear history of moves after updating strategies

    def act(self, x, y, vel, pipe):
        # Get parameters from state (get_state()) and decide using the q-table an appropriate action (flap or not)
        # store the transition from previous state to current state
        state = self.get_state(x, y, vel, pipe)
        if self.isTrainable:
            self.moves.append((self.previousState, self.previousAction, state))  # add the experience to history
            self.previousState = state  # update the last_state with the current state
        # Best action with respect to current state, default is 0 (do nothing), 1 is flap
        if self.qTable[state][0] >= self.qTable[state][1]:
            self.previousAction = 0
        elif self.qTable[state][0] < self.qTable[state][1]:
            self.previousAction = 1
        # Epsilon greedy policy for action, chance to explore
        # Remove since exploration is not efficient or required for this agent and environment
        #if random.random() <= self.epsilon:
        #    self.previous_action = random.choice([0, 1])
        return self.previousAction

    def plot_performance(self, window, logy, xlim=None, ylim=None):
        """Plot the training performance."""
        episodes = list(range(1, self.episode+1))
        scores = self.scores
        max_scores = self.maxScores
        fig, ax = plt.subplots()
        plt.ylabel('Score', fontsize=16)
        plt.xlabel('Episode', fontsize=16)
        if logy:
            ax.set_yscale('log')
            plt.ylabel('log(Score)', fontsize=14)
            scores = [x+1 for x in scores]
            max_scores = [x+1 for x in max_scores]
        plt.scatter(episodes, scores, label='scores', color='b', s=3)
        plt.plot(episodes, max_scores, label='max_score', color='g')
        plt.plot(episodes, np.convolve(scores, np.ones((window,)) / window, mode='same'),
                 label='rolling_mean_score', color='orange')
        if xlim:
            plt.xlim(xlim)
        if ylim:
            plt.ylim(ylim)
        ax.tick_params(axis="x", labelsize=12)
        ax.tick_params(axis="y", labelsize=12)
        plt.legend(loc='upper left', fontsize=14)
        fig.tight_layout()
        plt.show()
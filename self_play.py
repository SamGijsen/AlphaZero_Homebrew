import numpy as np
import torch
import pickle
import math

from neural_nets import ValueNet, PolicyNet
from TTT_env import TTT
from mcts import MCTS, Node



class Self_Play():

    def __init__(self, parameter_path, games=1, depth=50, temperature=1):

        self.games = games
        self.depth = depth
        self.temperature = temperature
        self.parameter_path = parameter_path


    def play(self, version, random_opponent):

        state_log = np.empty((0, 9), int)
        mcts_log = np.empty((0, 9), float)
        win_log = np.empty((0,1), int)
        beginner_score = 0

        for i in range(self.games):

            state = [0]*9
            env = TTT()
            move = 0

            while env.check_terminality() == None:

                legal_moves = env.legal_moves(state)[0]

                if random_opponent and ((move % 2) == (i % 2)):

                    # random action
                    pi = np.ones(len(legal_moves))/len(legal_moves)
                    action = np.random.choice(np.arange(len(legal_moves), dtype="float64"), p=pi, size=1)[0].astype(int)

                else: 

                    # perform search
                    mcts = MCTS(state, turn=1, runs=self.depth, parameter_path=self.parameter_path)
                    root = mcts.search(version=version)

                    # select move
                    action, pi = root.sample_action(temperature=self.temperature)


                # log
                state_log = np.append(state_log, np.array(state).reshape(1,-1), axis=0)
                pi_masked = np.zeros(len(state))
                pi_masked[legal_moves] = pi 
                mcts_log = np.append(mcts_log, pi_masked.reshape(1,-1), axis=0)
 
                # make move
                state[legal_moves[action]] = 1 

                # prep board for 'opposing' player
                state = [-x for x in state] 
                env = TTT(state)
                move += 1

            z = env.check_terminality()

            if z != 0: # if not a draw
                if (move % 2) == 0: # move is even number
                    z = -1
                else:
                    z = 1

                if (i % 2) == 0: # random player wins
                    beginner_score += z 
                else:
                    beginner_score -= z

            for t in range(move):
                if t == 0: # code first state always as a'draw' ending - TODO: exclude from training
                    win_log = np.append(win_log, np.array(0).reshape(1, -1), axis=0)
                else:
                    win_log = np.append(win_log, np.array(z).reshape(1, -1), axis=0)
                z *= -1

        if not random_opponent:
            self.save_game_data(version, self.parameter_path, state_log, mcts_log, win_log)

        return state_log, mcts_log, win_log, beginner_score


    def save_game_data(self, version, parameter_path, state_log, mcts_log, win_log):

        fn = parameter_path + "game_data_v{}".format(version) + ".data"

        with open(fn, "wb") as f:
            pickle.dump([state_log, mcts_log, win_log], f)

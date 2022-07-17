import numpy as np
import torch
from neural_nets import ValueNet, PolicyNet
from TTT_env import TTT
import math


class MCTS():
    
    def __init__(self, state, turn, runs, parameter_path):
               
        self.state = state
        self.turn = turn
        self.runs = runs
        self.parameter_path = parameter_path

    def search(self, version):

        torch.manual_seed(version)
        
        pnet = PolicyNet()
        vnet = ValueNet()

        if version > 0: # load parameters from previous versions
            pnet.load_state_dict(torch.load(self.parameter_path + "pnet_v{}".format(version-1)))
            vnet.load_state_dict(torch.load(self.parameter_path + "vnet_v{}".format(version-1)))

        # expand the root
        p_policy = pnet.predict(torch.tensor(self.state).float())[0]
        root = Node(self.state, self.turn)
        root.expand(self.turn, prior=p_policy.copy())

        # do MCTS steps
        for run in range(self.runs):

            node_t = root
            search_path = [node_t]

            while node_t.expanded:
                node_t = self.select_child(node_t)
                search_path.append(node_t)


            env = TTT(node_t.state, turn=node_t.turn) 
            Q = env.check_terminality() # requires board-state to be from the right POV
            if Q == None: # not terminal
                Q = vnet.predict(torch.tensor(node_t.state).float())
                p_policy = pnet.predict(torch.tensor(node_t.state).float())[0]
                node_t.expand(node_t.turn, prior=p_policy)

            # backup
            for i in range(len(search_path)-1, -1, -1):
                search_path[i].value += Q*search_path[i].turn
                search_path[i].N += 1

        return root


    def UCB_scoring(self, node):

        ucb_scores = []

        for a in range(len(node.child)):

            score = 3 * node.child[a].prior * math.sqrt(node.N) / (node.child[a].N+1)
            if node.child[a].N > 0:
                v = -node.child[a].value / node.child[a].N
            else:
                v = 0
            ucb_scores.append(v + score)

        return ucb_scores


    def select_child(self, node):

        scores = self.UCB_scoring(node)
        a = np.argmax(scores)

        return node.child[a]


class Node():
    
    def __init__(self, state, turn):
        
        self.N = 0 # visits
        self.prior = 0
        self.child = {}
        self.state = state
        self.expanded = False
        self.value = 0
        self.turn = turn

    def expand(self, turn, prior):

        self.expanded = True
        # 1. generate child
        # 2. assign priors
        count = 0

        env = TTT(state=self.state.copy())
        legal_moves = env.legal_moves(self.state)[0]
        prior = prior[legal_moves]
        prior /= np.sum(prior)

        for i, a in enumerate(prior):
            env = TTT(state=self.state.copy())
            env.step(legal_moves[i], 1)
            state = [-x for x in env.state.copy()] # flip the board
            self.child[i] = Node(state, turn*-1)
            self.child[i].prior = prior[i].item()

    def sample_action(self, temperature=1):

        visits = [self.child[i].N**temperature for i in range(len(self.child))]
        pi = np.array(visits, dtype="float64")/np.sum(visits)
        a = np.random.choice(np.arange(len(visits), dtype="float64"), p=pi, size=1)[0]
        
        return a.astype(int), pi

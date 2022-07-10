import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.notebook import trange
import numpy as np
import pickle

class ValueNet(torch.nn.Module):
    
    def __init__(self):
        super(ValueNet, self).__init__()
        self.l1 = nn.Linear(9,64)
        self.l2 = nn.Linear(64,64)
        self.l3 = nn.Linear(64,1)
    
    def forward(self, x):
        x = self.l1(x)
        x = F.relu(x)
        x = self.l2(x)
        x = F.relu(x)
        x = self.l3(x)
        return torch.tanh(x)

    def predict(self, state):
        state = torch.FloatTensor(state)
        self.eval()
        with torch.no_grad():
            p = self.forward(state)

        return p.cpu().numpy()
    
class PolicyNet(torch.nn.Module):
    
    def __init__(self):
        super(PolicyNet, self).__init__()
        self.l1 = nn.Linear(9,32)
        self.l2 = nn.Linear(32,9)
        self.sm = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.l1(x)
        x = F.relu(x)
        x = self.l2(x)
        x = self.sm(x) 
        return x

    def predict(self, state):

        self.eval()
        state = state.view(1, len(state))
        with torch.no_grad():
            p = self.forward(state)

        return p.cpu().numpy()

class Training():

    def __init__(self):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("Device=", self.device)


    def train_policy(self, state_log, mcts_log, win_log, version, parameter_path, lr=0.02, batchsize=32, epochs=10):

        model = PolicyNet().to(self.device)

        if version > 0: # load parameters from previous versions
            model.load_state_dict(torch.load(parameter_path + "pnet_v{}".format(version-1)))

        loss_function = nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0, weight_decay=1e-4)
        losses = []

        for i in trange( int(len(win_log)*epochs / batchsize) ):
            
            idx = np.random.randint(0, high=len(win_log), size=batchsize)
            x = torch.tensor(state_log[idx,:]).float().requires_grad_()
            y = torch.tensor(mcts_log[idx,:]).float().requires_grad_()
            x, y = x.to(self.device), y.to(self.device)
            
            optimizer.zero_grad()
            output = model(x)
            
            loss = self.CrossEntropy(output, y).mean()
            losses.append(loss.item())
            
            loss.backward()
            optimizer.step()

        self.save_param_loss(version, parameter_path, model, losses, "pnet")

        return model, losses

    def CrossEntropy(self, output, y):

        return -(y * torch.log(output)).sum(dim=0)


    def train_value(self, state_log, mcts_log, win_log, version, parameter_path, lr=0.02, batchsize=32, epochs=10):

        model = ValueNet().to(self.device)

        if version > 0: # load parameters from previous versions
            model.load_state_dict(torch.load(parameter_path + "vnet_v{}".format(version-1)))

        loss_function = nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0, weight_decay=1e-4)
        losses = []

        for i in trange( int((len(win_log) / batchsize)*epochs) ):
            
            idx = np.random.randint(0, high=len(win_log), size=batchsize)
            x = torch.tensor(state_log[idx,:]).float().requires_grad_()
            y = torch.tensor(win_log[idx]).float().requires_grad_()
            x, y = x.to(self.device), y.to(self.device)
            
            optimizer.zero_grad()
            output = model(x)

            loss = loss_function(output, y)

            losses.append(loss.item())
            
            loss.backward()
            optimizer.step()

        self.save_param_loss(version, parameter_path, model, losses, "vnet")

        return model, losses


    def save_param_loss(self, version, parameter_path, model, losses, net_prefix):

        # Save losses
        loss_fn = parameter_path + net_prefix + "_loss_v{}".format(version) + ".data"

        with open(loss_fn, "wb") as f:
            pickle.dump(losses, f)

        # Save parameters
        p_fn = parameter_path + net_prefix + "_v{}".format(version)
        torch.save(model.state_dict(), p_fn)
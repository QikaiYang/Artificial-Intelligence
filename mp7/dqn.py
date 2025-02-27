import gym
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import copy

import utils
from policies import QPolicy

def make_dqn(statesize, actionsize):
    """
    Create a nn.Module instance for the q leanring model.

    @param statesize: dimension of the input continuous state space.
    @param actionsize: dimension of the descrete action space.

    @return model: nn.Module instance
    """
    return torch.nn.Sequential(\
        torch.nn.Linear(statesize, 64, bias = True),\
        torch.nn.ELU(),\
        torch.nn.Linear(64, 32, bias = True),\
        torch.nn.ELU(),\
        torch.nn.Linear(32, 32, bias = True),\
        torch.nn.ELU(),\
        torch.nn.Linear(32, 32, bias = True),\
        torch.nn.ELU(),\
        torch.nn.Linear(32, actionsize, bias = True),\
)

class DQNPolicy(QPolicy):
    """
    Function approximation via a deep network
    """

    def __init__(self, model, statesize, actionsize, lr, gamma):
        """
        Inititalize the dqn policy

        @param model: the nn.Module instance returned by make_dqn
        @param statesize: dimension of the input continuous state space.
        @param actionsize: dimension of the descrete action space.
        @param lr: learning rate 
        @param gamma: discount factor
        """
        super().__init__(statesize, actionsize, lr, gamma)
        self.statesize = statesize
        self.actionsize = actionsize
        self.lr = lr
        self.gamma = gamma
        self.count = 0
        #--------------------------------------------------
        self.model = model
        #self.target_model = copy.deepcopy(model)
        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=lr)
        self.loss_fn = F.smooth_l1_loss

    def qvals(self, state):
        """
        Returns the q values for the states.

        @param state: the state
        
        @return qvals: the q values for the state for each action. 
        """
        self.model.eval()
        with torch.no_grad():
            states = torch.from_numpy(state).type(torch.FloatTensor)
            qvals = self.model(states)
        return qvals.numpy()

    def td_step(self, state, action, reward, next_state, done):
        """
        One step TD update to the model

        @param state: the current state
        @param action: the action
        @param reward: the reward of taking the action at the current state
        @param next_state: the next state after taking the action at the
            current state
        @param done: true if episode has terminated, false otherwise
        @return loss: total loss the at this time step
        """
        # if(self.count % 50==0): #To Modify
        #     self.target_model = copy.deepcopy(self.model)
        #-------------------------------------------------
        self.optimizer.zero_grad()
        if len(state) == 4:
            state = torch.tensor([[state[i] for i in range(4)]])
        if len(next_state) == 4:
            next_state = torch.tensor([[next_state[i] for i in range(4)]])
        #--------------------preprocess-------------------
        q_eval = self.model(state)[0][action]
        q_next = max(self.model(next_state)[0])
        q_target = torch.tensor(reward) + self.gamma * q_next if done != True \
                   else torch.tensor(reward)
        loss = self.loss_fn(q_eval, q_target)
        loss.backward()
        self.optimizer.step()
        #self.count += 1
        return loss.item()

    def save(self, outpath):
        """
        saves the model at the specified outpath
        """        
        torch.save(self.model, outpath)


if __name__ == '__main__':
    args = utils.hyperparameters()

    env = gym.make('CartPole-v1')
    statesize = env.observation_space.shape[0]
    actionsize = env.action_space.n

    policy = DQNPolicy(make_dqn(statesize, actionsize), statesize, actionsize, lr=args.lr, gamma=args.gamma)

    utils.qlearn(env, policy, args)

    torch.save(policy.model, 'models/dqn.model')

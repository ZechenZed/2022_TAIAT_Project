'''
@Author: Wenhao Ding
@Email: wenhaod@andrew.cmu.edu
@Date: 2020-01-24 13:52:10
LastEditTime: 2022-04-10 22:03:33
@Description: 
'''

import os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils as utils
from torch.autograd import Variable
import torch.nn.init as init


def CUDA(var):
    return var.cuda() if torch.cuda.is_available() else var


def kaiming_init(m):
    if isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)   # ??
        if m.bias is not None:
            m.bias.data.fill_(0)


pi = CUDA(Variable(torch.FloatTensor([math.pi])))
def normal(x, mu, sigma_sq):
    a = (-1*(CUDA(Variable(x))-mu).pow(2)/(2*sigma_sq)).exp()
    b = 1/(2*sigma_sq*pi.expand_as(sigma_sq)).sqrt()
    return a*b


class HD_Autoregressive_Policy(nn.Module):
    def __init__(self):
        super(HD_Autoregressive_Policy, self).__init__()
        input_size = 30*2+1  ## ？
        hidden_size_1 = 32  # ？higher hidden dim ?

        self.a_os = 1
        self.b_os = 1
        self.c_os = 1
        self.d_os = 1

        self.fc_input = nn.Sequential(
            nn.Linear(input_size, hidden_size_1),
            nn.ReLU(inplace=True)
        )

        self.fc_action_a = nn.Sequential(
            nn.Linear(hidden_size_1, self.a_os*2),
        )

        self.fc_action_b = nn.Sequential(
            nn.Linear(1+hidden_size_1, self.b_os*2),
        )

        self.fc_action_c = nn.Sequential(
            nn.Linear(1+1+hidden_size_1, self.c_os*2),
        )

        self.fc_action_d = nn.Sequential(
            nn.Linear(1+1+1+hidden_size_1, self.d_os*2),
        )

    def sample_action(self, normal_action, action_os):
        # get the mu and sigma
        #mu = torch.tanh(normal_action[:, :action_os])
        mu = normal_action[:, :action_os]  # concatenated
        sigma = F.softplus(normal_action[:, action_os:])

        # calculate the probability by mu and sigma of normal distribution
        eps = CUDA(Variable(torch.randn(mu.size())))
        action = (mu + sigma.sqrt()*eps)  # reparameterization

        return action, mu, sigma

    def forward(self, x):
        # p(s)
        x = x.view(1, -1)
        s = self.fc_input(x)
        
        # p(a|s)
        normal_a = self.fc_action_a(s)
        action_a, mu_a, sigma_a = self.sample_action(normal_a, self.a_os)

        # p(b|a,s) 
        state_sample_a = torch.cat((s, action_a), dim=1)
        normal_b = self.fc_action_b(state_sample_a)
        action_b, mu_b, sigma_b = self.sample_action(normal_b, self.b_os)

        # p(c|a,b,s)
        state_sample_a_b = torch.cat((s, action_a, action_b), dim=1)
        normal_c = self.fc_action_c(state_sample_a_b)
        action_c, mu_c, sigma_c = self.sample_action(normal_c, self.c_os)

        # p(d|a,b,c,s)
        state_sample_a_b_c = torch.cat((s, action_a, action_b, action_c), dim=1)
        normal_d = self.fc_action_d(state_sample_a_b_c)
        action_d, mu_d, sigma_d = self.sample_action(normal_d, self.d_os)

        return [mu_a, mu_b, mu_c, mu_d], [sigma_a, sigma_b, sigma_c, sigma_d], [action_a[0], action_b[0], action_c[0], action_d[0]]

    # deterministic output
    def deterministic_forward(self, x):
        # p(s)
        x = x.view(1, -1)
        s = self.fc_input(x)
        
        # p(a|s)
        normal_a = self.fc_action_a(s)
        _, mu_a, sigma_a = self.sample_action(normal_a, self.a_os)

        # p(b|a,s) 
        state_sample_a = torch.cat((s, mu_a), dim=1)
        normal_b = self.fc_action_b(state_sample_a)
        _, mu_b, sigma_b = self.sample_action(normal_b, self.b_os)

        # p(c|a,b,s)
        state_sample_a_b = torch.cat((s, mu_a, mu_b), dim=1)
        normal_c = self.fc_action_c(state_sample_a_b)
        _, mu_c, sigma_c = self.sample_action(normal_c, self.c_os)

        # p(d|a,b,c,s)
        state_sample_a_b_c = torch.cat((s, mu_a, mu_b, mu_c), dim=1)
        normal_d = self.fc_action_d(state_sample_a_b_c)
        _, mu_d, sigma_d = self.sample_action(normal_d, self.d_os)

        print('================================================== Test Action space ==================================================')
        print('normal A: {} {}'.format(mu_a, sigma_a))
        print('normal B: {} {}'.format(mu_b, sigma_b))
        print('normal C: {} {}'.format(mu_c, sigma_c))
        print('normal D: {} {}'.format(mu_d, sigma_d))
        print('======================================================================================================================')

        # output the mean value to be the deterministic action
        return mu_a[0][0], mu_b[0][0], mu_c[0][0], mu_d[0][0]


class REINFORCE:
    def __init__(self, lr, gamma, model_id=0, model_path='./model'):
        self.model = CUDA(HD_Autoregressive_Policy())
        self.model.apply(kaiming_init)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.model.train()

        self.model_id = model_id
        self.model_path = model_path
        self.gamma = gamma

    def select_action(self, state):
        state = CUDA(Variable(torch.from_numpy(state)))
        mu_bag, sigma_bag, action_bag = self.model(state)

        # calculate the probability that this distribution outputs this action
        prob_a = normal(action_bag[0], mu_bag[0], sigma_bag[0])
        prob_b = normal(action_bag[1], mu_bag[1], sigma_bag[1])
        prob_c = normal(action_bag[2], mu_bag[2], sigma_bag[2])
        prob_d = normal(action_bag[3], mu_bag[3], sigma_bag[3])
        log_prob = prob_a.log() + prob_b.log() + prob_c.log() + prob_d.log()

        # calculate the entropy 
        entropy_a = -0.5*((sigma_bag[0]+2*pi.expand_as(sigma_bag[0])).log()+1)  # define of pi?
        entropy_b = -0.5*((sigma_bag[1]+2*pi.expand_as(sigma_bag[1])).log()+1)
        entropy_c = -0.5*((sigma_bag[2]+2*pi.expand_as(sigma_bag[2])).log()+1)
        entropy_d = -0.5*((sigma_bag[2]+2*pi.expand_as(sigma_bag[2])).log()+1)  # ? typo ?
        entropy = entropy_a + entropy_b + entropy_c + entropy_d

        a_1 = action_bag[0][0].detach().cpu().numpy()
        a_2 = action_bag[1][0].detach().cpu().numpy()
        a_3 = action_bag[2][0].detach().cpu().numpy()
        a_4 = action_bag[3][0].detach().cpu().numpy()

        return [a_1, a_2, a_3, a_4], log_prob, entropy

    def deterministic_action(self, state):
        with torch.no_grad():
            state = CUDA(Variable(torch.from_numpy(state)))
            action_a, action_b, action_c, action_d = self.model.deterministic_forward(state)

        #print('score a: {}'.format(score_a.cpu().numpy()))
        #print('score b: {}'.format(score_b.cpu().numpy()))
        #print('score c: {}'.format(score_c.cpu().numpy()))
        return [action_a.cpu().numpy(), action_b.cpu().numpy(), action_c.cpu().numpy(), action_d.cpu().numpy()]

    def update_parameters(self, rewards, log_probs, entropies):
        R = CUDA(torch.zeros(1, 1))
        loss = 0
        for i in reversed(range(len(rewards))):
            R = self.gamma * R + rewards[i]
            loss = loss - (log_probs[i]*(Variable(R).expand_as(log_probs[i]))).sum() # - (0.001*entropies[i]).sum()
        loss = loss / len(rewards)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_model(self):
        states = {'parameters': self.model.state_dict()}
        filepath = os.path.join(self.model_path, 'model.'+str(self.model_id)+'.torch')
        with open(filepath, 'wb+') as f:
            torch.save(states, f)

    def load_model(self):
        filepath = os.path.join(self.model_path, 'model.'+str(self.model_id)+'.torch')
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                checkpoint = torch.load(f)
            self.model.load_state_dict(checkpoint['parameters'])

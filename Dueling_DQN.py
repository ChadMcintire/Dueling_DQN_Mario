import torch
from model import model
from torch import optim
import numpy as np
import os
import torch.nn.functional as F

class Dueling_DQN(object):
    def __init__(self, n_frames, env, args):
        #Todo change parameter into arg parse arguments
        learning_rate = args.lr
        self.gamma = args.gamma
        self.device = torch.device('cuda' if torch.cuda.is_available() 
                                                           else 'cpu') 
        self.batch_size = args.batch_size
        self.eps = args.epsilon
        self.q = model(n_frames, env.action_space.n, self.device).to(self.device)
        self.q_target = model(n_frames, env.action_space.n, self.device).to(self.device)
        self.optimizer = optim.Adam(self.q.parameters(), lr=learning_rate)


    def select_action(self, state, env, evaluation=False):
        if self.eps > np.random.rand():
            action = env.action_space.sample()
        else:
            if self.device == "cpu":
                action = np.argmax(self.q(state).detach().numpy())
            else:
                action = np.argmax(self.q(state).cpu().detach().numpy())
        return action

    def update_parameters(self, memory):
        state, reward, action, next_state, done = list(map(list, zip(*memory.sample(self.batch_size))))
        state = np.array(state).squeeze()
        next_state = np.array(next_state).squeeze()
        action_max = self.q(next_state).max(1)[1].unsqueeze(-1)
        reward = torch.FloatTensor(reward).unsqueeze(-1).to(self.device)
        done = torch.FloatTensor(done).unsqueeze(-1).to(self.device)
        with torch.no_grad():
            y = reward + self.gamma * self.q_target(next_state).gather(1, action_max) * done
        action = torch.tensor(action).unsqueeze(-1).to(self.device)
        q_value = torch.gather(self.q(state), dim=1, index=action.view(-1, 1).long())

        loss = F.smooth_l1_loss(q_value, y).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss


    def save_checkpoint(self,  suffix="", ckpt_path="checkpoints"):
        ch_pt = "check_points/" 
        name = "check_point1.pth"
        if not os.path.exists(ch_pt + ckpt_path + "/"):
            os.makedirs(ch_pt + ckpt_path + "/")
        #if ckpt_path is None:
        #print("Save models to {}".format(ckpt_path + "/"))
        torch.save({"q_net_state_dict": self.q.state_dict(),
                    "q_net_state_target_dict": self.q_target.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict()}, ch_pt + ckpt_path + "/" + name)


    def load_checkpoint(self, ckpt_path, evaluate=True):
        print("Loading models from {}".format(ckpt_path))
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)
            self.q.load_state_dict(checkpoint['q_net_state_dict'])
            self.q_target.load_state_dict(checkpoint["q_net_state_target_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            if evaluate:
                self.q.eval()
                self.q_target.eval()

            else:
                self.q.train()
                self.q_target.train()

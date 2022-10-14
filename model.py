import torch.nn as nn
from utils import init_weights
import torch

class model(nn.Module):
    def __init__(self, n_frame, n_action, device):
        super(model, self).__init__()
        self.layer1 = nn.Conv2d(n_frame, 32, 8, 4)
        self.layer2 = nn.Conv2d(32, 64, 3, 1)
        self.fc = nn.Linear(20736, 512)
        self.q = nn.Linear(512, n_action)
        self.v = nn.Linear(512, 1)

        self.device = device
        self.seq = nn.Sequential(self.layer1, self.layer2, self.fc, 
                                 self.q, self.v)

        self.seq.apply(init_weights)

    def forward(self, x):
        if type(x) != torch.Tensor:
            x = torch.FloatTensor(x).to(self.device)
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = x.view(-1, 20736)
        x = torch.relu(self.fc(x))
        adv = self.q(x)
        v = self.v(x)
        q = v + (adv -1 / adv.shape[-1] * adv.max(-1, True)[0])

        return q



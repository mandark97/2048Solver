import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, outputs):
        super(DQN, self).__init__()
        self.fn1 = nn.Linear(4 * 16, 300)
        self.fn2 = nn.Linear(300, 300)
        self.fn3 = nn.Linear(300, 200)
        self.fn4 = nn.Linear(200, 200)
        self.fn5 = nn.Linear(200, 100)
        self.fn6 = nn.Linear(100, 100)

        self.head = nn.Linear(100, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.fn1(x.view(x.size(0), -1)))
        x = F.relu(self.fn2(x))
        x = F.relu(self.fn3(x))
        x = F.relu(self.fn4(x))
        x = F.relu(self.fn5(x))
        x = F.relu(self.fn6(x))

        return self.head(x)

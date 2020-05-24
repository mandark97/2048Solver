import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    # 4 * 16
    def __init__(self, input_shape, outputs):
        super(DQN, self).__init__()

        self.fn1 = nn.Linear(input_shape, 300)
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


class DuelingDQN(nn.Module):
    def __init__(self, input_shape, outputs):
        super(DuelingDQN, self).__init__()

        self.fn1 = nn.Linear(input_shape, 300)
        self.fn2 = nn.Linear(300, 300)
        self.fn3 = nn.Linear(300, 200)
        self.fn4 = nn.Linear(200, 200)
        self.fn5 = nn.Linear(200, 100)
        # self.fn6 = nn.Linear(100, 100)

        self.adv1 = nn.Linear(100, 100)
        self.adv2 = nn.Linear(100, outputs)

        self.val1 = nn.Linear(100, 60)
        self.val2 = nn.Linear(60, 1)

    def forward(self, x):
        x = F.relu(self.fn1(x.view(x.size(0), -1)))
        x = F.relu(self.fn2(x))
        x = F.relu(self.fn3(x))
        x = F.relu(self.fn4(x))
        x = F.relu(self.fn5(x))

        adv = F.relu(self.adv1(x))
        adv = self.adv2(adv)

        val = F.relu(self.val1(x))
        val = self.val2(val)

        return val + adv - adv.mean()


class Actor(nn.Module):
    def __init__(self, input_shape, outputs):
        super().__init__()

        self.outputs = outputs
        self.fn1 = nn.Linear(input_shape, 300)
        self.fn2 = nn.Linear(300, 300)
        self.fn3 = nn.Linear(300, 200)
        self.fn4 = nn.Linear(200, 200)
        self.fn5 = nn.Linear(200, 100)
        self.fn6 = nn.Linear(100, 100)

        self.head = nn.Linear(100, outputs)


    def forward(self, x):
        x = F.relu(self.fn1(x.view(x.size(0), -1)))
        x = F.relu(self.fn2(x))
        x = F.relu(self.fn3(x))
        x = F.relu(self.fn4(x))
        x = F.relu(self.fn5(x))
        x = F.relu(self.fn6(x))

        return F.softmax(self.head(x), dim=1)

class Critic(nn.Module):
    def __init__(self, input_shape):
        super().__init__()

        self.fn1 = nn.Linear(input_shape, 300)
        self.fn2 = nn.Linear(300, 300)
        self.fn3 = nn.Linear(300, 200)
        self.fn4 = nn.Linear(200, 200)
        self.fn5 = nn.Linear(200, 100)
        self.fn6 = nn.Linear(100, 100)

        self.head = nn.Linear(100, 1)

    def forward(self, x):
        x = F.relu(self.fn1(x.view(x.size(0), -1)))
        x = F.relu(self.fn2(x))
        x = F.relu(self.fn3(x))
        x = F.relu(self.fn4(x))
        x = F.relu(self.fn5(x))
        x = F.relu(self.fn6(x))

        return self.head(x)

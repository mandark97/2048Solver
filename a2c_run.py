# Imports
# %matplotlib inline
import logging
from collections import Counter
from types import MethodType
from typing import Tuple

import numpy as np
import torch
import torch.optim as optim
import matplotlib
import matplotlib.pyplot as plt
import torch.nn.functional as F

from env import GameEnv
from rewards import Reward3
from trainers.a2c import A2CTrainer

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
plt.ion()

env = GameEnv(reward_class=Reward3)


# define transformation function

# 4*4 matrix
def get_state(self):
    with np.errstate(divide='ignore'):
        board = np.where(self.env.board != 0, np.log2(self.env.board), 0)
        return torch.from_numpy(np.ascontiguousarray(board)).unsqueeze(0).float().to(self.device)


# 4*16 matrix
def state2(self):
    with np.errstate(divide='ignore'):
        board = np.where(env.board != 0, np.log2(env.board).astype(np.int), 0)
    board = np.vectorize(np.binary_repr)(board, width=4).astype(str)
    board = np.array([list(''.join(line)) for line in board]).astype(np.int)
    return torch.from_numpy(np.ascontiguousarray(board)).unsqueeze(0).float().to(self.device)


# define constants
num_episodes = 3000
BATCH_SIZE = 256
GAMMA = 0.999
EPS_START = 1.
EPS_END = 0.1
EPS_DECAY = 9999
TARGET_UPDATE = 50
MEMORY_SIZE = 10000
optimizer = optim.Adam
optimizer_params = {"lr": 0.0005}
# define trainer

logging.basicConfig(level=logging.DEBUG, filename="run.log")
trainer = A2CTrainer(memory_size=MEMORY_SIZE, batch_size=BATCH_SIZE, gamma=GAMMA, eps_start=EPS_START, eps_end=EPS_END,
                     eps_decay=EPS_DECAY, target_update=TARGET_UPDATE, env=env, input_shape=4*16, optimizer_klass=optimizer,
                     optimizer_params=optimizer_params,
                     loss_f=F.smooth_l1_loss, is_ipython=is_ipython, log_dir="runs/a2c_run3")

# assign state method
trainer.get_state = MethodType(state2, trainer)

# train
trainer.train(num_episodes)
trainer.write_results()

import math
import random
import json
from itertools import count
from typing import Optional, Dict
import torch
import os
import matplotlib.pyplot as plt
import torch
from IPython import display
from collections import Counter

from env import GameEnv
from replay_memory import ReplayMemory, Transition


class Trainer(object):
    def __init__(self, memory_size, batch_size, gamma, eps_start, eps_end, eps_decay, target_update,
                 env: GameEnv, model=None,
                 optimizer_klass=None, optimizer_params: Optional[Dict] = None,
                 loss_f=None, loss_params: Optional[Dict] = None,
                 is_ipython=False, log_dir: str = None):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.is_ipython = is_ipython
        self.log_dir = log_dir

        self.BATCH_SIZE = batch_size
        self.GAMMA = gamma
        self.EPS_START = eps_start
        self.EPS_END = eps_end
        self.EPS_DECAY = eps_decay
        self.TARGET_UPDATE = target_update

        self.env = env
        self.env.reset()
        self.memory = ReplayMemory(memory_size)
        self.get_state = None

        self.steps_done = 0
        self.episode_scores = []
        self.highest_scores = []

        self.init_models(model)

        self.optimizer_params = optimizer_params or {}
        self.optimizer = optimizer_klass(
            self.policy_net.parameters(), **self.optimizer_params)

        if loss_params is None:
            loss_params = {}
        self.loss = loss_f
        self.loss_params = loss_params

    def init_models(self, model):
        self.n_actions = self.env.action_space.n
        self.target_net = model(self.n_actions).to(self.device)
        self.policy_net = model(self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1

        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.n_actions)]], device=self.device, dtype=torch.long)

    def plot_scores(self, save=None):
        plt.figure(2)
        plt.clf()
        scores_t = torch.tensor(self.episode_scores, dtype=torch.float)
        plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Scores')
        plt.plot(scores_t.numpy())
        # Take 100 episode averages and plot them too
        if len(scores_t) >= 100:
            means = scores_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        if save is not None:
            plt.savefig(save, dpi=300, bbox_inches='tight')

        plt.pause(0.001)  # pause a bit so that plots are updated

        if self.is_ipython:
            display.clear_output(wait=True)
            # display.display(plt.gcf())

    def optimize_model(self):
        if len(self.memory) < self.BATCH_SIZE:
            return
        transitions = self.memory.sample(self.BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(
            state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
        next_state_values[non_final_mask] = self.next_state_action(
            non_final_next_states)
        # Compute the expected Q values
        expected_state_action_values = (
            next_state_values * self.GAMMA) + reward_batch

        # Compute Huber loss
        loss = self.loss(state_action_values, expected_state_action_values.unsqueeze(
            1), **self.loss_params)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def train(self, num_episodes: int):
        for i_episode in range(num_episodes):
            if not self.is_ipython:
                print(i_episode)
            # Initialize the environment and state
            self.env.reset()
            current_board = self.get_state()
            old_board = self.get_state()
            for t in count():
                # Select and perform an action
                action = self.select_action(current_board)
                _, reward, done, _ = self.env.step(action.item())
                reward = torch.tensor([reward], device=self.device)
                current_board = self.get_state()
                if not done:
                    next_state = current_board
                else:
                    next_state = None

                # Store the transition in memory
                self.memory.push(old_board, action, next_state, reward)

                # Perform one step of the optimization (on the target network)
                self.optimize_model()
                if done:
                    if not self.is_ipython:
                        self.env.render()
                    self.episode_scores.append(self.env.score)
                    self.highest_scores.append(self.env.highest())
                    self.plot_scores()
                    break
            # Update the target network, copying all weights and biases in DQN
            if i_episode % self.TARGET_UPDATE == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

        self.env.close()

    def next_state_action(self, non_final_next_states):
        return self.target_net(non_final_next_states).max(1)[0].detach()

    def get_highest_scores(self):
        return torch.tensor(self.highest_scores).numpy()

    def write_results(self):
        try:
            os.makedirs(self.log_dir)
        except OSError:
            pass
        self.plot_scores(save=f"{self.log_dir}/plot.png")
        torch.save(self.target_net, f"{self.log_dir}/target.pt")
        torch.save(self.policy_net, f"{self.log_dir}/policy.pt")

        with open(f"{self.log_dir}/highest_scores.json", "w") as json_file:
            scores = {str(k): v for k, v in Counter(
                self.get_highest_scores()).items()}
            json.dump(scores, json_file)

        with open(f"{self.log_dir}/train_params.json", "w") as json_file:
            params = {
                "batch_size": self.BATCH_SIZE,
                "gamma": self.GAMMA,
                "eps_start": self.EPS_START,
                "eps_end": self.EPS_END,
                "eps_decay": self.EPS_DECAY,
                "target_update": self.TARGET_UPDATE,
                **self.loss_params
            }
            json.dump(params, json_file)

from trainers.dqn import DQNTrainer
from networks import Actor, Critic
from itertools import count
import torch
import os
from collections import Counter
import json
import logging
from replay_memory import EpisodeMemory
import numpy as np


class A2CTrainer(DQNTrainer):
    def __init__(self, memory_size, batch_size, gamma, eps_start, eps_end, eps_decay, target_update, env, input_shape=None, optimizer_klass=None, optimizer_params=None, loss_f=None, loss_params=None, is_ipython=False, log_dir=None):
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
        # self.memory = ReplayMemory(memory_size)
        self.get_state = None  # added method from main for state processing

        self.steps_done = 0
        self.episode_scores = []
        self.highest_scores = []

        self.init_models(input_shape)

        self.optimizer_params = optimizer_params or {}
        self.actor_optimizer = optimizer_klass(
            self.actor.parameters(), **self.optimizer_params)
        self.critic_optimizer = optimizer_klass(
            self.critic.parameters(), **self.optimizer_params)

        if loss_params is None:
            loss_params = {}
        self.loss = loss_f
        self.loss_params = loss_params

        self.memory = EpisodeMemory()

    def init_models(self, input_shape):
        self.n_actions = self.env.action_space.n
        self.actor = Actor(input_shape, self.n_actions).to(self.device)
        self.critic = Critic(input_shape).to(self.device)

    # def select_action(self, state):
    #     probs = self.actor(state)
    #     dist = torch.distributions.Categorical(probs=probs)
    #     action = dist.sample()

    #     return action

    # def optimize_model(self):
    #     return super().optimize_model()

    # def next_state_action(self, non_final_next_states):
    #     return super().next_state_action(non_final_next_states)

    def optimize_model(self, q_val):
        values = torch.stack(self.memory.values)
        q_vals = np.zeros((len(self.memory), 1))

        # target values are calculated backward
        # it's super important to handle correctly done states,
        # for those cases we want our to target to be equal to the reward only
        q_val = q_val.item()
        for i, (_, _, reward, done) in enumerate(self.memory.reversed()):
            q_val = reward + self.GAMMA*q_val*(1.0-done)
            # store values from the end to the beginning
            q_vals[len(self.memory)-1 - i] = q_val

        advantage = torch.Tensor(q_vals).to(self.device) - values

        critic_loss = advantage.pow(2).mean()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = (-torch.stack(self.memory.log_probs)
                      * advantage.detach()).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

    def train(self, num_episodes):
        max_steps = 2000
        for i_episode in range(num_episodes):
            if not self.is_ipython:
                print(i_episode)
            # Initialize the environment and state
            self.env.reset()
            steps = 0

            current_board = self.get_state()
            for t in count():
                probs = self.actor(current_board)
                dist = torch.distributions.Categorical(probs=probs)
                action = dist.sample()

                logging.debug(action.item())
                logging.debug(f"\n {self.env.board}")

                _, reward, done, _ = self.env.step(action.item())
                steps += 1
                # reward = torch.tensor([reward], device=self.device)
                self.memory.add(dist.log_prob(action), self.critic(
                    current_board), reward, done)
                current_board = self.get_state()

                if done or (steps % max_steps == 0):
                    if (steps % max_steps == 0):
                        print("max steps reached")
                    if not self.is_ipython:
                        self.env.render()

                    last_q_val = self.critic(current_board)
                    self.optimize_model(last_q_val)
                    self.memory.clear()
                    self.episode_scores.append(self.env.score)
                    self.highest_scores.append(self.env.highest())
                    self.plot_scores()
                    break

        self.env.close()

    def write_results(self):
        try:
            os.makedirs(self.log_dir)
        except OSError:
            pass
        self.plot_scores(save=f"{self.log_dir}/plot.png")
        torch.save(self.actor, f"{self.log_dir}/actor.pt")
        torch.save(self.critic, f"{self.log_dir}/critic.pt")

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

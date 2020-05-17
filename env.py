import sys

import gym
import numpy as np
from gym import spaces
from gym.utils import seeding


class GameEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3

    def __init__(self, dim=4, seed=None, reward_class=None):
        self.size = dim
        self.action_space = spaces.Discrete(4)

        squares = self.size * self.size
        self.observation_space = spaces.Box(0, 2 ** squares, (self.size, self.size,), dtype=np.int)
        self.reward_range = (0., float(2 ** squares))
        self.seed(seed=seed)
        self.reset()
        self.reward_class = reward_class

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        try:
            score, changed = self._move(action)
            self.score += score
            self._add_tile()
            done = self._game_finished()
            if done:
                reward = self.reward_class.final_state_reward(score)
            else:
                reward = score
        except IllegalMove as e:
            done = False
            reward = self.reward_class.invalid_move_reward()

        observation = self.board
        info = {"max_tile": self.highest()}

        return observation, reward, done, info

    def reset(self):
        self.board = np.zeros((self.size, self.size), np.int)
        self.score = 0

        self._add_tile()
        self._add_tile()

        return self.board

    def render(self, mode='human'):
        """Rendering for standard output of score, highest tile reached and
        board-matrix of game."""
        outfile = sys.stdout
        s = 'Score: {}\n'.format(self.score)
        s += 'Highest: {}\n'.format(self.highest())
        npa = np.array(self.board)
        grid = npa.reshape((self.size, self.size))
        s += "{}\n".format(grid)
        outfile.write(s)
        return outfile

    def _add_tile(self):
        possible_values = np.array([2, 4])
        tile_probabilities = np.array([0.9, 0.1])
        val = np.random.choice(possible_values, 1, p=tile_probabilities)[0]
        empty_tiles = self._empty_tiles()
        choice = np.random.choice(empty_tiles.shape[0])
        choice = empty_tiles[choice]

        self.board[choice[0]][choice[1]] = val

    def _empty_tiles(self):
        return np.argwhere(self.board == 0)

    def _move(self, direction, test=False):
        changed = False
        move_score = 0.

        self._flip_board(direction)

        for i in range(self.size):
            new_line, line_score, line_changed = self._move_line(self.board[i])
            if not test:
                self.board[i] = new_line
            move_score += line_score
            changed |= line_changed

        self._flip_board(direction)
        if not changed:
            raise IllegalMove

        return move_score, changed

    def _game_finished(self) -> bool:
        for direction in range(4):
            try:
                self._move(direction, test=True)
                return False
            except IllegalMove:
                pass

        return True

    def _move_line(self, board_line: np.array):
        tiles = [val for val in board_line if val != 0]
        skip = False
        new_line = np.zeros(self.size, np.int)
        index = 0
        move_score = 0.
        for value_pair in zip(tiles, tiles[1:]):
            if skip:
                skip = False
                continue
            new_line[index] = value_pair[0]

            # combine tiles
            if value_pair[0] == value_pair[1]:
                new_line[index] += value_pair[1]
                move_score += self.reward_class.reward_function(value_pair)
                skip = True

            index += 1
        if len(tiles) > 0 and not skip:
            new_line[index] = tiles[-1]

        return (new_line, move_score, (new_line != board_line).any())

    def _flip_board(self, direction: int):
        if direction == self.LEFT:
            pass
        elif direction == self.RIGHT:
            self.board = np.flip(self.board, 1)
        elif direction == self.UP:
            self.board = self.board.T
        elif direction == self.DOWN:
            self.board = np.flip(self.board.T, 0)

    def highest(self):
        return np.max(self.board)


class IllegalMove(Exception):
    pass

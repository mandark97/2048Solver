import gym


class GameEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, dim):
        super(GameEnv, self).__init__()

    def step(self, action):
        pass

    def reset(self):
        pass

    def render(self, mode='human'):
        pass


import numpy as np


class Game(object):
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3

    def __init__(self, board_size: int = 4, start_tiles: int = 2):
        super().__init__()
        self.board_size = board_size
        self.start_tiles = start_tiles

    def _init_board(self):
        self.score = 0
        self.ended = False
        self.won = False
        self.board = np.zeros((self.board_size, self.board_size), np.int)

    def _empty_tiles(self):
        return np.argwhere(self.board == 0)

    def _add_tile(self):
        possible_values = np.array([2, 4])
        tile_probabilities = np.array([0.9, 0.1])
        val = np.random.choice(possible_values, 1, p=tile_probabilities)[0]
        empty_tiles = self._empty_tiles()
        choice = np.random.choice(empty_tiles.shape[0])
        choice = empty_tiles[choice]

        self.board[choice[0]][choice[1]] = val

    def _set(self, x: int, y: int, value: int) -> None:
        self.board[x][y] = value

    def move(self, direction: int):
        changed = False
        move_score = 0

        self.flip_board(direction)

        for i in range(self.board_size):
            new_line, line_score, line_changed = self.move_line(self.board[i])
            self.board[i] = new_line
            move_score += line_score
            changed |= line_changed

        self.flip_board(direction)
        # if self._game_finished():


    def _game_finished(self) -> bool:
        return np.where(self.board == 0)[0].size == 0

    def move_line(self, board_line: np.array):
        tiles = np.argwhere(board_line != 0)
        skip = False
        new_line = np.zeros(self.board_size, np.int)
        index = 0
        move_score = 0
        for value_pair in zip(tiles, tiles[1:]):
            if skip:
                skip = False
                continue
            new_line[index] = value_pair[0]

            # combine tiles
            if value_pair[0] == value_pair[1]:
                new_line[index] += value_pair[1]
                move_score += value_pair[0] + value_pair[1]
                skip = True

            index += 1
        if board_line and not skip:
            new_line[index] = board_line[-1]

        return (new_line, move_score, new_line == board_line)

    def flip_board(self, direction: int):
        if direction == self.LEFT:
            pass
        elif direction == self.RIGHT:
            self.board = np.flip(self.board, 1)
        elif direction == self.UP:
            self.board = self.board.T
        elif direction == self.DOWN:
            self.board = np.flip(self.board.T, 0)

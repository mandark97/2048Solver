from typing import Tuple
import numpy as np

class Reward1():
    @staticmethod
    def reward_function(score) -> float:
        return score / 80

    @staticmethod
    def final_state_reward(score: float = None) -> float:
        return -1.

    @staticmethod
    def invalid_move_reward() -> float:
        return 0.

class Reward3():
    @staticmethod
    def reward_function(value_pair: Tuple) -> float:
        return min(value_pair[0] / 4., 1.)

    @staticmethod
    def final_state_reward(score: float = None) -> float:
        return -1.

    @staticmethod
    def invalid_move_reward() -> float:
        return 0.


class Reward2():
    @staticmethod
    def reward_function(value_pair: Tuple) -> float:
        return min(value_pair[0] / 4., 1.)

    @staticmethod
    def final_state_reward(score: float = None) -> float:
        return score

    @staticmethod
    def invalid_move_reward() -> float:
        return -1.

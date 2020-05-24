from typing import Tuple


class Reward(object):
    @staticmethod
    def reward_function(value_pair: Tuple) -> float:
        raise NotImplementedError

    @staticmethod
    def final_state_reward(score=None) -> float:
        raise NotImplementedError

    @staticmethod
    def invalid_move_reward() -> float:
        raise NotImplementedError


class Reward1(Reward):
    @staticmethod
    def reward_function(value_pair: Tuple) -> float:
        return min(value_pair[0] / 4., 1.)

    @staticmethod
    def final_state_reward(score: float = None) -> float:
        return score

    @staticmethod
    def invalid_move_reward() -> float:
        return 0.

class Reward3(Reward):
    @staticmethod
    def reward_function(value_pair: Tuple) -> float:
        return min(value_pair[0] / 4., 1.)

    @staticmethod
    def final_state_reward(score: float = None) -> float:
        return -1

    @staticmethod
    def invalid_move_reward() -> float:
        return 0.


class Reward2(Reward):
    @staticmethod
    def reward_function(value_pair: Tuple) -> float:
        return min(value_pair[0] / 4., 1.)

    @staticmethod
    def final_state_reward(score: float = None) -> float:
        return score

    @staticmethod
    def invalid_move_reward() -> float:
        return -1.

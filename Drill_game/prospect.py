from abc import ABC
from statistics import mean
from typing import Tuple
import numpy as np
from tqdm import tqdm

from terrain import Terrain


class Prospect(ABC):
    def __init__(self, trials: int, terrain: Terrain, random_seed=None) -> None:
        super().__init__()
        self.trials = trials
        self.rewards = []
        self.terrain = terrain
        self.knowledge = np.zeros_like(terrain.grid)
        self.rng = np.random.default_rng(random_seed)

    def decide_next_coordinates(self) -> Tuple[int, int]:
        pass

    def prospect_once(self) -> None:
        x, y = self.decide_next_coordinates()
        self.rewards.append(((x, y), self.terrain.get_reward(x, y)))
        tile_draws = [reward for loc, reward in self.rewards if loc == (x, y)]
        self.knowledge[x, y] = mean(tile_draws)

    def prospect(self) -> None:
        for _ in tqdm(range(self.trials), desc=self.__class__.__name__):
            self.prospect_once()

    def get_total_reward(self) -> int:
        return sum([reward for _, reward in self.rewards])


class UniformProspect(Prospect):
    """Prospecting agent with a uniform policy"""

    def decide_next_coordinates(self) -> Tuple[int, int]:
        return self.terrain.get_random_coordinate()


class EGreedyProspect(Prospect):
    """Prospecting agent with an e-greedy policy"""

    def __init__(self, epsilon: float = 0.1, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.epsilon = epsilon

    def decide_next_coordinates(self) -> Tuple[int, int]:
        exploit = self.rng.random() >= self.epsilon
        if exploit:
            return self.rng.choice(
                np.transpose(np.nonzero(self.knowledge == np.max(self.knowledge)))
            )
        else:
            return self.terrain.get_random_coordinate()


class SoftmaxProspect(Prospect):
    """Prospecting agent with a softmax policy"""
    # TODO: random choice!

    def __init__(self, tau: float = 1, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.tau = tau

    def decide_next_coordinates(self) -> Tuple[int, int]:
        probs = np.exp(self.knowledge / self.tau)
        probs /= np.sum(probs)
        return self.rng.choice(np.transpose(np.nonzero(probs == np.max(probs))))


class UCB1Prospect(Prospect):
    """Prospecting agent with a UCB1 policy"""
    # TODO: random choice!

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.trials_count = np.zeros_like(self.terrain.grid)

    def prospect_once(self) -> None:
        x, y = self.decide_next_coordinates()
        self.rewards.append(((x, y), self.terrain.get_reward(x, y)))
        tile_draws = [reward for loc, reward in self.rewards if loc == (x, y)]
        self.knowledge[x, y] = mean(tile_draws)
        self.trials_count[x, y] += 1

    def decide_next_coordinates(self) -> Tuple[int, int]:
        n = len(self.rewards)
        if not n:
            return self.terrain.get_random_coordinate()
        probs = self.knowledge + np.sqrt(
            np.divide(
                2 * np.log(n),
                self.trials_count,
                out=np.zeros_like(self.trials_count),
                where=self.trials_count > 0,
            )
        )
        return self.rng.choice(np.transpose(np.nonzero(probs == np.max(probs))))

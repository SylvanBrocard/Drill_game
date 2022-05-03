from typing import Tuple
from scipy import stats
import numpy as np
from abc import ABC

class Terrain(ABC):
    pass

class DiscreteTerrain(Terrain):
    def __init__(self, X:int, Y:int, random_seed=None) -> None:
        self.X = X
        self.Y = Y
        self.rng = np.random.default_rng(random_seed)
        self.grid = self.rng.random(size=(X,Y))

    def get_reward(self, x:int, y:int) -> int:
        return stats.bernoulli.rvs(self.grid[x,y])

    def get_random_coordinate(self) -> Tuple[int, int]:
        return self.rng.choice(self.X), self.rng.choice(self.Y)

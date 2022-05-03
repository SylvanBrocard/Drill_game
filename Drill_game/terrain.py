from typing import Tuple
from scipy import stats
import numpy as np
from abc import ABC

class Terrain(ABC):
    pass

class DiscreteTerrain(Terrain):
    def __init__(self, X:int, Y:int) -> None:
        self.X = X
        self.Y = Y
        rng = np.random.default_rng()
        self._grid = rng.random(size=(X,Y))

    def get_reward(self, x:int, y:int) -> int:
        return stats.bernoulli.rvs(self._grid[x,y])

    def get_random_coordinates(self) -> Tuple[int, int]:
        return self.rng.randint(self.X), self.rng.randint(self.Y)

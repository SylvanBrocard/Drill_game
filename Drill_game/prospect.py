from abc import ABC
from statistics import mean
from typing import Tuple
import numpy as np

from prospect import Terrain

class Prospect(ABC):
    def __init__(self, trials:int, terrain:Terrain) -> None:
        super().__init__()
        self.trials = trials
        self.rewards = []
        self.terrain = terrain
        self.knowledge = np.empty_like(terrain.grid)

    def decide_next_coordinates(self) -> Tuple[int, int]:
        pass

    def prospect_once(self) -> None:
        x, y = self.decide_next_coordinates()
        self.rewards.append(((x,y),self.terrain.get_reward(x,y)))
        tile_draws = [reward for loc, reward in self.rewards if loc == (x,y)]
        self.knowledge[x,y] = mean(tile_draws)

    def prospect(self) -> None:
        for _ in range(self.trials):
            self.prospect_once()

    def get_total_reward(self) -> int:
        return sum([reward for _, reward in self.rewards])

class UniformProspect(Prospect):
    def decide_next_coordinates(self) -> None:
        return self.terrain.get_random_coordinate()

class EGreedyProspect(Prospect):
    pass

class UCB1Prospect(Prospect):
    pass
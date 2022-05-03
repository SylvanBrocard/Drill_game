import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from terrain import DiscreteTerrain

from prospect import Prospect, UniformProspect, EGreedyProspect, SoftmaxProspect

class Vizualizer():
    '''Classe du vizualiser afin de voir des graphiques de performances'''
    #prospect_perf matrice 
    def __init__(self,prospect_perf:np.ndarray) -> None:
        self.x, self.y = prospect_perf.shape
        self.prospect_perf = prospect_perf
        pass

    def graph_3d(self):
        ax = plt.axes(projection='3d') # Création d'un objet "axe 3D"
        for l in range(self.x):
            for c in range(self.y):
                ax.scatter(l, c, self.prospect_perf[l,c], c=self.prospect_perf[l,c])


    def graph_2D(self, prospect:Prospect) -> None:
        plt.plot(self.get_cumulative_sum(prospect))
        plt.grid()

    def get_cumulative_average(self, prospect:Prospect) -> float:
        return np.cumsum([reward for _, reward in prospect.rewards]) / np.arange(1, len(prospect.rewards) + 1)

    
    def get_cumulative_sum(self, prospect:Prospect) -> float:
        return np.cumsum([reward for _, reward in prospect.rewards])


if __name__ == '__main__':
    # matrice = np.array([[1,2,3],[2,5,4]])
    # viz = Vizualizer(matrice)

    # viz.graph_3d

    terrain = DiscreteTerrain(10, 10)

    for prospect_class in [UniformProspect, EGreedyProspect, SoftmaxProspect]:
        prospect = prospect_class(trials=100, terrain=terrain)
        prospect.prospect()
        viz = Vizualizer(prospect.knowledge)
        viz.graph_2D(prospect)
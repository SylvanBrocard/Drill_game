import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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


    def graph_2D(self):

np.matrice = [[1,2,3],[2,5,4]]
viz = Vizualizer(matrice)

viz.graph_3d
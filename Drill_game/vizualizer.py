import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Vizualizer():
    '''Classe du vizualiser afin de voir des graphiques de performances'''
    #prospect_perf matrice 
    def __init__(self,prospect_perf:np.ndarray) -> None:
        self.X, self.Y = prospect_perf.shape
        self.prospect_perf = prospect_perf
        pass

    def graph_3d(self):
        import matplotlib.pyplot as plt
        from matplotlib import cm
        from matplotlib.ticker import LinearLocator
        import numpy as np
        
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        # Make data.
        x_ = np.arange(self.X)
        y_ = np.arange(self.Y)
        x_, y_ = np.meshgrid(x_, y_)
        z_ = self.prospect_perf

        # Plot the surface.
        surf = ax.plot_surface(x_, y_, z_, cmap=cm.coolwarm, linewidth=0, antialiased=False)    

        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)

        plt.show()
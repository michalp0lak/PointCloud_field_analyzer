import numpy as np
from geomdl import BSpline
from geomdl import utilities
from geomdl.visualization import VisMPL
from sklearn.neighbors import KDTree
from matplotlib import cm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from configs.config_plot_stats_evaluator import plot_grid_Parameters
from configs.config_plot_stats_evaluator import surface_fit_Parameters

class VolumeEvaluator(object):

    #Median method -  as a reference height in given interval, median of raw points z-coordinate is taken
    #Surface method - NURBS surface fit is used to model surface of given field, in given interval median of z-coordinate of surface points is taken as reference height

    def __init__(self, points: np.ndarray, area_size: float, method: str, visualize: bool):

        assert (method == 'raw') or (method == 'surface'), 'Method of volume evaluation has to be median or surface'
        assert (type(area_size) == float) and (area_size > 0), 'Argument area_size has to be positive float value'

        self.points = points
        self.area_size = area_size
        self.method = method
        self.visualize = visualize


    def sliding_window(self, points: np.ndarray, stepSize: float, windowSize: float):
            # slide a window across the x-y hyperplane
            for x in np.arange(int(np.floor(points[:,0].min())), int(np.ceil(points[:,0].max())), stepSize):
                for y in np.arange(int(np.floor(points[:,1].min())), int(np.ceil(points[:,1].max())), stepSize):
                    # yield the current window
                    yield (points[(points[:,0] >= x) & (points[:,0] < x+windowSize) & (points[:,1] >= y) & (points[:,1] < y+windowSize)])


    def terrain_grid(self, points: np.ndarray, metric: str, K: int, grid_resolution: int):

        terrain_grid = []
        tree = KDTree(points[:,:2], metric = metric)

        x_min = np.floor(points[:,0].min())
        x_max = np.ceil(points[:,0].max())
        y_min = np.floor(points[:,1].min())
        y_max = np.ceil(points[:,1].max())

        for x in np.linspace(x_min, x_max, grid_resolution).tolist():

            for y in np.linspace(y_min, y_max, grid_resolution).tolist():

                point = np.array([x,y])

                _, index = tree.query(point.reshape(1, -1), k = K+1)
                index = index[0]
                terrain_grid.append([x,y,points[index,2].mean()])

        return np.array(terrain_grid)


    def surface_fit(self, terrain_grid, grid_resolution: int, u_degree: int, v_degree: int, delta: float):

        # Create a BSpline surface instance
        surf = BSpline.Surface()

        # Set evaluation delta
        surf.delta = delta

        # Set up the surface
        surf.degree_u = u_degree
        surf.degree_v = v_degree

        control_points = terrain_grid.tolist()
        surf.set_ctrlpts(control_points, grid_resolution, grid_resolution)

        surf.knotvector_u = utilities.generate_knot_vector(surf.degree_u, grid_resolution)
        surf.knotvector_v = utilities.generate_knot_vector(surf.degree_v, grid_resolution)

        # Evaluate surface points
        surf.evaluate()

        return surf

    def surface_visualizer(self, surface):

        #%matplotlib

        # Create a visualization configuration instance with no legend, no axes and set the resolution to 120 dpi
        vis_config = VisMPL.VisConfig(ctrlpts = False, axes_equal = False)
        # Create a visualization method instance using the configuration above
        vis_obj = VisMPL.VisSurface(vis_config)
        # Set the visualization method of the curve object
        surface.vis = vis_obj
        surface.render(colormap = cm.cool, plot=False)


    def compute_volume(self, points: np.ndarray, area_size: float):

        volume = 0

        for area in self.sliding_window(points, stepSize=area_size, windowSize=area_size):
            if area.shape[0]: volume += np.median(area[:,2]) * area_size**2

        return volume

    def _execute(self):

        pgP = plot_grid_Parameters()
        sfP = surface_fit_Parameters()

        volume = 0

        if self.method == 'raw':

            volume = self.compute_volume(self.points, self.area_size)

            stats = {'volume': volume, 'height_median': np.median(self.points[:,2]), 'height_variability': np.var(self.points[:,2])}

        elif self.method == 'surface':

            grid = self.terrain_grid(self.points, metric=pgP.METRIC, K = pgP.K, grid_resolution = pgP.GRID_RESOLUTION)
            surface = self.surface_fit(grid, grid_resolution = pgP.GRID_RESOLUTION, u_degree = sfP.U_DEGREE,
            v_degree = sfP.V_DEGREE, delta = sfP.DELTA)
            surface_points = np.array(surface.evalpts)

            volume = self.compute_volume(surface_points, self.area_size)

            stats = {'surface_volume': volume, 'surface_height_median': np.median(surface_points[:,2]), 'surface_height_variability': np.var(surface_points[:,2])}

            if self.visualize: self.surface_visualizer(surface)

        return stats
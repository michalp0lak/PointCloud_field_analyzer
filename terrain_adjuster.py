import os
import numpy as np
from laspy.file import File
from geomdl import BSpline
from geomdl import utilities
from geomdl.visualization import VisMPL
from sklearn.neighbors import KDTree
import random
from matplotlib import cm
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from utils import get_point_cloud
import time

from utils import save_img, get_point_cloud, store_point_cloud, get_header

from configs.config_terrain_adjuster import terrain_evaluator_Parameters
from configs.config_terrain_adjuster import outlier_detector_Parameters
from configs.config_terrain_adjuster import terrain_filter_Parameters
from configs.config_terrain_adjuster import terrain_grid_Parameters
from configs.config_terrain_adjuster import surface_fit_Parameters
from configs.config_global import global_Parameters

class TerrainEvaluator(object):

    def __init__(self, lasFile_path: str, filename: str, downsampling_rate: float, visualize = True):

        assert (type(lasFile_path) == str) and os.path.exists(lasFile_path), 'LASFILE_PATH parameter has to be string and existing folder directory'
        assert (type(filename) == str) and (filename.split('.')[1] == 'las') and os.path.exists(lasFile_path+filename), 'FILENAME parameter has to be string, it has to be las type format and it has to be localized in {}'.format(lasFile_path)
        assert (downsampling_rate > 0) and (downsampling_rate <= 1), 'DOWNSAMPLING_RATE parameter has to be bigger than 0 and smaller or equal than 1'

        self.path = lasFile_path
        self.filename = filename
        self.dw_rate = downsampling_rate
        self.header = get_header(self.path + self.filename)
        self.point_cloud = get_point_cloud(self.path+self.filename, self.header)
        self.dws_point_cloud = self.point_cloud[random.sample(range(0,self.point_cloud.shape[0]), round(self.point_cloud.shape[0]*self.dw_rate)),:]
        self.visualize = visualize
        

    def sliding_window(self, points: np.ndarray, stepSize: int, windowSize: int):
	
        # slide a window across the x-y hyperplane
        for x in range(int(np.floor(points[:,0].min())), int(np.ceil(points[:,0].max())), stepSize):
            for y in range(int(np.floor(points[:,1].min())), int(np.ceil(points[:,1].max())), stepSize):
                # yield the current window
                yield (points[(points[:,0] >= x) & (points[:,0] < x+windowSize) & (points[:,1] >= y) & (points[:,1] < y+windowSize)])


    def outlier_detector(self, points: np.ndarray, metric: str, method: str, deviance: float, radius: float, K: int):


        assert (type(metric) == str) and metric in ["euclidean", "manhattan", "chebyshev", "minkowski", "wminkowski", 
        "seuclidean", "mahalanobis"], 'Parameter metric has to be string value and one of(euclidean, manhattan, chebyshev, minkowski, wminkowski, seuclidean, mahalanobis)'
        assert method == 'perimeter' or method == 'nn', 'Parameter method has to be perimeter (neighbors are in given distance far away from examined point) or nn (fixed number of closest neighbors is taken)'
        assert (type(deviance) == float) and (deviance > 0), 'Parameter deviance has to be positive float'
        assert (type(radius) == float) and (radius > 0), 'Parameter radius has to be positive float value'
        assert (type(K) == int) and (K > 0), 'Parameter K has to be positive integer value'

        neighbour_dist = np.zeros_like(0, shape = points.shape[0])
        neighbourhood_size_flag = np.zeros_like(False, shape = points.shape[0])

        tree = KDTree(points[:,:2], metric = metric)

        for i, point in enumerate(tqdm(points)):

            if method == 'nn': 

                _, index = tree.query(point[:2].reshape(1, -1), k = K+1)
                index = index[0]
                neighbors = points[index,:]

            elif method == 'perimeter':

                index = tree.query_radius(point[:2].reshape(1, -1), radius)[0]
                neighbors = points[index, :]

                if neighbors.shape[0] < 5: neighbourhood_size_flag[i] = True
        
            point_frame = np.repeat(point.reshape(1,-1), repeats = neighbors.shape[0], axis=0)
            mean_dist = np.mean(np.abs(neighbors[: , 2] - point_frame[: , 2]))

            neighbour_dist[i] = mean_dist

        flag = neighbour_dist < neighbour_dist.mean() + deviance*neighbour_dist.var()**(0.5)
        flag[neighbourhood_size_flag] = False
        print(flag)
        return flag


    def terrain_filter(self, points: np.ndarray, method: str, window_size: int, window_stride: int, k: int, quantile: float):

        assert method == 'quantile' or method == 'k-values', 'Method argument has to be quantile (only given percentage of lowest points are selected) or k-values (k-lowest-values are selected) for terrain model'
        assert (type(window_size) == float) and (window_size > 0) and (window_size < np.min([np.max(points[:,0]), np.max(points[:,1])])), 'Window size can not extend size of analyzed area'
        assert (type(window_stride) == int) and (window_stride > 0) and (window_stride < window_size), 'Parameter window_stride has to be positive integer value'
        assert (type(k) == int) and (k > 0), 'Parameter k has to be positive integer value'
        assert (type(quantile) == float) and  (quantile > 0) and (quantile < 0.1), 'Parameter quantile has to be positive float value, but not bigger than 0.1'


        terrain_points = []

        for area in self.sliding_window(points, stepSize=window_stride, windowSize=window_size):

            if method == 'quantile' and area.shape[0] > 0: terrain_points.append(area[np.quantile(area[:,2], quantile) >= area[:,2],:])


            elif method == 'k-values' and area.shape[0] >= k:
        
                part = np.argpartition(area[:,2], k)
                terrain_points.append(area[part[:k],:])

            elif method == 'k-values' and area.shape[0] < k:

                terrain_points.append(area)
            
        return np.unique(np.vstack(terrain_points), axis=0)


    def terrain_grid(self, points: np.ndarray, metric: str, K: int, grid_resolution: int):

        assert metric in ["euclidean", "manhattan", "chebyshev", "minkowski", "wminkowski", "seuclidean", 
        "mahalanobis"], 'Parameter METRIC has to be string value and one of(euclidean, manhattan, chebyshev, minkowski, wminkowski, seuclidean, mahalanobis)'
        assert (type(K) == int) and (K > 0), 'Parameter K has to be positive integer value'
        assert (type(grid_resolution) == int) and (grid_resolution > 0), 'Parameter grid_resolution has to be positive integer value'

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

    def surface_fit(self, terrain_grid: np.ndarray, grid_resolution: int, u_degree: int, v_degree: int, delta: float):

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

    def _execute(self):

        print('Process of field surface evaluation started')
        start = time.time()
        if not os.path.exists(self.path + 'surface_evaluation/'): os.makedirs(self.path + 'surface_evaluation/')
        
        odP = outlier_detector_Parameters()
        tfP = terrain_filter_Parameters()
        tgP = terrain_grid_Parameters()
        sfP = surface_fit_Parameters()

        out_flag = self.outlier_detector(points = self.dws_point_cloud, metric = odP.METRIC,  method = odP.METHOD,
        radius = odP.RADIUS, deviance = odP.DEVIANCE, K = odP.K)
        
        store_point_cloud(self.path + 'surface_evaluation/' + "clean_roi.las", self.dws_point_cloud[out_flag,:], self.header)
        save_img(self.dws_point_cloud[out_flag,:],'clean_roi_0_0', self.path + 'surface_evaluation/', 0, 0)
        save_img(self.dws_point_cloud[out_flag,:],'clean_roi_0_90', self.path + 'surface_evaluation/', 0, 90)
        
        print('outliers removed .....')
        terrain_points = self.terrain_filter(points = self.dws_point_cloud[out_flag,:], method = tfP.METHOD,
        window_size = tfP.WINDOW_SIZE, window_stride = tfP.WINDOW_STRIDE, quantile = tfP.WINDOW_QUANTILE, k = tfP.K)
        print('terrain points founded .....')
        terrain_grid_points = self.terrain_grid(points = terrain_points, metric = tgP.METRIC, K = tgP.K,
        grid_resolution = tgP.GRID_RESOLUTION)
        print('terrain grid was formed .....')

        surface = self.surface_fit(terrain_grid = terrain_grid_points, grid_resolution = tgP.GRID_RESOLUTION,
        u_degree = sfP.U_DEGREE, v_degree = sfP.V_DEGREE, delta = sfP.DELTA)
        # Create a visualization configuration instance with no legend, no axes and set the resolution to 120 dpi
        vis_config = VisMPL.VisConfig(ctrlpts = False, axes_equal = False)
        # Create a visualization method instance using the configuration above
        vis_obj = VisMPL.VisSurface(vis_config)
        # Set the visualization method of the curve object
        surface.vis = vis_obj
        surface.render(filename = self.path + 'surface_evaluation/' + "terrain.png", colormap = cm.cool, plot=False)
        print('surface was modeled .....')

        

        # Get the evaluated points
        surface_points = np.array(surface.evalpts)
        print('de-terrain points process')
        
        deTerreained_points = np.zeros_like(self.dws_point_cloud[out_flag,:])

        for i, point in enumerate(tqdm(self.dws_point_cloud[out_flag,:])):

            clean_point = point.copy()
            point_dist = np.array(((surface_points[:,0]-point[0])**2 + (surface_points [:,1]-point[1])**2)**(0.5))

            clean_point[2] = clean_point[2] - surface_points[np.argmin(point_dist),2]
            deTerreained_points[i] = clean_point

        deTerreained_points[:,2] = deTerreained_points[:,2] + np.abs(deTerreained_points[:,2].min())
        store_point_cloud(self.path + 'surface_evaluation/' + "de-terrained_roi.las", deTerreained_points, self.header)

        save_img(deTerreained_points,'de-terrained_roi_0_0', self.path + 'surface_evaluation/', 0, 0)
        save_img(deTerreained_points,'de-terrained_roi_0_90', self.path + 'surface_evaluation/', 0, 90)
        print('point cloud was de-terrained ......')
        end = time.time()

        print('Time:{}'.format(end-start))
if __name__ == '__main__':

    gP = global_Parameters()
    teP = terrain_evaluator_Parameters()

    TH = TerrainEvaluator(lasFile_path = gP.LASFILE_PATH, filename = 'roi.las', downsampling_rate = teP.DOWNSAMPLING_RATE)
    TH._execute()
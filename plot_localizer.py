import os
import shutil
import numpy as np
import warnings
import random
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from tqdm import tqdm, trange
from utils import compute_signal, rotate_points, get_point_cloud, get_header
from fourier import get_peaks

warnings.filterwarnings("ignore")
#https://www.pythonpool.com/matplotlib-draw-rectangle/

from configs.config_global import global_Parameters
from configs.config_plot_localizer import plot_localizer_Parameters

class PlotLocalizer(object):

    def __init__(self, path: str, plots_num: int, signal_span: float):

        self.path = path
        self.header = get_header(self.path + 'roi.las')
        self.field = get_point_cloud(self.path + 'surface_evaluation/clean_field.las', self.header)
        self.plots_num = plots_num
        self.signal_span = signal_span

    def visualize_plots(self, points: np.ndarray, borders: np.ndarray, coord: int, x: float, y: float, 
    downsample_rate = 0.2, top = False):

        indexes = random.sample(range(0, points.shape[0]), round(points.shape[0]*downsample_rate))
        verts_side = []
        verts_top = []

        if not bool(coord):

            azimuth = 0
            xm = points[:,0].min()
            xx = points[:,0].max()
            zm = points[:,2].min()
            zx = points[:,2].max()

            for border in borders:

                verts_side.append([[xm, border, zm],[xm, border, zx]])
                verts_top.append([[xx, border, zx],[xm, border, zx]])

        if bool(coord):

            azimuth = 90
            ym = points[:,1].min()
            yx = points[:,1].max()
            zm = points[:,2].min()
            zx = points[:,2].max()


            for border in borders:

                verts_side.append([[border ,ym, zm],[border, ym, zx]])
                verts_top.append([[border, yx, zx],[border, ym, zx]])

        if top:

            plt.figure(figsize=[30, 20])
            ax = plt.axes(projection='3d')
            ax.view_init(90,azimuth)
            ax.scatter(points[indexes,0], points[indexes,1], points[indexes,2], c = points[indexes,2] , s= 1, marker='o')

            for vert in verts_top: ax.add_collection3d(Poly3DCollection(vert, facecolors='white', linewidths=1, edgecolors='r', alpha=.20))

            plt.savefig(self.path + 'plots/block_{}_{}.png'.format(x,y))

        else:

            plt.figure(figsize=[30, 20])
            ax = plt.axes(projection='3d')
            ax.view_init(0,azimuth)
            ax.scatter(points[indexes,0], points[indexes,1], points[indexes,2], c = points[indexes,2] , s= 1, marker='o')

            for vert in verts_side: ax.add_collection3d(Poly3DCollection(vert, facecolors='white', linewidths=1, edgecolors='r', alpha=.20))

            plt.savefig(self.path + 'plots/block_{}_{}.png'.format(x,y))


    def get_borders(self, points: np.ndarray, coord: int, subplots_num: int, signal_span: float):

        projection = compute_signal(points, signal_span, int(not bool(coord)))

        mins, maxs = get_peaks(projection[:,0], projection[:,1], method = 'numpy')

        if mins[0] is None:

            mins, maxs = get_peaks(projection[:,0], projection[:,1], method = 'lm')
            print('Lm model used for fourier fit')

        else: print('Numpy model used for fourier fit')

        if mins[0].shape[0] == subplots_num + 1: borders = mins[0]

        elif mins[0].shape[0] == subplots_num:

            if (mins[0][0] > maxs[0][0]) and (mins[0][-1] > maxs[0][-1]): borders = [np.max([mins[0][0] - np.median(np.diff(mins[0])), points[:,int( not bool(coord))].min()])] + list(mins[0])
            elif (mins[0][0] < maxs[0][0]) and (mins[0][-1] < maxs[0][-1]): borders = list(mins[0]) + [np.min([mins[0][-1] + np.median(np.diff(mins[0])), points[:,int(not bool(coord))].max()])]
            else: 

                if np.abs(mins[0][0]-points[:,int(not bool(coord))].min()) > np.abs(mins[0][-1]-points[:,int(not bool(coord))].max()): borders = [np.max([mins[0][0] - np.median(np.diff(mins[0])), points[:,int( not bool(coord))].min()])] + list(mins[0])
                elif np.abs(mins[0][0]-points[:,int(not bool(coord))].min()) < np.abs(mins[0][-1]-points[:,int(not bool(coord))].max()): borders = list(mins[0]) + [np.min([mins[0][-1] + np.median(np.diff(mins[0])), points[:,int(not bool(coord))].max()])]

        else: borders = np.linspace(projection[:,0].min(), projection[:,0].max(), subplots_num+1)

        return borders


    def _execute(self):

        print('Subplot localization of batch of plots started')
        if os.path.exists(self.path + 'plots/'): shutil.rmtree(self.path + 'plots/')
        if not os.path.exists(self.path + 'plots/'): os.makedirs(self.path + 'plots/')
        
        #Import metadata
        with open(self.path + 'block_location_metadata.json', "r") as read_file: data = json.load(read_file)

        for _, block in enumerate(tqdm(data['blocks'])):

            #rotate point clouds
            rot_field = rotate_points(self.field, data['rotation_angle'])
            points = rot_field[(rot_field[:,0] >= block['x_min']) & (rot_field[:,0] < block['x_max']) & (rot_field[:,1] >= block['y_min']) & (rot_field[:,1] < block['y_max'])]

            #compute borders
            borders = self.get_borders(points, data['dominant_coordinate'], self.subplots_num, self.signal_span)

            #Define plots metadata
            metafile = {'x': block['x_min'], 'y': block['y_min'], 'dominant_coordinate': data['dominant_coordinate'], 'rotation_angle': data['rotation_angle'], 'borders': list(borders)}
            #Save plots metadata to JSON
            with open(self.path + 'block_{}_{}_metadata.json'.format(block['x_min'], block['y_min']), "w") as write_file: json.dump(metafile, write_file)
            #Visualize result of subplot detection
            self.visualize_plots(points, borders, data['dominant_coordinate'], block['x_min'], block['y_min'])
            print('{}_{} block was processed'.format(block['x_min'], block['y_min']))
            
if __name__ == '__main__':

    gB = global_Parameters()
    pdP = plot_localizer_Parameters()

    PL = PlotLocalizer(gB.LASFILE_PATH, gB.PLOT_NUM, pdP.SIGNAL_SPAN)
    PL._execute()
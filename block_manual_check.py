import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import json
from laspy.file import File
import numpy as np
from utils import rotate_points
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

from configs.config_global import global_Parameters

class PlotBoundariesVisualizer(object):

    def __init__(self,path):

        self.path = path


    def visualizer(self):

        points = np.loadtxt(self.path + 'field_evaluation/labels.csv', delimiter=",")
        field_las = File(self.path + 'surface_evaluation/clean_field.las', mode='r')

        points[:,0] = points[:,0] - field_las.header.min[0]
        points[:,1] = points[:,1] - field_las.header.min[1]
        points[:,2] = points[:,2] - field_las.header.min[2]

        with open(self.path + "plots/plots_metadata.json", "r") as read_file: data = json.load(read_file)


        points_rot = rotate_points(points, data['rotation_angle'])

        plt.figure(figsize=[10, 7])
        ax = plt.axes()
    
        ax.scatter(points_rot[:,0], points_rot[:,1], c = points_rot[:,2], s = 0.1, marker='o')
        for plot in data['plots']: ax.add_patch(Rectangle(xy=(plot['x_min'], plot['y_min']), width=plot['x_max']-plot['x_min'], height=plot['y_max']-plot['y_min'], linewidth = 1, color = plot['color'][1], fill=False))

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.show()

if __name__ == '__main__':

    gP = global_Parameters()
    
    PBV = PlotBoundariesVisualizer(gP.LASFILE_PATH)
    PBV.visualizer()
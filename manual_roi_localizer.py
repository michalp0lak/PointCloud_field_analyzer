import os
import json
from laspy.file import File
from matplotlib.patches import Rectangle
import random
import matplotlib.pyplot as plt
import numpy as np
from utils import get_point_cloud, store_point_cloud, get_header

from configs.config_global import  global_Parameters
from configs.config_manual_roi_localizer import roi_cropper_Parameters

class RoiCropper(object):

    def __init__(self, path, filename, ds_rate, low_quantile, up_quantile):

        self.path = path
        self.filename = filename
        self.ds_rate = ds_rate
        self.low_quantile = low_quantile
        self.up_quantile = up_quantile
        self.header = get_header(self.path + self.filename)
        self.point_cloud = get_point_cloud(self.path + self.filename, self.header)
        
    def visualizer(self, downsample_rate = 0.01):

        if not os.path.exists(self.path + 'roi_metadata.json'):

            #Define ROI area
            coords = {'x_min': self.point_cloud[:,0].mean() - 1, 'x_max': self.point_cloud[:,0].mean() + 1,
            'y_min': self.point_cloud[:,1].mean() - 1, 'y_max': self.point_cloud[:,1].mean() + 1}
            #Save plots metadata to JSON
            with open(self.path + 'roi_metadata.json', "w") as write_file: json.dump(coords, write_file)


        with open(self.path + 'roi_metadata.json', "r") as read_file: coords = json.load(read_file)

        #Crop area with rectangle
        field = self.point_cloud[(self.point_cloud[:,0] >= coords['x_min']) & (self.point_cloud[:,0] < coords['x_max']) & 
        (self.point_cloud[:,1] >= coords['y_min']) & (self.point_cloud[:,1] < coords['y_max'])]
        store_point_cloud(self.path + "roi.las", field, self.header)
     

        #Visualize
        reduced_pc = self.point_cloud[(self.point_cloud[:,2] > np.quantile(self.point_cloud[:,2], self.low_quantile)) & (self.point_cloud[:,2] < np.quantile(self.point_cloud[:,2], self.up_quantile)),:]
        reduced_pc = reduced_pc[(reduced_pc[:,0] > np.quantile(reduced_pc[:,0], 0.01)) & (reduced_pc[:,0] < np.quantile(reduced_pc[:,0], 0.99)),:]
        reduced_pc = reduced_pc[(reduced_pc[:,1] > np.quantile(reduced_pc[:,1], 0.01)) & (reduced_pc[:,1] < np.quantile(reduced_pc[:,1], 0.99)),:]

        indexes = random.sample(range(0, reduced_pc.shape[0]), round(reduced_pc.shape[0]*self.ds_rate))

        fig = plt.figure(figsize=[30, 30])
        fig.patch.set_facecolor('white')
        fig.suptitle('Region of interest', fontsize=50)

        ax = plt.axes()
        ax.axis('equal')
        ax.set_aspect('equal')
        ax.set_xlabel('x', fontsize=35)
        ax.set_ylabel('y', fontsize=35)
        ax.tick_params(labelsize=30)

        ax.scatter(reduced_pc[indexes,0], reduced_pc[indexes,1], c = reduced_pc[indexes,2], s = 0.01, marker='o',  cmap='turbo')
        ax.add_patch(Rectangle(xy=(coords['x_min'], coords['y_min']), width = coords['x_max']-coords['x_min'], height=coords['y_max']-coords['y_min'], linewidth=1, color='red', fill=False))
        
        X = reduced_pc[:,0]
        Y = reduced_pc[:,1]

        ax.set_xlim(X.min(), X.max())
        ax.set_ylim(Y.min(), Y.max())

        plt.show()




if __name__ == '__main__':

    gP = global_Parameters()
    rcP = roi_cropper_Parameters()
    RC = RoiCropper(gP.LASFILE_PATH, gP.FILENAME, rcP.DOWNSAMPLING_RATE, rcP.LOW_HEIGHT_QUANTILE, rcP.UP_HEIGHT_QUANTILE)
    RC.visualizer()
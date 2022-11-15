import numpy as np
import random
import os, sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors
import json
from tqdm import tqdm, trange
import warnings
from sklearn.preprocessing import normalize
from scipy import stats
warnings.filterwarnings("ignore")

from utils import find_angle, rotate_points, determine_plots_orientation, compute_projections, compute_projection, compute_signal, get_header
from fourier import get_peaks
from configs.config_global import global_Parameters
from configs.config_block_localizer import block_localizer_Parameters

class BlockDetector(object):

    def __init__(self, path, block_num, signal_span_z, signal_span_e, edge_level, w, LAMBDA, q_dom, q_notdom):

        self.path = path
        self.header = get_header(self.path + 'roi.las')
        self.block_num = block_num
        self.signal_span_z = signal_span_z
        self.signal_span_e = signal_span_e
        self.edge_level = edge_level
        self.w = w
        self.LAMBDA = LAMBDA
        self.q_dom = q_dom
        self.q_notdom = q_notdom

    #function for creation of chunks (for plots color specification)
    def chunkIt(self, seq, num):

        avg = len(seq) / float(num)
        chunks = []
        last = 0.0

        while last < len(seq):
            chunks.append(seq[int(last):int(last + avg)])
            last += avg

        return chunks

    def get_colors(self):

        # Sort colors by hue, saturation, value and name.
        by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgb(color))),
                            name, color)
                            for name, color in mcolors.CSS4_COLORS.items())

        #get colors and its codes
        color_names = [(name, color) for hsv, name, color in by_hsv]

        #create chunks
        chunks = self.chunkIt(color_names, self.block_num)

        colors = []
        #sample random colors for plots
        for i in range(0, self.block_num): colors.append(random.sample(chunks[i], 1))

        return colors


    def find_plot_seeds(self, points: np.ndarray, block_num: int, coord: int, offset: float):

        #Seeds localization of given number of plots according in dominant coordinate
        dominant_signal = compute_signal(points, 0.01, int(coord))
        _, maxs_nu = get_peaks(dominant_signal[:,0], dominant_signal[:,1])
        dom_coords = list(maxs_nu[0])
        
        if len(dom_coords) == block_num:

            notdom_coords = []

            for seed in dom_coords:

                #Seeds localization in not dominant coordinate as maximu of z-coordinate signa;
                seed_area = points[(points[:,coord] > seed-offset) & (points[:,coord] < seed+offset),:]
                not_dominant_signal = compute_projection(seed_area, 0.01, int(not bool(coord)))
                notdom_coords.append(not_dominant_signal[np.argmax(not_dominant_signal[:,1]),0])

            if not coord: seeds = np.reshape(dom_coords + notdom_coords, (2, block_num)).T
            elif coord: seeds = np.reshape(notdom_coords + dom_coords, (2, block_num)).T

            return seeds

        else:

            print('Seeds were not find')

            seeds = None
            return seeds

    def get_edge_weight(self, edge_points: np.ndarray, width: float, lamb: float):

        Uniformity = []
        Power = []

        for point in edge_points:

            x_range = edge_points[(edge_points[:,0] > (point[0] - width)) & (edge_points[:,0] < (point[0] + width)),:]
            y_range = edge_points[(edge_points[:,1] > (point[1] - width)) & (edge_points[:,1] < (point[1] + width)),:]

            #Power atribute for x and y - how many other edge points is on similar level in x and y as analyzed point
            power = np.array([x_range.shape[0], y_range.shape[0]])

            #Uniformity atribute - testing uniformity of analyzed point level for x and y with KS test
            uniformity = np.array([stats.kstest(x_range[:,0], stats.uniform(loc = edge_points[:,0].min(), 
            scale = edge_points[:,0].max()).cdf, alternative = 'two-sided').pvalue,stats.kstest(y_range[:,1], 
            stats.uniform(loc = edge_points[:,1].min(), scale = edge_points[:,1].max()).cdf, alternative = 'two-sided').pvalue])

            Uniformity.append(uniformity)
            Power.append(power)

        Uniformity = np.array(Uniformity)
        Power = np.array(Power)

        #Normalizing uniformity atribute
        exp_x = int(str(np.median(Uniformity[:,0])).split('-')[1])
        exp_y = int(str(np.median(Uniformity[:,1])).split('-')[1])
        Uniformity[:,0] = Uniformity[:,0]**(1/exp_x)
        Uniformity[:,1] = Uniformity[:,1]**(1/exp_y)
        Uniformity = normalize(Uniformity, axis=0, norm='max')
        Power = normalize(Power, axis=0, norm='max')
        
        Weights = (lamb*Power + (1-lamb)*(1-Uniformity))

        return normalize(Weights, axis=0, norm='max')

    def find_block_rectangle(self, seed: np.ndarray, edge_points: np.ndarray, dominant_coord: int):


        if not dominant_coord:

            x_thresh = self.q_dom
            y_thresh = self.q_notdom

        else:

            x_thresh = self.q_notdom
            y_thresh = self.q_dom

        xp = compute_signal(edge_points[:,[0,1,2]], self.signal_span_e, 0)
        yp = compute_signal(edge_points[:,[0,1,3]], self.signal_span_e, 1)

        xpn = xp[xp[:,0] < seed[0],:]
        xpp = xp[xp[:,0] > seed[0],:]

        ypn = yp[yp[:,0] < seed[0],:]
        ypp = yp[yp[:,0] > seed[0],:]

        xpn[xpn[:,1] < np.quantile(xpn[:,1], x_thresh),1] = 0
        xpp[xpp[:,1] < np.quantile(xpp[:,1], x_thresh),1] = 0

        ypn[ypn[:,1] < np.quantile(ypn[:,1], y_thresh),1] = 0
        ypp[ypp[:,1] < np.quantile(ypp[:,1], y_thresh),1] = 0

        xn_peaks = []

        for i,x in enumerate(xpn):

            if i > 0 and i < xpn.shape[0]-1:

                if (xpn[i,1] > xpn[i-1,1]) and (xpn[i,1] > xpn[i+1,1]): xn_peaks.append(xpn[i,:])

        xp_peaks = []

        for i,x in enumerate(xpp):

            if i > 0 and i < xpp.shape[0]-1:

                if (xpp[i,1] > xpp[i-1,1]) and (xpp[i,1] > xpp[i+1,1]): xp_peaks.append(xpp[i,:])

        yn_peaks = []

        for i,y in enumerate(ypn):

            if i > 0 and i < ypn.shape[0]-1:

                if (ypn[i,1] > ypn[i-1,1]) and (ypn[i,1] > ypn[i+1,1]): yn_peaks.append(ypn[i,:])

        yp_peaks = []

        for i,y in enumerate(ypp):

            if i > 0 and i < ypp.shape[0]-1:

                if (ypp[i,1] > ypp[i-1,1]) and (ypp[i,1] > ypp[i+1,1]): yp_peaks.append(ypp[i,:])

        xn_peaks = np.array(xn_peaks)
        xp_peaks = np.array(xp_peaks)
        yn_peaks = np.array(yn_peaks)
        yp_peaks = np.array(yp_peaks)

        x_min = xn_peaks[np.argmin(seed[0] - xn_peaks[:,0]),0]
        x_max = xp_peaks[np.argmin(xp_peaks[:,0] - seed[0]),0]

        y_min = yn_peaks[np.argmin(seed[1] - yn_peaks[:,0]),0]
        y_max = yp_peaks[np.argmin(yp_peaks[:,0] - seed[1]),0]


        return x_min,x_max,y_min,y_max


    def form_blocks(self, points: np.ndarray, angle: float, dominant_coordinate: int):


        #Get crop points
        crop_points = points[points[:,3] == 1,:3]

        seeds = self.find_plot_seeds(crop_points, self.block_num, dominant_coordinate, offset = 1)

        #Get edge points
        edge_points = points[(points[:,4] < np.quantile(points[:,4], self.edge_level)),:]

        #Get edge point weights
        weights = self.get_edge_weight(edge_points, self.w, self.LAMBDA)
        edges = np.hstack([edge_points[:,:2],weights])

        #Get colors for plots visualizing
        colors = self.get_colors()

        blocks = []
        block_width = 0.45*((points[:,dominant_coordinate].max() - points[:,dominant_coordinate].min())/self.block_num)

        for i, seed in enumerate(tqdm(seeds)):

            seed_area = edges[(edges[:,dominant_coordinate] > seed[dominant_coordinate]-block_width) & (edges[:,dominant_coordinate] < seed[dominant_coordinate]+block_width),:]

            x_min,x_max,y_min,y_max = self.find_block_rectangle(seed, seed_area, dominant_coordinate)

            #define block metadata
            blocks.append({'color': colors[i][0], 'seed_x': seed[0], 'seed_y': seed[1],'x_min': x_min, 'x_max': x_max, 'y_min': y_min, 'y_max': y_max})

       
        #Define plots metadata
        metafile = {'dominant_coordinate': dominant_coordinate, 'rotation_angle': angle, 'blocks': blocks}
        #Save plots metadata to JSON
        with open(self.path + "block_metadata.json", "w") as write_file: json.dump(metafile, write_file)

        #Save plots visualization
        plt.figure(figsize=[15, 20])
        ax = plt.axes()
        ax.scatter(points[:,0], points[:,1], c = points[:,2], s = 0.1, marker='o')

        for block in blocks:
            
            ax.add_patch(Rectangle(xy=(block['x_min'], block['y_min']), width=block['x_max'] - block['x_min'], height=block['y_max'] - block['y_min'], linewidth = 3, color = block['color'][1], fill = False))
 
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.savefig(self.path +'block_boundaries.png')

    def _execute(self):


        print('Import of point clouds')

        points = np.loadtxt(self.path + 'field_evaluation/labels.csv', delimiter=",")
        points[:,0] = points[:,0] - self.header.min[0]
        points[:,1] = points[:,1] - self.header.min[1]
        points[:,2] = points[:,2] - self.header.min[2]

        print('PLOT DETECTION STARTED')
        print('##################################################################')
        if os.path.exists(self.path + 'block_location_metadata.json'): os.remove(self.path + 'block_location_metadata.json')

        #Find optimal rotation of plots just with crop points
        print('Computation of optimal point cloud rotation')
        #angle = find_angle(points[points[:,3] == 1,:3], self.signal_span_z)
        angle = 0.05688120403435537

        #Rotate point clouds with optimal angle
        points_rot = rotate_points(points, angle)

        #Decide blocks orientation
        px,py = compute_projections(points_rot, self.signal_span_z)
        dominant_coordinate = determine_plots_orientation(px,py)


        #In this part should be boarders of plots computed, now manual vaules are used
        #boundaries = [(6.6, 16.6, 1.7, 86.7), (18.4, 28.4, 1.7, 86.7),(30.6, 40.6, 1.9, 86.9),(42.6, 52.6, 2.3, 87.3), (54.9, 64.9, 2.5, 87.5)]
        #boundaries = [(7.5, 17.5, 0.5, 85.5), (19.3, 29.3, 0.5, 85.5),(31.5, 41.5, 0.7, 85.7),(43.5, 53.5, 1.1, 86.1), (55.8, 65.8, 1.3, 86.3)]

        print('Detection of block seeds and block boundaries')

        self.form_blocks(points_rot, angle, dominant_coordinate)


if __name__ == '__main__':

    gP = global_Parameters()
    bdP = block_localizer_Parameters()

    BD = BlockDetector(gP.LASFILE_PATH, gP.BLOCK_NUM, bdP.SIGNAL_SPAN_Z, bdP.SIGNAL_SPAN_E, bdP.EDGE_LEVEL,
    bdP.w, bdP.LAMBDA, bdP.DOMINANT_QUANTILE, bdP.NOTDOMINANT_QUANTILE)
    BD._execute()
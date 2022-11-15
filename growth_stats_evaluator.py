import numpy as np
from laspy.file import File
import json
from volume_evaluation import VolumeEvaluator
import pandas as pd
import os
import shutil
from tqdm import tqdm, trange
from utils import rotate_points, get_point_cloud, get_header
import warnings
warnings.filterwarnings("ignore")

from configs.config_global import global_Parameters
from configs.config_growth_stats_evaluator import cut_border_Parameters
from configs.config_growth_stats_evaluator import point_filter_Parameters
from configs.config_growth_stats_evaluator import volume_evaluator_Parameters

class ComputeFieldStats(object):

    def __init__(self, lasFile_path, major_border_pct: float, minor_border_pct:float, height_quantile: float,
    height_threshold: float, filter_method: str):

        assert (type(lasFile_path) == str) and os.path.exists(lasFile_path), 'Argument lasFile_path has to be string and existing folder directory'
        assert (type(major_border_pct) == float) and (major_border_pct > 0) and (major_border_pct <= 1), 'Argument major_border_pct has to be float value bigger than 0 and smaller than 1'
        assert (type(minor_border_pct) == float) and (minor_border_pct > 0) and (minor_border_pct <= 1), 'Argument minor_border_pct has to be float value bigger than 0 and smaller than 1'
        assert (type(height_quantile) == float) and (height_quantile > 0) and (height_quantile <= 1), 'Argument height_quantile has to be float value bigger than 0 and smaller than 1'
        assert (type(height_threshold) == float) and (height_threshold > 0), 'Argument height_threshold has to be positive float value'
        assert (filter_method == 'quantile') or (filter_method == 'threshold'), 'Argument filter_method has to be quantile or threshold'

        self.path = lasFile_path
        self.major_border_pct = major_border_pct
        self.minor_border_pct = minor_border_pct
        self.height_quantile = height_quantile
        self.height_threshold = height_threshold
        self.fmethod = filter_method

        self.header = get_header(self.path + 'field.las')
        self.field = get_point_cloud(self.path + 'surface_evaluation/de-terrained_field.las', self.header)


    def cut_border(self, points: np.ndarray, coord: int, major_border_pct: float, minor_border_pct: float):

        major_adjuster = ((points[:,coord].max() - points[:,coord].min()) * major_border_pct)/2
        minor_adjuster = ((points[:,int(not bool(coord))].max() - points[:,int(not bool(coord))].min()) * minor_border_pct)/2

        return points[(points[:,coord] > points[:,coord].min()+major_adjuster) & (points[:,coord] < points[:,coord].max()-major_adjuster) & (points[:,int(not bool(coord))] > points[:,int(not bool(coord))].min()+minor_adjuster) & (points[:,int(not bool(coord))] < points[:,int(not bool(coord))].max()-minor_adjuster),:]

    def low_point_filter(self, points: np.ndarray, method: str, height_quantile: float, height_threshold: float):

        if method == 'quantile': return points[points[:,2] > np.quantile(points[:,2], height_quantile),:]
        elif method == 'threshold': return points[points[:,2] > height_threshold,:]

    def _execute(self):

        if os.path.exists(self.path + 'structured_results/'): shutil.rmtree(self.path + 'structured_results/')
        if not os.path.exists(self.path + 'structured_results/'): os.makedirs(self.path + 'structured_results/')

        print('FIELD STATS EVALUATION IS BEING PROCESSED')

        print('Data import')
        with open(self.path + 'block_{}_{}_metadata.json', "r") as read_file: data = json.load(read_file)

        rot_field = rotate_points(self.field, data['rotation_angle'])

        for _, plot in enumerate(tqdm(data['plots'])):

            structured_data = []
            veP = volume_evaluator_Parameters()

            with open(self.path + 'plots/block_{}_{}_metadata.json'.format(plot['x_min'], plot['y_min']), "r") as read_file: border_data = json.load(read_file)
            
            plot_points = rot_field[(rot_field[:,0] > plot['x_min']) & (rot_field[:,0] < plot['x_max']) & 
            (rot_field[:,1] > plot['y_min']) & (rot_field[:,1] < plot['y_max'])]

            print('Plot features evaluation')
            for i, _ in enumerate(tqdm(border_data['borders'])):

                if i < len(border_data['borders']) - 1:

                    #Computation of experimental unit statistic
                    raw_subplot = plot_points[(plot_points[:,int(not bool(data['dominant_coordinate']))] > border_data['borders'][i]) & (plot_points[:,int(not bool(data['dominant_coordinate']))] < border_data['borders'][i+1])]
                    cropped_subplot = self.cut_border(raw_subplot, data['dominant_coordinate'], self.major_border_pct, self.minor_border_pct)

                    cropped_subplot_size = (cropped_subplot[:,int(not bool(data['dominant_coordinate']))].max() - cropped_subplot[:,int(not bool(data['dominant_coordinate']))].min()) * (cropped_subplot[:,data['dominant_coordinate']].max() - cropped_subplot[:,data['dominant_coordinate']].min())

                    cleaned_cropped_subplot  = self.low_point_filter(cropped_subplot, method = self.fmethod, 
                    height_quantile = self.height_quantile,height_threshold = self.height_threshold)

                    CRVE = VolumeEvaluator(cleaned_cropped_subplot, veP.BLOCK_SIZE, 'raw', False)
                    raw_stat = CRVE._execute()
                    cropped_raw_EV = raw_stat['volume']/cropped_subplot_size

                    CSVE = VolumeEvaluator(cleaned_cropped_subplot, veP.BLOCK_SIZE, 'surface', False)
                    surf_stat = CSVE._execute()
                    cropped_surface_EV = surf_stat['surface_volume']/cropped_subplot_size
                    
                    x_min = cropped_subplot[:,0].min()
                    x_max = cropped_subplot[:,0].max()
                    y_min = cropped_subplot[:,1].min()
                    y_max = cropped_subplot[:,1].max()

                    center_x = np.mean(cropped_subplot[:,0])
                    center_y = np.mean(cropped_subplot[:,1])


                    structured_data.append(dict(zip(('plot', 'x_min', 'x_max', 'y_min', 'y_max','x_center','y_center',
                    'volume', 'height_median', 'height_variability', 'height_EV', 'surface_volume', 'surface_height_median',
                    'surface_height_variability', 'surface_height_EV'),
                    ('{}_{}'.format(border_data['x'], border_data['y']), x_min, x_max, y_min, y_max, center_x, center_y,
                    raw_stat['volume'], raw_stat['height_median'], raw_stat['height_variability'], cropped_raw_EV,
                    surf_stat['surface_volume'], surf_stat['surface_height_median'], surf_stat['surface_height_variability'],
                    cropped_surface_EV
                    ))))

            df = pd.DataFrame(structured_data)
            df = df[['plot', 'x_min', 'x_max', 'y_min', 'y_max','x_center','y_center', 'volume', 'height_median', 
                    'height_variability', 'height_EV', 'surface_volume', 'surface_height_median', 'surface_height_variability', 
                    'surface_height_EV']]

            df.to_excel(self.path + 'structured_results/' + '/block_{}_{}_stats.xlsx'.format(border_data['x'],border_data['y']),engine='openpyxl')
            print('Plot was succesfully analyzed and structured dataframe block_{}_{}_stats.xlsx exported'.format(border_data['x'],border_data['y']))


if __name__ == '__main__':

    gP = global_Parameters()
    cbP = cut_border_Parameters()
    pfP = point_filter_Parameters()

    CFS = ComputeFieldStats(gP.LASFILE_PATH, cbP.CROP_QUANTILE_DOMINANT, cbP.CROP_QUANTILE_NOTDOMINANT, pfP.HEIGHT_QUANTILE,
    pfP.HEIGHT_THRESHOLD, pfP.METHOD)
    CFS._execute()
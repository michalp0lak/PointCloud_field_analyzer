
import numpy as np
import random
import os

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.neighbors import KDTree

from featurizer import CloudFeaturizer
from classifier import VegetationClassifier
from edge_detector import EdgeDetector

from utils import save_img, get_point_cloud, store_point_cloud, get_header
from configs.config_cloud_evaluator import cloud_evaluator_Parameters
from configs.config_cloud_evaluator import cloud_featurizer_Parameters
from configs.config_cloud_evaluator import edge_detector_Parameters
from configs.config_global import global_Parameters

class CloudEvaluator(object):

    def __init__(self, lasFile_path: str, filename: str, downsampling_rate: float):

        self.path = lasFile_path
        self.filename = filename
        self.dw_rate = downsampling_rate
        self.header = get_header(self.path + 'roi.las')
        self.point_cloud = get_point_cloud(self.path + self.filename, self.header)
        self.dws_point_cloud = self.point_cloud[random.sample(range(0,self.point_cloud.shape[0]), round(self.point_cloud.shape[0]*self.dw_rate)),:]

    def _execute(self):

        if not os.path.exists(self.path + 'field_evaluation/'): os.makedirs(self.path + 'field_evaluation/')

        print('EVALUATION OF POINT CLOUD STARTED')

        #Initialize CloudFeaturizer class
        cfP = cloud_featurizer_Parameters()
        featurizer = CloudFeaturizer(self.dws_point_cloud, cfP.METRIC, cfP.DIMENSIONS)

        print('POINT FEATURIZATION')
        #Compute features
        features = featurizer.collect_cloud_features()
        fkeys = list(features.keys())

        print('POINT CLASSIFICATION')
        #Initialize VegetationClassifier class
        classifier = VegetationClassifier(self.dws_point_cloud, features)
        #Compute classification
        crop_label, valid_idx = classifier.classify_vegetation()
        #sample valid (classified) points for further analysis
        classified_points = self.dws_point_cloud[valid_idx,:]
        #Visualize na save classification result
        save_img(np.hstack((classified_points, crop_label.reshape(-1,1))), 'classification', self.path + 'field_evaluation/', 90, 0)
        
        print('EDGE DETECTION')
        #Initialize EdgeDetector class
        edP = edge_detector_Parameters()
        detector = EdgeDetector(classified_points, crop_label, edP.METRIC,edP.ENTROPY_QUANTILE, edP.K)
        edge_label = detector.get_edge_points()
        save_img(np.hstack((classified_points, edge_label.reshape(-1,1))), 'edge', self.path + 'field_evaluation/', 90, 0)

        valid_points = classified_points
        valid_points[:,0] = valid_points[:,0] + self.header.min[0]
        valid_points[:,1] = valid_points[:,1] + self.header.min[1]
        valid_points[:,2] = valid_points[:,2] + self.header.min[2]

        np.savetxt(self.path + 'field_evaluation/' + '{}_features.csv'.format(fkeys[0]), features[fkeys[0]])
        np.savetxt(self.path + 'field_evaluation/' + '{}_features.csv'.format(fkeys[1]), features[fkeys[1]])
        np.savetxt(self.path + 'field_evaluation/' + 'labels.csv', np.hstack((valid_points, crop_label.reshape(-1,1), edge_label.reshape(-1,1))), delimiter=",")
        
        print('EVALUATION OF POINT CLOUD FINISHED')
        
if __name__ == '__main__':

    ceP = cloud_evaluator_Parameters()
    gP = global_Parameters()
    CE = CloudEvaluator(gP.LASFILE_PATH, ceP.FILENAME, ceP.DOWNSAMPLING_RATE)
    CE._execute()
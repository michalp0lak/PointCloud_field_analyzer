import os

class cloud_evaluator_Parameters(object):

    def __init__(self):

        """
        Parameter FILENAME determines name of module output las file.
        """
        self.FILENAME = 'surface_evaluation/de-terrained_roi.las'

        """
        Parameter DOWNSAMPLING_RATE determines percentage number of downsampled points of de-terrained points
        which will be processed with module. Since point clouds are huge files in general, to use 
        subset of points can significantly accelerate computation and decrease processing time.
        """
        self.DOWNSAMPLING_RATE = 0.8


class cloud_featurizer_Parameters(object):

    def __init__(self):

        """
        Parameter METRIC determines neighborhood shape of analyzed point. For neighborhood evaluation we are using KDTree, so
        metrics available in sklearn.neighbors.DistanceMetric can be used in our software.
        """
        self.METRIC = 'minkowski'

        """
        Parameter DIMENSIONS determines which axis (x,y,z) are used for neighborhood computation. We are using xy-plane 
        to determine neighboring points
        """
        self.DIMENSIONS = [0,1]

        """
        Parameter RADIUS defines neighbourhood points of analyzed point for perimeter neighbourhood method. 
        Evaluation of features for analyzed point is based on these points.
        """
        self.RADIUS = 0.07

        """
        Parameter K defines neighbourhood points of analyzed point for k-nearest neighbors neighbourhood method.
        Evaluation of features for analyzed point is based on these points.
        """
        self.K = 30


class edge_detector_Parameters(object):

    def __init__(self):

        """
        Parameter METRIC determines neighbourhood shape of analyzed point. For neighbourhood evaluation we are using KDTree, so
        metrics available in sklearn.neighbors.DistanceMetric can be used in our software.
        """
        self.METRIC = 'minkowski'
 
        """
        Parameter K defines fixed number of closest points to analyzed point, which are used for
        computation of edge_entropy criterium for given point.
        """
        self.K = 50

        """
        Parameter ENTROPY_QUANTILE defines percentage of points with smallest edge_entropy criterium.
        These points are considered as edge points.
        """
        self.ENTROPY_QUANTILE = 0.01
import os

class roi_cropper_Parameters(object):

    def __init__(self):

        """
        Parameter DOWNSAMPLING_RATE determines percentage number of downsampled points of raw point cloud
        which will be visualized with module. Since point clouds are huge files in general, to use 
        subset of points can significantly accelerate computation and decrease processing time.
        """
        self.DOWNSAMPLING_RATE = 0.01


        """
        Parameter LOW_HEIGHT_QUANTILE determines percentage amount of points with lowest height, which are not displayed
        for manual ROI localization.
        """
        self.LOW_HEIGHT_QUANTILE = 0.05


        """
        Parameter UP_HEIGHT_QUANTILE determines percentage amount of points with biggest height, which are not displayed
        for manual ROI localization.
        """
        self.UP_HEIGHT_QUANTILE =  0.995

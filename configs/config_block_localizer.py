import os


class block_localizer_Parameters(object):

    def __init__(self):

        """
        Parameter SIGNAL_SPAN_Z defines size of step (in meters) for x and y axis, which is used for z-coordinate
        signal computation. Value of z-coordinate of points localized in the range is used to compute some statistic 
        (sum, mean, etz.). Smaller SIGNAL_SPAN_Z parameter means detail we are able detect.
        Big SIGNAL_SPAN values are not recommended, it can cause loss of structure in computed signal.
        """
        self.SIGNAL_SPAN_Z = 0.01

        """
        Parameter SIGNAL_SPAN_E defines size of step (in meters) for x and y axis, which is used for edge
        signal computation. Value of edge weight of points localized in the range is used to compute some statistic 
        (sum, mean, etz.). Smaller SIGNAL_SPAN_E parameter means detail we are able detect.
        Big SIGNAL_SPAN values are not recommended, it can cause loss of structure in computed signal.
        """
        self.SIGNAL_SPAN_E = 0.01


        """
        Parameter EDGE_LEVEL determines percentage of most significant points based on edge entropy, which are considered as
        edge points and used for block boundaries detection.
        """
        self.EDGE_LEVEL = 0.01

        """
        Parameter w defines the size (in meters) of analyzed edge point area in x and y coordinate.
        """
        self.w = 0.1

        """
        Parameter LAMBDA defines the smoothing coefficient defining the mutual significancy of cardinality and 
        uniformity for the edge points weight computation.
        """
        self.LAMBDA = 0.5

        """
        Parameter DOMINANT_QUANTILE determines the percentage of reduced edge signal values in the dominant
        coordinate, which are not considered as the candidates for the experimental block border.
        """
        self.DOMINANT_QUANTILE = 0.8

        """
        Parameter NOTDOMINANT_QUANTILE determines the percentage of reduced edge signal values in not-dominant
        coordinate, which are not considered as the candidates for the experimental block border.
        """
        self.NOTDOMINANT_QUANTILE = 0.98
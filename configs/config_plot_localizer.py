import os


class plot_localizer_Parameters(object):

    def __init__(self):

        """
        Parameter SIGNAL_SPAN_Z defines size of step (in meters) for x and y axis, which is used for z-coordinate
        signal computation. Value of z-coordinate of points localized in the range is used to compute some statistic 
        (sum, mean, etz.). Smaller SIGNAL_SPAN_Z parameter means detail we are able detect.
        Big SIGNAL_SPAN values are not recommended, it can cause loss of structure in computed signal.
        """
        self.SIGNAL_SPAN = 0.01
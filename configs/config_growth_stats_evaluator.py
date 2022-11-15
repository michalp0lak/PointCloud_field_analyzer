import os

class cut_border_Parameters(object):

    def __init__(self):

        """
        Parameter CROP_QUANTILE_DOMINANT determines percentage part which is removed from size of dominant coordinate.
        It means that half of MAJOR_BORDER_PCT value is removed from lower and half from upper part of subplot. The 
        reason is to remove noisy borders of subplot.
        """
        self.CROP_QUANTILE_DOMINANT = 0.04

        """
        Parameter CROP_QUANTILE_NOTDOMINANT determines percentage part which is removed from size of not dominant coordinate.
        It means that half of MINOR_BORDER_PCT value is removed from lower and half from upper part of subplot. The 
        reason is to remove noisy borders of subplot.
        """
        self.CROP_QUANTILE_NOTDOMINANT = 0.30


class point_filter_Parameters(object):

    def __init__(self):

        """
        Parameter METHOD determines the way how are points with low z-coordinate value cleaned oout from subplot point cloud.
        Quantile method filters out given percentage of lowest points. Threshold method filters out points up to certain level
        of z-coordinate value.
        """
        self.METHOD = 'quantile'

        """
        Parameter HEIGHT_QUANTILE filters out from subplot point cloud given percentage of points with lowest value
        of z-coordinate.
        """
        self.HEIGHT_QUANTILE = 0.2

        """
        Parameter HEIGHT_THRESHOLD filters out from subplot point cloud points with value of z-coordinate lower than
        HEIGHT_THRESHOLD value.
        """
        self.HEIGHT_THRESHOLD = 0.3


class plot_grid_Parameters(object):

    def __init__(self):
 
        """
        Parameter METRIC determines neighborhood shape of analyzed point. For neighborhood evaluation we are using KDTree, so
        metrics available in sklearn.neighbors.DistanceMetric can be used in our software.
        """
        self.METRIC = 'minkowski'

        """
        Parameter K defines fixed number of closest terrain points to given grid point, which are used for
        computation computation of grid point z-coordinate value as mean value of z-coordinate of k-closest
        terrain points.
        """
        self.K = 10

        """
        Parameter GRID_RESOLUTION defines point density of terrain grid. GRID_RESOLUTION is number of equidistantly
        distributed points in x axis range and the same number of points is euqidistantly distributed in y axis range.
        In total gird is formed with GRID_RESOLUTION x GRID_RESOLUTION points.
        """
        self.GRID_RESOLUTION = 100


class surface_fit_Parameters(object):

    """
    NURBS-Python (geomdl) is a self-contained, object-oriented pure Python B-Spline and NURBS library with implementations
    of curve, surface and volume generation and evaluation algorithms. It also provides convenient and easy-to-use data
    structures for storing curve, surface and volume descriptions.

    In order to generate a spline shape with NURBS-Python, you need 3 components: degree, knot vector, control points.
    The number of components depend on the parametric dimensionality of the shape regardless of the spatial dimensionality.
    Surface is parametrically 2-dimensional (or 2-manifold).
    """

    def __init__(self):
            
        """
        Parameter U_DEGREE defines polynomial order of spline for first dimension of parametrical space of surface.
        """
        self.U_DEGREE = 2

        """
        Parameter V_DEGREE defines polynomial order of spline for second dimension of parametrical space of surface.
        """
        self.V_DEGREE = 2

        """
        Evaluation delta is used to change the number of evaluated points. Increasing the number of points will
        result in a bigger evaluated points array and decreasing will reduce the size of the array. Therefore,
        evaluation delta can also be used to change smoothness of the plots generated using the visualization modules.
        """
        self.DELTA = 0.01

class volume_evaluator_Parameters(object):

     def __init__(self):
            
        """
        Parameter AREA_SIZE defines size of squared sliding window, which is sliding around subplot area and is used for
        volume computation in this small area of sliding window. The principle of subplot volume computation is the same as
        as integration. The volume is sum of volumes calculated in all sliding window areas in subplot area
        """
        self.BLOCK_SIZE = 0.1


        """
        Parameter METHOD determines the way, how volume is computed in area of sliding window. Raw method computes height 
        of block as median of z-coordinate. Surface method uses B-Spline fitted on the surface of subplot. Height of block
        given by sliding window is calculated as median of B-Spline values in given area.
        """
        self.METHOD = 'surface'

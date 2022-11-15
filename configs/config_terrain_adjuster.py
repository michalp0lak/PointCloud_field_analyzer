import os

class terrain_evaluator_Parameters(object):

    def __init__(self):

        """
        Parameter DOWNSAMPLING_RATE determines percentage number of downsampled points of raw points
        which will be processed with module. Since point clouds are huge files in general, to use 
        subset of points can significantly accelerate computation and decrease processing time.
        """
        self.DOWNSAMPLING_RATE = 1


class outlier_detector_Parameters(object):

    def __init__(self):

        """
        Parameter METRIC determines neighborhood shape of analyzed point. For neighborhood evaluation we are using KDTree, so
        metrics available in sklearn.neighbors.DistanceMetric can be used in our software.
        """
        self.METRIC = 'minkowski'

        """
        Parameter METHOD determines the algorithm, how neighbors of analyzed point are defined. First approach (nn) k-nearest neighbors
        selects fixed value of closest points. Second approach (perimeter) selects points in given distance from analyzed points.
        """
        self.METHOD = 'perimeter'

        """
        Parameter DEVIANCE is essential for outlier detection. For each point average euclidean distance of z-coordinate in its 
        perimeter neighbourhood is computed. We got sample of average euclidean distance of z-coordinate and we compute mean and standard
        deviation of the sample. DEVIANCE parameter defines how many standard deviations higher than mean distance has to be distance of 
        single point to consider point as outlier.
        """
        self.DEVIANCE = 3.0

        """
        Parameter RADIUS defines size of neighbourhood for perimeter neighbourhood method.
        """
        self.RADIUS = 0.2

        """
        Parameter K defines number of neighbours for k-nearest neighbors neighbourhood method.
        """
        self.K = 50


class terrain_filter_Parameters(object):

    def __init__(self):

        """
        Parameter METHOD determines the algorithm, how points with lowest value of z-coordinate are selected. Quantile method selects given
        percentage of lowest points. K-values method selects fixed number of points with lowest z-coordinate value.
        """
        self.METHOD = 'quantile'

        """
        Parameter WINDOW_SIZE defines size (in meters) of square sliding window in xy-plane), which used
        for terrain points selection in given area of window. It's important to define the WINDOW_SIZE big enough,
        so terrain points will be always in the window area. It should not happen that filtered terrain points in
        window area will include points of crop.
        """
        self.WINDOW_SIZE = 10.0

        """
        Parameter WINDOW_STRIDE defines how is sliding window moved around analyzed area. Sliding windows starts in
        origin of coordinates and is moved in x and y coordinate direction with given step (in meters). This step
        is WINDOW_STRIDE.
        """
        self.WINDOW_STRIDE = 2

        """
        Parameter K defines fixed number of points with lowest value of z-coordinate in sliding window, which
        are selected as terrain points.
        """
        self.K = 30

        """
        Parameter QUANTILE defines percentage of points with lowest value of z-coordinate in sliding window, which
        are selected as terrain points.
        """
        self.WINDOW_QUANTILE = 0.01


class terrain_grid_Parameters(object):

    def __init__(self):
 
        """
        Parameter METRIC determines neighbourhood shape of analyzed point. For neighbourhood evaluation we are using KDTree, so
        metrics available in sklearn.neighbors.DistanceMetric can be used in our software.
        """
        self.METRIC = 'minkowski'

        """
        Parameter K defines fixed number of closest terrain points to given grid point, which are used for
        computation of grid point z-coordinate.
        """
        self.K = 30

        """
        Parameter GRID_RESOLUTION defines point density of terrain grid. GRID_RESOLUTION is number of equidistantly
        distributed points in x axis range and the same number of points is euqidistantly distributed in y axis range.
        In total gird is formed with GRID_RESOLUTION x GRID_RESOLUTION points 
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

     
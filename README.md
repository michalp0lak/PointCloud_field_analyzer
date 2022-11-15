# UAV-crop-analyzer

## Authors
Michal Polak [1]

[1] Palacky Univ, Fac Sci, Laboratory of Growth Regulators, Olomouc 77900, Czech Republic

## Introduction

UAV-crop-analyzer is a software developed for purpose of growth analysis of various crops based on laser signal. Expected input is point cloud in **las** format. Each module of processing pipeline generates its own output. Last module generates structured csv file providing information about crop growth in specific locations. Software assumes regular grid of rectangular-shaped blocks-plots, this is key assumption for succesfull analysis. Software was prototyped with data generated with **Ricopter** UAV machine manufactured by **Riegl** enterprise. **UAV-crop-analyzer** is composed of 6 independent modules and its goal is to compute growth statistics of individual plots in experimental blocks. Advantage of software is that algorithms used in **UAV-crop-analyzer** are *unsupervised* and it's use is *semi-automated*. Each module have several parameters configurable by user, which effects final output of module. The software time performance needs to be optimizied with parallelization and GPU usage.

## Modules

All modules (Python script) are individually callable from server terminal.

### 1. Manual ROI localization
Usage of first **manual_roi_localizer.py** module is optional and depends on scanned area. The purpose of this module is to define rectangular-shape region of interest (ROI) for further analysis. This module visualizes raw point cloud (with a given downsampling rate) and generates a **roi_metadata.json** file. The user defines x-y coordinates of rectanglar-shape ROI and controls visualization with 3 parameters. The user should define ROI as small as possible around field, so unnecessary noisy points are not included.

```
**roi_cropper_Parameters**

Parameter DOWNSAMPLING_RATE determines percentage number of downsampled points of raw point cloud
which will be visualized with module. Since point clouds are huge files in general, to use 
subset of points can significantly accelerate computation and decrease processing time.

Parameter LOW_HEIGHT_QUANTILE determines percentage amount of points with lowest height, which are not displayed
for manual ROI localization.

Parameter UP_HEIGHT_QUANTILE determines percentage amount of points with biggest height, which are not displayed
for manual ROI localization.
```


![alt text](https://github.com/UPOL-Plant-phenotyping-research-group/UAV-crop-analyzer/blob/main/readme_images/ROI.png?raw=true)

### 2. Terrain adjustment

As the ROI terrain is not naturally flat, it would effect plot growth statistic values. Because of this fact a digital terrain trend removing step was included into the algorithm to remove terrain global trend of ROI. This is performed via the **terrain\_adjuster.py** module. This module performs two essential steps, Outliers removal and Terrain effect removal. First, outliers are detected and removed from the raw ROI point cloud. After outlier removal cleaned ROI point cloud **clean\_roi.las** file is generated.

##### Raw point cloud of field
![alt text](https://github.com/UPOL-Plant-phenotyping-research-group/UAV-crop-analyzer/blob/main/readme_images/field.png?raw=true)

##### Point cloud of field without outliers
![alt text](https://github.com/UPOL-Plant-phenotyping-research-group/UAV-crop-analyzer/blob/main/readme_images/clean_field.png?raw=true)


Further, it is necessary to compute the digital terrain model (DTM) and use it for the terrain effect removal, to allow the measuring of the canopy height and not the effect in the terrain beneath it. This procedure consists of several steps. Points belonging to the terrain are identified by a square sliding window. The window defined by user-defined *size* and *stride* slides around the ROI and in each window area a percentage of points with the lowest z-coordinate value is sampled as "terrain" points with user-defined parameter the *window quantile*. The *size* parameter of sliding window is important parameter and has to be determined by user according to shape and size of the ROI. The window size should be big enough, so it cannot happen that in the window will be only points denoting crops. It is recommended to define size of the sliding window so as to be at least as big as smaller dimension of rectangular-shaped experimental block.

##### Terrain points
![alt text](https://github.com/UPOL-Plant-phenotyping-research-group/UAV-crop-analyzer/blob/main/readme_images/terrain_points.png?raw=true)

As next the regular terrain grid is formed. Important parameter of grid is the *grid resolution*, it determines the density of points in the grid. Bigger resolution means more detailed and precise estimation of the DTM. 

##### Terrain grid
![alt text](https://github.com/UPOL-Plant-phenotyping-research-group/UAV-crop-analyzer/blob/main/readme_images/terrain_grid.png?raw=true)

The final step of the DTM evaluation is fitting the surface spline to grid points using [https://nurbs-python.readthedocs.io/en/5.x/](NURBS) Python library and generating set of spline points.
				
##### Terrain fit with B-spline
![alt text](https://github.com/UPOL-Plant-phenotyping-research-group/UAV-crop-analyzer/blob/main/readme_images/terrain_spline.png?raw=true)

Then spline points are used to remove the terrain effect of cleaned ROI points.

##### Deterrained field
![alt text](https://github.com/UPOL-Plant-phenotyping-research-group/UAV-crop-analyzer/blob/main/readme_images/detrendim.tiff?raw=true)

```
**terrain_evaluator_Parameters**

Parameter DOWNSAMPLING_RATE determines percentage number of downsampled points of raw points
which will be processed with module. Since point clouds are huge files in general, to use 
subset of points can significantly accelerate computation and decrease processing time.


**outlier_detector_Parameters**

Parameter METRIC determines neighborhood shape of analyzed point. For neighborhood evaluation we are using KDTree, so
metrics available in sklearn.neighbors.DistanceMetric can be used in our software.

Parameter METHOD determines the algorithm, how neighbors of analyzed point are defined. First approach (nn) k-nearest neighbors
selects fixed value of closest points. Second approach (perimeter) selects points in given distance from analyzed points.

Parameter DEVIANCE is essential for outlier detection. For each point average euclidean distance of z-coordinate in its 
perimeter neighbourhood is computed. We got sample of average euclidean distance of z-coordinate and we compute mean and standard
deviation of the sample. DEVIANCE parameter defines how many standard deviations higher than mean distance has to be distance of 
single point to consider point as outlier.

Parameter RADIUS defines size of neighbourhood for perimeter neighbourhood method.

Parameter K defines number of neighbours for k-nearest neighbors neighbourhood method.


**terrain_filter_Parameters**

Parameter METHOD determines the algorithm, how points with lowest value of z-coordinate are selected. Quantile method selects given
percentage of lowest points. K-values method selects fixed number of points with lowest z-coordinate value.

Parameter WINDOW_SIZE defines size (in meters) of square sliding window in xy-plane), which used
for terrain points selection in given area of window. It's important to define the WINDOW_SIZE big enough,
so terrain points will be always in the window area. It should not happen that filtered terrain points in
window area will include points of crop.

Parameter WINDOW_STRIDE defines how is sliding window moved around analyzed area. Sliding windows starts in
origin of coordinates and is moved in x and y coordinate direction with given step (in meters). This step
is WINDOW_STRIDE.

Parameter K defines fixed number of points with lowest value of z-coordinate in sliding window, which
are selected as terrain points.

Parameter QUANTILE defines percentage of points with lowest value of z-coordinate in sliding window, which
are selected as terrain points.


**terrain_grid_Parameters**

Parameter METRIC determines neighbourhood shape of analyzed point. For neighbourhood evaluation we are using KDTree, so
metrics available in sklearn.neighbors.DistanceMetric can be used in our software.

Parameter K defines fixed number of closest terrain points to given grid point, which are used for
computation of grid point z-coordinate.

Parameter GRID_RESOLUTION defines point density of terrain grid. GRID_RESOLUTION is number of equidistantly
distributed points in x axis range and the same number of points is euqidistantly distributed in y axis range.
In total gird is formed with GRID_RESOLUTION x GRID_RESOLUTION points.


**surface_fit_Parameters**

Parameter U_DEGREE defines polynomial order of spline for first dimension of parametrical space of surface.

Parameter V_DEGREE defines polynomial order of spline for second dimension of parametrical space of surface.

Evaluation delta is used to change the number of evaluated points. Increasing the number of points will
result in a bigger evaluated points array and decreasing will reduce the size of the array. Therefore,
evaluation delta can also be used to change smoothness of the plots generated using the visualization modules.
```

### 3. Point featurization, classification and edge detection

The following module of the pipeline a **cloud_evaluator.py** performs the point featurization, the ground-crop classification and the edge points detection. For each de-terrained point the neighbourhood of the k-nearest neighbours and the neighbourhood of given radius is determined with the KDTree algorithm. For each neighbourhood [https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html?highlight=pca#sklearn.decomposition.PCA](PCA) is applied, to compute eigenvalues and eigenvectors. On the basis of computed eigenvalues and eigenvectors several, features are computed (sum of eigenvalues sum, omnivariance, eigenentropy, anisotropy, planarity, linearity, surface variation, sphericity, verticality, first order moment, average distance in neighbourhood). It's convenient to use downsampling to reduce the time of processing. It's possible to localize the experimental blocks with significantly reduced amount of point. After the  de-terrained points featurization, two sets of features are generated. Set for the radius neighbourhood and set for the k-nearest neighbours neighbourhood. For both sets of features the [https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html?highlight=aussianmixture#examples-using-sklearn-mixture-gaussianmixture](Expectation–Maximization clustering using GMM) algorithm with two components is evaluated. Based on average height of clusters, crop class and the ground class is easily determined for the both clusterings. A point is considered to be crop class only if it was clustered as this class for both sets of features.

![alt text](https://github.com/UPOL-Plant-phenotyping-research-group/UAV-crop-analyzer/blob/main/readme_images/classification.png?raw=true)
				
To precisely localize the experimental blocks, it's necessary to detect its rectangular-shaped boundaries. Because of this it seems convenient to try detect edge points in the featurized point cloud. We tried to exploit the existing algorithms for edge point detection in point cloud like, but it didn't produce nicely structured edges of the experimental blocks. New method for the exp. block edges detection was developed and it's using result of the clustering from previous step. All e-terrained points are labeled with 0 (ground point) or 1 (crop point).	To detect the edge point, each labeled point is analyzed on its local neighbourhood. As a first the k-nearest neighbours in xy-plane is determined with the given *metric* and *K* parameters using the KDTree algorithm. Then neighbourhood is divided into the two sub-neighbourhoods using the SVM (Support vector machine) with the linear kernel. Using the labels as information for the linear SVM supervised training, the optimal separation of neighborhood in analyzed point is given with the plane (linear kernel).
				
![alt text](https://github.com/UPOL-Plant-phenotyping-research-group/UAV-crop-analyzer/blob/main/readme_images/svm_plane.png?raw=true)

		
The sub-neighborhood with the higher amount of crop points and the second with less crop points is determined. It is expected that on the borders of blocks, plane will separate space in the way that majority of ground points will occur under the plane and majority of crop points above the plane (or opposite). The *edge entropy* is non-negative and lower the value is, point is more considered as edge point. User can determine edge points with the quantile which specifies a percentage of points with the lowest value of *edge entropy*.
				
![alt text](https://github.com/UPOL-Plant-phenotyping-research-group/UAV-crop-analyzer/blob/main/readme_images/edges.png?raw=true)

```
**cloud_evaluator_Parameters**

Parameter DOWNSAMPLING_RATE determines percentage number of downsampled points of de-terrained points
which will be processed with module. Since point clouds are huge files in general, to use 
subset of points can significantly accelerate computation and decrease processing time.

**cloud_featurizer_Parameters**

Parameter METRIC determines neighborhood shape of analyzed point. For neighborhood evaluation we are using KDTree, so
metrics available in sklearn.neighbors.DistanceMetric can be used in our software.

Parameter DIMENSIONS determines which axis (x,y,z) are used for neighborhood computation. We are using xy-plane 
to determine neighboring points

Parameter RADIUS defines neighbourhood points of analyzed point for perimeter neighbourhood method. 
Evaluation of features for analyzed point is based on these points.

Parameter K defines neighbourhood points of analyzed point for k-nearest neighbors neighbourhood method.
Evaluation of features for analyzed point is based on these points.

**edge_detector_Parameters**

Parameter METRIC determines neighbourhood shape of analyzed point. For neighbourhood evaluation we are using KDTree, so
metrics available in sklearn.neighbors.DistanceMetric can be used in our software.

Parameter K defines fixed number of closest points to analyzed point, which are used for
computation of edge_entropy criterium for given point.

Parameter ENTROPY_QUANTILE defines percentage of points with smallest edge_entropy criterium.
These points are considered as edge points.
```


### 4. Block localization

The module **block\_localizer.py** performs the experimental blocks localization. Our method assumes the regular grid of experimental blocks with rectangular shape. First, points classified as crop points are used for the computation of the optimal point cloud rotation and its orientation. The ranges of coordinate x and y of is divided into certain number of intervals with the user-defined *signal span* parameter which determines the size (in meters) of these intervals. In each interval the mean value of the z-coordinate (the height signal) for the x axis and the y axis is computed. For points rotation in the xy-plane, the rotation matrix is defined. The objective is to find such a rotation, which maximizes the sum of the height signal variation in the rotated coordinates x' and y'.

![alt text](https://github.com/UPOL-Plant-phenotyping-research-group/UAV-crop-analyzer/blob/main/readme_images/crop_rotation.png?raw=true)

Then with the optimal value of rotation, the crop points  are rotated into the new coordinate system  x'y'z. The coordinate of x'y'-plane with the higher value of the height signal variation is defined as *dominant* and coordinate with the lower value as *not-dominant*. The experimental blocks orientation is determined with the *dominant*/*not-dominant* coordinate evaluation. In this way the block borders in the rotated coordinate system will be parallel with the x' and y' axis, so it can be defined with the rectangular-shaped border. 
			
Next step is the localization of the experimental block seeds. The block seed is a point which is located within the experimental block border. The height signal with the user-defined *signal span* parameter evaluated in the *dominant* coordinate of the rotated crop points is used as signal for the seeds localization. The seed locations are computed with the Fourier transform as the coordinate of the curve maximum peaks. The user has to specify the number of blocks, it is the important input for the block localization. Without the correct value it's not possible to localize the experimental blocks.

![alt text](https://github.com/UPOL-Plant-phenotyping-research-group/UAV-crop-analyzer/blob/main/readme_images/rotated_vegetation.png?raw=true)

![alt text](https://github.com/UPOL-Plant-phenotyping-research-group/UAV-crop-analyzer/blob/main/readme_images/plot_signal.png?raw=true)

After the seeds localization in x'y'-plane the edge points are rotated into the $x'y'z$ coordinate system using the $R_{xy}$ rotation matrix. For each analyzed edge point two features are evaluated for the x' coordinate and the y' coordinate. First feature the *cardinality* expresses number of points in narrow interval with center in given coordinate. Second feature the *uniformity* measures the uniformity of  narrow interval with center in given coordinate. Kolmogorov-Smirnov test was used to describe uniformity of given coordiante. Evaluated features are used to compute edge point weights as weighted average of features. The normalized weights of the edge points and the exp. block seeds are used for the exp. block borders localization. The border rectangular shape is assumed, it means that we need to find *x'_{min}, y'_{min}, x'_{max}, y'_{max}* border values in the x'y'z coordinate system to determine the block location. For each seed, seed are is determined as area around the seed defined with *dominant* coordinate range. The ranges of the coordinates x' and y' of seed area are divided into the certain number of intervals with the user-defined *signal span* parameter which determines the size (in meters) of these intervals. For each interval of the x' and y' coordinate, the mean value of normalized weight is computed. The seed area edge signal is divided into the four sub-areas (smaller than seed x' coordinate, bigger than seed x' coordinate, smallr than seed y' coordinate, bigger than seed y' coordinate). Further depending on the exp. blocks orientation, edge signal for x' and for y' is reduced to 0 with the user-defined *dominant quantile* and *not-dominant quantile* parameters. Edge signal values lower than these quantiles are reduced to 0. Based on the exp. blocks orientation, one quantile is *dominant quantile* and second *not-dominant quantile*. For the remaining active (positive edge signal) points, stationary points are identified. The stationary points of four defined areas closest to seed are determined as exp. block border.

![alt text](https://github.com/UPOL-Plant-phenotyping-research-group/UAV-crop-analyzer/blob/main/readme_images/plots.png?raw=true)

The result of the block localization is saved in the *block_metadata.json* file. If automatic block localization fails, there is the possibility of defining the block borders manually. The **block_manual_check.py** module visualizes a top view of the ROI and provides the user with x' and y' coordinate information via the mouse cursor. In this way the user can find coordinates of rectangular-shaped borders. This information has to be modified in the **block_metadata.json** file 
(via arbitrary text editor modify *x'_{min}, y'_{min}, x'_{max}, y'_{max}* attributes of given blocks and save file changes). 

```
**block_localizer_Parameters**

Parameter SIGNAL_SPAN_Z defines size of step (in meters) for x and y axis, which is used for z-coordinate
signal computation. Value of z-coordinate of points localized in the range is used to compute some statistic 
(sum, mean, etz.). Smaller SIGNAL_SPAN_Z parameter means detail we are able detect.
Big SIGNAL_SPAN values are not recommended, it can cause loss of structure in computed signal.
 
Parameter SIGNAL_SPAN_E defines size of step (in meters) for x and y axis, which is used for edge
signal computation. Value of edge weight of points localized in the range is used to compute some statistic 
(sum, mean, etz.). Smaller SIGNAL_SPAN_E parameter means detail we are able detect.
Big SIGNAL_SPAN values are not recommended, it can cause loss of structure in computed signal.

Parameter EDGE_LEVEL determines percentage of most significant points based on edge entropy, which are considered as
edge points and used for block boundaries detection.

Parameter w defines the size (in meters) of analyzed edge point area in x and y coordinate.

Parameter LAMBDA defines the smoothing coefficient defining the mutual significancy of cardinality and 
uniformity for the edge points weight computation.

Parameter DOMINANT_QUANTILE determines the percentage of reduced edge signal values in the dominant
coordinate, which are not considered as the candidates for the experimental block border.

Parameter NOTDOMINANT_QUANTILE determines the percentage of reduced edge signal values in not-dominant
coordinate, which are not considered as the candidates for the experimental block border.
```


### 5. Plot localization

After the exp. blocks localization, each block is identically analyzed individually. In this step of the processing pipeline we perform localization of the plots in the raw (not de-terrained) block area with the **plot_localizer.py** module. Again, it is assumed that a single block is made up of a certain number of rectangular-shaped and parallel plots separated by small gaps. The height signal with the user-defined *signal span* parameter is evaluated in *not-dominant* coordinate of rotated x'y'z coordinate system. Plot borders location in *not-dominant* coordinate is computed with Fourier transform as coordinates of curve minimum peaks. The user has to specify the number of plots, it is important input for the plot localization. The result of the plot border localization is saved in the **block_x_y_metadata.json** file, where x and y specify experimental block.

![alt text](https://github.com/UPOL-Plant-phenotyping-research-group/UAV-crop-analyzer/blob/main/readme_images/subplots.png?raw=true)

To find structure of subplots *Fourier transform* is applied on raw (not de-terrained) point cloud of plot.

![alt text](https://github.com/UPOL-Plant-phenotyping-research-group/UAV-crop-analyzer/blob/main/readme_images/subplot_fourier.png?raw=true)

And with little adjusment borders of subplots can be defined.

![alt text](https://github.com/UPOL-Plant-phenotyping-research-group/UAV-crop-analyzer/blob/main/readme_images/subplot_borders.png?raw=true)

```
**plot_localizer_Parameters**

Parameter SIGNAL_SPAN_Z defines size of step (in meters) for x and y axis, which is used for z-coordinate
signal computation. Value of z-coordinate of points localized in the range is used to compute some statistic 
(sum, mean, etz.). Smaller SIGNAL_SPAN_Z parameter means detail we are able detect.
Big SIGNAL_SPAN values are not recommended, it can cause loss of structure in computed signal.
```

### 6. Growth statistic evaluation

The last part of pipeline is **growth_stats_evaluator.py** module which performs the growth statistic evaluation for each plot. It analyses a whole batch of plots of a single block and creates a structured result in *xlsx* format. For the growth analysis only de-terrained rotated points are used. Points of the single exp. block are cropped from de-terrained point cloud with the border coordinates. Single plot is cropped from the exp. block with the borders in *not-dominant* coordinate.

![alt text](https://github.com/UPOL-Plant-phenotyping-research-group/UAV-crop-analyzer/blob/main/readme_images/subplot.png?raw=true)

There are two preprocessing steps applied on the plot point cloud. First, border area in x'y'-plane of plot is cropped. This is defined by the user with *crop quantile dominant* and *crop quantile not dominant* parameters. Second step is to remove points which has the low height value. These points are not considered as crop points and are filtered out with *height quantile* parameter. After the cropping and the low point filtering, the plot point cloud is cleaned and prepared for the growth statistic evaluation. 

![alt text](https://github.com/UPOL-Plant-phenotyping-research-group/UAV-crop-analyzer/blob/main/readme_images/cleaned_subplot.png?raw=true)

There are two sets of the growth statistics generated by software. First set is evaluated with the raw points of cleaned plot points. Second approach is using surface B-spline. It fits spline to cleaned plot points and evaluate the growth statistics with the surface spline points. Again Python NURBS library is used for the B-spline fitting. First, regular grid for cleaned plot point cloud is constructed with user-defined *grid resolution* parameter. Height of each grid point is computed as median of the k-nearest neighbours of cleaned plot points determined with the KDTree algorithm and user-defined *K* parameter. The spline fitting to the grid points is again configured with the parameters *u*, *v* and *delta*. 
		
![alt text](https://github.com/UPOL-Plant-phenotyping-research-group/UAV-crop-analyzer/blob/main/readme_images/crop_surface.png?raw=true)		
		
Set of the raw features and set of the spline-based features is evaluated. For both approaches 4 growth statistics are evaluated for each plot. Median of the z-coordinate, variance of the z-coordinate, volume and height expected value. The median is standard 0.5 quantile and the variance is standard sum of squared deviations. The volume is sum of the little blocks defined in x'y'-plane of plot rectangular area. Blocks fully cover plot area and don’t overlap. Blocks have square shape and it’s size is determined by user-defined  *block size* parameter. The height of each block is derived as the median of the z-coordinate in the given block area. The expected value of height is easily determined as the volume divided by plot area.

![alt text](https://github.com/UPOL-Plant-phenotyping-research-group/UAV-crop-analyzer/blob/main/readme_images/heatmap.png?raw=true)

```
**cut_border_Parameters**

Parameter CROP_QUANTILE_DOMINANT determines percentage part which is removed from size of dominant coordinate.
It means that half of MAJOR_BORDER_PCT value is removed from lower and half from upper part of subplot. The 
reason is to remove noisy borders of subplot.

Parameter CROP_QUANTILE_NOTDOMINANT determines percentage part which is removed from size of not dominant coordinate.
It means that half of MINOR_BORDER_PCT value is removed from lower and half from upper part of subplot. The 
reason is to remove noisy borders of subplot.

**point_filter_Parameters**

Parameter METHOD determines the way how are points with low z-coordinate value cleaned oout from subplot point cloud.
Quantile method filters out given percentage of lowest points. Threshold method filters out points up to certain level
of z-coordinate value.

Parameter HEIGHT_QUANTILE filters out from subplot point cloud given percentage of points with lowest value
of z-coordinate.

Parameter HEIGHT_THRESHOLD filters out from subplot point cloud points with value of z-coordinate lower than
HEIGHT_THRESHOLD value.

**plot_grid_Parameters**

Parameter METRIC determines neighborhood shape of analyzed point. For neighborhood evaluation we are using KDTree, so
metrics available in sklearn.neighbors.DistanceMetric can be used in our software.

Parameter K defines fixed number of closest terrain points to given grid point, which are used for
computation computation of grid point z-coordinate value as mean value of z-coordinate of k-closest
terrain points.

Parameter GRID_RESOLUTION defines point density of terrain grid. GRID_RESOLUTION is number of equidistantly
distributed points in x axis range and the same number of points is euqidistantly distributed in y axis range.
In total gird is formed with GRID_RESOLUTION x GRID_RESOLUTION points.

**surface_fit_Parameters**

Parameter U_DEGREE defines polynomial order of spline for first dimension of parametrical space of surface.

Parameter V_DEGREE defines polynomial order of spline for second dimension of parametrical space of surface.


Evaluation delta is used to change the number of evaluated points. Increasing the number of points will
result in a bigger evaluated points array and decreasing will reduce the size of the array. Therefore,
evaluation delta can also be used to change smoothness of the plots generated using the visualization modules.

**volume_evaluator_Parameters**

Parameter AREA_SIZE defines size of squared sliding window, which is sliding around subplot area and is used for
volume computation in this small area of sliding window. The principle of subplot volume computation is the same as
as integration. The volume is sum of volumes calculated in all sliding window areas in subplot area

Parameter METHOD determines the way, how volume is computed in area of sliding window. Raw method computes height 
of block as median of z-coordinate. Surface method uses B-Spline fitted on the surface of subplot. Height of block
given by sliding window is calculated as median of B-Spline values in given area.

```

## TO DO / IMPROVEMENTS 

-   Software computation acceleration: Numba, GPU, Concurent programming, definiton of new arrays with predefined dimensions (not arbitrary)
-   local point density paremeter used in neghborhood computations
-   better understanding of feature selection
-   not just rectangular shapes or grid shapes of fields -> contour detection
-   improvement of subplot localization algorithm


## INSTALLATION

### Prerequisities
1. As a first install Python up to date version (not older than **3.9.X**) on your server. Follow instructions [of this website] (https://realpython.com/installing-python) to install python required version.
3. Python virtualenv package is required. Open terminal and execute `python3 -m pip install virtualenv` (for Unix) or `py -m pip install --user virtualenv` (for Windows) command to install this package.

### Configure local environment
Create your own local environment, for more see https://pipa.io/en/latest/user_guide/, and install dependencies requirements.txt contains list of packages and can be installed as

```
Michals-MacBook-Pro:Repos michal$ cd UAV-crop-analyzer/ 
Michals-MacBook-Pro:UAV-crop-analyzer michal$ `python -m venv UAV`
Michals-MacBook-Pro:UAV-crop-analyzer michal$ source UAV/bin/activate  
Michals-MacBook-Pro:UAV-crop-analyzer michal$ pip install -r requirements.txt
Michals-MacBook-Pro:UAV-crop-analyzer michal$ deactivate  
```

## Use

### Activalte local virtual environment

```
Michals-MacBook-Pro:Repos michal$ cd UAV-crop-analyzer/ 
Michals-MacBook-Pro:UAV-crop-analyzer michal$ source UAV/bin/activate  
```

### Configure global variables
In configs folder edit with text editor file **config_global.py** and configure global variablles:
```
Parameter LASFILE_PATH navigates software in folder, where target las file is
stored and where output files of this module will be generated.
```

```
Parameter FILENAME determines name of processed las file.
```

```
Parameter PLOT_NUM is number of experimental blocks in field.
```

```
Parameter SUBPLOT_NUM is number of experimental units in single experimental block.
```


### Manual ROI localizer
To manually localize field/region of interest in raw point cloud, use **manual_field_localizer.py** module.

1. Call module from terminal: ```(UAV) Michals-MacBook-Pro:UAV-crop-analyzer michal$ python manual_field_localizer.py```
2. Find x-y coordinates of rectangular field with visualization tool and in created file **field metadata.json** modify borders of ractangle.
3. Run module again to visually check field borders: ```(UAV) Michals-MacBook-Pro:UAV-crop-analyzer michal$ python manual_field_localizer.py```


### Terrain adjuster
To remove outlier and de-terrain point cloud of field, use **terrain_adjuster.py** module. 

1. In configs folder edit with text editor file **config_terrain_adjuster.py** and configure module parameters.
2. Call module from terminal: ```(UAV) Michals-MacBook-Pro:UAV-crop-analyzer michal$ python terrain_adjuster.py```

### Cloud evaluator
To evaluate features of de-terrain field point cloud, use **cloud_evaluator.py** module. 

1. In configs folder edit with text editor file **config_cloud_evaluator.py** and configure module parameters.
2. Call module from terminal: ```(UAV) Michals-MacBook-Pro:UAV-crop-analyzer michal$ python cloud_evaluator.py```
  
### Block localizer
To localize plots/experimental blocks in field point cloud, use **block_localizer.py** module. 

1. In configs folder edit with text editor file **config_plot_localizer.py** and configure module parameters (we are going to mention just important parameters):
2. Call module from terminal: ```(UAV) Michals-MacBook-Pro:UAV-crop-analyzer michal$ python block_localizer.py```


### Plot localizer
To localize plots in block point cloud, use **plot_localizer.py** module. 

1. In configs folder edit with text editor file **config_plot_localizer.py** and configure module parameters.
2. Call module from terminal: ```(UAV) Michals-MacBook-Pro:UAV-crop-analyzer michal$ python plot_localizer.py```

### Growth stats evaluator
To evaluate growth statistics for all plots in given plot point cloud (it's possible to evaluate batch of blocks), use **growth_stat_evaluator.py** module.

1. In configs folder edit with text editor file **config_growth_stat_evaluator.py** and configure module parameters. 
2. Call module from terminal: ```(UAV) Michals-MacBook-Pro:UAV-crop-analyzer michal$ python growth_stat_evaluator.py```

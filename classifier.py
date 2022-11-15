from sklearn.mixture import GaussianMixture
import numpy as np
from sklearn.preprocessing import StandardScaler

class VegetationClassifier():

    def __init__(self,points, FEATURES):

        self.points = points
        self.FEATURES = FEATURES
        self.features_keys = list(self.FEATURES.keys())

        self.featurized_points_indexes = self.get_featurized_point_indexes(self.FEATURES)
        self.featurized_points = self.points[self.featurized_points_indexes,:]

        self.nn_features = StandardScaler().fit_transform(self.FEATURES[self.features_keys[0]][self.featurized_points_indexes,:])
        self.per_features = StandardScaler().fit_transform(self.FEATURES[self.features_keys[1]][self.featurized_points_indexes,:])

    def get_featurized_point_indexes(self, FEATURES):

        #Some points in point cloud can't be featurized. This is caused with perimeter neighborhood.
        #For this neighborhood given radius is defined. It's always good to examine really small neighborhood of point.
        #But because of various density in point cloud, there can be very few point in perimeter neighborhood.
        #PCA algorithm can't compute covariance tensor for 3 components with neighborhood of < 3 points.
        #All features of this points are None and can't be used in clusteringg algorithm.

        #Idea -> make XY point density heatmap and define radius of perimeter neighborhood with this heatmap, to
        #get better featurization.

        features_keys = list(FEATURES.keys())

        return (np.sum(np.isnan(FEATURES[features_keys[0]]), axis = 1) == 0) & (np.sum(np.isnan(FEATURES[features_keys[1]]), axis = 1) == 0)

    def fix_label(self, points: np.ndarray, label: np.ndarray):

        if np.mean(points[label == 0,2]) > np.mean(points[label == 1,2]):

            new_label = np.zeros(label.shape[0])
            new_label[label == 0] = 1

        else: new_label = label

        return new_label

    def classify_vegetation(self):

        #For ground X non-ground classification, gaussian mixture clustering with two clusters is used.
        #We apply this model on two sets of features. One is computed with nn_neighborhood, second with perimeter neighborhood.
        #We join result of this two clustering together and compute combined class. Final non-ground point has to be non-ground in both               #classifications.

        # define the model
        model = GaussianMixture(n_components=2)

        # fit the model
        model.fit(self.nn_features)
        # assign a cluster to each example
        nn_class = model.predict(self.nn_features)
        nn_class = self.fix_label(self.featurized_points, nn_class)

        # fit the model
        model.fit(self.per_features)
        # assign a cluster to each example
        per_class = model.predict(self.per_features)
        per_class = self.fix_label(self.featurized_points, per_class)

        return nn_class*per_class, self.featurized_points_indexes
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.neighbors import KDTree
from tqdm import tqdm, trange

from configs.config_cloud_evaluator import cloud_featurizer_Parameters




class CloudFeaturizer():

    def __init__(self, point_cloud: np.ndarray, metric: str, dimensions:list):

        assert metric in ["euclidean", "manhattan", "chebyshev", "minkowski", "wminkowski", "seuclidean", 
        "mahalanobis"], 'Parameter METRIC has to be string value and one of(euclidean, manhattan, chebyshev, minkowski, wminkowski, seuclidean, mahalanobis)'

        self.point_cloud = point_cloud
        self.metric = metric
        self.dimensions = dimensions
        self.x = StandardScaler().fit_transform(self.point_cloud)
        self.tree = KDTree(self.x[:,self.dimensions], metric = self.metric)

    def get_neighborhood(self, point, tree, method, radius, K):

        assert method == 'perimeter' or method == 'n-neighbors', 'Method argument has to be perimeter (neighbors are in given distance far away from examined point) or n-neighbors (neighborhood is K closest points)'
        
        if method == 'n-neighbors':
                
            _, neighbors_index = tree.query(point[self.dimensions].reshape(1, -1), k = K+1)
            neighbors = neighbors_index[0]

        elif method == 'perimeter':

            neighbors = tree.query_radius(point[self.dimensions].reshape(1, -1), radius)[0]

        return neighbors


    def compute_point_features(self, point, points, indexes):

        if indexes.any() and indexes.shape[0] > 3:

            neighbors = points[indexes,:]

            pca = PCA(n_components=3)
            pca.fit_transform(neighbors)

            eigenvalues = pca.explained_variance_
            eigenvectors = pca.components_

            #Sum of eigenvalues
            SUM =  eigenvalues.sum()

            #Omnivariance
            OMNIVARIANCE = eigenvalues.prod()**(1/3)

            #Eigenentropy
            EIGENENTROPY = np.sum([-x*np.log(x) for x in eigenvalues])

            #Anisotropy
            ANISOTROPY = (eigenvalues[0]-eigenvalues[2])/eigenvalues[0]

            #Planarity
            PLANARITY = (eigenvalues[1]-eigenvalues[2])/eigenvalues[0]

            #Linearity
            LINEARITY = (eigenvalues[0]-eigenvalues[1])/eigenvalues[0]

            #Surface variation
            SURFACEV = eigenvalues[2]/(eigenvalues[0]+eigenvalues[1]+eigenvalues[2])

            #Sphericity
            SPHERICITY = eigenvalues[2]/eigenvalues[0]

            #Verticality
            VERTICALITY = 1 - np.abs(eigenvectors[2][2])

            #First order moment
            FOM = np.sum([np.dot((neighbor-point), eigenvectors[0]) for neighbor in neighbors])**2 / np.sum([np.dot((neighbor-point),eigenvectors[0])**2 for neighbor in neighbors])

            #Average distance in neighborhood
            point_frame = np.repeat(point.reshape(1,-1), repeats = neighbors.shape[0], axis=0)
            dist = 0

            for dim in range(0, point.shape[0]):

                dist =+ (neighbors[: , dim] - point_frame[: , dim])**2

            DISTANCE = np.mean(dist**(0.5))

            return eigenvalues.tolist() + eigenvectors.flatten().tolist() + [SUM, OMNIVARIANCE, EIGENENTROPY, ANISOTROPY, PLANARITY, LINEARITY, SURFACEV, SPHERICITY, VERTICALITY, FOM, DISTANCE]

        else: 
            
            return list(np.repeat(None, 23))


    def collect_cloud_features(self):

        cfP = cloud_featurizer_Parameters()
        
        #nn_feat = np.zeros_like(0, shape = (self.x.shape[0],23))
        #per_feat = np.zeros_like(0, shape = (self.x.shape[0],23))
        nn_feat = []
        per_feat = []
        for i, point in enumerate(tqdm(self.x)):
            
            nn_neighborhood = self.get_neighborhood(point, self.tree, method = 'n-neighbors', radius = cfP.RADIUS, K=cfP.K)
            per_neighborhood = self.get_neighborhood(point, self.tree, method = 'perimeter', radius = cfP.RADIUS, K=cfP.K)
            
            nn_feat.append(self.compute_point_features(point, self.x, nn_neighborhood))
            per_feat.append(self.compute_point_features(point, self.x, per_neighborhood))

        NN_FEATURES = StandardScaler().fit_transform(nn_feat)
        PER_FEATURES = StandardScaler().fit_transform(per_feat)

        return {'nn_neighborhood': NN_FEATURES, 'perimeter_neighborhood':PER_FEATURES}
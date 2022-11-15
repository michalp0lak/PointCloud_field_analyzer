from sklearn.svm import SVC
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KDTree

class EdgeDetector():

    def __init__(self, points: np.ndarray, labels: np.ndarray, metric: str, entropy_quantile: float, K: int):

        assert (type(entropy_quantile == float)) and (entropy_quantile > 0) and (entropy_quantile < 0.1), 'Parameter entropy_quantile has to be positive float value smaller than 0.1'
        assert (type(K == int)) and (entropy_quantile > 0), 'Parameter K has to be positive integer value'

        self.points = points
        self.metric = metric
        self.x = StandardScaler().fit_transform(self.points)
        self.labels = labels
        self.entropy_quantile = entropy_quantile
        self.K = K

    def edge_entropy(self, edge_candidate: np.ndarray, neighbors: np.ndarray, class_label: np.ndarray):

        if np.unique(class_label).shape[0] == 1: entropy = 10**(9)

        else:

            # Fit the data with an svm
            svc = SVC(kernel='linear')
            svc.fit(neighbors, class_label)
            p = svc.coef_[0]

            sigma_n = np.zeros_like(False, shape = self.K+1)
            sigma_p = np.zeros_like(False, shape = self.K+1)
            

            for i, point in enumerate(neighbors):

                value = p[0]*(point[0] - edge_candidate[0]) + p[1]*(point[1] - edge_candidate[1]) + p[2]*(point[2] - edge_candidate[2])

                if value > 0: sigma_p[i] = True
                elif value < 0: sigma_n[i] = True


            if np.sum(sigma_p) == 0 or np.sum(sigma_n) == 0: entropy = 10**(9)

            else:

                sum_n = np.sum(class_label[sigma_n])
                sum_p = np.sum(class_label[sigma_p])

                #Determination of plant and ground area
                if sum_n >= sum_p: sum_plant = sum_n; sum_ground = sum_p;n_plant = sigma_n.shape[0];n_ground = sigma_p.shape[0]
                elif sum_n < sum_p: sum_plant = sum_p; sum_ground = sum_n;n_plant = sigma_p.shape[0];n_ground = sigma_n.shape[0]

                #Computation of edge entropy
                entropy = ((sum_ground+1)/(sum_plant+1))*(n_plant/n_ground)

        return entropy


    def get_edge_points(self):

        points_entropy = []

        tree = KDTree(self.x[:,:2], metric = self.metric)

        for point in self.x:

            _ , index = tree.query(point[:2].reshape(1, -1), k = self.K+1)

            points_entropy.append(self.edge_entropy(point, self.x[index[0],:], self.labels[index[0]])) 

        #return points_entropy < np.quantile(points_entropy, self.entropy_quantile)
        return np.array(points_entropy)
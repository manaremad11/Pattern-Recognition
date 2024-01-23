import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.utils import shuffle


def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


def manhatan_distance(x1, x2):
    return np.sum(np.abs(x1 - x2))


class KMeans:

    def __init__(self, K=3, max_iters=100):
        self.K = K
        self.max_iters = max_iters
        self.clusters = [[] for _ in range(self.K)]
        self.centroids = []
        self.X = []
        self.n_samples = 0
        self.n_features = 0

    def predict(self, X):
        self.X = X
        self.n_samples, self.n_features = X.shape

        # initialize random centroids from the data
        random_sample_idxs = np.random.choice(self.n_samples, self.K, replace=False)
        self.centroids = X[random_sample_idxs]

        # optimize clusters
        for _ in range(self.max_iters):
            # assign samples to closest centroids (create clusters)
            self.clusters = self._create_clusters(self.centroids)

            # calculate new centroids from the clusters
            centroids_old = self.centroids
            self.centroids = self._get_centroids(self.clusters)

            if self._is_converged(centroids_old, self.centroids):
                break

        # classify samples as the index of their clusters
        return self._get_cluster_labels(self.clusters)

    def _get_cluster_labels(self, clusters):
        # each sample will get the label of the cluster it was assigned to
        labels = np.empty(self.n_samples)
        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                labels[sample_idx] = cluster_idx

        return labels

    def _create_clusters(self, centroids):
        # assign the samples to the closest centroids
        clusters = [[] for _ in range(self.K)]
        for idx, sample in enumerate(self.X):
            centroid_idx = self._closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters

    def _closest_centroid(self, sample, centroids):
        # distance of the current sample to each centroid
        distances = [manhatan_distance(sample, point) for point in centroids]
        closest_idx = np.argmin(distances)
        return closest_idx

    def _get_centroids(self, clusters):
        # assign mean value of clusters to centroids
        centroids = np.zeros((self.K, self.n_features))
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[cluster_idx] = cluster_mean
        return centroids

    def _is_converged(self, centroids_old, centroids):
        # distances between old and new centroids, for all centroids
        distances = [manhatan_distance(centroids_old[i], centroids[i]) for i in range(self.K)]
        return sum(distances) == 0


if __name__ == "__main__":
    data = load_iris()
    X = data['data']

    print('data shape', X.shape)

    X = shuffle(X, random_state=0)

    model = KMeans(K=3, max_iters=150)
    y = model.predict(X)
    for x in model.clusters:
        print(len(x))
    print('centroids', model.centroids)

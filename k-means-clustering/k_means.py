import k_means_base
from matplotlib import pyplot as plt
import numpy as np


class KMeansClassifier(k_means_base.BaseKMeansClassifier):
    def __init__(self, input_file_name, clusters_count, available_colors=None, save_history=False):
        super(KMeansClassifier, self).__init__(input_file_name, clusters_count, available_colors, save_history)

        # randomly choose centroids
        self.centroids = self.points[np.random.choice(self.points.shape[0], self.CLUSTERS_COUNT, replace=False)]
        self.save_history = save_history
        if save_history:
            self.history_centroids = [self.centroids]

    @staticmethod
    def get_distances(point0, points):
        return np.linalg.norm(points - point0, axis=1)

    @staticmethod
    def get_closest_centroid_indices(points, centroids):
        closest_indices = [0] * len(points)

        for i, point in enumerate(points):
            distances = KMeansClassifier.get_distances(point, centroids)
            closest_indices[i] = int(distances.argmin())

        return closest_indices

    def plot_history_centroids(self):
        if self.save_history:
            for i in range(1, len(self.history_centroids)):
                p_iteration = self.history_centroids[i - 1]
                c_iteration = self.history_centroids[i]

                for j, c_centroid in enumerate(c_iteration):
                    p_centroid = p_iteration[j]
                    p0 = [p_centroid[0], c_centroid[0]]
                    p1 = [p_centroid[1], c_centroid[1]]

                    plt.plot(p0, p1, "c-")

            for iteration in self.history_centroids:
                for i, centroid in enumerate(iteration):
                    plt.scatter(centroid[0], centroid[1], c=self.AVAILABLE_COLORS[i], marker="^")

    def reinitialize_centroids(self):
        closest_centroid_indices = self.predict(self.points)
        new_centroids = np.zeros(self.centroids.shape)
        for i in range(self.centroids.shape[0]):
            closest_points = np.array([p for j, p in enumerate(self.points) if closest_centroid_indices[j] == i])
            new_centroids[i] = np.average(closest_points, 0)

        continue_learning = not np.array_equal(self.centroids, new_centroids)
        self.centroids = new_centroids

        if self.save_history:
            self.history_centroids.append(self.centroids)

        return continue_learning

    def calculate_best_centroids(self):
        while self.reinitialize_centroids():
            pass

    def predict(self, points):
        return KMeansClassifier.get_closest_centroid_indices(points, self.centroids)


def main():
    INPUT_FILE_NAME = "data.csv"
    CLUSTERS_COUNT = 3

    k_means = KMeansClassifier(INPUT_FILE_NAME, CLUSTERS_COUNT, save_history=True)
    k_means.calculate_best_centroids()

    k_means.plot_points()
    k_means.plot_history_centroids()
    plt.show()


if __name__ == "__main__":
    main()

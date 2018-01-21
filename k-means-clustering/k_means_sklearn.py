import k_means_base
from sklearn.cluster import KMeans


class KMeansClassifier(k_means_base.BaseKMeansClassifier):
    def __init__(self, input_file_name, clusters_count, available_colors=None):
        super(KMeansClassifier, self).__init__(input_file_name, clusters_count, available_colors)

        # all the logic of finding centroids
        self.k_means = KMeans(n_clusters=self.CLUSTERS_COUNT).fit(self.points)
        self.centroids = self.k_means.cluster_centers_

    def predict(self, points):
        return self.k_means.predict(points)


def main():
    INPUT_FILE_NAME = "data.csv"
    CLUSTERS_COUNT = 3

    k_means = KMeansClassifier(INPUT_FILE_NAME, CLUSTERS_COUNT)
    k_means.show_plot()


if __name__ == "__main__":
    main()

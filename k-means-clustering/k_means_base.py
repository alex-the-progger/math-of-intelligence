import pandas as pd
from matplotlib import pyplot as plt


class BaseKMeansClassifier:
    def __init__(self, input_file_name, clusters_count, available_colors=None, save_history=False):
        self.INPUT_FILE_NAME = input_file_name
        self.CLUSTERS_COUNT = clusters_count
        self.AVAILABLE_COLORS = ["b", "g", "r", "c", "m", "y", "k", "w"] if available_colors is None else available_colors

        # read the data set
        df = pd.read_csv(self.INPUT_FILE_NAME)
        self.points = df.as_matrix()

    def plot_points(self):
        closest_centroid_indices = self.predict(self.points)
        for i, index in enumerate(closest_centroid_indices):
            point = self.points[i]
            color = self.AVAILABLE_COLORS[index]
            plt.scatter(point[0], point[1], c=color, marker="o", s=2)

    def plot_centroids(self):
        for i, centroid in enumerate(self.centroids):
            plt.scatter(centroid[0], centroid[1], c=self.AVAILABLE_COLORS[i], marker="^")

    def plot(self):
        self.plot_points()
        self.plot_centroids()

    def show_plot(self):
        self.plot()
        plt.show()

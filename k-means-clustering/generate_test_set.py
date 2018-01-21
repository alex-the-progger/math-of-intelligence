from matplotlib import pyplot as plt
import pandas as pd
import random


MAX_CLUSTER_WIDTH = 10
MAX_CLUSTER_HEIGHT = 10
CLUSTER_SIZE = 50
OUTPUT_FILE_NAME = "data.csv"


class Point:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.color = color

    def generate_point(self, color):
        x_delta = (random.random() - 0.5) * MAX_CLUSTER_WIDTH
        y_delta = (random.random() - 0.5) * MAX_CLUSTER_HEIGHT

        x = self.x + x_delta
        y = self.y + y_delta

        return Point(x, y, color)


centroids = [
    Point(10, 5, "r"),
    Point(40, 15, "g"),
    Point(25, 50, "b")
]

random_points = []

for centroid in centroids:
    cluster_points = [centroid.generate_point("k") for i in range(CLUSTER_SIZE)]
    random_points.extend(cluster_points)

for point in random_points + centroids:
    plt.scatter(point.x, point.y, c=point.color)

plt.show()

df = pd.DataFrame(data=[(p.x, p.y) for p in random_points], columns=["x", "y"])
df.to_csv(OUTPUT_FILE_NAME, index=False)

print("Data was successfully generated and saved to", OUTPUT_FILE_NAME)

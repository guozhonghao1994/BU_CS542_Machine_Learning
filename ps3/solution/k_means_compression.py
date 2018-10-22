from __future__ import absolute_import

import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import misc

from solution.k_means_clustering import run_k_means, choose_random_centroids, find_closest_centroids


def main():
    datafile = 'data/bird_small.png'
    # This creates a three-dimensional matrix bird_small whose first two indices
    # identify a pixel position and whose last index represents red, green, or blue.
    bird_small = scipy.misc.imread(datafile)

    print("bird_small shape is ", bird_small.shape)
    plt.imshow(bird_small)
    # Divide every entry in bird_small by 255 so all values are in the range of 0 to 1
    bird_small = bird_small / 255.

    # Unroll the image to shape (16384,3) (16384 is 128*128)
    bird_small = bird_small.reshape(-1, 3)

    # Run k-means on this data, forming 16 clusters, with random initialization
    num_centroids = 16
    clusters, centroid_history = run_k_means(bird_small, choose_random_centroids(bird_small, num_centroids), n_iter=10)

    # Now I have 16 centroids, each representing a color.
    # Let's assign an index to each pixel in the original image dictating
    # which of the 16 colors it should be
    clusters = find_closest_centroids(bird_small, centroid_history[-1])

    final_centroids = centroid_history[-1]
    # Now loop through the original image and form a new image
    # that only has 16 colors in it
    final_image = np.zeros((clusters.shape[0], 3))
    for pixel in range(final_image.shape[0]):
        final_image[pixel] = final_centroids[int(clusters[pixel])]
    # Reshape the original image and the new, final image and draw them
    # To see what the "compressed" image looks like
    plt.figure()
    plt.imshow(bird_small.reshape(128, 128, 3))
    plt.figure()
    plt.imshow(final_image.reshape(128, 128, 3))

    plt.show()


if __name__ == '__main__':
    main()

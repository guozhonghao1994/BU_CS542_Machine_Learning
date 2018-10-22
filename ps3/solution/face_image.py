import matplotlib.pyplot as plt
import numpy as np
import pylab
import scipy.io
import scipy.misc

from solution.pca import feature_normalize, get_usv, project_data, recover_data


def get_datum_img(row):
    """
    Function that is handed a single np array with shape 1x1032,
    crates an image object from it, and returns it
    """
    width, height = 32, 32
    square = row.reshape(width, height)
    return square.T


def display_data(samples, num_rows=10, num_columns=10):
    """
    Creates an image object from a single np array with shape 1x1032
    :param row: a single np array with shape 1x1032
    :return: the constructed image
    """
    width, height = 32, 32
    num_rows, num_columns = num_rows, num_columns

    big_picture = np.zeros((height * num_rows, width * num_columns))

    row, column = 0, 0
    for index in range(num_rows * num_columns):
        if column == num_columns:
            row += 1
            column = 0
        img = get_datum_img(samples[index])
        big_picture[row * height:row * height + img.shape[0], column * width:column * width + img.shape[1]] = img
        column += 1
    plt.figure(figsize=(10, 10))
    img = scipy.misc.toimage(big_picture)
    plt.imshow(img, cmap=pylab.gray())


def main():
    datafile = 'data/faces.mat'
    mat = scipy.io.loadmat(datafile)
    samples = mat['X']
    display_data(samples)
    # Feature normalize
    means, stds, sample_norm = feature_normalize(samples)
    # Run SVD
    U, S, V = get_usv(sample_norm)
    # Visualize the top 36 eigenvectors found
    display_data(U[:, :36].T, num_rows=6, num_columns=6)
    # Project each image down to 36 dimensions
    z = project_data(sample_norm, U, K=36)
    # Attempt to recover the original data
    recovered_samples = recover_data(z, U, K=36)
    # Plot the dimension-reduced data
    display_data(recovered_samples)
    plt.show()


if __name__ == '__main__':
    main()

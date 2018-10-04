import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.io


def feature_normalize(samples):
    """
    Feature-normalize samples
    :param samples: samples.
    :return: normalized feature
    """
    means = np.mean(samples,axis=0)
    X_normalized = samples - means

    std = np.std(samples,axis=0,ddof=0)
    X_normalized = X_normalized/std

    return X_normalized


def get_usv(sample_norm):
    m = sample_norm.shape[0]
    Sigma = (1/m)*np.matmul(sample_norm.T,sample_norm)
    U,S,V = scipy.linalg.svd(Sigma)

    return U,S,V


def project_data(samples, U, K):
    """
    Computes the reduced data representation when
    projecting only on to the top "K" eigenvectors
    """

    # Reduced U is the first "K" columns in U
    reduced_U = U[:,0:K]
    reduced_samples = np.matmul(samples,reduced_U)

    return reduced_samples


def recover_data(Z, U, K):
    recovered_sample = np.matmul(Z,U[:,0:K].T)

    return recovered_sample


def main():
    datafile = 'data/data1.mat'
    mat = scipy.io.loadmat(datafile)
    samples = mat['X']

    plt.figure(figsize=(7, 7))
    plt.scatter(samples[:, 0], samples[:, 1], s=30, facecolors='none', edgecolors='b')
    plt.title("Example Dataset", fontsize=18)
    plt.grid(True)
    #plt.show()
    # Feature normalize

    samples_norm = feature_normalize(samples)

    # Run SVD

    U,S,V = get_usv(samples_norm)

    # output the top principal component (eigen- vector) found
    # should expect to see an output of about [-0.707 -0.707]"
    print('Top principal component is ', U[:, 0])


    plt.figure(figsize=(7, 7))
    plt.scatter(samples[:, 0], samples[:, 1], s=30, facecolors='none', edgecolors='b')
    plt.title("Example Dataset: PCA Eigenvectors Shown", fontsize=18)
    plt.xlabel('x1', fontsize=18)
    plt.ylabel('x2', fontsize=18)
    plt.grid(True)
    # To draw the principal component, you draw them starting
    # at the mean of the data

    # IMPLEMENT PLOT
    x_mean = np.mean(samples[:,0])
    y_mean = np.mean(samples[:,1])
    d_x1 = S[0]*U[:,0][0]
    d_y1 = S[0]*U[:,0][1]
    d_x2 = S[1]*U[:,1][0]
    d_y2 = S[1]*U[:,1][1]
    line1 = plt.plot([x_mean,x_mean+d_x1],[y_mean,y_mean+d_y1],color='r',label='First Principal Component',linewidth=3)
    line2 = plt.plot([x_mean,x_mean+d_x2],[y_mean,y_mean+d_y2],color='magenta',label='Second Principal Component',linewidth=3)
    plt.legend(loc=4)

    # project the first example onto the first dimension
    # should see a value of about 1.481"

    z = project_data(samples_norm, U, 1)
    print('Projection of the first example is %0.3f.' % float(z[0]))
    recovered_sample = recover_data(z, U, 1)
    print('Recovered approximation of the first example is ', recovered_sample[0])

    plt.figure(figsize=(7, 7))
    plt.scatter(samples_norm[:, 0], samples_norm[:, 1], s=30, facecolors='none',
                edgecolors='b', label='Original Data Points')
    plt.scatter(recovered_sample[:, 0], recovered_sample[:, 1], s=30, facecolors='none',
                edgecolors='r', label='PCA Reduced Data Points')

    plt.title("Example Dataset: Reduced Dimension Points Shown", fontsize=14)
    plt.xlabel('x1 [Feature Normalized]', fontsize=14)
    plt.ylabel('x2 [Feature Normalized]', fontsize=14)
    plt.grid(True)

    for x in range(samples_norm.shape[0]):
        plt.plot([samples_norm[x, 0], recovered_sample[x, 0]], [samples_norm[x, 1], recovered_sample[x, 1]], 'k--')

    plt.legend(loc=4)
    plt.xlim((-2.5, 2.5))
    plt.ylim((-2.5, 2.5))
    plt.show()


if __name__ == '__main__':
    main()

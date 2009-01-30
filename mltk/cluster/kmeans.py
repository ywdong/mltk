import random
import numpy.matlib as ml
from numpy import array, ndarray, zeros, inf, dot
from scipy.spatial import distance_matrix

__docformat__ = 'restructuredtext'
__all__ = ('kmeans')

def kmeans(data, k, threshold=1e-15, ntry=10,
        seeds=None, verbose=False):
    """\
    k-means clustering algorithm.

    :Parameters:
        data : ndarray
            a N-by-D data array where each row is a vector for data
            point and each column means a *feature*.
        k : int
            number of clusters.
        threshold : float
            the threshold to stop the iteration.
        ntry : int
            number of times to run k-means, useful to avoid accidentally
            trapped in local maximum. If ``seeds`` is given, the algorithm
            will run only once because the result is always the same with
            the same seeds.
        verbose : bool
            if ``True``, print progress to standard output, useful for
            debugging.

    :Returns:
        centroids : ndarray
            a k-by-D array of indicating k centroids.
        labels : ndarray
            a length-N int array of the label for each
            point.
    """
    minJ = inf
    min_centroids = None
    min_labels = None

    if seeds is None:
        N = data.shape[0]
        for i in range(ntry):
            seeds = random.sample(range(N), k)
            centroids, labels, J = \
                    _kmeans(data, threshold, data[seeds], verbose)
            if J < minJ:
                minJ = J
                min_centroids = centroids
                min_labels = labels
    else:
        D = data.shape[1]
        if seeds.shape != (k,D):
            raise ValueError('seeds should be a %d-by-%d array' % (k,D))
        min_centroids, min_labels, minJ = \
                _kmeans(data, threshold, seeds, verbose)
    return min_centroids, min_labels


def _kmeans(data, threshold, centroids, verbose):
    """\
    The *raw* version of k-means.
    """
    # initialize J
    Jprev = inf
    # initialize iteration count
    iter = 0

    # iterations
    while True:
        # calculate the distance from x to each centroids
        dist = distance_matrix(data, centroids)
        # assign x to nearst centroids
        labels = dist.argmin(axis=1)
        # re-calculate each center
        for j in range(len(centroids)):
            idx_j = (labels == j).nonzero()
            centroids[j] = data[idx_j].mean(axis=0)
        # calculate J
        # Note, if you would like to compare the J here to that
        # of k-medoids, here should be 
        #   (((...).sum(axis=1))**0.5).sum()
        J = ((data-centroids[labels])**2).sum()

        iter += 1
        if verbose:
            print '[kmeans] iter %d (J=%.4f)' % (iter, J)

        if Jprev-J < threshold:
            break
        Jprev = J

    return centroids, labels, J

import random
import numpy.matlib as ml
from numpy import array, ndarray, zeros, inf

def kmedoids(distmat, k, threshold=1e-15, maxiter=300,
        seeds=None, verbose=False):
    """\
    k-medoids clustering algorithm.
      distmat: a N-by-N dissimilarity matrix.
      k: number of clusters
      threshold: the threshold to stop the iteration
      maxiter: max iteration steps
      seeds: a length-k int array of index (into distmat)
             representing the initial selection of medoid 
             points.
      verbose: if True, print progress to standard output,
               useful for debugging.

    return (imedoids, labels)
      imedoids: a length-k int array of index indicating
                the points selected as final medoids.
      labels: a length-N int array of the label for each
              point.
    """
    N = len(distmat)
    # initialize medoids
    if seeds is None:
        imedoids = random.sample(range(N), k)
    else:
        if not isinstance(seeds, ndarray):
            seeds = array(seeds)
        imedoids = seeds
        if imedoids.shape != (k,):
            raise ValueError('seeds should be a length-%d array' % k)

    # initialize J
    Jprev = inf
    # initialize iteration count
    iter = 0

    # iterations
    while True:
        # distance from medoids to all other points
        dist = distmat[imedoids]
        # assign x to nearst medoid
        labels = dist.argmin(axis=0)
        J = 0
        # re-choose each medoids
        for j in range(k):
            idx_j = (labels == j).nonzero()[0]
            distj = distmat[idx_j][:, idx_j]
            distsum = ml.sum(distj, axis=1)
            idxmin = distsum.argmin()
            imedoids[j] = idx_j[idxmin]
            J += distsum[idxmin]

        iter += 1
        if verbose:
            print '[kmedoids] iter %d (J=%.4f)' % (iter, J)

        if Jprev-J < threshold:
            break
        Jprev = J
        if iter >= maxiter:
            break

    return imedoids, labels

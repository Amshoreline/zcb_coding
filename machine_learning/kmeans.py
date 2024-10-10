import numpy as np


def kmeans(X, k):
    '''
    Parameters:
        X: np.array(n, d), points, dtype=np.float
        k: int, number of centers
    Return(s):
        C: np.array(k, d), centers, dtype=np.float
    '''
    n, d = X.shape
    print(n, d)
    max_iters = 100
    C = np.random.randn(k, d)
    for _ in range(max_iters):
        # square_dist.shape = (n, k)
        square_dist = np.sum(np.square(X.reshape(n, 1, d) - C.reshape(1, k, d)), axis=2)
        # assigns.shape = (n, k)
        assigns = (np.argmin(square_dist, axis=1).reshape(n, 1) == np.arange(k).reshape(1, k)).astype(np.float)
        # num_points.shape = (k, 1)
        num_points = np.sum(assigns, axis=0).reshape(k, 1) + 1e-12
        # feature_sum.shape = (k, d)
        feature_sum = np.sum(assigns.reshape(n, k, 1) * X.reshape(n, 1, d), axis=0)
        C = feature_sum / num_points
    return C


def kmeans_torch(X, k):
    pass


if __name__ == '__main__':
    import itertools
    cluster_1 = np.array(list(itertools.product(range(-4, 2), range(-4, 2), range(-4, 4)))).astype(np.float)
    cluster_2 = np.array(list(itertools.product(range(100, 102), range(100, 102), range(300, 320)))).astype(np.float)
    cluster_3 = np.array(list(itertools.product(range(-100, -20), range(-1000, -990), range(-500, -450)))).astype(np.float)
    X = np.concatenate([cluster_1, cluster_2, cluster_3], axis=0)
    print(kmeans(X, 3))
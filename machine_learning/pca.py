import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# From https://github.com/aim-uofa/AdelaiDet/blob/master/adet/modeling/MEInst/LME/utils.py
def transform(X, components_, explained_variance_, mean_=None, whiten=False):
    """Apply dimensionality reduction to X.
    X is projected on the first principal components previously extracted
    from a training set.
    Parameters
    ----------
    X: array-like, shape (n_samples, n_features)
        New data, where n_samples is the number of samples
        and n_features is the number of features.
    components_: array-like, shape (n_components, n_features)
    mean_: array-like, shape (n_features,)
    explained_variance_: array-like, shape (n_components,)
                        Variance explained by each of the selected components.
    whiten : bool, optional
        When True (False by default) the ``components_`` vectors are divided
        by ``n_samples`` times ``components_`` to ensure uncorrelated outputs
        with unit component-wise variances.
        Whitening will remove some information from the transformed signal
        (the relative variance scales of the components) but can sometimes
        improve the predictive accuracy of the downstream estimators by
        making data respect some hard-wired assumptions.
    Returns
    -------
    X_new : array-like, shape (n_samples, n_components)
    """
    if mean_ is not None:
        X = X - mean_
    X_transformed = np.dot(X, components_.T)
    if whiten:
        X_transformed /= np.sqrt(explained_variance_)
    return X_transformed


def inverse_transform(X, components_, explained_variance_, mean_=None, whiten=False):
    """Transform data back to its original space.
    In other words, return an input X_original whose transform would be X.
    Parameters
    ----------
    X : array-like, shape (n_samples, n_components)
        New data, where n_samples is the number of samples
        and n_components is the number of components.
    components_: array-like, shape (n_components, n_features)
    mean_: array-like, shape (n_features,)
    explained_variance_: array-like, shape (n_components,)
                        Variance explained by each of the selected components.
    whiten : bool, optional
        When True (False by default) the ``components_`` vectors are divided
        by ``n_samples`` times ``components_`` to ensure uncorrelated outputs
        with unit component-wise variances.
        Whitening will remove some information from the transformed signal
        (the relative variance scales of the components) but can sometimes
        improve the predictive accuracy of the downstream estimators by
        making data respect some hard-wired assumptions.

    Returns
    -------
    X_original array-like, shape (n_samples, n_features)
    """
    if whiten:
        X_transformed = np.dot(X, np.sqrt(explained_variance_[:, np.newaxis]) * components_)
    else:
        X_transformed = np.dot(X, components_)
    if mean_ is not None:
        X_transformed = X_transformed + mean_
    return X_transformed



def main():
    # Plot X
    plt.subplot(221)
    plt.title('X')
    num_points = 32
    X = 10 * np.random.rand(num_points, 2) + np.arange(num_points).reshape(-1, 1) - num_points // 2
    xlim = (np.min(X[:, 0]), np.max(X[:, 0]))
    ylim = (np.min(X[:, 1]), np.max(X[:, 1]))
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.plot(X[:, 0], X[:, 1], 'rx')
    # Apply PCA
    pca = PCA(n_components=2)
    pca.fit(X)
    # Plot T(X)
    plt.subplot(222)
    plt.title('Y = T(X)')
    Y = transform(X, pca.components_, pca.explained_variance_, pca.mean_)
    whiten_Y = transform(X, pca.components_, pca.explained_variance_, pca.mean_, True)
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.plot(Y[:, 0], [0] * num_points, 'gx')
    # Plot whiten T(X)
    plt.subplot(223)
    plt.title('Y = T(X, whiten=True)')
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.plot(whiten_Y[:, 0], [0] * num_points, 'gx')
    # Plot T'T(X)
    plt.subplot(224)
    plt.title('Z = T\'T(X) w/o or w/ whiten')
    Z = inverse_transform(Y[:, : 1], pca.components_[: 1], pca.explained_variance_[: 1], pca.mean_)
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.plot(Z[:, 0], Z[:, 1], 'bx')
    plt.show()

if __name__ == '__main__':
    main()

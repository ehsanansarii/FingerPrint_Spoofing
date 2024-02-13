import numpy as np
import matplotlib.pyplot as plt
import fingerprints_dataset


def vcol(vector):
    """ Convert vector to column vector """
    return vector.reshape((vector.size, 1))


def dataset_mean(dataset):
    """ Compute the mean of each feature """
    return np.mean(dataset, axis=1, keepdims=True)


def pca(dataset, m):
    """ Perform PCA and return the projected dataset and the principal components """
    # Subtract the mean
    mu = dataset_mean(dataset)
    zero_mean_dataset = dataset - mu

    # Compute the covariance matrix
    covariance_matrix = np.dot(zero_mean_dataset, zero_mean_dataset.T) / zero_mean_dataset.shape[1]

    # Perform eigendecomposition
    s, U = np.linalg.eigh(covariance_matrix)

    # Select the top m eigenvectors
    P = U[:, ::-1][:, :m]

    # Project the dataset onto the new subspace
    projected_dataset = np.dot(P.T, zero_mean_dataset)

    return projected_dataset, P


def pca_svd(dataset, m):
    """ Perform PCA using SVD and return the projected dataset """
    mu = dataset_mean(dataset)
    zero_mean_dataset = dataset - mu

    # Perform SVD
    U, s, Vh = np.linalg.svd(zero_mean_dataset, full_matrices=False)

    # Select the top m components
    P = U[:, :m]

    # Project the dataset
    projected_dataset = np.dot(P.T, zero_mean_dataset)

    return projected_dataset


def explained_variance(dataset):
    # Center the data
    mu = np.mean(dataset, axis=1).reshape(-1, 1)
    zero_mean_dataset = dataset - mu

    # Compute the covariance matrix
    covariance_matrix = np.dot(zero_mean_dataset, zero_mean_dataset.T) / zero_mean_dataset.shape[1]

    # Compute eigenvalues
    eigenvalues, _ = np.linalg.eigh(covariance_matrix)

    # Sort eigenvalues in descending order
    sorted_eigenvalues = np.sort(eigenvalues)[::-1]

    # Compute the explained variance
    total_variance = np.sum(sorted_eigenvalues)
    explained_variance_ratio = sorted_eigenvalues / total_variance

    # Compute the cumulative explained variance
    cumulative_explained_variance = np.cumsum(explained_variance_ratio)

    return cumulative_explained_variance



if __name__ == '__main__':
    # Load dataset
    dataset, labels = fingerprints_dataset.load("/content/train.txt")

    # Perform PCA
    dataset_with_PCA, _ = pca(dataset, 5)

    # Plotting the PCA result
    fingerprints_dataset.plot_scatter(dataset_with_PCA, labels)

    """
    # Calculate explained variance
    cum_explained_variance = explained_variance(dataset)

    # Plot the explained variance
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(cum_explained_variance) + 1), cum_explained_variance, marker='o')
    plt.xlabel('PCA Dimensions')
    plt.ylabel('Fraction of Explained Variance')
    plt.title('PCA: Explained Variance')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    """

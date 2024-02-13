import numpy as np
import matplotlib.pyplot as plt
import math

def dataset_mean(dataset):
    """Compute the mean of each feature in the dataset."""
    return np.mean(dataset, axis=1, keepdims=True)

def num_samples(dataset):
    """Return the number of samples in the dataset."""
    return dataset.shape[1]

def vcol(vector):
    """Convert a vector into a column vector."""
    return vector.reshape((-1, 1))

def vrow(vector):
    """Convert a vector into a row vector."""
    return vector.reshape((1, -1))

def logpdf_GAU_ND_single_sample(x, mu, C):
    """Compute log-density of a Gaussian for a single sample."""
    _, log_det_cov_mat = np.linalg.slogdet(C)
    inv_cov_mat = np.linalg.inv(C)
    diff = x - mu
    result = -0.5 * (np.log(2 * np.pi) * mu.shape[0] + log_det_cov_mat + diff.T @ inv_cov_mat @ diff)
    return result.item()

def logpdf_GAU_ND(X, mu, C):
    """Compute log-density of a Gaussian for multiple samples."""
    return np.array([logpdf_GAU_ND_single_sample(X[:, i:i+1], mu, C) for i in range(X.shape[1])])

def ML_estimate(X):
    """Maximum Likelihood Estimation of mean and covariance matrix."""
    mu_ml = dataset_mean(X)
    X_centered = X - mu_ml
    covariance_matrix_ml = X_centered @ X_centered.T / num_samples(X)
    return mu_ml, covariance_matrix_ml

def loglikelihood(X, mu_ml, C_ml):
    """Compute the log-likelihood of the data given the model."""
    return np.sum(logpdf_GAU_ND(X, mu_ml, C_ml))

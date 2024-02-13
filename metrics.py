
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

def vector_column(v):
    """Convert a vector to a column vector."""
    return v.reshape((-1, 1))

def vector_row(v):
    """Convert a vector to a row vector."""
    return v.reshape((1, -1))

def whitening(data):
    """Whiten the given dataset."""
    covariance = np.cov(data)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    whitening_matrix = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues + 1e-5)) @ eigenvectors.T
    return whitening_matrix @ data

def z_score_normalization(dataset):
    """Normalize the dataset using Z-score normalization."""
    mean = np.mean(dataset, axis=1, keepdims=True)
    std = np.std(dataset, axis=1, keepdims=True)
    return (dataset - mean) / std

def load_iris_data():
    """Load the iris dataset."""
    data, labels = load_iris(return_X_y=True)
    return data.T, labels

def split_db_2to1(D, L, seed=0):
    """Split the dataset into training and testing sets."""
    nTrain = int(D.shape[1] * 0.2)  # Adjusted to use 20% for training for a more common split
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    idxTrain, idxTest = idx[:nTrain], idx[nTrain:]
    return D[:, idxTrain], L[idxTrain], D[:, idxTest], L[idxTest]

def compute_confusion_matrix(actual, predicted, num_classes=2):
    """Compute the confusion matrix."""
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    np.add.at(matrix, (actual, predicted), 1)
    return matrix

def compute_DCF(p1, cfn, cfp, confusion_matrix):
    """Compute the Detection Cost Function."""
    FNR = confusion_matrix[0, 1] / np.sum(confusion_matrix[1])
    FPR = confusion_matrix[1, 0] / np.sum(confusion_matrix[0])
    return (p1 * cfn * FNR + (1 - p1) * cfp * FPR) / min(p1 * cfn, (1 - p1) * cfp)

def plot_ROC(scores, labels):
    """Plot the ROC curve."""
    thresholds = np.sort(scores)
    FPRs, TPRs = [], []
    for t in thresholds:
        predicted_labels = scores > t
        cm = compute_confusion_matrix(labels, predicted_labels.astype(int))
        TPR = 1 - cm[0, 1] / np.sum(cm[1])
        FPR = cm[1, 0] / np.sum(cm[0])
        FPRs.append(FPR)
        TPRs.append(TPR)
    plt.plot(FPRs, TPRs, marker='*')
    plt.plot([0, 1], [0, 1], linestyle='--', color='navy')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.grid(True)
    plt.show()




def plot_Bayes_error(scores, labels, compute_DCF, compute_DCFmin):
    """Plot the Bayes error curves for DCF and min DCF."""
    effPriorLogOdds = np.linspace(-3, 3, 21)
    DCFs, minDCFs = [], []

    for p in effPriorLogOdds:
        eff_p = 1 / (1 + np.exp(-p))
        preds = compute_results_with_diff_triplet(scores, eff_p, 1, 1)
        confusion_matrix = compute_confusion_matrix(labels, preds, num_classes=2)
        DCFs.append(compute_DCF(eff_p, 1, 1, confusion_matrix))
        minDCFs.append(compute_DCFmin(scores, labels, eff_p, 1, 1))

    plt.plot(effPriorLogOdds, DCFs, label='DCF', color='red')
    plt.plot(effPriorLogOdds, minDCFs, label='min DCF', color='blue')
    plt.ylim([0, 1.1])
    plt.xlim([-3, 3])
    plt.xlabel('Effective Prior Log Odds')
    plt.ylabel('Cost')
    plt.title('Bayes Error Plot')
    plt.legend()
    plt.show()



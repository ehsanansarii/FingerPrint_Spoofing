import numpy as np
import scipy.optimize as optimize
from metrics import *
from PCA import *

def vcol(vector):
    """Convert a vector into a column vector."""
    return vector.reshape((-1, 1))

def vrow(vector):
    """Convert a vector into a row vector."""
    return vector.reshape((1, -1))

def num_samples(dataset):
    """Return the number of samples in a dataset."""
    return dataset.shape[1]

def logreg_obj_wrap(DTR, LTR, l, prior):
    """Wrap the logistic regression objective function."""
    def logreg_obj(v):
        w, b = v[:-1], v[-1]
        z = 2 * LTR - 1
        sigmoid = lambda x: 1 / (1 + np.exp(-x))
        log_likelihood = z * (np.dot(w.T, DTR) + b)
        regularization = (l / 2) * np.sum(w ** 2)
        J = np.sum(np.logaddexp(0, -log_likelihood)) / num_samples(DTR) + regularization
        return J

    return logreg_obj

def quad_logreg_obj_wrap(DTR, LTR, l, prior):
    """Wrap the quadratic logistic regression objective function."""
    def quad_logreg_obj(v):
        w, b = v[:-1], v[-1]
        z = 2 * LTR - 1
        X_aug = np.vstack([np.kron(DTR[:, i], DTR[:, i]) for i in range(num_samples(DTR))]).T
        X_aug = np.vstack((X_aug, DTR))  # Augment with linear terms
        log_likelihood = z * (np.dot(w.T, X_aug) + b)
        regularization = (l / 2) * np.sum(w ** 2)
        J = np.sum(np.logaddexp(0, -log_likelihood)) / num_samples(DTR) + regularization
        return J

    return quad_logreg_obj

def quad_logreg_obj_wrap_with_grad(DTR, LTR, l, prior):
    """Wrap the quadratic logistic regression objective function and its gradient."""
    def quad_logreg_obj(v):
        # Same as quad_logreg_obj_wrap
        pass  # Placeholder for the actual implementation

    def quad_logreg_grad(v):
        # Compute the gradient of the quadratic logistic regression objective function
        pass  # Placeholder for the actual implementation

    return quad_logreg_obj, quad_logreg_grad



def bin_logistic_regression_train(DTR, LTR, l, prior):
    """Train a binary logistic regression model."""
    logreg_obj = logreg_obj_wrap(DTR, LTR, l, prior)
    x0 = np.zeros(DTR.shape[0] + 1)  # Initial guess
    v, _, _ = optimize.fmin_l_bfgs_b(logreg_obj, x0, approx_grad=True, iprint=-1)
    print(f"J(w*,b*) = {logreg_obj(v)}")
    return v[:-1], v[-1]  # Return w_best, b_best

def bin_quadratic_logistic_regression_train(DTR, LTR, l, prior):
    """Train a binary quadratic logistic regression model."""
    logreg_obj = quad_logreg_obj_wrap(DTR, LTR, l, prior)
    x0 = np.zeros(DTR.shape[0] ** 2 + DTR.shape[0] + 1)  # Initial guess for quadratic terms
    v, _, _ = optimize.fmin_l_bfgs_b(logreg_obj, x0, approx_grad=True, iprint=-1)
    print(f"J(w*,b*) = {logreg_obj(v)}")
    return v[:-1], v[-1]  # Return w_best, b_best

def bin_quadratic_logistic_regression_with_grad_train(DTR, LTR, l, prior):
    """Train a binary quadratic logistic regression model with gradient."""
    logreg_obj, logreg_grad = quad_logreg_obj_wrap_with_grad(DTR, LTR, l, prior)
    x0 = np.zeros(DTR.shape[0] ** 2 + DTR.shape[0] + 1)  # Initial guess for quadratic terms
    v, _, _ = optimize.fmin_l_bfgs_b(logreg_obj, x0, fprime=logreg_grad, iprint=-1)
    print(f"J(w*,b*) = {logreg_obj(v)}")
    return v[:-1], v[-1]  # Return w_best, b_best

def quad_logistic_regression_eval(DTE, LTE, w_best, b_best, prior):
    """Evaluate a quadratic logistic regression model."""
    scores = np.dot(w_best.T, np.vstack([np.kron(DTE[:, i], DTE[:, i]) for i in range(DTE.shape[1])]).T) + b_best
    preds = np.where(scores > math.log(prior / (1 - prior)), 1, 0)
    accuracy = np.mean(preds == LTE)
    return preds, accuracy, scores

def logistic_regression_eval(DTE, LTE, w_best, b_best, prior):
    """Evaluate a logistic regression model."""
    scores = np.dot(w_best.T, DTE) + b_best
    preds = np.where(scores > math.log(prior / (1 - prior)), 1, 0)
    accuracy = np.mean(preds == LTE)
    return preds, accuracy, scores




def loadFile(file):
    """Load dataset and labels from a file."""
    data, labels = [], []
    with open(file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            data.append([float(value) for value in parts[:10]])
            labels.append(int(parts[-1]))
    return np.array(data).T, np.array(labels)

def K_fold(K, train_set, labels, l, prior, cfn, cfp):
    """Perform K-fold cross-validation."""
    np.random.seed(0)
    indices = np.random.permutation(len(labels))
    fold_size = len(labels) // K
    min_DCFs = []

    for i in range(K):
        val_mask = np.zeros(len(labels), dtype=bool)
        val_mask[i * fold_size: (i + 1) * fold_size] = True
        DTR, LTR = train_set[:, ~val_mask], labels[~val_mask]
        DTE, LTE = train_set[:, val_mask], labels[val_mask]

        w_best, b_best = bin_quadratic_logistic_regression_with_grad_train(DTR, LTR, l, prior)
        _, _, scores = quad_logistic_regression_eval(DTE, LTE, w_best, b_best, prior)

        min_DCF = compute_DCFmin(scores, LTE, prior, cfn, cfp)
        min_DCFs.append(min_DCF)

    return np.mean(min_DCFs)

if __name__ == '__main__':
    D, L = loadFile("Train.txt")
    DTE, LTE = loadFile("Test.txt")

    D_pca8, P = pca(D, 8)
    D_pca8_z = z_score_normalization(D_pca8)
    DTE_pca8 = P.T @ DTE
    DTE_pca8_z = z_score_normalization(DTE_pca8)

    pi_tilde = 0.5 * 1 / (0.5 * 1 + 0.5 * 10)

    (DTR, LTR), (DTC, LTC) = split_db_2to1(D_pca8_z, L, seed=0)
    w_best, b_best = bin_quadratic_logistic_regression_with_grad_train(DTR, LTR, 0.001, pi_tilde)

    scores_train_cal = quad_logistic_regression_eval(DTC, LTC, w_best, b_best, pi_tilde)[2]
    w_cal, b_cal = bin_logistic_regression_train(np.array([scores_train_cal]), LTC, 0.001, pi_tilde)

    scores_UNCAL = quad_logistic_regression_eval(DTE_pca8_z, LTE, w_best, b_best, pi_tilde)[2]
    predicted, _, scores_CAL = logistic_regression_eval(np.array([scores_UNCAL]), LTE, w_cal, b_cal, pi_tilde)
    plot_Bayes_error(scores_CAL, LTE)

    actual_dcf = compute_DCF(pi_tilde, 1, 1, compute_confusion_matrix(LTE, predicted, 2))
    min_dcf = compute_DCFmin(scores_CAL, LTE, pi_tilde, 1, 1)
    print(f"actual dcf: {actual_dcf}")
    print(f"min dcf: {min_dcf}")

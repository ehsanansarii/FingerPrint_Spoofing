import numpy as np
import scipy.optimize as optimize
from metrics import *  # Ensure metrics module provides necessary functionality


def logreg_obj_wrap(DTR, LTR, l, prior):
    """Wrap the logistic regression objective function."""

    def logreg_obj(v):
        w, b = v[:-1], v[-1]  # Unpack weights and bias
        z = 2 * LTR - 1  # Transform labels to {-1, 1}
        linear_combination = np.dot(w, DTR) + b  # Vectorized computation
        J_regularization = (l / 2) * np.sum(w ** 2)  # Regularization term

        # Compute the logistic loss in a vectorized form
        logistic_losses = np.logaddexp(0, -z * linear_combination)

        # Apply class weighting
        nt = np.sum(LTR == 1)  # Number of positive samples
        nf = np.sum(LTR == 0)  # Number of negative samples
        weights = np.where(LTR == 1, prior / nt, (1 - prior) / nf)
        J_loss = np.sum(weights * logistic_losses)  # Weighted logistic loss

        J = J_regularization + J_loss
        return J

    return logreg_obj


def bin_logistic_regression_train(DTR, LTR, l, prior):
    """Train binary logistic regression model."""
    logreg_obj = logreg_obj_wrap(DTR, LTR, l, prior)
    x0 = np.zeros(DTR.shape[0] + 1)  # Initial guess
    v, _, _ = optimize.fmin_l_bfgs_b(logreg_obj, x0, approx_grad=True, iprint=-1)
    w_best, b_best = v[:-1], v[-1]
    print(f"J(w*,b*) = {logreg_obj(v)}")
    return w_best, b_best


def logistic_regression_eval(DTE, LTE, w_best, b_best, prior):
    """Evaluate logistic regression model."""
    score = np.dot(w_best.T, DTE) + b_best
    preds = np.where(score > np.log(prior / (1 - prior)), 1, 0)

    accuracy = np.mean(preds == LTE)
    scores = score - np.log(prior / (1 - prior))  # Adjusted scores for decision

    return preds, accuracy, scores.tolist()


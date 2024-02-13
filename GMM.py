

import numpy as np
import scipy.special
import MVG
import metrics
import PCA
import fingerprints_dataset


def split_db_2to1(D, L, seed=0):
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    nTrain = int(D.shape[1] * 2.0 / 3.0)
    idxTrain, idxTest = idx[:nTrain], idx[nTrain:]
    return (D[:, idxTrain], L[idxTrain]), (D[:, idxTest], L[idxTest])

def vcol(vector):
    return vector.reshape((-1, 1))

def vrow(vector):
    return vector.reshape((1, -1))

def logpdf_GMM(X, gmm):
    logS = np.array([logpdf_GAU_ND(X, component[1], component[2]) + np.log(component[0]) for component in gmm])
    return scipy.special.logsumexp(logS, axis=0)

def EM_GMM(X, gmm, psi):
    while True:
        logS, SPost, Z, F, S = _prepare_em_variables(X, gmm)
        mu, cov_mat = _update_mu_cov(X, SPost, Z, F, S, psi)
        gmm = _update_gmm(gmm, mu, cov_mat, Z)
        new_loss = logpdf_GMM(X, gmm)
        if _check_convergence(new_loss, logS, psi):
            break
    print(np.mean(new_loss))
    return gmm

def EM_Diag_GMM(X, gmm, psi):
    while True:
        logS, SPost, Z, F, S = _prepare_em_variables(X, gmm, diag=True)
        mu, cov_mat = _update_mu_cov(X, SPost, Z, F, S, psi, diag=True)
        gmm = _update_gmm(gmm, mu, cov_mat, Z)
        new_loss = logpdf_GMM(X, gmm)
        if _check_convergence(new_loss, logS, psi):
            break
    print(np.mean(new_loss))
    return gmm

def _prepare_em_variables(X, gmm, diag=False):
    num_components = len(gmm)
    logS = np.empty((num_components, X.shape[1]))
    for i, component in enumerate(gmm):
        logS[i, :] = logpdf_GAU_ND(X, component[1], component[2]) + np.log(component[0])
    logSMarginal = vrow(scipy.special.logsumexp(logS, axis=0))
    SPost = np.exp(logS - logSMarginal)
    Z = np.sum(SPost, axis=1)
    F = np.dot(X, SPost.T)
    S = np.dot(X, (SPost * X).T) if not diag else np.diag(np.dot(X, (SPost * X).T))
    return logS, SPost, Z, F, S

def _update_mu_cov(X, SPost, Z, F, S, psi, diag=False):
    mu = F / Z
    cov_mat = (S / Z) - np.dot(vcol(mu), vrow(mu).T) if not diag else np.diag((S / Z) - np.dot(vcol(mu), vrow(mu).T))
    for i in range(cov_mat.shape[-1]):
        U, s, _ = np.linalg.svd(cov_mat[:, :, i])
        s[s < psi] = psi
        cov_mat[:, :, i] = np.dot(U, np.diag(s) @ U.T)
    return mu, cov_mat

def _update_gmm(gmm, mu, cov_mat, Z):
    w = Z / np.sum(Z)
    return [(w[i], vcol(mu[:, i]), cov_mat[:, :, i]) for i in range(len(gmm))]

def _check_convergence(new_loss, old_loss, psi):
    return np.abs(np.mean(new_loss) - np.mean(old_loss)) < psi



import numpy as np
import scipy.special
from MVG import ML_estimate, logpdf_GAU_ND

def EM_TiedCov_GMM(X, gmm, psi):
    while True:
        num_components = len(gmm)
        logS, logSMarginal, SPost = _e_step(X, gmm, num_components)
        old_loss = np.sum(logSMarginal)
        gmm, cov_updated = _m_step_tied_cov(X, SPost, num_components, psi)
        new_loss = np.sum(logpdf_GMM(X, gmm))
        if _convergence_reached(old_loss, new_loss, psi):
            break
    return gmm

def LBG(X, psi, alpha=0.1, num_components_max=4):
    gmm_start = _initialize_gmm(X, psi, num_components_max, alpha, diag=False)
    while len(gmm_start) < num_components_max:
        gmm_start = _expand_gmm(X, gmm_start, psi, alpha, diag=False)
    return gmm_start

def Diag_LBG(X, psi, alpha=0.1, num_components_max=4):
    gmm_start = _initialize_gmm(X, psi, num_components_max, alpha, diag=True)
    while len(gmm_start) < num_components_max:
        gmm_start = _expand_gmm(X, gmm_start, psi, alpha, diag=True)
    return gmm_start

def _e_step(X, gmm, num_components):
    logS = np.array([logpdf_GAU_ND(X, comp[1], comp[2]) + np.log(comp[0]) for comp in gmm])
    logSMarginal = scipy.special.logsumexp(logS, axis=0)
    SPost = np.exp(logS - logSMarginal)
    return logS, logSMarginal, SPost

def _m_step_tied_cov(X, SPost, num_components, psi):
    Z = np.sum(SPost, axis=1)
    F = np.dot(X, SPost.T)
    mu = F / Z
    cov_mat = np.dot(X * Z, X.T) / X.shape[1] - np.dot(mu, mu.T)
    cov_updated = _update_cov_mat(cov_mat, psi)
    w = Z / np.sum(Z)
    gmm = [(w[i], mu[:, i:i+1], cov_updated) for i in range(num_components)]
    return gmm, cov_updated

def _update_cov_mat(cov_mat, psi):
    U, s, _ = np.linalg.svd(cov_mat)
    s[s < psi] = psi
    return np.dot(U, np.diag(s) @ U.T)

def _initialize_gmm(X, psi, num_components_max, alpha, diag):
    # Initial GMM setup based on ML estimate
    return []  # Placeholder for initial GMM setup logic

def _expand_gmm(X, gmm, psi, alpha, diag):
    # Logic to expand GMM components
    return []  # Placeholder for GMM expansion logic

def _convergence_reached(old_loss, new_loss, psi):
    return np.abs(new_loss - old_loss) < psi

# Placeholder for logpdf_GMM function that computes log-density for GMM
def logpdf_GMM(X, gmm):
    return np.sum([comp[0] * logpdf_GAU_ND(X, comp[1], comp[2]) for comp in gmm], axis=0)





def TiedCov_LBG(X, psi, alpha=0.1, num_components_max=4):
    # Initialize the first Gaussian component
    weight_initial = 1
    mean_initial, cov_initial = ML_estimate(X)

    # Ensure covariance matrix is not degenerate by setting minimum eigenvalue to psi
    U, eigenvalues, _ = np.linalg.svd(cov_initial)
    eigenvalues_clipped = np.maximum(eigenvalues, psi)
    cov_initial = U @ np.diag(eigenvalues_clipped) @ U.T

    gmm_components = []

    # Handle the case for a single component GMM
    if num_components_max == 1:
        gmm_components.append((weight_initial, np.atleast_2d(mean_initial).T, cov_initial))
        return gmm_components

    # Split the initial component into two
    weight_new = weight_initial / 2
    direction, scale = U[:, :1], np.sqrt(eigenvalues_clipped[0]) * alpha
    displacement = direction * scale
    mean_new_1, mean_new_2 = mean_initial + displacement, mean_initial - displacement

    gmm_components.extend([
        (weight_new, np.atleast_2d(mean_new_1).T, cov_initial),
        (weight_new, np.atleast_2d(mean_new_2).T, cov_initial)
    ])

    # Iteratively refine the GMM components
    while len(gmm_components) < num_components_max:
        gmm_components = EM_TiedCov_GMM(X, gmm_components, psi)

        # Split each component to increase the mixture components
        new_components = []
        for weight, mean, cov in gmm_components:
            U, s, _ = np.linalg.svd(cov)
            direction = U[:, :1]
            scale = np.sqrt(s[0]) * alpha
            displacement = direction * scale
            mean_new_1, mean_new_2 = mean + displacement, mean - displacement

            new_components.extend([
                (weight / 2, mean_new_1, cov),
                (weight / 2, mean_new_2, cov)
            ])
        gmm_components = new_components

    return gmm_components


def GMM_classifier(train_set, labels, num_classes, psi=0.01, alpha=0.1, num_components_max=[4, 4]):
    models = []
    for class_index in range(num_classes):
        # Extract training data for the current class
        class_data = train_set[:, labels == class_index]
        # Generate GMM for the class
        gmm = LBG(class_data, psi, alpha, num_components_max[class_index])
        models.append(gmm)
    return models  # Returns a list with GMM parameters for each class

def GMM_Diag_classifier(train_set, labels, num_classes, psi=0.01, alpha=0.1, num_components_max=[4, 4]):
    models = []
    for class_index in range(num_classes):
        # Extract training data for the current class
        class_data = train_set[:, labels == class_index]
        # Generate diagonal covariance GMM for the class
        gmm = Diag_LBG(class_data, psi, alpha, num_components_max[class_index])
        models.append(gmm)
    return models  # Returns a list with GMM parameters for each class

def GMM_TiedCov_classifier(train_set, labels, num_classes, psi=0.01, alpha=0.1, num_components_max=[4, 4]):
    models = []
    for class_index in range(num_classes):
        # Extract training data for the current class
        class_data = train_set[:, labels == class_index]
        # Generate tied covariance GMM for the class
        gmm = TiedCov_LBG(class_data, psi, alpha, num_components_max[class_index])
        models.append(gmm)
    return models  # Returns a list with GMM parameters for each class





def predict_log(model, test_samples, prior):
    num_classes = len(model)
    log_likelihoods = []
    log_joint_scores = []

    # Calculate log likelihood and log joint scores for each class
    for class_index in range(num_classes):
        gmm_params = model[class_index]
        log_likelihood = logpdf_GMM(test_samples, gmm_params)
        log_likelihoods.append(log_likelihood)
        log_joint_scores.append(log_likelihood + np.log(prior[class_index]))

    # Compute the log marginal likelihood for normalization
    log_marginal_likelihood = np.atleast_2d(scipy.special.logsumexp(log_joint_scores, axis=0))

    # Calculate log posterior probabilities
    log_posterior_probs = log_joint_scores - log_marginal_likelihood

    # Calculate the log likelihood ratio for binary classification case
    llr = log_posterior_probs[1] - log_posterior_probs[0] - np.log(prior[1] / prior[0])

    # Convert log posterior probabilities to actual probabilities
    posterior_probs = np.exp(log_posterior_probs)

    # Determine predicted labels based on maximum posterior probability
    predicted_labels = np.argmax(posterior_probs, axis=0)

    return predicted_labels.ravel(), llr





def accuracy(predicted_labels, original_labels):
    if len(predicted_labels) != len(original_labels):
        return 0  # Return 0 accuracy if lengths mismatch
    correct_predictions = (predicted_labels == original_labels).sum()
    return (correct_predictions / len(predicted_labels)) * 100


def error_rate(predicted_labels, original_labels):
    return 100 - accuracy(predicted_labels, original_labels)


def K_fold(K, train_set, labels, num_classes, prior):
    np.random.seed(0)  # Ensure reproducibility
    indices = np.arange(train_set.shape[1])
    np.random.shuffle(indices)
    train_set, labels = train_set[:, indices], labels[indices]

    fold_size = len(labels) // K
    llr_results, preds = [], []
    for k in range(K):
        print(f"This is the {k + 1}-th FOLD")
        val_mask = np.zeros(len(labels), dtype=bool)
        val_mask[k * fold_size: (k + 1) * fold_size] = True

        # Split data into training and validation sets
        X_train, y_train = train_set[:, ~val_mask], labels[~val_mask]
        X_val, y_val = train_set[:, val_mask], labels[val_mask]

        # Train model and predict on validation set
        model = GMM_TiedCov_classifier(X_train, y_train, num_classes, num_components_max=[2, 1])
        pred_labels, llr = predict_log(model, X_val, prior)

        # Store results
        llr_results.append(llr)
        preds.append(pred_labels)

    # Evaluate overall performance
    total_labels = np.concatenate(preds)
    confusion_matrix = metrics.compute_confusion_matrix(labels[:len(total_labels)], total_labels, num_classes)
    dcf = metrics.compute_DCF(1 / 11, 1, 1, confusion_matrix)
    metrics.plot_Bayes_error(np.concatenate(llr_results), labels[:len(total_labels)])

    print(f'GMM DCF = {dcf}')




if __name__ == '__main__':
    # Load training and test datasets
    DTR, LTR = fingerprints_dataset.load("Train.txt")
    DTE, LTE = fingerprints_dataset.load("Test.txt")

    # Perform PCA for dimensionality reduction to 9 principal components
    DTR_reduced, projection_matrix = PCA.pca(DTR, 9)
    # Project the test dataset using the same PCA transformation
    DTE_reduced = np.dot(projection_matrix.T, DTE)

    # Define the effective prior for the application context
    effective_prior = 1 / 11  # (0.5, cfn=1, cfp=10)

    # Uncomment to train and evaluate different GMM models
    '''
    models = [
        GMM_classifier(DTR_reduced, LTR, num_classes=2, num_components_max=[8, 1]),
        GMM_Diag_classifier(DTR_reduced, LTR, num_classes=2, num_components_max=[8, 1]),
        GMM_TiedCov_classifier(DTR_reduced, LTR, num_classes=2, num_components_max=[8, 1])
    ]

    for idx, model in enumerate(models):
        pred_labels, llr = predict_log(model, DTE_reduced, [1-effective_prior, effective_prior])
        confusion_matrix = metrics.compute_confusion_matrix(LTE, pred_labels, num_classes=2)
        dcf = metrics.compute_DCF(effective_prior, 1, 1, confusion_matrix)
        dcf_min = metrics.compute_DCFmin(llr, LTE, effective_prior, 1, 1)
        print(f"-------\nDCF Model {idx+1}: {dcf}\nMIN DCF Model {idx+1}: {dcf_min}")
    '''

    # Perform K-fold validation using the reduced dataset
    K_fold(10, DTR_reduced, LTR, num_classes=2, prior=[1 - effective_prior, effective_prior])



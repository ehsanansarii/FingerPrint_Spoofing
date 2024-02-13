# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 22:15:06 2023

@author: Ehsan Ansari
"""

import fingerprints_dataset
import numpy as np
import PCA
import metrics
from LDA import compute_Sw
from scipy import special
import MVG

def vcol(vector):
    return vector.reshape((vector.shape[0],1))

def vrow(vector):
    return vector.reshape((1,vector.shape[0]))

def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1]*2.0/3.0)
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]
    return (DTR, LTR), (DTE, LTE)

def calculate_mvg_model(train_data, class_labels, total_classes):
    mvg_model = []
    for class_id in range(total_classes):
        # Filter training data for the current class
        current_class_data = train_data[:, class_labels == class_id]
        # Estimate mean and covariance for the current class
        mean_vector, covariance_matrix = MVG.ML_estimate(current_class_data)
        # Append the mean and covariance to the model
        mvg_model.extend([mean_vector, covariance_matrix])
    return mvg_model

def calculate_mvg_tied_model(train_data, class_labels, total_classes):
    tied_model = []
    # Compute shared within-class covariance matrix
    shared_covariance = compute_Sw(train_data, class_labels, total_classes)
    for class_id in range(total_classes):
        # Filter training data for the current class
        current_class_data = train_data[:, class_labels == class_id]
        # Estimate mean for the current class, ignoring its covariance
        mean_vector, _ = MVG.ML_estimate(current_class_data)
        # Append the mean and shared covariance to the model
        tied_model.extend([mean_vector, shared_covariance])
    return tied_model


def calculate_naive_bayes_model(training_data, labels, number_of_classes):
    naive_bayes_model = calculate_mvg_model(training_data, labels, number_of_classes)

    for class_index in range(number_of_classes):
        covariance_matrix_index = 2 * class_index + 1
        # Convert covariance matrix to diagonal matrix for Naive Bayes assumption
        naive_bayes_model[covariance_matrix_index] = np.diag(np.diag(naive_bayes_model[covariance_matrix_index]))

    return naive_bayes_model


def calculate_mvg_tied_naive_bayes_model(training_data, labels, number_of_classes):
    tied_naive_bayes_model = calculate_mvg_tied_model(training_data, labels, number_of_classes)

    for class_index in range(number_of_classes):
        covariance_matrix_index = 2 * class_index + 1
        # Apply Naive Bayes assumption by diagonalizing the shared covariance matrix
        tied_naive_bayes_model[covariance_matrix_index] = np.diag(
            np.diag(tied_naive_bayes_model[covariance_matrix_index]))

    return tied_naive_bayes_model


def compute_log_likelihood_ratios(model, test_samples, priors):
    """
    Computes log likelihood ratios for test samples given a model and priors.

    :param model: The trained model with methods to compute log likelihoods.
    :param test_samples: Numpy array of test samples.
    :param priors: List or numpy array of prior probabilities for each class.
    :return: Numpy array of log likelihood ratios for all test samples.
    """
    # Placeholder for log likelihood ratios
    log_likelihood_ratios = np.zeros(len(test_samples))

    # Assuming the model has a method 'log_likelihood' that returns log likelihoods for each class
    for i, sample in enumerate(test_samples):
        log_likelihood_class_1 = model.log_likelihood(sample, class_id=1)
        log_likelihood_class_2 = model.log_likelihood(sample, class_id=2)

        # Compute log likelihood ratio and store it
        ratio = log_likelihood_class_1 - log_likelihood_class_2
        log_likelihood_ratios[i] = ratio + np.log(priors[1] / priors[0])

    return log_likelihood_ratios


def predict(model, test_samples, priors):
    num_classes = len(model) // 2
    joint_scores = []

    for class_index in range(num_classes):
        mean_vector = model[2 * class_index]
        covariance_matrix = model[2 * class_index + 1]
        class_likelihood = np.exp(MVG.logpdf_GAU_ND(test_samples, mean_vector, covariance_matrix))
        joint_scores.append(class_likelihood * priors[class_index])

    marginal_score = np.sum(joint_scores, axis=0)
    class_posteriors = [joint_score / marginal_score for joint_score in joint_scores]
    predicted_labels = np.argmax(class_posteriors, axis=0)

    return predicted_labels.ravel()


def predict_log(model, test_samples, priors):
    num_classes = len(model) // 2
    log_joint_scores = []

    for class_index in range(num_classes):
        mean_vector = model[2 * class_index]
        covariance_matrix = model[2 * class_index + 1]
        class_log_likelihood = MVG.logpdf_GAU_ND(test_samples, mean_vector, covariance_matrix)
        log_joint_scores.append(class_log_likelihood + np.log(priors[class_index]))

    log_marginal = special.logsumexp(log_joint_scores, axis=0)
    log_posteriors = np.array(log_joint_scores) - log_marginal
    predicted_labels = np.argmax(np.exp(log_posteriors), axis=0)
    llr = log_posteriors[1, :] - log_posteriors[0, :] - np.log(priors[1] / priors[0])

    return predicted_labels.ravel(), llr



def calculate_accuracy(predicted, original):
    if len(predicted) != len(original):
        raise ValueError("Lengths of predicted and original labels must match.")
    correct_predictions = np.sum(predicted == original)
    return (correct_predictions / len(predicted)) * 100

def calculate_error_rate(predicted, original):
    return 100 - calculate_accuracy(predicted, original)

def normalize_dataset_z_score(dataset):
    mean_vector = np.mean(dataset, axis=1, keepdims=True)
    std_vector = np.std(dataset, axis=1, ddof=1, keepdims=True)  # ddof=1 for sample standard deviation
    return (dataset - mean_vector) / std_vector


def K_fold(K, train_set, labels, num_classes, prior):
    np.random.seed(0)
    indices = np.arange(train_set.shape[1])
    np.random.shuffle(indices)
    train_set, labels = train_set[:, indices], labels[indices]

    fold_size = train_set.shape[1] // K
    results1, results2, results3, results4 = [], [], [], []

    for i in range(K):
        start, end = i * fold_size, (i + 1) * fold_size
        val_indices = np.arange(start, end)
        train_indices = np.delete(np.arange(train_set.shape[1]), val_indices)

        curr_tr_set, curr_val_set = train_set[:, train_indices], train_set[:, val_indices]
        curr_tr_labels, curr_val_labels = labels[train_indices], labels[val_indices]

        model1 = calculate_mvg_model(curr_tr_set, curr_tr_labels, num_classes)
        model2 = calculate_naive_bayes_model(curr_tr_set, curr_tr_labels, num_classes)
        model3 = calculate_mvg_tied_model(curr_tr_set, curr_tr_labels, num_classes)
        model4 = calculate_mvg_tied_naive_bayes_model(curr_tr_set, curr_tr_labels, num_classes)

        llr1 = predict_log(model1, curr_val_set, prior)[1]
        llr2 = predict_log(model2, curr_val_set, prior)[1]
        llr3 = predict_log(model3, curr_val_set, prior)[1]
        llr4 = predict_log(model4, curr_val_set, prior)[1]

        results1.append(llr1)
        results2.append(llr2)
        results3.append(llr3)
        results4.append(llr4)

    dcfmin1 = metrics.compute_DCFmin(np.concatenate(results1), labels[:2320], prior[1], 1, 1)
    dcfmin2 = metrics.compute_DCFmin(np.concatenate(results2), labels[:2320], prior[1], 1, 1)
    dcfmin3 = metrics.compute_DCFmin(np.concatenate(results3), labels[:2320], prior[1], 1, 1)
    dcfmin4 = metrics.compute_DCFmin(np.concatenate(results4), labels[:2320], prior[1], 1, 1)

    print(f'MVG = {dcfmin1}')
    print(f'NaiveBayes = {dcfmin2}')
    print(f'Tied = {dcfmin3}')
    print(f'Tied Naive Bayes= {dcfmin4}')



def main():
    # Load training and evaluation data
    DTR, LTR = fingerprints_dataset.load("Train.txt")
    DTE, LTE = fingerprints_dataset.load("Test.txt")

    # Normalize and apply PCA to the training data
    DTR_normalized = normalize_dataset_z_score(DTR)
    DTR_PCA, _ = PCA.pca(DTR_normalized, components=9)

    # Perform K-fold cross-validation on the PCA-transformed data
    effective_prior = 1 / 11  # Adjust as per the specific application requirement
    K_fold(K=10, train_set=DTR_PCA, labels=LTR, num_classes=2, prior=[1-effective_prior, effective_prior])


    # Uncomment and modify the following lines as needed based on cross-validation results

    # model = MVG_classifier(DTR, LTR, num_classes=2)
    # predicted_labels, _ = predict_log(model, DTE, prior=[0.5, 0.5])
    # err = error_rate(predicted_labels, LTE)
    # print(f"Error Rate: {err}%")

if __name__ == '__main__':
    main()



import numpy as np
import fingerprints_dataset


def lda(dataset, labels, num_components):
    """Perform Linear Discriminant Analysis (LDA)"""
    # Calculate the overall mean
    mean_overall = np.mean(dataset, axis=1).reshape(-1, 1)

    # Calculate the within-class and between-class scatter matrices
    S_W = np.zeros((dataset.shape[0], dataset.shape[0]))
    S_B = np.zeros((dataset.shape[0], dataset.shape[0]))

    class_labels = np.unique(labels)

    for c in class_labels:
        # Compute per class mean vector
        data_c = dataset[:, labels == c]
        mean_c = np.mean(data_c, axis=1).reshape(-1, 1)

        # Within-class scatter
        S_W += np.dot((data_c - mean_c), (data_c - mean_c).T)

        # Between-class scatter
        n_c = data_c.shape[1]
        mean_diff = (mean_c - mean_overall)
        S_B += n_c * (mean_diff).dot(mean_diff.T)

    # Solve the generalized eigenvalue problem for discriminant directions
    eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))

    # Sort eigenvectors by eigenvalues in descending order
    pairs = [(np.abs(eigenvalues[i]), eigenvectors[:, i]) for i in range(len(eigenvalues))]
    pairs = sorted(pairs, key=lambda x: x[0], reverse=True)

    # Stack the eigenvectors to form a transformation matrix
    W = np.hstack([pairs[i][1].reshape(dataset.shape[0], 1) for i in range(num_components)])

    # Project the dataset onto the new feature subspace
    transformed = W.T.dot(dataset)

    return transformed


if __name__ == '__main__':
    # Load the dataset
    dataset, labels = fingerprints_dataset.load("Train.txt")

    # Perform LDA
    transformed_dataset = lda(dataset, labels, num_components=2)

    # Plotting the LDA result (you need to define this function)
    # fingerprints_dataset.plot_scatter(transformed_dataset, labels)

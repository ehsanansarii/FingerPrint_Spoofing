import numpy as np
import matplotlib.pyplot as plt


def mcol(v):
    return np.reshape(v, (-1, 1))


def load(filename):
    with open(filename) as f:
        data = [line.strip().split(',') for line in f if line.strip()]
    D = np.array([[float(attr) for attr in row[:10]] for row in data])
    labels = np.array([int(row[-1]) for row in data], dtype=int)
    return D.T, labels  # Transpose D to match original structure


def plot_heatmap(D, L):
    # Heatmap for all data
    plt.figure(figsize=(10, 8))
    corr_matrix_all = np.corrcoef(D)
    plt.imshow(corr_matrix_all, cmap='seismic')
    plt.colorbar()
    plt.title('Pearson Correlation Heatmap - All data')
    plt.tight_layout()
    plt.savefig('heatmap_all_data.png', dpi=300)
    plt.show()

    # Heatmap for authentic fingerprints
    plt.figure(figsize=(10, 8))
    corr_matrix_authentic = np.corrcoef(D[:, L == 1])
    plt.imshow(corr_matrix_authentic, cmap='seismic')
    plt.colorbar()
    plt.title('Pearson Correlation Heatmap - Authentic fingerprints')
    plt.tight_layout()
    plt.savefig('heatmap_authentic_fingerprints.png', dpi=300)
    plt.show()

    # Heatmap for spoofed fingerprints
    plt.figure(figsize=(10, 8))
    corr_matrix_spoofed = np.corrcoef(D[:, L == 0])
    plt.imshow(corr_matrix_spoofed, cmap='seismic')
    plt.colorbar()
    plt.title('Pearson Correlation Heatmap - Spoofed fingerprints')
    plt.tight_layout()
    plt.savefig('heatmap_spoofed_fingerprints.png', dpi=300)
    plt.show()


def plot_scatter(D, L):
    num_features = D.shape[0]  # Get the number of features from the dataset shape
    D0 = D[:, L == 0]
    D1 = D[:, L == 1]

    for dIdx1 in range(num_features):
        for dIdx2 in range(num_features):
            if dIdx1 == dIdx2:
                continue
            plt.figure()
            plt.xlabel(f'Feature {dIdx1}')
            plt.ylabel(f'Feature {dIdx2}')
            plt.scatter(D0[dIdx1, :], D0[dIdx2, :], label='Spoofed-fingerprint')
            plt.scatter(D1[dIdx1, :], D1[dIdx2, :], label='Authentic-fingerprint')

            plt.legend()
            plt.tight_layout()  # Use with non-default font size to keep axis label inside the figure
            plt.savefig(f'scatter_{dIdx1}_{dIdx2}.png')
            plt.close()  # Close the figure to avoid displaying it with plt.show()


def plot_hist(D, L):
    feature_names = [f'Feature_{i}' for i in range(10)]
    spoofed_color, authentic_color = "#0000FF", "#FF0000"  # Blue, Red Hex codes

    for feature_index in range(10):
        plt.figure()
        plt.xlabel(feature_names[feature_index])

        # Plot histograms for spoofed and authentic fingerprints
        plt.hist(D[feature_index, L == 0], bins=80, density=True, alpha=0.6, color=spoofed_color,
                 label='Spoofed-fingerprint')
        plt.hist(D[feature_index, L == 1], bins=80, density=True, alpha=0.6, color=authentic_color,
                 label='Authentic-fingerprint')

        plt.legend()
        plt.tight_layout()
        plt.savefig(f'hist_{feature_index}.png')
        plt.close()  # Close the figure to avoid displaying it with plt.show()


if __name__ == '__main__':
    plt.rc('font', size=16)  # Adjust font size
    D, L = load('/content/train.txt')
    # plot_heatmap(D, L)
    plot_scatter(D, L)

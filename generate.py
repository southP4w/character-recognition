import numpy as np
from matplotlib import pyplot as plt


def plot_random_samples(X, y, samples_per_set: int):
    fig, axes   = plt.subplots(10, samples_per_set, figsize=(samples_per_set, 10))
    for digit in range(10):
        index       = np.flatnonzero(y==digit)
        selected    = np.random.choice(index, samples_per_set, replace=False)
        for i, j in enumerate(selected):
            axes[digit, i].imshow(X[j].reshape(16, 16), cmap='gray', vmin=-1, vmax=1)
            axes[digit, i].axis('off')
    plt.tight_layout()
    plt.show()


def plot_mean_image(X, y, *digit_sets: int):
    if not digit_sets:
        data        = X
    else:
        selected    = []
        for digit_set in digit_sets:
            selected.append(X[y==digit_set])
        data        = np.vstack(selected)
    image   = data.mean(axis=0).reshape(16, 16)
    plt.figure()
    plt.imshow(image, cmap='gray', vmin=-1, vmax=1)
    plt.title(f'Mean Training Image for digit(s) {digit_sets}')
    plt.axis('off')
    plt.show()


def plot_median_image(X, y, *digit_sets: int):
    if not digit_sets:
        data        = X
    else:
        selected    = []
        for digit_set in digit_sets:
            selected.append(X[y==digit_set])
        data    = np.vstack(selected)
    image   = np.median(data, axis=0).reshape(16, 16)
    plt.figure()
    plt.imshow(image, cmap='gray', vmin=-1, vmax=1)
    plt.title(f'Median Training Image for digit(s) {digit_sets}')
    plt.axis('off')
    plt.show()


def plot_mode_image(X, y, *digit_sets: int):
    if not digit_sets:
        data        = X
    else:
        selected    = []
        for digit_set in digit_sets:
            selected.append(X[y==digit_set])
        data        = np.vstack(selected)
    modes   = np.zeros(data.shape[1])
    for i in range(data.shape[1]):
        values, counts  = np.unique(data[:, i], return_counts=True)
        modes[i]        = values[counts.argmax()]
    image   = modes.reshape(16, 16)
    plt.figure()
    plt.imshow(image, cmap='gray', vmin=-1, vmax=1)
    plt.title(f'Mode Training Image for digit(s) {digit_sets}')
    plt.axis('off')
    plt.show()

import numpy as np


def mixup_data(batch, batch_labels, alpha):

    lam = np.random.beta(alpha, alpha)
    batch_size = batch.shape[0]
    permutation = np.random.permutation(batch_size)
    mix_x = lam*batch + (1 - lam)*batch[permutation, :, :, :]
    y_a = batch_labels
    y_b = batch_labels[permutation, :]

    return batch, mix_x, y_a, y_b, lam
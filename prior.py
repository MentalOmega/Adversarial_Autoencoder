

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from math import sin,cos,sqrt

def gaussian_mixture(batch_size, n_dim=2, n_labels=10, x_var=0.5, y_var=0.1, label_indices=None):
    if n_dim % 2 != 0:
        raise Exception("n_dim must be a multiple of 2.")

    def sample(x, y, label, n_labels):
        shift = 1.4
        r = 2.0 * np.pi / float(n_labels) * float(label)
        new_x = x * cos(r) - y * sin(r)
        new_y = x * sin(r) + y * cos(r)
        new_x += shift * cos(r)
        new_y += shift * sin(r)
        return np.array([new_x, new_y]).reshape((2,))

    x = np.random.normal(0, x_var, (batch_size, n_dim // 2))
    y = np.random.normal(0, y_var, (batch_size, n_dim // 2))
    z = np.empty((batch_size, n_dim), dtype=np.float32)
    for batch in range(batch_size):
        for zi in range(n_dim // 2):
            if label_indices is not None:
                z[batch, zi*2:zi*2+2] = sample(x[batch, zi], y[batch, zi], label_indices[batch], n_labels)
            else:
                z[batch, zi*2:zi*2+2] = sample(x[batch, zi], y[batch, zi], np.random.randint(0, n_labels), n_labels)

    return z


def swiss_roll(batch_size, n_dim=2, n_labels=10, label_indices=None):
    def sample(label, n_labels):
        uni = np.random.uniform(0.0, 1.0) / float(n_labels) + float(label) / float(n_labels)
        r = sqrt(uni) * 3.0
        rad = np.pi * 4.0 * sqrt(uni)
        x = r * cos(rad)
        y = r * sin(rad)
        return np.array([x, y]).reshape((2,))

    z = np.zeros((batch_size, n_dim), dtype=np.float32)
    for batch in range(batch_size):
        for zi in range(n_dim // 2):
            if label_indices is not None:
                z[batch, zi*2:zi*2+2] = sample(label_indices[batch], n_labels)
            else:
                z[batch, zi*2:zi*2+2] = sample(np.random.randint(0, n_labels), n_labels)
    return z



def uniform(batch_size, n_dim, n_labels=10, minv=-1, maxv=1, label_indices=None):
#     z = np.random.uniform(minv, maxv, (batch_size, n_dim)).astype(np.float32)
#     return z
    def sample(label, n_labels):
        num = int(np.ceil(np.sqrt(n_labels)))
        size = (maxv-minv)*1.0/num
        x, y = np.random.uniform(-size/2, size/2, (2,))
        i = label / num
        j = label % num
        x += j*size+minv+0.5*size
        y += i*size+minv+0.5*size
        return np.array([x, y]).reshape((2,))

    z = np.empty((batch_size, n_dim), dtype=np.float32)
    for batch in range(batch_size):
        for zi in range(n_dim//2):
            if label_indices is not None:
                z[batch, zi*2:zi*2+2] = sample(label_indices[batch], n_labels)
            else:
                z[batch, zi*2:zi*2+2] = sample(np.random.randint(0, n_labels), n_labels)
    return z


def gaussian(batch_size, n_dim, mean=0, var=1):
    z = np.random.normal(mean, var, (batch_size, n_dim)).astype(np.float32)
    return z



def plot_latent_variable(X, Y):
    # print '%d samples in total' % X.shape[0]
    if X.shape[1] != 2:
        pca = PCA(n_components=2)
        X = pca.fit_transform(X)
        print(pca.explained_variance_ratio_)
    plt.figure(figsize=(16, 16))
#     plt.axes().set_aspect('equal')
    color = plt.cm.Spectral(np.linspace(0, 1, 10))
    for l, c in enumerate(color):
        inds = np.where(Y==l)
        # print '\t%d samples of label %d' % (len(inds[0]), l)
        plt.scatter(X[inds, 0], X[inds, 1], color=c, label=l)
    # plt.xlim([-5.0, 5.0])
    # plt.ylim([-5.0, 5.0])
    plt.legend()
    plt.show()


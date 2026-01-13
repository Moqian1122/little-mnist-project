import matplotlib.pyplot as plt
import numpy as np

def mnist_image_display(nrows: int, ncols: int, x_train, t_train):

    fig, axes = plt.subplots(nrows = nrows, ncols = ncols, figsize=(7,7))

    for i in range(nrows):
        for j in range(nrows):
            axes[i][j].imshow(x_train[t_train == i][j].reshape(28, 28), cmap='grey')
            axes[i][j].set_axis_off()
            if j == nrows - 1:
                axes[i][j].text(2, 0.5, f'class "{i}"',
                    horizontalalignment='center',
                    verticalalignment='center',
                    rotation="horizontal",
                    transform=axes[i][j].transAxes)

    fig.suptitle(t="Illustration of data of class 0 - 9")

def normalize_image_array(x):
    return x / np.max(x)
import matplotlib.pyplot as plt
import math


def plot_pair(im_l, im_r):
    plt.subplot(1, 2, 1)
    plt.imshow(im_l)
    plt.subplot(1, 2, 2)
    plt.show()
    plt.imshow(im_r)


def multi_plot(ims):
    """function for plotting either single image or list of images
        argument: image, or list of images"""

    def plot_subplot(ims):
        num_im = len(ims)
        x_grid = math.floor(num_im ** 0.65)
        y_grid = math.ceil(num_im ** 0.65)
        for idx, im in enumerate(ims, start=1):
            plt.subplot(y_grid, x_grid, idx)
            plt.imshow(im.astype('uint8'))
            plt.axis('off')
        plt.show()

    if isinstance(ims, list):
        plot_subplot(ims)
    elif isinstance(ims, dict):
        plot_subplot(ims.values())
    else:
        plot_subplot([ims])

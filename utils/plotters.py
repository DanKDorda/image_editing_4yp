import matplotlib.pyplot as plt


def plot_pair(im_l, im_r):
    plt.subplot(1, 2, 1)
    plt.imshow(im_l)
    plt.subplot(1, 2, 2)
    plt.imshow(im_r)
    plt.show()

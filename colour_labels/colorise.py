import numpy as np
import torch


###############################################################################
# Code from
# https://github.com/ycszen/pytorch-seg/blob/master/transform.py
# Modified so it complies with the Citscape label map colors
###############################################################################


def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])


def labelcolormap(N):
    if N == 35:  # cityscape
        cmap = np.array([(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (111, 74, 0), (81, 0, 81),
                         (128, 64, 128), (244, 35, 232), (250, 170, 160), (230, 150, 140), (70, 70, 70),
                         (102, 102, 156), (190, 153, 153),
                         (180, 165, 180), (150, 100, 100), (150, 120, 90), (153, 153, 153), (153, 153, 153),
                         (250, 170, 30), (220, 220, 0),
                         (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142),
                         (0, 0, 70),
                         (0, 60, 100), (0, 0, 90), (0, 0, 110), (0, 80, 100), (0, 0, 230), (119, 11, 32), (0, 0, 142)],
                        dtype=np.uint8)
    else:
        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r, g, b = 0, 0, 0
            id = i
            for j in range(7):
                str_id = uint82bin(id)
                r = r ^ (np.uint8(str_id[-1]) << (7 - j))
                g = g ^ (np.uint8(str_id[-2]) << (7 - j))
                b = b ^ (np.uint8(str_id[-3]) << (7 - j))
                id = id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b
    return cmap


class Colorize(object):
    def __init__(self, n=35):
        self.cmap = labelcolormap(n)
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.size()
        color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)
        # total = 0
        for label in range(0, len(self.cmap)):
            mask = (label == gray_image[0]).cpu()
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]
            # total += mask[mask > 0].numel()
            # print('label: {}, colours: {} : {} : {}, swapped: {}\n'.format(label, self.cmap[label][0],
            #                                                               self.cmap[label][1], self.cmap[label][2],

        # assert total == 1024 * 512
        return color_image


def test(usedata=None):
    import cv2
    import matplotlib.pyplot as plt
    data = '../data/good_results/res_uncold/epoch116_synthesized_image.jpg'
    data2 = '../data/good_results/res_uncold/epoch116_real_image.jpg'
    if usedata is None:
        usedata = data
    im0 = cv2.imread(usedata)

    im = torch.tensor(im0)
    im = im.permute(2, 0, 1)
    col = Colorize()
    im2 = col(im)
    im2 = im2.permute(1, 2, 0).numpy()

    def plotting():
        print(im2.shape)
        plt.subplot(2, 1, 1)
        plt.imshow(im2)
        plt.axis('off')
        plt.subplot(2, 1, 2)
        plt.imshow(im0)
        plt.axis('off')
        plt.show()

    def saving():
        name = usedata.rstrip('.pngj') + '_col.jpg'
        cv2.imwrite(name, im2)

    saving()


datas = [
    '../data/good_results/res_uncold/epoch116_real_image.jpg',
    '../data/good_results/res_uncold/epoch116_synthesized_image.jpg',
    '../data/good_results/res_uncold/epoch032_real_image.jpg',
    '../data/good_results/res_uncold/epoch032_synthesized_image.jpg',
]

for d in datas:
    test(d)

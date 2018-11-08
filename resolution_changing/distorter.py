import os
import numpy as np
import utils.file_utils as fu
from PIL import Image
import matplotlib.pyplot as plt


class Resizer:

    def __init__(self, im):
        self.im = np.array(im)  # is this neccessary? ???
        # store a PIL or numpy image???

        self.SIMPLE = 0
        self.ONE_HOT = 1
        self.BOUND_RECT = 2

    def get_resize(self, mode, scale=None):

        if not scale:
            scale = 16

        if mode == 0:
            return self.resize_simple(scale)
        elif mode == 1:
            return self.resize_one_hot(scale)
        elif mode == 2:
            return self.resize_bound(scale)
        else:
            raise ValueError('bad mode')

    def resize_simple(self, scale):
        # this work on PIL
        temp_im = Image.fromarray(self.im)
        w = int(temp_im.size[0] / scale)
        h = int(temp_im.size[1] / scale)
        size = (w, h)
        downscaled = temp_im.resize(size, Image.NEAREST)
        upscaled = downscaled.resize((temp_im.size[0], temp_im.size[1]), Image.BILINEAR)
        return np.array(upscaled)

    def resize_one_hot(self, scale):
        # this is a np operation
        return self.im

    def resize_bound(self, scale):
        # ditto
        return self.im

    def show_resizes(self):
        im_dict = {}
        plt.subplot(2, 2, 1)
        plt.axis('off')
        plt.imshow(self.im)

        for i in range(3):
            im_dict[i] = self.get_resize(i)
            plt.subplot(2, 2, i + 2)
            plt.imshow(im_dict[i])
            plt.axis('off')

        plt.show()


class Distorter:

    def __init__(self, t_dir):
        self.file_list = fu.get_files(t_dir)
        self.num_files = len(self.file_list)

    def load_im(self, idx):
        im = Image.open(self.file_list[idx])
        return np.array(im)

    def distort(self, im, range, start=0):
        assert range + start < self.num_files

        for file_name in self.file_list:
            im = self.load_im(file_name)
            self.save_distorted(im)

    def save_distorted(self, im):
        im_path = 'test'
        im_pil = Image.fromarray(im)
        im_pil.save(im_path)


test_im = Image.open('../data/test_dir/aachen_000001_000019_gtFine_labelIds.png')
r = Resizer(test_im)
r.show_resizes()

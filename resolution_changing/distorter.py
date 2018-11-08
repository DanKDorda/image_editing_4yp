import os
import numpy as np
import utils.file_utils as fu
from PIL import Image
import matplotlib.pyplot as plt


class Resizer:

    def __init__(self, im):
        self.im = im  # is this neccessary? ???
        # store a PIL or numpy image???

        self.SIMPLE = 0
        self.ONE_HOT = 1
        self.BOUND_RECT = 2

    def get_resize(self, mode, *args):
        if mode == 0:
            return self.resize_simple(self.im, args[0])
        elif mode == 1:
            return self.resize_one_hot(self.im, args)
        elif mode == 2:
            return self.resize_bound(self.im, args)
        else:
            raise ValueError('bad mode')

    def resize_simple(self, im, scale):
        # this work on PIL
        w = im.shape[0] / scale
        h = im.shape[1] / scale
        size = (w, h)
        return im.resize(size, Image.LANCZOS)

    def resize_one_hot(self, scale):
        # this is a np operation
        pass

    def resize_bound(self, scale):
        # ditto
        pass

    def show_resizes(self):
        im_dict = {}
        for i in range(3):
            im_dict[i] = self.get_resize(i)
            plt.subplot(1, 3, i)
            plt.imshow(im_dict[i])
            plt.axes('off')

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

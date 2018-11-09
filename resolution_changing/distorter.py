from os.path import join
import numpy as np
import utils.file_utils as fu
from PIL import Image
import matplotlib.pyplot as plt
import resolution_changing.bounding as bou
import cv2


class Resizer:

    SIMPLE = 0
    ONE_HOT = 1
    BOUND_RECT = 2

    def __init__(self, im):
        self.im = np.array(im)  # is this neccessary? ???
        # store a PIL or numpy image???

    def get_resize(self, mode, scale=None, return_format='?'):

        if not scale:
            scale = 8

        if mode == 0:
            return self.resize_simple(scale)
        elif mode == 1:
            return self.resize_one_hot(scale)
        elif mode == 2:
            return self.resize_bound(scale)
        else:
            raise ValueError('bad mode')

    def resize_simple(self, scale, image=None):
        # this work on PIL
        if image is None:
            temp_im = Image.fromarray(self.im)
        else:
            temp_im = Image.fromarray(image)
        w = int(temp_im.size[0] / scale)
        h = int(temp_im.size[1] / scale)
        size = (w, h)
        downscaled = temp_im.resize(size, Image.NEAREST)
        upscaled = downscaled.resize((temp_im.size[0], temp_im.size[1]), Image.NEAREST)
        return np.array(upscaled)

    def resize_one_hot(self, scale):
        # this is a np operation -> si self.im is right type
        hots, cols = bou.make_one_hots(self.im)
        resize_list = [self.resize_simple(scale, h[0]) for h in
                       np.split(hots, hots.shape[0])]
        glued_im = bou.glue(resize_list, cols)
        return glued_im

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
        self.t_dir = t_dir
        self.file_list = fu.get_files(t_dir)
        self.num_files = len(self.file_list)

    def load_im(self, idx):
        im = Image.open(self.file_list[idx])
        return np.array(im)

    def distort_directory(self, im, range, start=0, mode=Resizer.SIMPLE, res_dir='../results'):
        assert range + start < self.num_files
        fu.safe_dir(join(self.t_dir, res_dir))

        for file_name in self.file_list:
            path = join(self.t_dir, file_name)
            im = Image.open(path)
            r = Resizer(im)
            im2 = r.get_resize(mode)
            self.save_distorted(im2)

    def save_distorted(self, im, og_file_name):
        new_name = og_file_name.strip('.png') + 'distorted.png'
        save_path = join(self.t_dir, new_name)
        im_pil = Image.fromarray(im)
        im_pil.save(save_path)


test_im = Image.open('../data/test_img.bmp')
res = Resizer(test_im)
res.show_resizes()

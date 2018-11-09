from os.path import join
import numpy as np
import utils.file_utils as fu
import utils.plotters as p
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

    def get_resize(self, mode, scale=None):

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

    def get_hierarchy(self, mode, scales=[2, 4, 8, 16, 32, 64]):
        im_dict = {}

        for scale in scales:
            rescaled = self.get_resize(mode, scale)
            im_dict[str(scale)] = rescaled

        return im_dict


class Distorter:

    def __init__(self, t_dir, res_dir='results'):
        self.t_dir = t_dir
        self.res_dir = res_dir
        self.file_list = fu.get_files(t_dir, 'results', 'results_np', 'res_results')
        self.num_files = len(self.file_list)

    def distort_directory(self, images_to_change=0, start=0, mode=Resizer.SIMPLE):
        assert images_to_change + start < self.num_files
        fu.safe_dir(join(self.t_dir, self.res_dir))
        if images_to_change < 1:
            images_to_change = self.num_files

        for i, file_name in enumerate(self.file_list[start:start + images_to_change]):
            if i % 50 == 0:
                print('Done {}%'.format(i / self.num_files * 100))
            path = join(self.t_dir, file_name)
            with Image.open(path) as im:
                r = Resizer(im)
                resized = r.get_hierarchy(mode)
                for key, val in resized.items():
                    subdir = 's' + key
                    fu.safe_dir(join(self.t_dir, self.res_dir, subdir))
                    self.__save_distorted__(val, file_name, subdir, '_' + key + '.png')

    def __save_distorted__(self, im, og_file_name, sub_dir='', postfix='distorted.png'):
        new_name = og_file_name.strip('.png') + postfix
        save_path = join(self.t_dir, self.res_dir, sub_dir, new_name)
        im_pil = Image.fromarray(im)
        im_pil.save(save_path)


laptop_test = '../data/test_img.bmp'
desktop_test = '../data/test_dir/aachen_000001_000019_gtFine_labelIds.png'
test_im = Image.open(desktop_test)
res = Resizer(test_im)
# res.show_resizes()
# hierarchy = res.get_hierarchy(Resizer.SIMPLE)
# p.multi_plot(hierarchy)
desk_test_dir = '../data/test_dir'
real_deal = '../data/traning_instances'
d = Distorter(real_deal, 'res_results')
d.distort_directory()

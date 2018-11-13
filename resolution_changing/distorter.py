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
    BLEND = 3
    BLEND_HOTS = 4

    def __init__(self, im):
        self.im = np.array(im)  # is this neccessary? ???
        # store a PIL or numpy image???

    def get_resize(self, mode, scale=None):

        if not scale:
            scale = 16

        if mode == 0:
            return self.resize_simple(scale)
        elif mode == 1:
            return self.resize_one_hot(scale)
        elif mode == 2:
            return self.resize_bound(scale)
        elif mode == 3:
            return self.resize_blend(scale)
        elif mode == 4:
            return self.resize_blend_hots(scale)
        else:
            raise ValueError('bad mode')

    def resize_simple(self, scale, image=None, no_upscale=False):
        # this work on PIL
        if image is None:
            temp_im = Image.fromarray(self.im)
        else:
            temp_im = Image.fromarray(image)
        w = int(temp_im.size[0] / scale)
        h = int(temp_im.size[1] / scale)
        size = (w, h)
        downscaled = temp_im.resize(size, Image.NEAREST)
        if no_upscale:
            return np.array(downscaled)
        else:
            upscaled = downscaled.resize((temp_im.size[0], temp_im.size[1]), Image.NEAREST)
            return np.array(upscaled)

    def resize_one_hot(self, scale, params={'ker_size': (41, 41), 'thresh': 100}):
        # this is a np operation -> si self.im is right type
        # make resize
        # then one hot
        # blur
        # threshold with cv2
        # glue
        # params = {'ker_size': (31, 31), 'thresh': 100}

        def blur_and_thresh(im, params):
            im = cv2.GaussianBlur(im, params['ker_size'], 0)
            _, im = cv2.threshold(im, params['thresh'], 255, cv2.THRESH_BINARY)
            return im

        small_img = self.resize_simple(scale)
        hots, cols = bou.make_one_hots(small_img)
        blurred_thresh_list = [blur_and_thresh(h[0], params) for h in np.split(hots, hots.shape[0])]

        # gon shelve this idea of doing this to specific layers only, might be useful somewhere else
        # li = [0, 1, 3]
        # resize_list = [self.resize_simple(scale, h) if i in li else h for i, h in enumerate(hot_list)]
        glued_im = bou.glue(blurred_thresh_list, cols)
        return glued_im

    def resize_bound(self, scale):
        # better idea for this
        # get small resize then blend between low and high res
        # then threshold
        hots, cols = bou.make_one_hots(self.im)
        disarray = bou.distort_array(hots, [0, 1, 3])
        hot_list = [h[0] for h in np.split(disarray, hots.shape[0])]
        # p.multi_plot(hot_list)
        glued = bou.glue(hot_list, cols)
        final_im = self.resize_simple(scale, glued)
        return final_im

    def resize_blend(self, scale):
        small = self.resize_simple(scale)
        pills = Image.fromarray(self.im), Image.fromarray(small)
        blend = Image.blend(pills[0], pills[1], 0.5)

        return np.array(blend)

    def resize_blend_hots(self, scale):
        small = self.resize_simple(scale)
        big_hot, col = bou.make_one_hot_list(self.im)
        lil_hot, col = bou.make_one_hot_list(small)

        def blend_n_thresh(im1, im2, alpha):
            assert 0 <= alpha <= 1
            # im1_pill, im2_pill = Image.fromarray(im1), Image.fromarray(im2)
            # blend = np.array(Image.blend(im1_pill, im2_pill, alpha))
            blend = alpha * im1 + (1 - alpha) * im2
            blend = cv2.GaussianBlur(blend, (31, 31), 0)
            _, threshed = cv2.threshold(blend, 100, 255, cv2.THRESH_BINARY)
            return threshed

        thresh_list = [blend_n_thresh(big, lil, 0.3) for big, lil in zip(big_hot, lil_hot)]
        glued = bou.glue(thresh_list, col)
        return glued

    def show_resizes(self):
        plt.subplot(2, 3, 1)
        plt.axis('off')
        plt.imshow(self.im)

        for i in range(5):
            new_im = self.get_resize(i)
            plt.subplot(2, 3, i + 2)
            print(i)
            plt.imshow(new_im)
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


def distort_in_avl():
    desk_test_dir = '../data/test_dir'
    real_deal = '../data/traning_instances'
    d = Distorter(real_deal, 'res_results')
    d.distort_directory()


def test(device=1, mode='methods'):
    if device == 1:
        test_im_path = '../data/test_img.bmp'
    else:
        test_im_path = '../data/test_dir/aachen_000001_000019_gtFine_labelIds.png'
    test_im = Image.open(test_im_path)
    print('test im shape ', np.array(test_im).shape)
    res = Resizer(test_im)
    if mode == 'methods':
        res.show_resizes()
    elif mode == 'hierarchies':
        hierarchy = res.get_hierarchy(Resizer.SIMPLE)
        p.multi_plot(hierarchy)


def test_hot_params(device=1):
    if device == 1:
        test_im_path = '../data/test_img.bmp'
    else:
        test_im_path = '../data/test_dir/aachen_000001_000019_gtFine_labelIds.png'
    test_im = Image.open(test_im_path)
    res = Resizer(test_im)
    params = {'ker_size': (9, 9), 'thresh': 50}
    idx = 1
    for k_size in range(59, 80, 10):
        params['ker_size'] = (k_size, k_size)
        for t in range(50, 151, 50):
            params['thresh'] = t
            im = res.resize_one_hot(8, params)
            plt.subplot(3, 3, idx)
            plt.imshow(im)
            plt.title('K: {} T: {}'.format(params['ker_size'], params['thresh']))
            plt.axis('off')
            idx += 1

    plt.show()


TEST_LAPTOP = 1
TEST_AVL = 2
test(TEST_LAPTOP)
# test_hot_params(TEST_LAPTOP)

import numpy as np
import utils.plotters as p
from PIL import Image
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
            return self.resize_simple(scale, no_upscale=False)
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

    def resize_one_hot(self, scale, params={'ker_size': (41, 41), 'thresh': 100}, recursion=True):
        # this is a np operation -> si self.im is right type
        # make resize
        # then one hot
        # blur
        # threshold with cv2
        # glue
        # params = {'ker_size': (31, 31), 'thresh': 100}

        def blur_and_thresh(im, params):
            #im = cv2.dilate(im, np.ones((10, 10)))
            im = cv2.GaussianBlur(im, params['ker_size'], 0)
            _, im = cv2.threshold(im, params['thresh'], 255, cv2.THRESH_BINARY)
            return im

        if scale > 2 and recursion:
            # speed up by remembering past image
            small_img = self.resize_one_hot(scale / 2)
        else:
            small_img = self.resize_simple(scale, self.im)

        hots, cols = bou.make_one_hots(small_img)
        blurred_thresh_list = [blur_and_thresh(h[0], params) for h in np.split(hots, hots.shape[0])]
        p.multi_plot(blurred_thresh_list)

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

    def get_hierarchy(self, mode, scales=[2, 4, 8, 16, 32, 64]):
        im_dict = {}

        for scale in scales:
            rescaled = self.get_resize(mode, scale)
            im_dict[str(scale)] = rescaled
            print('done scale ', scale)

        return im_dict


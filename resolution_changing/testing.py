import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

import resolution_changing.bounding as bou
from resolution_changing.resizing import Resizer
import utils.plotters as p


class ResizerTest(Resizer):

    def __init__(self, im):
        super(ResizerTest, self).__init__(im)

    def show_resizes(self):
        plt.subplot(2, 3, 1)
        plt.axis('off')
        plt.imshow(self.im)

        for i in range(5):
            new_im = self.get_resize(i, 64)
            plt.subplot(2, 3, i + 2)
            print(i)
            plt.imshow(new_im)
            plt.axis('off')

        plt.show()

    def resize_one_hot(self, scale, params={'ker_size': (41, 41), 'thresh': 100}, past_ims=None, recursion=True):
        # this is a np operation -> si self.im is right type
        # make resize
        # then one hot
        # blur
        # threshold with cv2
        # glue
        # params = {'ker_size': (31, 31), 'thresh': 100}

        def blur_and_thresh(im, params):
            im1 = cv2.dilate(im, np.ones((10, 10)), iterations=4)
            im2 = cv2.GaussianBlur(im1, params['ker_size'], 0)
            _, im3 = cv2.threshold(im2, params['thresh'], 255, cv2.THRESH_BINARY)
            #p.multi_plot([im, im1, im2, im3])
            return im3

        def recusrive_blur():
            if past_ims is None:
                return past_ims
            current_im = None
            if scale == 2:
                return current_im
            else:
                return self.resize_one_hot(scale/2, params)

        # resize by 2 and do the gaussian funking
        small_img = self.resize_simple(2, self.im)
        hots, cols = bou.make_one_hots(small_img)
        blurred_thresh_list = [blur_and_thresh(h[0], params) for h in np.split(hots, hots.shape[0])]
        #p.multi_plot(blurred_thresh_list)

        # gon shelve this idea of doing this to specific layers only, might be useful somewhere else
        # li = [0, 1, 3]
        # resize_list = [self.resize_simple(scale, h) if i in li else h for i, h in enumerate(hot_list)]
        glued_im = bou.glue(blurred_thresh_list, cols)
        return glued_im


def test(device=1, mode='methods'):
    if device == 1:
        test_im_path = '../data/test_img.bmp'
    else:
        test_im_path = '../data/test_dir/aachen_000001_000019_gtFine_labelIds.png'
    test_im = Image.open(test_im_path)
    print('test im shape ', np.array(test_im).shape)
    res = ResizerTest(test_im)
    if mode == 'methods':
        res.show_resizes()
    elif mode == 'hierarchies':
        hierarchy_oh = res.get_hierarchy(Resizer.ONE_HOT)
        hierarchy_sim = res.get_hierarchy(Resizer.SIMPLE)
        all_images = list(hierarchy_sim.values()) + list(hierarchy_oh.values())
        hierarchy_oh['1'] = res.im
        p.multi_plot(hierarchy_oh)
    elif mode == '64':
        im_64 = res.resize_one_hot(64)
        im_64_no_rec = res.resize_one_hot(64, recursion=False)
        p.multi_plot([res.im, im_64, im_64_no_rec])
    else:
        raise ValueError('unknown test mode')


def test_hot_params(device=1):
    if device == 1:
        test_im_path = '../data/test_img.bmp'
    else:
        test_im_path = '../data/test_dir/aachen_000001_000019_gtFine_labelIds.png'
    test_im = Image.open(test_im_path)
    res = ResizerTest(test_im)
    params = {'ker_size': (9, 9), 'thresh': 50}
    idx = 1
    for k_size in range(79, 100, 10):
        params['ker_size'] = (k_size, k_size)
        for t in range(10, 31, 10):
            params['thresh'] = t
            im = res.resize_one_hot(64, params)
            plt.subplot(3, 3, idx)
            plt.imshow(im)
            plt.title('K: {} T: {}'.format(params['ker_size'], params['thresh']))
            plt.axis('off')
            idx += 1

    plt.show()


TEST_LAPTOP = 1
TEST_AVL = 2
#test(TEST_AVL, mode='64')
test_hot_params(TEST_AVL)

import numpy as np
import utils.plotters as p
from PIL import Image
import resolution_changing.bounding as bou
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage


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

        if not isinstance(mode, int):
            mode, no_upscale = mode[0], mode[1]

        if mode == 0:
            return self.resize_simple(scale, no_upscale=no_upscale)
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
            # im = cv2.dilate(im, np.ones((10, 10)))
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
            # print('done scale ', scale)

        # print('done all scales\n')
        return im_dict


def loop_dist(distorted):
    ker_size = (5, 5)
    ker = np.ones(ker_size)
    for i in range(5):
        distorted = cv2.erode(distorted, ker)
        distorted = cv2.blur(distorted, ker_size)
        distorted = cv2.dilate(distorted, ker)
    return distorted


def get_points(vx, vy, x, y, mag=3000):
    stretch_left = mag
    x1 = x + stretch_left * vx
    y1 = y + stretch_left * vy

    stretch_right = mag
    x2 = x - stretch_right * vx
    y2 = y - stretch_right * vy

    p1 = (x1, y1)
    p2 = (x2, y2)

    return [p1, p2]


def get_midpoint(p1, p2):
    xmid = int((p1[0] + p2[0]) / 2)
    ymid = int((p1[1] + p2[1]) / 2)
    return xmid, ymid


def get_linspaced_points(p1, p2, n):
    xs = np.linspace(p1[0], p2[0], n)
    ys = np.linspace(p1[1], p2[1], n)

    points = [(int(x), int(y)) for x, y in zip(xs, ys)]
    return points


class StrokeResizer():
    def __init__(self, im, num_labels=35, method='one', do_skips=True):
        self.im = im
        self.num_labels = num_labels
        self.distorted = None
        self.method = method
        self.do_skips = do_skips

    def get_onehot(self):
        onehot = np.eye(self.num_labels)[np.array(self.im).reshape(-1)]
        onehot = onehot.reshape(list(self.im.shape) + [self.num_labels])
        return onehot.astype('uint8')

    def glue_onehot(self, onehot):
        glued = np.argmax(onehot, axis=2)
        return glued

    def get_grads(self, im_in=None):
        if im_in is None:
            grads = self.get_grads(self.im)
        else:
            grads = cv2.Laplacian(im_in, cv2.CV_8U)
        return grads

    def make_distort(self):
        distorted = self.get_onehot()

        oh_grads = self.get_grads(distorted)
        oh_grads = cv2.dilate(oh_grads, np.ones((10, 10)))
        # self.peek_at_layers(oh_grads)
        # distorted = distorted * oh_grads
        lines = np.zeros_like(distorted)
        im_area = np.prod(distorted[:, :, 0].shape)

        distorted_layers = np.split(distorted, distorted.shape[2], axis=2)
        for i, dlay in enumerate(distorted_layers):
            _, contours, _ = cv2.findContours(dlay.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            line = np.zeros_like(dlay[:, :, 0])
            perp_line = np.zeros_like(dlay[:, :, 0])
            for c in contours:
                c = c[:, 0, :]

                # fit line through contour
                [vx, vy, x, y] = cv2.fitLine(c, cv2.DIST_L2, 0, 0.01, 0.01)
                [bound1, bound2] = get_points(vx, vy, x, y)

                # make line stay within contour bounding box
                rect = cv2.boundingRect(c)
                if self.do_skips:
                    rect_area = rect[2]*rect[3]
                    if rect_area >= 0.90 * im_area:
                        print('he rect too big for he goddam image')
                        continue

                _, bound1, bound2 = cv2.clipLine(rect, bound1, bound2)

                # get n points along the line
                dis = np.sqrt((bound1[0] - bound2[0]) ** 2 + (bound1[1] - bound2[1]) ** 2)
                n = dis // 40
                if n < 2:
                    n = 2
                elif n > 30:
                    n = 30

                points = get_linspaced_points(bound1, bound2, n)
                # rec_width = np.sqrt((points[0][0]-points[1][0])**2 + (points[0][1]-points[1][1])**2)

                # draw the contour
                contour = cv2.drawContours(np.zeros_like(dlay), [c], -1, (255), cv2.FILLED)[:, :, 0]

                p_prev = bound1
                for p in points[1:]:
                    # plot a line perpendicular to main line through current point
                    perp_line.fill(0)
                    [p1, p2] = get_points(vy, -vx, p[0], p[1])
                    cv2.line(perp_line, p1, p2, 255, 10)

                    # move the perp_line or not lol, if you just plot a new line every time loool
                    # multiply the contour by perp_line
                    perp_line = cv2.multiply(perp_line, contour)
                    # find the centroid of contour fragment
                    # make that the new point
                    if perp_line.sum() == 0:
                        print('zero sum')
                        new_p = p
                    else:
                        x_cent, y_cent = ndimage.measurements.center_of_mass(perp_line)
                        new_p = (int(y_cent), int(x_cent))

                    # print('{} xy original\n{} xy new\n'.format(p, (x_cent, y_cent)))
                    # print('{} difference\n'.format((p[0]-y_cent, p[1] - x_cent)))

                    # draw a line between the points
                    # line = cv2.add(line, perp_line)
                    cv2.line(line, p_prev, new_p, 255, 3)
                    p_prev = new_p

                """
                contour = cv2.drawContours(np.zeros_like(dlay), [c], -1, (255), -1)
                contour = cv2.erode(contour, np.ones((5, 5)))
                contour = cv2.blur(contour, (80, 80))
                _, contour = cv2.threshold(contour, 0, 255, cv2.THRESH_BINARY)
                line = cv2.multiply(line, contour)
                """
                lines[:, :, i] += line

        # self.peek_at_layers(distorted)
        zero = np.zeros_like(lines[:, :, 0])
        np.dstack((zero, lines))
        distorted = self.glue_onehot(lines)
        self.distorted = distorted

    def peek_at_layers(self, layered):
        num_subplots = layered.shape[2]
        vert = np.ceil(np.sqrt(num_subplots))
        horz = np.ceil(np.sqrt(num_subplots))

        print(layered.dtype)

        for i in range(num_subplots):
            plt.subplot(vert, horz, i + 1)
            # hist = cv2.calcHist([layered[:, :, i]], [0], None, [256], [0, 256])
            plt.imshow(layered[:, :, i])

        plt.show()

    def show_current_distort(self, save=False):
        plt.subplot(2, 1, 1)
        plt.imshow(self.im)
        plt.title('image at {}'.format(self.distorted.shape))
        plt.subplot(2, 1, 2)
        plt.imshow(self.distorted)
        plt.title('distort at {}'.format(self.distorted.shape))
        plt.show()
        if save:
            ## TODO: save the distorted image
            pass

    def get_distort(self):
        return self.distorted


def experiment2(im, tic):
    sr = StrokeResizer(im)
    sr.make_distort()

    toc = time.time()
    dist = sr.get_distort()

    print('\nTime elapsed: {}'.format(toc - tic))
    tic = time.time()
    sr.do_skips = False
    sr.make_distort()
    toc = time.time()

    dist2 = sr.get_distort()
    print('\nTime elapsed: {}'.format(toc - tic))
    plotters.multi_plot([sr.im, dist, dist2])


if __name__ == "__main__":
    import time
    from utils import plotters

    tic = time.time()
    print('running stroke resizer test\n')

    image = Image.open('../data/test_dir/test3.png')
    im = np.array(image).astype('uint8')

    print(im.shape)

    sr = StrokeResizer(im)
    sr.make_distort()

    toc = time.time()
    sr.show_current_distort()

    print('\nTime elapsed: {}'.format(toc - tic))

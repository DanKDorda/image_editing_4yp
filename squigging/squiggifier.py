import matplotlib.pyplot as plt
import numpy as np
import cv2
import random
from scipy import spatial
from PIL import Image


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


def get_linspaced_points(p1, p2, n):
    xs = np.linspace(p1[0], p2[0], n)
    ys = np.linspace(p1[1], p2[1], n)

    points = [(int(x), int(y)) for x, y in zip(xs, ys)]
    return points


def get_closest_points(contour_points, projected_points, max, min, axis=0):
    select = (projected_points[:, :, axis] < max) & (min < projected_points[:, :, axis])
    close_points = contour_points[select.ravel(), :, :]
    return close_points


"""
def get_distance(p1, p2, pc):
    dist = abs((p2[1] - p1[1]) * pc[0] - (p2[0] - p1[0]) * pc[1] + p1[1] * p2[0] - p1[0] * p2[1]) / (
        np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2))
"""


def project_points(c, grad, x, y):
    c[:, :, 0] = c[:, :, 0] - x
    c[:, :, 1] = c[:, :, 1] - y
    c_line_project = np.dot(c, grad) * grad
    c_line_project = c_line_project.transpose(0, 2, 1)
    c_line_project[:, :, 0] = c_line_project[:, :, 0] + x
    c_line_project[:, :, 1] = c_line_project[:, :, 1] + y
    # c[:, :, 0] = c[:, :, 0] + x
    # c[:, :, 1] = c[:, :, 1] + y
    return c_line_project


def get_centroid(projected):
    mean = np.mean(projected, axis=0)
    mean_x, mean_y = int(mean[0, 0]), int(mean[0, 1])
    return mean_x, mean_y


def get_height(projected, scale=0.9):
    D = spatial.distance.pdist(projected[:, 0, :])
    height = np.max(D) / 2
    return height * scale


def draw_sin(im, p1, p2, phase, n=5, grad=None):
    # approximate a sinusoid from p1 to p2 with n piecewiese linear elements
    points = get_linspaced_points(p1, p2, n)
    vx = p1[0] - p2[0]
    vy = p1[1] - p2[1]
    dist = np.sqrt(vx ** 2 + vy ** 2)
    if grad is None:
        grad = np.array([vx, vy])
        if (abs(grad[0]) < 1e-6) and (abs(grad[1]) < 1e-6):
            # print(f'{grad}grad 2 smol')
            return cv2.line(im, p1, p2, 255, 3)

        grad = grad / np.sqrt(np.dot(grad.transpose(), grad))
        if grad.any() > 1.0:
            print('points r fucky')
            return cv2.line(im, p1, p2, 255, 3)

    mag = phase * (dist / 30) ** 1.3
    prev_p = points[0]
    for i, p in enumerate(points[1:]):
        flip = mag * np.sin((i + 2) * 2 * np.pi / (n))
        j = i / n
        # flip = flip*(j**2 - n*j + n+1)
        xmov = flip * grad[1]
        ymov = flip * grad[0]
        new_p = (int(p[0] + xmov), int(p[1] - ymov))
        cv2.line(im, prev_p, new_p, 255, 3)
        prev_p = new_p

    return im


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
        # contour_points = []
        for i, dlay in enumerate(distorted_layers):
            _, contours, _ = cv2.findContours(dlay.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            line = np.zeros_like(dlay[:, :, 0])
            perp_line = np.zeros_like(dlay[:, :, 0])
            for c in contours:
                if len(c) < 3:
                    continue
                # fit line through contour
                [vx, vy, x, y] = cv2.fitLine(c, cv2.DIST_L2, 0, 0.01, 0.01)

                # get bounding rect of contour, skip if dodgy contour
                rect = cv2.boundingRect(c)
                if self.do_skips:
                    rect_area = rect[2] * rect[3]
                    if rect_area >= 0.95 * im_area:
                        print('he rect too big for he gotdam image')
                        continue

                # get projected contour
                grad = np.array([vx, vy])
                c_line_proj = project_points(c, grad, x, y)

                [bound1, bound2] = get_points(vx, vy, x, y)

                # make line stay within contour bounding box
                _, bound1, bound2 = cv2.clipLine(rect, bound1, bound2)

                # get n points along the line
                dis = np.sqrt((bound1[0] - bound2[0]) ** 2 + (bound1[1] - bound2[1]) ** 2)
                n = dis // 20
                """
                if j < 20:
                    n = j
                else:
                    n = 20
                if j < 3:
                    n = 3
                if j > 40:
                    n = 40
                """
                if n < 3:
                    n = 3
                elif n > 40:
                    n = 40
                points = get_linspaced_points(bound1, bound2, n)

                # for each point along line
                p_prev = bound1
                # pnum = 0
                yea = True
                for p in points:
                    # pnum += 1
                    # print(pnum)
                    distance = 5
                    # find relevant points
                    # plot a sinusoid, going thorough centre, with height h

                    # get min and max coords
                    relevant_c = []
                    while len(relevant_c) < 2:
                        min_projected_position = (p[0] - distance * vx, p[1] - distance * vy)
                        max_projected_position = (p[0] + distance * vx, p[1] + distance * vy)
                        xwidth = (max_projected_position[0] - min_projected_position[0]) ** 2
                        ywidth = (max_projected_position[1] - min_projected_position[1]) ** 2

                        if xwidth > ywidth:
                            axis = 0
                        else:
                            axis = 1
                        max = max_projected_position[axis] if max_projected_position[axis] > min_projected_position[
                            axis] else min_projected_position[axis]
                        min = min_projected_position[axis] if min_projected_position[axis] < max_projected_position[
                            axis] else max_projected_position[axis]

                        relevant_c = get_closest_points(c, c_line_proj, max,
                                                        min, axis)
                        distance = distance * 1.5
                        # print(len(relevant_c), len(points), distance)
                        if distance > 10000:
                            print('ono')
                            print(c)
                            print(c_line_proj)
                            print(min, max)
                            raise ValueError('distance unresonably large')

                    # get their height and centre
                    grad = grad = np.array([vy, -vx])
                    projected_relevants = project_points(relevant_c, grad, p[0], p[1])
                    x_cent, y_cent = get_centroid(projected_relevants)
                    cent = (x_cent, y_cent)

                    scale = random.choice([0.75, 0.8, 0.9, 0.9, 0.95])
                    height = get_height(projected_relevants, scale)

                    # draw a sinusoid between the points
                    # line = cv2.add(line, perp_line)
                    p1, p2 = get_points(vy, -vx, p[0], p[1], mag=height)
                    # p1, p2 = get_points(vy, -vx, x_cent, y_cent, mag=height)
                    yea = not yea
                    if yea:
                        new_p = p1
                        phase = -1
                    else:
                        new_p = p2
                        phase = 1

                    phase = phase
                    line = draw_sin(line, p_prev, new_p, phase=phase, n=40)
                    # cv2.line(line, p_prev, new_p, 255, 5)
                    p_prev = new_p

                lines[:, :, i] += line

        # self.peek_at_layers(distorted)
        zero = np.zeros_like(lines[:, :, 0])
        np.dstack((zero, lines))
        distorted = self.glue_onehot(lines)
        self.distorted = distorted
        # plt.hist(np.array(contour_points), bins='auto')
        # plt.show()

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


def test_draw_sin():
    from utils import plotters
    im = np.zeros((1024, 2048))
    for i in range(20):
        rx = random.randint(0, 2048)
        ry = random.randint(0, 1024)
        p1 = (rx, ry)
        rx = random.randint(0, 2048)
        ry = random.randint(0, 1024)
        p2 = (rx, ry)
        # rn = random.randint(5, 80)
        im2 = draw_sin(im, p1, p2, n=10)
        cv2.line(im2, p1, p2, (255), 3)

    plotters.multi_plot(im2)


if __name__ == "__main__":
    import time
    from utils import plotters

    tic = time.time()
    print('running stroke resizer test\n')

    image = Image.open('../data/test_dir/test1.png')
    im = np.array(image).astype('uint8')

    print(im.shape)

    sr = StrokeResizer(im)
    sr.make_distort()

    toc = time.time()
    sr.show_current_distort()

    print('\nTime elapsed: {}'.format(toc - tic))

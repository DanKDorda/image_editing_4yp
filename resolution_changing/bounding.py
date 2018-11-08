from PIL import Image
import numpy as np
from bisect import bisect_left


def is_mono(im):
    if len(im.shape) == 2:
        return True
    elif len(im.shape) == 3:
        return False
    raise ValueError('unknown dimension')


def get_cols(im):
    try:
        pil_im = Image.fromarray(im)
    except TypeError:
        print('already a pill')
    colours = pil_im.getcolors()
    return colours


def make_one_hots(im, opts=None):
    """
    seperate images by distinct labels
    arguments: im - ndarray shape (h,w,c)
                 with im[i,j,:] having n_label unique values
    returns: hots - ndarray shape (nlabel,h,w)
             col_list - list of colours associated with each layer
    """

    # this works well for rgb images, but not for bw images of shape (h,w)
    # get colours and re-order them by size
    colours = get_cols(im)
    colours = sorted(colours, key=lambda x: -x[0])

    if opts:
        n_label = opts.n_label
    else:
        n_label = len(colours)
        # n_label = 35

    hots = np.zeros((n_label, im.shape[0], im.shape[1]))  # return size n_label, h, w
    col_array = np.zeros(n_label, dtype='int')
    # print('im dim: ', im.shape)
    for idx, colour_tups in enumerate(colours):
        colour = colour_tups[1]
        cm, i = get_col_map(im, colour, idx)
        col_array[i] = colour
        hots[i, :, :] = cm

    return hots.astype('uint8'), col_array


def get_col_map(im, colour, idx):
    # unpack values

    if is_mono(im):
        colour_map = im == colour
        #hack here!!! FOR NVIDIA MAKE idx === colour
        return 255 * colour_map, idx
    else:
        red, green, blue = im.T
        target_red, target_green, target_blue = colour
        colour_map = (red == target_red) & (green == target_green) & (blue == target_blue)
        return 255 * colour_map, idx




def get_order(lc):
    # get tuple of size x layer

    sizes = []
    ordered_l = []
    ordered_c = []
    for l, c in lc:
        size = l[l > 0].size
        idx = bisect_left(sizes, size)
        sizes.insert(idx, size)
        ordered_l.insert(-idx, l)
        ordered_c.insert(-idx, c)

    # print('dis whats coming back')
    # plot(ordered)
    return ordered_l, ordered_c


def glue(layers, colours):
    h = layers[0].shape[0]
    w = layers[0].shape[1]

    final_img = np.zeros((h, w))

    ordered_layers, ordered_colours = get_order(zip(layers, colours))

    for layer, colour in zip(ordered_layers, ordered_colours):
        #only need one colour!
        #broadcast_colour = (np.array(colour)[:, np.newaxis, np.newaxis])
        broadcast_colour = colour
        one_layer = 1 * (layer > 0)
        new_lay = (one_layer * broadcast_colour).astype('float64')  # h,w * 3
        new_lay = np.swapaxes(new_lay.T, 0, 1)

        # enable for funky graphics
        # new_lay = cv2.GaussianBlur(new_lay,(31,31),0)

        # overlay where image exists
        if is_mono(new_lay):
            mask = new_lay > 0
            final_img[mask] = new_lay[mask]
        else:
            mask = new_lay.sum(axis=2) > 0
            final_img[mask, :] = new_lay[mask, :]

        # change to += for funky graphics
        # final_img = cv2.GaussianBlur(final_img,(3,3),0)
        # final_img = cv2.add(final_img,new_lay)

    return final_img

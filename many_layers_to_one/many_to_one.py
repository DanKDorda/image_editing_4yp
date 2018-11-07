import os
from skimage import io
import utils.file_utils as fu
from PIL import Image


def make_one(file_dir = '../data/train_blocks_3_layer'):

    files = fu.get_files(file_dir)
    fu.safe_dir('blocks_1_layer')

    for im_name in files[:5]:
        print('working on: ', im_name)
        im_path = os.path.join(file_dir, im_name)
        im = io.imread(im_path)
        if len(im.shape) == 2:
            print('shape already 2d: ', im.shape)
            continue

        im_pil = Image.fromarray(im[:, :, 0])
        im_pil.save(im_path)

    print('done')

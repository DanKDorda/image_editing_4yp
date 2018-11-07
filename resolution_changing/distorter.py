import os
import numpy as np
import utils.file_utils as fu
from PIL import Image


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

from os.path import join
import utils.file_utils as fu
from PIL import Image
from resolution_changing.resizing import Resizer


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


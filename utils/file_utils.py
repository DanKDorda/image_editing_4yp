import os


def get_files(root, *args):
    cwd = os.getcwd()
    files = (os.listdir(os.path.join(cwd, root)))
    files = [f for f in files if f not in args]
    return files


def safe_dir(name):
    p = os.getcwd()
    try:
        os.mkdir(p+name)
    except FileExistsError:
        print('dir {} exists'.format(name))

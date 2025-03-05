import os

def _list_dir_recursively(basename, subdir='', list_dir=[], keep_only_dir=False):

    dirname = os.path.join(basename, subdir)
    skip_files = False

    for dire in os.listdir(dirname):
        scan_dir = os.path.join(dirname, dire)
        scan_subdir = os.path.join(subdir, dire)
        if os.path.isdir(scan_dir):
            _list_dir_recursively(basename, scan_subdir, list_dir, keep_only_dir)
        elif os.path.isfile(scan_dir) and not skip_files:
            if not keep_only_dir:
                # List file
                list_dir.append((subdir, dire))
            else:
                # List parent dir only one time
                subdir_split = subdir.split('/')
                subdir_split = ('/'.join(subdir_split[:-1]), subdir_split[-1])

                list_dir.append(subdir_split)
                skip_files = True  # Do not list again the same parent dir

    return list_dir


def list_dir_recursively(dirname, keep_only_dir=False):
    """List recursively a dir by listing all the files encountered and subdirs

        Arguments
        ---------
        basename: str
            Root dir path to list

        keep_only_dir:
            Whether to list subdirs containing plain files (True), or list the files (False)

    """
    return _list_dir_recursively(basename=dirname, subdir='', list_dir=[], keep_only_dir=keep_only_dir)
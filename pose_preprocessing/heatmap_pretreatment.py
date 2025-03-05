import re
import os
import argparse
import time
import warnings
import json
import cv2 as cv
import numpy as np
import pickle
import tqdm
from multiprocessing import Process
from utils import list_dir_recursively

# General variables
casiab_fn_fmt_re = re.compile('([0-9]{3})-(nm|cl|bg)-([0-9]{2})-([0-9]{3})')
tum_gaid_fn_fmt_re = re.compile('(p[0-9]{3})-((n|b|s)[0-9]{2})')
TERM_FN = '.npy'

# Useful functions
sort_dict = lambda d: dict(sorted(d.items()))

def fuse_heatmap_channels(hm: np.array, groups: list, agg_ope='gauss') -> np.array:

    """Fuses the heatmap joint channels from a heatmap array into one or more 
        heatmap channel groups. Fusion is performed by assuming each heatmap
        channel holds a gaussian distribution with mean 0 and std 1.

        Arguments
        ---------
        hm: Array
            Separate channel heatmaps to be fused

        groups: List of array-like
            List of joint index groups to be taken to build each heatmap group

        agg_ope: str ('gauss' or 'max')
            Aggregation operation for joint heatmaps groupping
            Allowed values: gauss or max


        Agg. operations
        ---------------

        - gauss: Each heatmap is assumed in a gaussian distribution centered
            at 0 with std 1. Thus, suming for each heatmap group is performed
            by element-wise sum of the joint heatmaps, and dividing the heatmap
            group by the square root of the number of joint grouped:

          ```
            hm_group_{i} =  \frac{\sum_{j \in J_g} hm_{j}} {\sqrt{n_{g}}} \forall_{i} \in G
          ```
          
          where $hm_{j}$ is the raw heatmap for joint j, $J_g$ is the set of
          joint indexes included in the heatmap group, and $n_{g}$ is the number
          of joints grouped. $i$ refers to the i-th heatmap group, while G
          is the set of groups

        - max: Heatmaps are grouped by maximizing the values among all joint
            heatmaps in the same position

        Return
        ------
        Array containing the heatmaps with the aggrupation
    """

    n_c, h, w = hm.shape
    n_groups = len(groups)

    # Group the heatmap joints into each group
    hm_group = np.zeros((n_groups, h, w), dtype=hm.dtype)

    if agg_ope == 'gauss':
        for i in range(n_groups):
            hm_group[i] = hm[groups[i]].sum(axis=0) / np.sqrt(len(groups[i]))
    elif agg_ope == 'max':
        for i in range(n_groups):
            hm_group[i] = hm[groups[i]].max(axis=0)
    else:
        raise ValueError(f'{agg_ope} operation not supported')

    return hm_group


def expand_to_square_size(img: np.array) -> np.array:
    """
        Expands the image width or height to convert into square size

        Arguments
        ---------
        img: Array
            Image to be expanded

        Returns
        -------
        Array: Expanded image
    """

    height, width = img.shape[-2:]
    channels = img.shape[0] if len(img.shape) == 3 else 1

    # No expansion is required
    if height == width:
        return img

    # Expansion required on the width or height
    max_dim = max(height, width)

    exp_img = np.zeros((channels, max_dim, max_dim), dtype=img.dtype)

    if width < max_dim:
        pad = max_dim - width
        half_pad = int(pad // 2)
        exp_img[:, :, half_pad: width + half_pad] = img
    else:
        pad = max_dim - height
        half_pad = int(pad // 2)
        exp_img[:, half_pad: height + half_pad, :] = img

    if channels == 1:
        exp_img = exp_img[0]

    return exp_img


def process_dir(dataset_name, save_results, heatmaps_dir, dirlist, group_data=None, height_out=64, agg_method='sum',
                replace_if_exists=False, frame_num_re='frame-([0-9]+)', process_id=None):
    # Frame regex finder
    frame_num_re = re.compile(frame_num_re)

    # Process all the videos and show a progress bar
    if process_id is None:
        pbar = tqdm.tqdm(dirlist, 'Videos processed')
    else:
        pbar = tqdm.tqdm(dirlist, f'Subprocess {process_id}-Videos processed', position=process_id)

    # Process all the videos and show a progress bar
    for subdir, fname in pbar:

        # Retrieve video subdirs
        hm_dir = os.path.join(heatmaps_dir, subdir, fname)

        # Parse subject id (sid), seq. type, and view following dataset struct.
        if dataset_name == 'CASIAB':
            match = casiab_fn_fmt_re.search(hm_dir)

            if not match:
                print(f'Found no regular video {fname}. Ignored')
                continue

            sid, seq, view = (match.group(1), match.group(2) + '-' + match.group(3), match.group(4))
        elif dataset_name == 'TUM_GAID':
            match = tum_gaid_fn_fmt_re.search(hm_dir)

            if not match:
                print(f'Found no regular video {fname}. Ignored')
                continue

            sid, seq, view = (match.group(1), match.group(2), '90')
        else:
            print(f'Dataset {dataset_name} not supported yet')
            exit(-1)

        vid_fname = f'{sid}-{seq}-{view}'

        # Create the subdirs sid/seq/view
        subdir_path = os.path.join(save_results, sid, seq, view)
        os.makedirs(subdir_path, exist_ok=True)

        # Discard already processed heatmaps
        if not replace_if_exists and os.path.isfile(os.path.join(subdir_path, f'{view}.pkl')):
            continue

        # List and sort all the raw heatmap frames within heatmap dir according to frame number
        frame_nums = dict()
        for raw_hm_fn in os.listdir(hm_dir):

            if not raw_hm_fn.endswith(TERM_FN):
                warnings.warn(f'Found non-valid numpy file {raw_hm_fn}. Skipping')
                continue

            ## Retrieve frame number from raw heatmap filename
            found_frame = frame_num_re.search(raw_hm_fn)
            try:
                frame_num = int(found_frame.group(1))
            except:
                warnings.warn(f'Cannot retrieve frame number for heatmap file {raw_hm_fn}. Skipping!!')
                continue

            if frame_num in frame_nums:
                warnings.warn(f'Found duplicated frame {frame_num} in video {vid_fname}. Skipping !!')
                continue
            else:
                frame_nums[frame_num] = os.path.join(hm_dir, raw_hm_fn)

        frame_nums = sort_dict(frame_nums)

        if not frame_nums:
            warnings.warn('"{}" does not contains any frame and will be discarded'.format(vid_fname))
            continue

        # Process all heatmap frames following body part rep. and store as single array
        hm_proces = []

        for frame_num, raw_hm_fn in frame_nums.items():
            raw_hm_data = np.load(raw_hm_fn)

            # Fuse joint heatmaps following body part representation
            if group_data is not None:
                if agg_method == 'sum':  # gauss sum
                    raw_hm_data = fuse_heatmap_channels(raw_hm_data, list(group_data.values()), agg_ope='gauss')
                elif agg_method == 'max':
                    raw_hm_data = fuse_heatmap_channels(raw_hm_data, list(group_data.values()), agg_ope='max')

            # Resize hm to the output dim file
            if raw_hm_data.shape[-2:] != (height_out, height_out):
                raw_hm_data = expand_to_square_size(raw_hm_data)
                ## CV requires aray dimension as [height_out, height_out, groups]
                if len(raw_hm_data.shape) == 3:
                    raw_hm_data = np.moveaxis(raw_hm_data, (0, 1, 2), (2, 0, 1))

                raw_hm_data = cv.resize(raw_hm_data, (height_out, height_out))

                if len(raw_hm_data.shape) == 3:
                    raw_hm_data = np.moveaxis(raw_hm_data, (2, 0, 1), (0, 1, 2))

            hm_proces.append(raw_hm_data)

        hm_proces = np.array(hm_proces)

        # Save array into file
        with open(os.path.join(subdir_path, f'{view}.pkl'), 'wb') as out_file:
            pickle.dump(hm_proces, out_file)

def main(args):

    # Collect input arguments
    heatmaps_dir = args.input_path
    save_results = args.output_path
    height_out = args.height_output
    dataset_name = args.dataset
    group_cfg = args.group_cfg
    replace_if_exists = args.replace_if_exists
    frame_format = args.frame_format
    agg_method = args.agg_method
    num_workers = args.num_workers

    # Check if output results directory exists or create it
    if not os.path.isdir(save_results):
        try:
            os.mkdir(save_results)
        except Exception as e:
            print('Failed to create output directory {}'.format(str(e)))
            exit(-1)

    # Load grouping information
    if group_cfg:
        with open(group_cfg, 'r') as f:
            group_data = json.load(f)
        group_data = group_data['groups']

    # Retrieve list of video heatmaps dir to scan
    print(f'Scanning {heatmaps_dir}')
    dirlist = list_dir_recursively(heatmaps_dir, keep_only_dir=True)

    # Split between the workers available
    works_per_proc = len(dirlist) // num_workers + int(len(dirlist) % num_workers > 0)

    if num_workers == 1:
        process_dir(dataset_name, save_results, heatmaps_dir,
                    dirlist, group_data, height_out, agg_method,
                    replace_if_exists, frame_format, process_id=None)
    else:

        # Initialize process workers
        subprocesses = []

        try:
            for i in range(num_workers):
                # Submit the data split to each worker
                spl_dirlist = dirlist[i * works_per_proc: (i + 1) * works_per_proc]

                # Run worker
                proc = Process(target=process_dir, args=(dataset_name, save_results,
                                                         heatmaps_dir, spl_dirlist, group_data, height_out,
                                                         agg_method, replace_if_exists, frame_format, i))
                subprocesses.append(proc)
                proc.start()

            time.sleep(2)

            # Wait for all the process to finish
            for proc in subprocesses:
                proc.join()

        except Exception as e:
            print('An exception ocurred: ', str(e))
        except KeyboardInterrupt:
            print('Interrupt signal received. Closing all the processes')
        finally:
            # Wait for all the process until finishing
            for proc in subprocesses:
                if proc.is_alive():
                    proc.terminate()

            time.sleep(10)

            # Kill all the processes
            for proc in subprocesses:
                if proc.is_alive():
                    proc.kill()

            for proc in subprocesses:
                proc.close()

if __name__ == '__main__':

    ### Input Arguments
    parser = argparse.ArgumentParser(description='Preprocess a set of raw pose heatmaps to a given input representation')

    parser.add_argument('-i', '--input_path', help='Directory path containing the raw pose heatmaps', type=str)
    #parser.add_argument('--input_videos', help='If you desire to process individual heatmaps annotation, use this argument to instead "-i"/"--input_path"', type=str, nargs='+', default=[])
    parser.add_argument('-o', '--output_path', help='Directory path where the processed heatmaps image files will be placed', type=str)
    parser.add_argument('--height_output', help='Output frame height', type=int, required=False, default=64)
    parser.add_argument('-d', '--dataset', default='CASIAB', type=str, help='Dataset for pretreatment.')
    parser.add_argument('-g', '--group_cfg', help='Json configuration file with the joint grouping', type=str)
    parser.add_argument('--replace_if_exists', action='store_true', help='Replace file if exists', default=False)
    parser.add_argument('--frame_format', help='Frame number format followed in the heatmap filenames', type=str, default='frame-([0-9]+)')
    parser.add_argument('--agg_method', help='Heatmap aggregation function', type=str, default='sum', choices=['sum', 'max'])
    parser.add_argument('--num_workers', help='Number of workers to process the dataset', type=int, default=1)

    args = parser.parse_args()
    
    # Run program
    main(args)

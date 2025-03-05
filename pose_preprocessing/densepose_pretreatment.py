import sys
import re
import os
import json
import warnings
import argparse
import time
import cv2
import numpy as np
import pickle
import tqdm
from multiprocessing import Process

try:
    from densepose.structures import DensePoseDataRelative
    DENSEPOSE_N_PART_LABELS = DensePoseDataRelative.N_PART_LABELS
except ImportError:
    DENSEPOSE_N_PART_LABELS = 24

# General variables
casiab_fn_fmt_re = re.compile('([0-9]{3})-(nm|cl|bg)-([0-9]{2})-([0-9]{3})')
tum_gaid_fn_fmt_re = re.compile('(p[0-9]{3})-((n|b|s)[0-9]{2})')
term_fn = '.png'

# Useful functions
sort_dict = lambda d: dict(sorted(d.items()))


def resize_to_human_center(img, size=64, oth_img=None):

    """
    Resizes an image while centering a human figure detected within it. This
    function ensures the human figure remains centered after resizing. If
    additional images (oth_img) are provided, they are resized and aligned accordingly.

    Arguments
    ---------
    img: numpy.ndarray
        Input image, assumed to have a shape of (H, W) or (H, W, C).
        If the image is grayscale, a channel dimension is added.

    size: int
        The target height for the resized image. The width is adjusted proportionally.
        Default 64

    oth_img: list of numpy.ndarray or None

    Optional list of additional images to resize and align with the primary image.
    Default None. These images should have the same dimensions as the original img before resizing.

    Return
    ------
    If oth_img is None, returns a single resized image (numpy.ndarray).
    If oth_img is provided, returns a list where the first element is the resized img,
        followed by the resized versions of oth_img (list of numpy.ndarray).
    """

    # Add channel dim if not exists
    if img.ndim < 3:
        img = img[..., np.newaxis]

    h, w, c = img.shape

    # Get a binnary mask holding the human shape
    human_mask = np.any(img > 0, axis=2).astype(np.uint8)
    assert(np.all((human_mask == 0) | (human_mask == 1)))

    # Get the upper and lower points
    y_sum = human_mask.sum(axis=1)
    y_top = (y_sum != 0).argmax(axis=0)
    y_btm = (y_sum != 0).cumsum(axis=0).argmax(axis=0)
    human_mask = human_mask[y_top: y_btm + 1, :]
    img = img[y_top: y_btm + 1, :]
    
    if oth_img is not None:
        oth_img = list(map(lambda oth: oth[y_top: y_btm + 1, :], oth_img))

    # As the height of a person is larger than the width,
    # use the height to calculate resize ratio.
    ratio = human_mask.shape[1] / human_mask.shape[0]
    human_mask = cv2.resize(human_mask, (int(size * ratio), size), interpolation=cv2.INTER_CUBIC)
    img = cv2.resize(img, (int(size * ratio), size), interpolation=cv2.INTER_CUBIC)

    if oth_img is not None:
        oth_img = [cv2.resize(oth, (int(size * ratio), size), interpolation=cv2.INTER_CUBIC) for oth in oth_img]

    # Restore third channel if resize has eliminated
    if img.ndim == 2:
        img = img[:, :, np.newaxis]

    if oth_img is not None:
        for i in range(len(oth_img)):
            if oth_img[i].ndim == 2:
                oth_img[i] = oth_img[i][:, :, np.newaxis]

    # Get the median of the x-axis and take it as the person's x-center.
    x_csum = human_mask.sum(axis=0).cumsum()
    x_center = None
    human_mask_sum = human_mask.sum()
    for idx, csum in enumerate(x_csum):
        if csum > human_mask_sum / 2:
            x_center = idx
            break

    if not x_center:
        raise RuntimeError(f'Image has no center.')

    # Get the left and right points
    half_width = size // 2
    left = x_center - half_width
    right = x_center + half_width
    if left <= 0 or right >= human_mask.shape[1]:
        left += half_width
        right += half_width
        _ = np.zeros((human_mask.shape[0], half_width, c), dtype=img.dtype)
        img = np.concatenate([_, img, _], axis=1)

        if oth_img is not None:
            oth_img = [np.concatenate([_, oth, _], axis=1) for oth in oth_img]

    # Adjust imgs width
    img = img[:, left: right, :]
    if oth_img is not None:
        oth_img = [oth[:, left: right, :] for oth in oth_img]

    return img if oth_img is None else ([img] + oth_img)

def segment_image(segment_img: np.array, segment_vals: list, img_list: np.array or list):

    """
        Segment an image according to a segmentation image
    """

    # Ensure input is an image of lists
    if not isinstance(img_list, (list, tuple)):
        img_list = [img_list]

    dtype = img_list[0].dtype
    ret_img = [[] for _ in img_list] 

    for seg in segment_vals:
        # Retrieve the mask covering the segment value and isolate the image area
        mask = (segment_img == seg)
        for i, img in enumerate(img_list):
            proc_img = mask * img

            ret_img[i].append(proc_img)

    # Convert segment images into array
    ret_img = list(map(np.array, ret_img))

    return ret_img if len(ret_img) > 1 else ret_img[0]

def fuse_channels(dp: np.array, groups: list) -> np.array:

    """Fuses the dp body part channels into one or more 
        densepose channel groups.

        Arguments
        ---------
        dp: Array
            Body-part segmented densepose image to be fused.

        groups: List of array-like
            List of body part groups to be build

        Return
        ------
        Array containing the grouped densepose
    """

    n_c, h, w = dp.shape
    n_groups = len(groups)

    # Group the heatmap joints into each group
    dp_group = np.zeros((n_groups, h, w), dtype=dp.dtype)

    for i in range(n_groups):
        dp_group[i] = dp[groups[i]].sum(axis=0)

    return dp_group

def process_dir(dataset_name, save_results, basedirs_IUV, dirlists_IUV, group_data=None, height_out=64,
            replace_if_exists=False, frame_num_re='frame-([0-9]+)', process_id=None):

    # Frame regex finder
    frame_num_re = re.compile(frame_num_re)

    # Retrieve base dir I, U and V, and the video subdirs
    denseposes_dir_I, denseposes_dir_U, denseposes_dir_V = basedirs_IUV
    dirlist_I, dirlist_U, dirlist_V = dirlists_IUV

    # Process all the videos and show a progress bar
    if process_id is None:
        pbar = tqdm.tqdm(range(len(dirlist_I)), 'Videos processed')
    else:
        pbar = tqdm.tqdm(range(len(dirlist_I)), f'Subprocess {process_id}-Videos processed', position=process_id)

    for i in pbar:

        fname_I = dirlist_I[i]
        fname_U = dirlist_U[i]
        fname_V = dirlist_V[i]

        dp_dir_I = os.path.join(denseposes_dir_I, fname_I)
        dp_dir_U = os.path.join(denseposes_dir_U, fname_U)
        dp_dir_V = os.path.join(denseposes_dir_V, fname_V)

        frame_nums = dict()

        if not (os.path.isdir(dp_dir_I) and os.path.isdir(dp_dir_U) and os.path.isdir(dp_dir_V)):
            continue

        # Create output directory for the processed denseposes
        if dataset_name == 'CASIAB':
            # I channel
            match = casiab_fn_fmt_re.search(dp_dir_I)
            sid_I, seq_I, view_I = (match.group(1), match.group(2) + '-' + match.group(3), match.group(4))
            # U channel
            match = casiab_fn_fmt_re.search(dp_dir_U)
            sid_U, seq_U, view_U = (match.group(1), match.group(2) + '-' + match.group(3), match.group(4))
            # V channel
            match = casiab_fn_fmt_re.search(dp_dir_V)
            sid_V, seq_V, view_V = (match.group(1), match.group(2) + '-' + match.group(3), match.group(4))

            subdir_path = os.path.join(save_results, sid_I, seq_I, view_I)
            os.makedirs(subdir_path, exist_ok=True)

            assert sid_I == sid_U == sid_V and seq_I == seq_U == seq_V and view_I == view_U == view_V

        elif dataset_name == 'TUM_GAID':
            match = tum_gaid_fn_fmt_re.search(dp_dir_I)

            if not match:
                print(f'WARNING: Found no regular video {fname_I}. Ignored')
                continue
            # I channel
            sid_I, seq_I, view_I = (match.group(1), match.group(2), '90')
            # V channel
            match = tum_gaid_fn_fmt_re.search(dp_dir_U)
            sid_U, seq_U, view_U = (match.group(1), match.group(2), '90')
            # U channel
            match = tum_gaid_fn_fmt_re.search(dp_dir_V)
            sid_V, seq_V, view_V = (match.group(1), match.group(2), '90')

            subdir_path = os.path.join(save_results, sid_I, seq_I, view_I)
            assert sid_I == sid_U == sid_V and seq_I == seq_U == seq_V
            os.makedirs(subdir_path, exist_ok=True)
        else:
            print('Dataset {} not supported yet'.format(dataset_name))
            exit(-1)

        # Check if densepose has been already processed
        if not replace_if_exists and os.path.isfile(os.path.join(subdir_path, f'{view_I}.pkl')):
            continue

        # Scans all the densepose frame files within video dir
        dp_files_I = list(sorted(os.listdir(dp_dir_I)))
        dp_files_U = list(sorted(os.listdir(dp_dir_U)))
        dp_files_V = list(sorted(os.listdir(dp_dir_V)))

        for j in range(len(dp_files_I)):

            if not (dp_files_I[j].endswith(term_fn) and dp_files_U[j].endswith(term_fn) and dp_files_V[j].endswith(term_fn)):
                continue

            # get frame number from filename and from previous filename
            match_frame_I = frame_num_re.search(dp_files_I[j])
            if not match_frame_I:
                warnings.warn(f'No frame identified on {dp_files_I[j]}. Skipping')
                continue
            frame_num_I = int(match_frame_I.group(1))

            match_frame_U = frame_num_re.search(dp_files_U[j])
            frame_num_U = int(match_frame_U.group(1))

            match_frame_V = frame_num_re.search(dp_files_V[j])
            frame_num_V = int(match_frame_V.group(1))

            if frame_num_I in frame_nums:
                warnings.warn(f'Additional human detected on frame {frame_num_I} for video {fname_I}. Skipping!!')
                continue

            # Note filenames
            frame_nums[frame_num_I] = [None, None, None]
            frame_nums[frame_num_I][0] = os.path.join(dp_dir_I, dp_files_I[j])
            frame_nums[frame_num_U][1] = os.path.join(dp_dir_U, dp_files_U[j])
            frame_nums[frame_num_V][2] = os.path.join(dp_dir_V, dp_files_V[j])

        # Order the frame files
        frame_nums = sort_dict(frame_nums)

        if not frame_nums:
            warnings.warn('"{}" does not contains any frame and will be discarded'.format(fname_I))
            continue

        # Check there is no missing frame
        frame_nums_keys = list(frame_nums.keys())
        for j in range(1, len(frame_nums_keys)):
            if (frame_nums_keys[j] - frame_nums_keys[j - 1]) > 1:
                warnings.warn(f'Frames missing after {frame_nums_keys[j - 1]} for video {fname_I}')

        # Process all densepose frames following body part rep. and store as single array
        dp_proces = []

        for j in frame_nums:

            # Load I, U and V frames
            frame_data_I = cv2.imread(frame_nums[j][0], cv2.COLOR_BGR2GRAY)
            frame_data_U = cv2.imread(frame_nums[j][1], cv2.COLOR_BGR2GRAY)
            frame_data_V = cv2.imread(frame_nums[j][2], cv2.COLOR_BGR2GRAY)

            segments = frame_data_I

            # Reescale channel I to [0, 255]
            frame_data_I = frame_data_I.astype(np.float32) * 255. / DENSEPOSE_N_PART_LABELS
            frame_data_I = frame_data_I.round()
            frame_data_I = frame_data_I.astype(np.uint8)

            # Segment the images
            segment_I, segment_U, segment_V = segment_image(segments, list(range(DENSEPOSE_N_PART_LABELS + 1)),
                                                            [frame_data_I, frame_data_U, frame_data_V])

            # Fuse densepose body parts following body part representation
            if group_data is not None:
                frame_data_I = fuse_channels(segment_I, list(group_data.values()))
                frame_data_U = fuse_channels(segment_U, list(group_data.values()))
                frame_data_V = fuse_channels(segment_V, list(group_data.values()))

                frame_data_I = np.moveaxis(frame_data_I, source=(0, 1, 2), destination=(2, 0, 1))
                frame_data_U = np.moveaxis(frame_data_U, source=(0, 1, 2), destination=(2, 0, 1))
                frame_data_V = np.moveaxis(frame_data_V, source=(0, 1, 2), destination=(2, 0, 1))

                # Sort I, U, V for every grouping
                frames_data = np.stack((frame_data_I, frame_data_U, frame_data_V), axis=-1)
                frames_data = frames_data.reshape(*frames_data.shape[:2], -1) #.swapaxes(-1, -2)

                try:
                    frames_data = resize_to_human_center(frames_data, size=height_out)
                except Exception as e:
                    warnings.warn(f'For video {fname_I} no center could be computed on frame {j}: {str(e)}. Skipping frame !!')
                    continue

                frames_data = np.moveaxis(frames_data, source=(0, 1, 2), destination=(1, 2, 0)) # [c, h, w]
            else:
                # Default grouping
                segment_I, segment_U, segment_V = segment_I[1:], segment_U[1:], segment_V[1:] # Exclude background

                # Preprocess the input frame to make squared-size and person-centered
                try:
                    oth_imgs = list(segment_I) + [frame_data_I] + list(segment_U) + [frame_data_U] + list(segment_V) + [frame_data_V]
                    frames_data = resize_to_human_center(frame_data_I[:, :, np.newaxis],
                                                                    size=height_out,
                                                                    oth_img=[img[:, :, np.newaxis] for img in oth_imgs])


                except Exception as e:
                    warnings.warn(f'For video {fname_I} no center could be computed on frame {j}: {str(e)}')
                    continue

                frames_data = np.array(frames_data[1:])
                frames_data = frames_data.reshape(frames_data.shape[:-1])

            dp_proces.append(frames_data)

        dp_proces = np.array(dp_proces)

        # Save pretreated densepose
        with open(os.path.join(subdir_path, f'{view_I}.pkl'), 'wb') as out_file:
            pickle.dump(dp_proces, out_file)

def main(args):

    # Collect input arguments
    denseposes_dir = args.input_path
    save_results = args.output_path
    height_out = args.height_output
    dataset_name = args.dataset
    group_cfg = args.group_cfg
    replace_if_exists = args.replace_if_exists
    dir_set = args.dir_set
    frame_format = args.frame_format
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
    else:
        group_data = None


    # Retrieve list of heatmaps dir to scan
    denseposes_dir_I = os.path.join(denseposes_dir, dir_set[0])
    denseposes_dir_U = os.path.join(denseposes_dir, dir_set[1])
    denseposes_dir_V = os.path.join(denseposes_dir, dir_set[2])
    dirlist_I = list(sorted(os.listdir(denseposes_dir_I)))
    dirlist_U = list(sorted(os.listdir(denseposes_dir_U)))
    dirlist_V = list(sorted(os.listdir(denseposes_dir_V)))

    # Split between the workers available
    works_per_proc = len(dirlist_I) // num_workers + int(len(dirlist_I) % num_workers > 0)

    if num_workers == 1:
        process_dir(dataset_name, save_results, (denseposes_dir_I, denseposes_dir_U, denseposes_dir_V),
                    (dirlist_I, dirlist_U, dirlist_V), group_data, height_out,
                    replace_if_exists, frame_format, process_id=None)
    else:

        # Initialize process workers
        subprocesses = []

        try:
            for i in range(num_workers):
                # Submit the data split to each worker
                spl_dirlist_I = dirlist_I[i * works_per_proc: (i+1) * works_per_proc]
                spl_dirlist_U = dirlist_U[i * works_per_proc: (i+1) * works_per_proc]
                spl_dirlist_V = dirlist_V[i * works_per_proc: (i+1) * works_per_proc]

                # Run worker
                proc = Process(target=process_dir, args=(dataset_name, save_results,
                                                         (denseposes_dir_I, denseposes_dir_U, denseposes_dir_V),
                                                         (spl_dirlist_I, spl_dirlist_U, spl_dirlist_V), group_data,
                                                         height_out, replace_if_exists, frame_format, i))
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
    parser = argparse.ArgumentParser(description='Preprocess raw densepose I,U and V frames following the body part grouping into the OpenGait input format.')

    parser.add_argument('-i', '--input_path', help='Root path dir containing I, U and V poses', type=str)
    parser.add_argument('-o', '--output_path', help='Output directory path for the pretreated denseposes', type=str)
    parser.add_argument('--height_output', help='Output frame height', type=int, required=False, default=64)
    parser.add_argument('-d', '--dataset', default='CASIAB', type=str, help='Dataset for pretreatment.')
    parser.add_argument('-g', '--group_cfg', help='Json configuration file with the body part grouping', type=str, default='')
    parser.add_argument('--dir_set', type=str, help='Subdirectory names containing the densepose subimages I,U and V', nargs=3, default=['I', 'U', 'V'])
    parser.add_argument('--replace_if_exists', action='store_true', help='Replace file if exists', default=False)
    parser.add_argument('--frame_format', help='Frame number format followed in the heatmap filenames', type=str, default='frame([0-9]+)')
    parser.add_argument('--num_workers', help='Number of workers to process the dataset', type=int, default=1)

    args = parser.parse_args()

    # Run program
    main(args)

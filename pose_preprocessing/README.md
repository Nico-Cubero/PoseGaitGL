# Pose body part preprocessing

The present directory contains all the scripts required to preprocess both **heatmaps** and **DensePose** raw poses into our studied **pose representations**.

Preprocessed poses are stored in pickle format, which is supported by **OpenGait**.

## Heatmaps

Heatmaps have been computed using [ViTPose](https://github.com/ViTAE-Transformer/ViTPose). Please refer to the repo to reproduce computation.

Raw Heatmaps should be structured as follows:

```
    HEATMAP ROOT PATH/
        video #1 / (a subdir for each video containing all frames)
            frame-0.npy (numpy array with size [17, H, W])
            frame-1.npy
            frame-2.npy
            ......

        video #2 /
            frame-0.npy
            frame-1.npy
            frame-2.npy
            ......

        ......
```

Note that each `.npy` file contains a numpy array with 17 channels (one for each skeleton joint).

Run following script to preprocess the raw heatmaps into the **hierarchical limb-based** representation:

```
python3 heatmap_pretreatment.py \
    -i <heatmap root path> \
    -o <out dir for preprocessed hms> \
    -d CASIAB \ # or TUM_GAID
    -g hm_representations_cfgs/hierarchical_limb-based.json # Or replace by any other available
```

Also, check the [hm_representations_cfgs](hm_representations_cfgs/) directory for additional pose representations studied in our work (such as Full body, Half body, Side body, etc.).

## DensePose

DensePose computations are done using [DensePose](https://github.com/facebookresearch/Densepose).

Raw DensePoses structure should follow:

```
    DENSEPOSE ROOT PATH/
        I / (main subdir for I images)
            video #1 / (a subdir for each video with I frames)
                frame-0.png (grayscale image)
                frame-1.png
                frame-2.png
                ......
            video #2 /
                ......
            ......
            
        U / (main subdir for U images)
            video #1 / (a subdir for each video with U frames)
                frame-0.png (grayscale image)
                frame-1.png
                frame-2.png
                ......
            video #2 /
                ......
            ......

        V / (main subdir for V images)
            video #1 / (a subdir for each video with V frames)
                frame-0.png (grayscale image)
                frame-1.png
                frame-2.png
                ......
            video #2 /
                ......
            ......
```

Run following script to preprocess the raw DensePoses into the **hierarchical limb-based** representation:

```
python3 densepose_pretreatment.py \
    -i <DensePose root path> \
    -o <out dir for preprocessed dps> \
    -d CASIAB \ # or TUM_GAID
    -g dp_representations_cfgs/hierarchical_limb-based.json \ # Or replace by any other available
    --dir_set I U V \ # Main subdirs splitting I, U and V images
    --num_workers <Num. of parallel subprocesses>
```

Additional pose representations are available at [dp_representations_cfgs](dp_representations_cfgs/)/ dir as well.

## Output format for OpenGait

The output of the preprocessed poses follows the default dataset structure supported by OpenGait:

```
    DATASET_ROOT/
        001 (subject)/
            bg-01 (type)/
                    000 (view)/
                        000.pkl (contains all frames)
                ......
            ......
        ......
```
# OpenGait: Heatmaps & DensePose

Following source code is derived from official [OpenGait](https://github.com/ShiqiYu/OpenGait) repo.
Please note that some sources have been edited, removed, or simplified focus in our implementation.

## Prepare pose dataset
Preprocess poses by following [pose preprocessing](../pose_preprocessing).

## Prepare *cfg* files

Prepare your cfg file from the template found in [config](configs/). Make sure to set your preprocessed pose directory in `data_cfg.dataset_root`.

## Train
Train a model by:
```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
                                    --nproc_per_node=2 \
                                    opengait/main.py \
                                        --cfgs ./configs/posegaitgl/posegaitgl-hm_CASIAB.yaml \
                                        --phase train
```
- `python -m torch.distributed.launch` [DDP](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) launch instruction.
- `--nproc_per_node` The number of gpus to use, and it must equal the length of `CUDA_VISIBLE_DEVICES`.
- `--cfgs` The path to config file.
- `--phase` Specified as `train`.
- `--iter` You can specify a number of iterations or use `restore_hint` in the config file and resume training from there.
- `--log_to_file` If specified, the terminal log will be written on disk simultaneously.
- `--out_dir` Path to output directory. By default, output checkpoints and logs will be placed at *output/* directory.

## Test
To evaluate the trained model, use the following command:
```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
                                    --nproc_per_node=2 \
                                    opengait/main.py \
                                        --cfgs ./configs/posegaitgl/posegaitgl-hm_CASIAB.yaml \
                                        --phase test
```
- `--phase` Specified as `test`.
- `--iter` Specify a iteration checkpoint.
- `--out_dir` Remember to set identical *output/* dir used in train.

**Tip**: Other arguments are the same as train phase.

### Checkpoints

You can evaluate also our trained models by downloading the model checkpoints found in [checkpoints](checkpoints/) dir.
Download the model weights and set their path in the `restore_hint` parameter.

----
**Tip**: Some train & test example scripts are included in [scripts](scripts/) dir.

## Warning
- In `DDP` mode, zombie processes may be generated when the program terminates abnormally. You can use this command [sh misc/clean_process.sh](./misc/clean_process.sh) to clear them.
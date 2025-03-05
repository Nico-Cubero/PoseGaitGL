#### Train PoseGaitGL-HM on CASIA-B
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch \
                                                  --nproc_per_node=8
                                                  opengait/main.py \
                                                    --cfg configs/posegaitgl/posegaitgl-hm_CASIAB.yaml \
                                                    --phase train --log_to_file
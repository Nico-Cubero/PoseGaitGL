import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base_model import BaseModel
from ..modules import SeparateFCs, BasicConv3d, PackSequenceWrapper, SeparateBNNecks
from .gaitgl import GLConv, GeMHPP

class PoseGaitGL(BaseModel):
    """
        PoseGaitGL model from publication
        Empirical Study of Human Pose representations for Gait Recognition
    """

    def check_rep_layers_cfg(cfg: dict = None):

        # Default configuration
        default_cfg = {
            'input conv3d': 1,
            'GLConvA0': 1,
            'GLConvA1': 1,
            'GLConvB2': 1,
            'Head0': 1
        }

        if cfg is None:
            cfg = {}

        # Filter cfg to only contain valid keys and update default_cfg
        valid_cfg = {k: v for k, v in cfg.items() if k in default_cfg}
        updated_cfg = default_cfg.copy()
        updated_cfg.update(valid_cfg)

        # Validate values
        invalid_layers = {k: v for k, v in updated_cfg.items() if v < 1}
        if invalid_layers:
            raise ValueError(f"Invalid layer values: {invalid_layers}. "
                             f"Number of repeated layers should be 1 or greater.")

        return updated_cfg

    def __init__(self, *args, **kargs):
        super(PoseGaitGL, self).__init__(*args, **kargs)

    @property
    def in_channels(self):
        return self._in_channels

    def build_network(self, model_cfg):
        in_channel = model_cfg['num_in_channels']
        in_c = model_cfg['channels']
        class_num = model_cfg['class_num']
        dataset_name = self.cfgs['data_cfg']['dataset_name']
        rep_layers = PoseGaitGL.check_rep_layers_cfg(model_cfg.get('repeat_layers', {}))

        # Channels to display on tensorboard
        self.displayed_channels = self.cfgs['data_cfg'].get('displayed_channels', [-1])
        assert isinstance(self.displayed_channels, (list, tuple)) and len(self.displayed_channels) <= 3,\
            '"displayed_channels" key should be a list with 3 max values'

        halving = model_cfg.pop('halving', [3, 3, 3])

        ## Input conv 3d layer
        if rep_layers['input conv3d'] > 1:
            self.conv3d = nn.Sequential(nn.Sequential(
                BasicConv3d(in_channel, in_c[0], kernel_size=(3, 3, 3),
                            stride=(1, 1, 1), padding=(1, 1, 1)),
                nn.LeakyReLU(inplace=True)
            ),
              *[nn.Sequential(
                BasicConv3d(in_c[0], in_c[0], kernel_size=(3, 3, 3),
                        stride=(1, 1, 1), padding=(1, 1, 1)),
                nn.LeakyReLU(inplace=True)
            ) for _ in range(rep_layers['input conv3d'] - 1)])
        else:
            self.conv3d = nn.Sequential(
             BasicConv3d(in_channel, in_c[0], kernel_size=(3, 3, 3),
                        stride=(1, 1, 1), padding=(1, 1, 1)),
             nn.LeakyReLU(inplace=True)
            )
        
        ## LTA layer
        self.LTA = nn.Sequential(
            BasicConv3d(in_c[0], in_c[0], kernel_size=(3, 1, 1),
                        stride=(3, 1, 1), padding=(0, 0, 0)),
            nn.LeakyReLU(inplace=True)
        )

        ## GLConvA0 layer
        self.GLConvA0 = nn.Sequential(
                                GLConv(in_c[0], in_c[1], halving=halving[0],
                                        fm_sign=False, kernel_size=(3, 3, 3),
                                        stride=(1, 1, 1), padding=(1, 1, 1)),
                                *[GLConv(in_c[1], in_c[1], halving=halving[0],
                                          fm_sign=False, kernel_size=(3, 3, 3),
                                          stride=(1, 1, 1), padding=(1, 1, 1)
                                        ) for _ in range(rep_layers['GLConvA0'] - 1)]
                                     )

        ## Max Pooling 0 layer
        self.MaxPool0 = self.MaxPool0 = nn.MaxPool3d(kernel_size=(1, 2, 2),
                                                     stride=(1, 2, 2))

        ## GLConvA1 layer
        self.GLConvA1 = nn.Sequential(
                                GLConv(in_c[1], in_c[2], halving=halving[1],
                                         fm_sign=False, kernel_size=(3, 3, 3),
                                         stride=(1, 1, 1), padding=(1, 1, 1)),
                                *[GLConv(in_c[2], in_c[2], halving=halving[1],
                                         fm_sign=False, kernel_size=(3, 3, 3),
                                         stride=(1, 1, 1), padding=(1, 1, 1)
                                         ) for _ in range(rep_layers['GLConvA1'] - 1)]
                                     )
        
        ## GLConvB2 layer
        self.GLConvB2 = nn.Sequential(
                                *[GLConv(in_c[2], in_c[2], halving=halving[2],
                                         fm_sign=True,  kernel_size=(3, 3, 3),
                                         stride=(1, 1, 1), padding=(1, 1, 1)
                                         ) for _ in range(rep_layers['GLConvB2'])]
                                 )

        ## TP & HPP layers
        self.TP = PackSequenceWrapper(torch.max)
        self.HPP = GeMHPP()
        ## Head 0 FC layer
        self.Head0 = nn.Sequential(
                                *[SeparateFCs(64, in_c[-1], in_c[-1]
                                             ) for _ in range(rep_layers['Head0'])]
                                )
        ## Head 1 FC layer
        if 'SeparateBNNecks' in model_cfg.keys():
            self.BNNecks = SeparateBNNecks(**model_cfg['SeparateBNNecks'])
            self.Bn_head = False
        else:
            self.Bn = nn.BatchNorm1d(in_c[-1])
            self.Head1 = SeparateFCs(64, in_c[-1], class_num)
            self.Bn_head = True

    def forward(self, inputs):
        ipts, labs, _, _, seqL = inputs
        seqL = None if not self.training else seqL
        if not self.training and len(labs) != 1:
            raise ValueError(
                'The input size of each GPU must be 1 in testing mode, but got {}!'.format(len(labs)))
        poses = ipts[0]
        del ipts

        if len(poses.size()) == 4:
            poses = poses.unsqueeze(2)
        poses = torch.transpose(poses, 1, 2) # [n, j, s, h, w] from [n, s, j, h, w]

        n, j, s, h, w = poses.size()

        if s < 3:
            repeat = 3 if s == 1 else 2
            poses = poses.repeat(1, 1, repeat, 1, 1)

        outs = self.conv3d(poses)
        outs = self.LTA(outs)

        outs = self.GLConvA0(outs)
        outs = self.MaxPool0(outs)

        outs = self.GLConvA1(outs)
        outs = self.GLConvB2(outs)  # [n, c, s, h, w]

        outs = self.TP(outs, seqL=seqL, options={"dim": 2})[0]  # [n, c, h, w]
        outs = self.HPP(outs)  # [n, c, p]

        gait = self.Head0(outs)  # [n, c, p]
        
        if self.Bn_head: # Original GaitGL Head
            bnft = self.Bn(gait)  # [n, c, p]
            logi = self.Head1(bnft)  # [n, c, p]
            embed = bnft
        else: # BNNechk as Head
            bnft, logi = self.BNNecks(gait)  # [n, c, p]   
            embed = gait  

        retval = {
            'training_feat': {
                'triplet': {'embeddings': embed, 'labels': labs},
                'softmax': {'logits': logi, 'labels': labs}
            },
            'visual_summary': {
                'image/sils': poses[:, self.displayed_channels].transpose(1, 2).reshape(n * s, len(self.displayed_channels), h, w)
            },
            'inference_feat': {
                'embeddings': embed
            }
        }
        return retval

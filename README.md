# Empirical Study of Human Pose representations for Gait Recognition

[![https://doi.org/10.1016/j.eswa.2025.126946](https://img.shields.io/badge/doi-10.1016/j.eswa.2025.126946-blue)](https://doi.org/10.1016/j.eswa.2025.126946)
[![https://www.researchgate.net/publication/389444094_Empirical_study_of_human_pose_representations_for_gait_recognition](https://img.shields.io/badge/%20-Read_on_ResearchGate-blue.svg?logo=researchgate&color=00CCBB&labelColor=555555)](https://www.researchgate.net/publication/389444094_Empirical_study_of_human_pose_representations_for_gait_recognition)

Source code for the **PoseGaitGL** approach presented in the publication for Expert Systems with Applications.
Some code parts are derived from [OpenGait](https://github.com/ShiqiYu/OpenGait)

## Abstract

Gait recognition has gained attention for its ability to identify individuals from afar. Current state-of-the-art approaches predominantly utilize visual information, such as silhouettes, or a combination of visual data and basic body pose information, including skeleton joint coordinates. However, the role of human pose in gait recognition is still underexplored, often leading to poorer results compared to visual approaches. In this work, we propose a novel hierarchical limb-based representation that enhances the depiction of body pose and can be applied to various pose descriptors. Our representation consists of three hierarchical levels: full body, body limbs (arms and legs), and middle limbs (forearms, lower arms, thighs, and shins). This structure enriches the gait description of the overall pose by incorporating the specific movements of each limb. Particularly, we investigate the application of our hierarchical arrangement using two different rich pose descriptors: heatmaps derived from 2D body skeletons and a dense representation obtained from pixel-wise estimation of body pose (i.e. DensePose). Furthermore, we introduce the PoseGaitGL family of models to better leverage the features derived from our pose representations. By employing our hierarchical pose representations, the proposed model achieves state-of-the-art results in pose-based gait recognition. Thus, the hierarchical heatmap-based and hierarchical DensePose representations attain Rank-1 accuracy of 82.2% and 92.0%, respectively, on the cross-view setup of CASIA-B, and 99.3% and 99.8%, respectively, on TUM-GAID, establishing a new benchmark for pose-based methods.

## Code usage

### Requirements

1. Install **pytorch>=2.0.0,<2.4.0** and **torchvision** versions with CUDA support from official [PyTorch](https://pytorch.org/get-started/locally/).

2. Install additional required dependencies with:
```bash
pip install -r requirements.txt
```

### 1. Pose computation

Compute raw **pose heatmaps** & **DensePoses** from code released at official [ViTPose](https://github.com/ViTAE-Transformer/ViTPose) and [DensePose](https://github.com/facebookresearch/Densepose) repos.  

### 2. Pose representation preprocessing

Follow instructions at [pose preprocessing](pose_preprocessing/) to compute our pose representations from the raw heatmaps and DensePoses.

### 3. Train & test gait recognition models

Use [OpenGait](OpenGait/) dir to train and test our gait recognition models from the computed pose representations.

## Citation

If our work is useful for you, please consider to cite us using this *bibtex* citation:

```
@article{CUBERO2025126946,
    title = {Empirical study of human pose representations for gait recognition},
    journal = {Expert Systems with Applications},
    volume = {275},
    pages = {126946},
    year = {2025},
    issn = {0957-4174},
    doi = {https://doi.org/10.1016/j.eswa.2025.126946},
    author = {Nicolás Cubero and Francisco M. Castro and Julián R. Cózar and Nicolás Guil and Manuel J. Marín-Jiménez},
}
```
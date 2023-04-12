# FIPT-data: data preparation for FIPT

### [Project Page](https://jerrypiglet.github.io/fipt-ucsd/) | [Paper]() | [Data download for FIPT](https://drive.google.com/drive/folders/1N8H1yR41MykUuSTyHvKGsZcuV2VjtWGr?usp=share_link)

## Overview
This repo (branch: `fipt`) contains the code for generating customized data for [FIPT](), from scratch. 

The repo is also useful for loading/generating data for other indoor inverse rendering pipelines, by adding `load_{DATASET}Scene3D.py` and `lib/class_{DATASET}Scene3D.py` for loading, and  customized formats to `lib/class_exporter.py` for export.

Currently supported datasets include:

1. indoor_synthetic
   - See [**README_indoor_synthetic.md**](README_indoor_synthetic.md) for details.
   - Based on [Mitsuba XML files by Benedikt Bitterli](https://benedikt-bitterli.me/resources/)
   - Scripts support: sampling poses, rendering numerous modalities, visualization and export to FIPT/Monosdf/FVP/Li22.
2. real
   - See [**README_real.md**](README_real.md) for details.
   - Captured for FIPT
   - Scripts support: visualization and export to FIPT/Monosdf/FVP/Li22.

See [## Related Works](#related-works) for a brief overview of aforementioned methods.

## Installation

Please refer to **README_env.md** for instructions for installing the environment.
## Related Works
- [**FIPT**](https://jerrypiglet.github.io/fipt-ucsd/)
  - Wu and Zhu et al. 2023, *FIPT: Factorized Inverse Path Tracing*
  - Optimization-based multi-view inverse rendering method.

- [**Monosdf**](https://niujinshuchong.github.io/monosdf/)
  - Yu et al. NeurIPS 2022, *MonoSDF: Exploring Monocular Geometric Cues for Neural Implicit Surface Reconstruction*
  - NeRF-like methods for multi-view scene reconstruction using SDF (signed-distance function) representation.
  - Used in FIPT for acquiring scene geometry (meshes).

- [**IPT**](https://arxiv.org/abs/1903.07145)
  - Azinović et al. CVPR 2019, *Inverse Path Tracing for Joint Material and Lighting Estimation*
  - Optimization-based multi-view inverse rendering method.
  - Used as baseline in FIPT.

- [**MILO**](https://ci.idm.pku.edu.cn/Yu_TPAMI23.pdf)
  - Yu et al. TPAMI 2023, *MILO: Multi-bounce Inverse Rendering for Indoor Scene with Light-emitting Objects*
  - Optimization-based multi-view inverse rendering method.
  - Used as baseline in FIPT.

- [**FVP**](https://repo-sam.inria.fr/fungraph/deep-indoor-relight/)
  - Philip et al. TOG 2021, *Free-viewpoint Indoor Neural Relighting from Multi-view Stereo*
  - Takes multiple images and aggregate multiview irradiance and albedo information to a pre-trained network to synthesize a relit image.
  - Used as baseline in FIPT.
  - [Our code]() | [Original code](https://gitlab.inria.fr/sibr/projects/indoor_relighting)

- [**Li22**](https://vilab-ucsd.github.io/ucsd-IndoorLightEditing/)
  - Li et al. ECCV 2022, *Physically-Based Editing of Indoor Scene Lighting from a Single Image*
  - Learning-based single image inverse rendering and relighting.
  - Used as baseline in FIPT.
  - [Our code]() | [Original code](https://github.com/ViLab-UCSD/IndoorLightEditing)

- [**NeILF**](https://machinelearning.apple.com/research/neural-incident-light-field)
  - Yao et. al. ECCV 2022, *NeILF: Neural Incident Light Field for Material and Lighting Estimation*
  - NeRF-like methods for multi-view inverse rendering by estimating neural representations of surface lighting and BRDF.
  - Used as baseline in FIPT.
  - [Our code]() | [Original code](https://github.com/apple/ml-neilf)

See Related Works section by the end of [FIPT website](https://jerrypiglet.github.io/fipt-ucsd/) for overview of most recent works.

## TODO

- [ ] Add code links for re-implemented baseline methods: FVP, NeILF, Li22

## Citation

If you find our work is useful, please consider cite:

```
```

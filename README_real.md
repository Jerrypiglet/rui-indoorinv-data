<!--Generate the TOC via: -->
<!-- (bash) ../gh-md-toc --insert README_real.md-->
<!--See https://github.com/ekalinin/github-markdown-toc#readme-->

<!--ts-->
- [Real scenes: capture, process and load](#real-scenes-capture-process-and-load)
  - [Load preprocessed scenes](#load-preprocessed-scenes)
  - [Dump to Monosdf format](#dump-to-monosdf-format)
    - [Additional notes on getting axis-aligned geometry](#additional-notes-on-getting-axis-aligned-geometry)
  - [Dump to FIPT format](#dump-to-fipt-format)
  - [Capturing guide](#capturing-guide)
    - [RAW data](#raw-data)
    - [environment](#environment)
    - [run the pipeline](#run-the-pipeline)

<!-- Created by https://github.com/ekalinin/github-markdown-toc -->
<!-- Added by: jerrypiglet, at: Mon Apr 10 01:29:36 PDT 2023 -->

<!--te-->

# Real scenes: capture, process and load

## Load preprocessed scenes

We include two scenes in the repo: `ConferenceRoom` and `ClassRoom`. They are preprocessed and ready to use, organized as below:

- data/
  - real/
    - RESULTS_monosdf/
      - ConferenceRoomV2_final_supergloo/
        - merged_images # HDR images in .exr
        - png_images # SDR images in .png
        - outLight0.xml # additional emitter for re-lighting
        - transforms....json # pose file
      - 20230306-072825-K-ConferenceRoomV2_final_supergloo_SDR_grids_trainval.ply # mesh file

To visualize the shape and camrea poses ([ConferenceRoom](https://i.imgur.com/Nf0J7ia.png), [ClassRoom](https://i.imgur.com/TaiSxoP.png)), run:

``` bash
python load_realScene3D.py --scene ConferenceRoomV2_final_supergloo
python load_realScene3D.py --scene ClassRoom
```

To visualize 2D modalities, run with `--vis_2d_plt` ([ConferenceRoom](https://i.imgur.com/gi4gTdd.png)).

## Dump to Monosdf format

Export to the format of[adapted MonoSDF for FIPT](https://github.com/Jerrypiglet/monosdf) project, for training geometry on indoor_synthetic dataset and real-world scenes in FIPT.

``` bash
python load_realScene3D.py --scene ConferenceRoomV2_final_supergloo --export --export_format monosdf
```

Then in [MonoSDF for FIPT](https://github.com/Jerrypiglet/monosdf) code, run:

``` bash
python training/exp_runner.py --conf confs/real.conf --conf_add confs/real_ConferenceRoomV2_final_supergloo_SDR.conf --exps_folder {path to monosdf/exps/} --prefix DATE-’
```

### Additional notes on getting axis-aligned geometry

You may have notices that parameter `reorient_blender_angles` is provided for the ConferenceRoom scene. The parameters are intended to re-orient the original scene poses (acquired from Colmap or Superglue) so that the scene geometry is axis-aligned (for improved results with MonoSDF by reducing aliasing in its feature grid). However it is not possible to acquire those angles beforehand if started with only poses but no corresponding geometry. 

Our solution is two-step training with MonoSDF:

(1) Set `if_reorient_y_up=False`, export the scene to MonoSDF, train a rough geometry (e.g. for 1 epoch).
(2) Load the trained mesh into Blender ([Before](https://i.imgur.com/IWEbwdP.jpg)). Use the rotation tool to make the mesh axis-aligned, and up direction is y+. Copy the rotations angles ([mid-right panel here](https://i.imgur.com/5Ij7vr3.jpg)) to the .conf file -> `reorient_blender_angles`, and the rough geometry to `shape_file`.
(3) Set `if_reorient_y_up=True`, re-export the scene to MonoSDF to get re-oriented poses (and rough shape), train the final geometry, and it will be axis-aligned.

To validate the re-oriented poses and rough shape, check the images under *data/real/EXPORT_monosdf/ConferenceRoomV2_final_supergloo/MiNormalGlobal_OVERLAY*, which overlays the normal map (acquired from the re-oriented rough shape and poses, to original RGB image) should like ![this](https://i.imgur.com/lmA7fU4.png).

## Dump to FIPT format

## Capturing guide

### RAW data
Please refer to the FITP paper ([TODO]) for details on capturing. The RAW capture` consist of S*N RAW images, where N is the number of poses, and S is the number of exposures within a bracketing (e.g. 9 for ConferenceRoom, and 5 for ClassRoom). Organize the data as below:

- data/
  - real/
    - Sony
      - calib_raw # calibration images, by taking photos of a checkerboard
    - ConferenceRoomV2_final_supergloo
      - raw_images # *.ARW for Sony A7M3; *.CR2 for Canon 5D Mark IV

### environment
Recommend to create a new conda environment for this step. Tested only on Ubuntu with Python 3.8, PyTorch 1.13.1, CUDA 11.7:
  
``` bash
pip install -r tools/real_capture/requirements.txt

# [nerfstudio](https://docs.nerf.studio/en/latest/quickstart/installation.html)
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
git clone git@github.com:nerfstudio-project/nerfstudio.git
cd nerfstudio
pip install --upgrade pip setuptools
pip install -e .

# [hloc](https://github.com/cvg/Hierarchical-Localization#installation)
git clone --recursive https://github.com/cvg/Hierarchical-Localization/
cd Hierarchical-Localization/
python -m pip install -e .
```

For Arm64 arch, you may need to manually compile [lensfunpy](https://github.com/letmaik/lensfunpy#installation-from-source-on-linuxmacos). For Colmap, install [pycolmap](https://github.com/colmap/pycolmap#getting-started) then [hloc package](https://github.com/cvg/Hierarchical-Localization#installation).

### run the pipeline
Then run the notebook `tools/real_capture/hdr_convert.ipynb` to generate HDR images and poses, yielding the following new files:

- data/
  - real/
    - ConferenceRoomV2_final_supergloo
      - png_images # SDR images, for reference
      - merged_images # HDR images in *.exr
      - transforms_superglue.json # if use SuperGlue, otherwise transforms_colmap.json if use Colmap
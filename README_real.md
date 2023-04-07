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
# OpenRooms_RAW_loader

A dataloader and visualizer for OpenRooms modalities. Given one scene of multi-view observation, the following modalities are supported:
- 2D per-pixel properties:
  - geometry
    - depth map
    - normals
  - BRDF
    - roughness map
    - albedo map
  - lighting
    - incoming radiance as envmaps (8x16)
    - incoming radiance approximated using Spherical Gaussian (SG) mixture
- per-frame properties:
  - camera pose
- full 3D properties:
  - per-object
    - semantic label
    - mesh
    - emitter properties (for lamps and windows only)
      - intensity
      - (windows) SGs approximation of directional lighting (see Li et al. - 2022 - Physically-Based Editing of Indoor Scene Lighting... -> Sec. 3.1)
  - global environment (outdoor lighting)

## Dependencies

``` bash
conda create --name or-py38 python=3.8 pip
conda activate or-py38
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113 # not tested with other versions of PyTorch
pip install -r requirements.txt
```

Hopefully that was everything. 

## Data structure

- {OR_RAW_ROOT} # assets for OpenRooms; for access, request from Zhengqin Li or Rui Zhu
  - layoutMesh # mesh files for room layout (i.e. walls, ceiling and floor)
  - uv_mapped # shape files from ShapeNet
  - EnvDataset # outdoor envmaps

- data
  - intrinsic.txt
  - colors
  - semanticLabelName_OR42.txt
  - openrooms_public_re_2
    - scenes
      - xml1
        - scene0552_00_more
          - cam.txt # camera poses where rows are origin, lookat, up
          - main.xml # the scene XML file
    - main_xml1 # renderings
      - scene0552_00_more
        - im_%d.png/im_%d.hdr # RGB (PNG is only for reference; scales may be inconsistent)
        - imdepth_%d.dat # depth maps
        - imnormal_%d.png # normals in 2D
        - imroughness_%d.png # roughness in 2D
        - imbaseColor_%d.png # albedo in 2D
        <!-- - box{light_id}.dat # emitter info in 3D
        - light_%d # per-frame emitter source info; should not be useful
          - ...
        - imcadmatobj_%d.dat # instance/material segmentation in 2D
        - imenv_%d.hdr # per-pixel lighting envmaps in 2D -->


## Usage

### 2D dataloader and visualizer

``` bash
python test_class_openroomsScene2D.py --vis_2d
```

This will load per-pixel modalities in 2D, and visualize them in a Matplotlib plot like this:
![](images/demo_all_2D.png)

### 3D dataloader and visualizer

```bash
python test_class_openroomsScene3D.py
```

See the help info of the argparse arguments and comments for usage of flags.

Note that there are two visualizer implemented: based on Matplotlib and Open3D respectively.

The Matplotlib visualizer supports basic visualization of 3D properties inclucing boundinb boxes and camera axes, but not meshes. 

Use ``--vis_3d_plt True`` to use the Matplotlib visualizer. The result is something like this:

![](images/demo_emitters_3D_re1.png)

The Open3D visualizer is based on Open3D, supporting meshes and more beautiful visualization. If you are on a remote machine, make sure you have a X serssion with OpenGL supported (tested with TurboVNC on Ubuntu 18). Alternatively it is recommended to run everything locally (tested on Mac and Window), with the data transferred to your local device.


Use ``--vis_o3d True`` to use the Matplotlib visualizer. The result is something like this:

![](images/demo_all_o3d.png)

### Mitsuba 3 based inference

On Mac, install Mitsuba 3 via:
``` bash
brew install llvm
```

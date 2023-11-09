<!--Generate the TOC via: -->
<!-- (bash) ../gh-md-toc --insert README_indoor_synthetic.md-->
<!--See https://github.com/ekalinin/github-markdown-toc#readme-->

<!--ts-->
- [Indoor Synthetic scenes for multi-view inverse rendering: rendering, load, visualize, and convert](#indoor-synthetic-scenes-for-multi-view-inverse-rendering-rendering-load-visualize-and-convert)
  - [Load preprocessed scenes](#load-preprocessed-scenes)
  - [Export to MonoSDF format](#export-to-monosdf-format)
  - [From scratch🪄: scene preparation and rendering guide](#from-scratch-scene-preparation-and-rendering-guide)
    - [Firstly](#firstly)
    - [Prepare scene files](#prepare-scene-files)
    - [Generate poses via sampling in 3D](#generate-poses-via-sampling-in-3d)
    - [Render all modalities](#render-all-modalities)
    - [HDR images with Mitsuba](#hdr-images-with-mitsuba)
- [Other datasets](#other-datasets)
    - [$I^2$-SDF](#i2-sdf)
    - [OpenRooms](#openrooms)
  - [Tools](#tools)
    - [MonoSDF scene loader](#monosdf-scene-loader)
- [TODO](#todo)

<!-- Created by https://github.com/ekalinin/github-markdown-toc -->
<!-- Added by: jerrypiglet, at: Mon Apr 10 01:32:14 PDT 2023 -->

<!--te-->

# Indoor Synthetic scenes for multi-view inverse rendering: rendering, load, visualize, and convert

This repo contains the code to:

- generate data (synthetic) for indoor inverse rendering from scratch (i.e. synthetic scenes from [Mitsuba XML files by Benedikt Bitterli](https://benedikt-bitterli.me/resources/));
- access tools to visualize/convert data to multiple formats (e.g [FIPT](https://jerrypiglet.github.io/fipt-ucsd/), [MonoSDF](https://niujinshuchong.github.io/monosdf/), [FVP](https://repo-sam.inria.fr/fungraph/deep-indoor-relight), [Li22](https://vilab-ucsd.github.io/ucsd-IndoorLightEditing/)).

## Load preprocessed scenes

We include one scene for demo: `kitchen_mi` from [Benedikt Bitterli](https://benedikt-bitterli.me/resources/)->'Country Kitchen'. They are preprocessed and ready to use. Download [here](https://drive.google.com/drive/folders/1FP2oO2nScm57RTH9hwUzObnJOynD3UDO?usp=share_link) (faster to download .zip files and unzip), and organize as below:

<!-- https://tree.nathanfriend.io -->

<!-- - data/
  - indoor_synthetic/
    - kitchen
      - models
      - test.xml
      - textures
      - intrinsic_mitsubaScene.txt
      - train
        - Image            # HDR images in .exr, SDR images in .png
        - albedo           # material reflectance $\mathbf{a}'$ (See FIPT->Sec. 5.1)
        - DiffCol          # diffuse reflectance $\mathbf{k}_d$ (See FIPT->Equ.1)
        - IndexMA          # part segmentation
        - Emit             # emission radiance
        - Roughness        # roughness
        - segmentation     # semantic segmentation
        - train.npy        # Blender poses
        - transforms.json  # poses in another format for FIPT -->

```
.
└── data/
    └── indoor_synthetic/
        └── kitchen/
            ├── models
            ├── test.xml
            ├── textures
            ├── intrinsic_mitsubaScene.txt
            └── train/
                ├── Image            # HDR images in .exr, SDR images in .png
                ├── albedo           # material reflectance $\mathbf{a}'$ (See FIPT->Sec. 5.1)
                ├── DiffCol          # diffuse reflectance $\mathbf{k}_d$ (See FIPT->Equ.1)
                ├── IndexMA          # part segmentation
                ├── Emit             # emission radiance
                ├── Roughness        # roughness
                ├── segmentation     # semantic segmentation
                ├── train.npy        # Blender poses
                └── transforms.json  # poses in another format for FIPT
```

To load the scene and view default modalities (e.g. poses, meshes from XML file, fused TSDF mesh), run:

``` bash
python load_mitsubaScene3D.py --scene kitchen_mi
```

## Export to MonoSDF format

To export the scene into MonoSDF format (used in [customized MonoSDF repo (branch: fipt)](https://github.com/Jerrypiglet/monosdf/tree/fipt) for FIPT), 

``` bash
python load_mitsubaScene3D.py --scene kitchen_mi --export --export_format monosdf --export_appendix _fipt --split train --vis_3d_o3d False --force

python load_mitsubaScene3D.py --scene kitchen_mi --export --export_format monosdf --export_appendix _fipt --split val --vis_3d_o3d False --force
```

Note that you will need to dump `train` first then `val`, so that `val` will re-use the normalization parameters from `train`.

You will arrive at a folder at `indoor_synthetic/EXPORT_monosdf/kitchen_mi`. Link the folder to your customized MonoSDF repo project path (e.g. `Projects/monosdf/data/indoor_synthetic/kitchen_mi`), and follow the scripts in the customized MonoSDF repo.

## From scratch🪄: scene preparation and rendering guide

### Firstly
Add/modify a entry in `lib/global_vars.py` for your device. By default there are two devices, named `apple` for a M1 Macbook and `mm1` for a Ubuntu machine with CUDA-capable GPU.

Then specify your desired device in the start of `load_mitsubaScene3D.py` (e.g. `host = 'apple'`). As a result, Mitsuba rendering variant will be `mi_variant_dict[host]`. For Blender rendering device, you may need to add/modify `cycles_device` and `compute_device_type` in *lib/class_renderer_blender_mitsubaScene_3D.py*.

### Prepare scene files
Start with a scene downloaded from https://benedikt-bitterli.me/resources/, organized as below:

<!-- - data
  - indoor_synthetic
    - kitchen_diy
      - models
      - textures
      - test.xml -->

<!-- https://tree.nathanfriend.io -->
```
.
└── data/
    └── indoor_synthetic/
        └── kitchen_diy/
            ├── models
            ├── textures
            └── test.xml
```

The original downloaed XML file is *scene_v3.xml*. We modified the scene a bit, including
- removed transparent objects;
- added id for all emitters (e.g. `<emitter type="area" id="lamp_oven_0">`)

Then create a file at [*confs/indoor_synthetic/kitchen_diy.conf*](confs/indoor_synthetic/kitchen_diy.conf), including scene name, and optionally manually specifying emitters/floor etc.:

- `merge_lamp_id_list` for grouping a set of emitters into one.
- `floor_id`, `ceiling_id` to use the corresponding shape as safe space for sampling poses; if not indicated, the method will assumes shapes of walls/ceiling/floor has 'wall', 'ceiling' or 'floor' in their `id` fields.

You will need to assign the `id` field in corresponding shapes/emitters in the XML file. Example of the eventual modified XML file for the kitchen scene: [*samples/test.xml*](samples/test.xml).

Also add a file at *data/indoor_synthetic/kitchen_diy/intrinsic_mitsubaScene.txt* with manually picked intrinsics:

``` txt
554.2562397718481 0.000000 320.000000 # fx, 0, cx(==im_W/2)
0.000000 554.2562397718481 160.000000 # 0, fy, cy(==im_H/2)
0.000000 0.000000 1.000000
```

Use `modality_list = ['shapes', 'layout']` in `scene_obj = mitsubaScene3D()` constructor.     Set `visualizer_3D_o3d.run_o3d(shapes_params={'if_ceiling': True, 'if_walls': True, ...` to draw walls and ceiling (indicted in the `id` tag of corresponding `shape` entry in scene XML file, e.g. `<shape type="obj" id="WallSocket_0001">`).

Preview the scene in 3D with (you should see [this](https://i.imgur.com/PWg0xCU.png)):

``` bash
python load_mitsubaScene3D.py
```

### Generate poses via sampling in 3D

Use `modality_list = ['shapes', 'poses', 'layout']` in `scene_obj = mitsubaScene3D()` constructor.

``` bash
python load_mitsubaScene3D.py --scene kitchen_diy --if_sample_poses --split train
```

The number of sampled frames can be set with `'sample_pose_num'` (default: 200 for train, 20 for val). 
You can set `'sample_pose_if_vis_plt': True` to draw the [sampled poses from bird's-eye view](images/demo_sample_pose_living-room.png).

Fine-grained hyper-parameters of the sampled cameras can be set, by copying a section `cam_params_dict = {...}` from *confs/indoor_synthetic.conf* to *kitchen_diy.conf* and modifying the entries (e.g. `heightMin/Max` for camera height, `distRays...` for controlling the distance between a camera and the closest object in the scene).

Set `'if_ceiling': False, 'if_walls': False` in `visualizer_3D_o3d.run_o3d(shapes_params={...})` to hide walls and ceiling for better seeing the poses.

You should get a visualization of the scene and sampled poses like [this](https://i.imgur.com/H4sT9UN.png) (press `[` key to switch to orthographic projection view).

Sampled poses and params will be dumped to *data/indoor_synthetic/kitchen_diy/cam.txt* and *data/indoor_synthetic/kitchen_diy/cam_params_dict.txt*. In *cam.txt*, poses are saved in OpenRooms format (see notes starting with `OpenRooms convention` in *lib/class_mitsubaScene3D.py*).

[Optionally], add `--eval_scene` to visualize view coverage map.

New files generated:

<!-- - data/indoor_synthetic/kitchen_diy/train
  - cam.txt # poses in OpenRooms format
  - cam_params_dict.txt # params for generating poses
  - train.npy # poses in angles for Blender
  - tmp_sample_poses_rendering # temporary files in sampling poses
    - vis_sampled_poses.png # bird's-eye view of sampled poses
    - depth_*.png # low-res depth map for sampled poses
    - normal_*.png # low-res normal map for sampled poses
    - valid_normal_mask_*.png # low-res mask for valid normal map (i.e. cameras which are not inside objects) for sampled poses -->

```
.
└── data/indoor_synthetic/kitchen_diy/train
    ├── cam.txt # poses in OpenRooms format
    ├── cam_params_dict.txt # params for generating poses
    ├── train.npy # poses in angles for Blender
    └── tmp_sample_poses_rendering # temporary files in sampling poses
        ├── vis_sampled_poses.png # bird's-eye view of sampled poses
        ├── depth_*.png # low-res depth map for sampled poses
        ├── normal_*.png # low-res normal map for sampled poses
        └── valid_normal_mask_*.png # low-res mask for valid normal map (i.e. cameras which are not inside objects) for sampled poses
```

To visualize the poses, add `'poses'` to `mitsubaScene3D(modality_list=[...` and `visualizer_scene_3D_o3d(modality_list_vis=[...`, and re-run without `--if_sample_poses`.

### Render all modalities
Adjust `spp` properly (e.g. 32 for fast rendering, 4096 for final rendering).

<!-- Set `mitsubaScene3D(modality_list=['poses'])` in *load_mitsubaScene3D.py*. -->

### HDR images with Mitsuba

We choose to use Mitsuba to render HDR images (instead of Blender) because of some known issues with Blender rendering (@Liwen).

``` bash
python load_mitsubaScene3D.py --scene kitchen_diy --render_2d --renderer mi
```

New files generated:

<!-- - data/indoor_synthetic/kitchen_diy/train
  - Image
    - %03d_0001.exr # HDR images (0-based index)
    - %03d_0001.png # SDR images (gamma=2.2) -->  

```
.
└── data/indoor_synthetic/kitchen_diy/train
    └── Image
        ├── %03d_0001.exr # HDR images (0-based index)
        └── %03d_0001.png # SDR images (gamma=2.2)
```

Next we render **other modalities** with Blender.

First create a Blender scene file (`test.blend`) from the Mitsuba scene file (`test.xml`). See notes at the beginning of `class renderer_blender_mitsubaScene_3D`.

Set desired modalities to render in `renderer_blender_mitsubaScene_3D(modality_list=[...])`.

``` bash
python load_mitsubaScene3D.py --scene kitchen_diy --render_2d --renderer blender
```

New files generated:

<!-- - data/indoor_synthetic/kitchen_diy/train
  - Depth
  - Normal
  - DiffCol # diffuse albedo
  - Roughness
  - Emit # emission
  - IndexMA # [TODO][???]
  - LightingEnvmap-8x16x256x512 # per-pixel envmaps of each frame; e.g. for each frame, 8x16 envmaps, each of resolution 256x512 -->

```
.
└── data/indoor_synthetic/kitchen_diy/train
    ├── Depth
    ├── Normal
    ├── DiffCol # diffuse albedo
    ├── Roughness
    ├── Emit # emission
    ├── IndexMA # [TODO][???]
    └── LightingEnvmap-8x16x256x512 # per-pixel envmaps of each frame; e.g. for each frame, 8x16 envmaps, each of resolution 256x512
```

For envmaps, params can be set in *confs/indoor_synthetic.conf* -> `lighting_params_dict`.

To combine envmaps into a single image ([demo](https://i.imgur.com/Y5lumVu.jpg)), set `mitsubaScene3D(modality_list=['im_sdr','poses','lighting_envmap']` and `visualizer_scene_2D(modality_list_vis=['im', 'lighting_envmap']`, then run:

``` bash
python load_mitsubaScene3D.py --scene kitchen_diy --vis_2d_plt
```

To visualize other modalities ([demo](https://i.imgur.com/24i0yjA.png)), set `mitsubaScene3D(modality_list` and `visualizer_scene_2D(modality_list_vis` to desired modalities, then run the same command.

# Other datasets

### $I^2$-SDF
Synthetic dataset from [$I^2$-SDF](https://jingsenzhu.github.io/i2-sdf/). Datasets can be downloaded from the project page (2 scenes by 04/12/2023; the convention is explained [here](https://github.com/jingsenzhu/i2-sdf/blob/main/DATA_CONVENTION.md)). [demo](images/demo_i2sdf.png)

- data/i2-sdf-dataset
  - scan332_bedroom_relight_0/
    - depth
    - normal
    - material
      - %04d_kd.exr         # diffuse albedo
      - %04d_ks.exr         # specular albedo
      - %04d_roughness.exr  # roughness
    - image
    - hdr
    - mask
    - light_mask
    - cameras.npz # similar format to propossed MonoSDF poses

``` bash
python load_i2sdfScene3D.py --vis_3d_o3d True --vis_2d_plt False
```

With additional `--eval_scene`, you can visualize [view coverage map](images/demo_i2sdf_viewcount.png).

### OpenRooms
[OpenRooms] scenes re-rendered with multi-view poses per-scene (original dataset is only intended for single view invese rendering). Current generated version is `public_re_0203`, which can be downloaded at [here](https://drive.google.com/drive/folders/1VBSRTSREzYUSd1DLiLJ4VFJSnnUh04uv?usp=share_link). 

- data/public_re_0203
  - scenes                  # scene files
    - intrinsic.txt         # camera intrinsics
    - xml1 | xml
      - scene0552_00_more
        - transform.dat     # scene geometry transformation file
        - cam.txt           # sampled camera poses
        - main.xml | mainDiffLight.xml | mainDiffMat.xml # Scene files for OptixRenderer
        - 
  - mainDiffLight_xml1 | mainDiffLight_xml | mainDiffMat_xml1 | mainDiffMat_xml | main_xml1 | main_xml
    - scene0552_00_more
      - im_*.hdr, im_*.png  # HDR and SDR images
      - imBaseColor_*.png   # diffuse albedo
      - imcadmatobj_*.dat   # material segmentation and object segmentation
      - imdepth_*.dat       # depth
      - immask_*.dat        # mask for valid geometry
      - imnormal_*.png      # normal (camera coordinates; OpenGL convention)
      - imroughness_*.png   # roughness

To load 2D modalities ([demo](images/demo_openrooms_2D_plt.png)):

``` bash
python load_openroomsScene2D.py
```

To load 3D modalities ([demo](images/demo_openrooms_3D_o3d.png)):

``` bash
python load_openroomsScene3D.py
```

Similarly, use `--eval_scene` or `--export` as with other datasets.

## Tools
### MonoSDF scene loader

To validate exported scenes in [MonoSDF](https://github.com/Jerrypiglet/monosdf) format (from all the datasets mentioned above), run:

``` bash
python load_monosdfScene3D.py
```

Noteably, set `if_normalize_shape_depth_from_pose` to `True` to normalize the shape into MonoSDF scene space, using center/scale loaded from the dump file *scale_mat.npy* (demo of the bedroom scene from $I^2$-SDF: [3D]](images/demo_monosdf_normalize_3D_o3d.png), [2D](images/demo_monosdf_normalize_2D_plt.png); note that dumped depth and shape are NOT normalized; need to manually normalize with `scale_mat_dict['scale'], scale_mat_dict['center']` to MonoSDF normalized space).

Dumped scenes for MonoSDF look like:

- .../EXPORT_monosdf/{scene_name}
  - scene.obj # scene geometry exported from loaded shape(s)
  - Image
    - %04d_0001.exr | %04d_0001.png # HDR and SDR images
  - ImMask
    - %04d_0001.png # mask for valid geometry
  - cameras.npz # camera poses for MonoSDF
  - scale_mat.npy # scale and center for MonoSDF normalization
  - K_list.txt # camera intrinsics
  - depth
    - %04d_0001.npy # un-normalized depth from self.mi scene
  - normal
    - %04d_0001.npy # normals from self.mi scene; (3, H, W), [0., 1.] in camera-coordinates (OpenCV convention)
  - MiDepth
    - %04d_0001.npy # un-normalized loaded depth
  - _MiNormalOpenCV | _MiNormalOpenCV_OVERLAY
    - %04d_0001.png # visualization of normal maps rendered from self.mi scene, in camera-coordinates (OpenCV convention)
  - _normal_OVERLAY
    - %04d_0001.png # visualization of loaded normal maps, in camera-coordinates (OpenCV convention)


# TODO
- [ ] Blender: how to NOT render HDR images?
- [ ] better if_autoscale_scene with Mustafa's method (using Colmap points?)
- [ ] deal with extra_transform
- [ ] fix bpy.ops.import_scene.mitsuba() error
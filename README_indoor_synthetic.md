<!--Generate the TOC via: -->
<!-- (bash) ../gh-md-toc --insert README_indoor_synthetic.md-->
<!--See https://github.com/ekalinin/github-markdown-toc#readme-->

<!--ts-->
- [Generating Indoor Synthetic dataset from scratch](#generating-indoor-synthetic-dataset-from-scratch)
  - [Firstly](#firstly)
  - [Prepare scnee files](#prepare-scnee-files)
  - [Generate poses](#generate-poses)
  - [Render all modalities](#render-all-modalities)
    - [Other modalities with Blender](#other-modalities-with-blender)
- [Other datasets](#other-datasets)
    - [i2-sdf](#i2-sdf)
- [TODO](#todo)

<!-- Created by https://github.com/ekalinin/github-markdown-toc -->
<!-- Added by: jerrypiglet, at: Mon Apr 10 01:32:14 PDT 2023 -->

<!--te-->

# Generating Indoor Synthetic dataset from scratch

## Firstly
Add/modify a entry in `lib/global_vars.py` for your device. By default there are two devices, named `apple` for a M1 Macbook and `mm1` for a Ubuntu machine with CUDA-capable GPU.

Then specify your desired device in the start of `load_mitsubaScene3D.py` (e.g. `host = 'apple'`). As a result, Mitsuba rendering variant will be `mi_variant_dict[host]`. For Blender rendering device, you may need to add/modify `        cycles_device = {∫
        compute_device_type = {


## Prepare scnee files
Start with a scene downloaded from https://benedikt-bitterli.me/resources/, organized as below:

<!-- - data
  - indoor_synthetic
    - kitchen_mi
      - models
      - textures
      - test.xml -->

<!-- https://tree.nathanfriend.io -->
```
.
└── data/
    └── indoor_synthetic/
        └── kitchen_mi/
            ├── models
            ├── textures
            └── test.xml
```

The original XML file is scene_v3.xml. We modified the scene a bit, including
- removed transparent objects;
- added id for all emitters (e.g. `<emitter type="area" id="lamp_oven_0">`)

Then add a file at *confs/indoor_synthetic/kitchen_mi.conf*, including scene name, and optionally `merge_lamp_id_list` for grouping a set of emitters into one.

Also add a file at *data/indoor_synthetic/kitchen_mi/intrinsic_mitsubaScene.txt* with manually picked intrinsics:

```
554.2562397718481 0.000000 320.000000 # fx, 0, cx(==im_W/2)
0.000000 554.2562397718481 160.000000 # 0, fy, cy(==im_H/2)
0.000000 0.000000 1.000000
```
Use `modality_list = ['shapes', 'layout']` in `scene_obj = mitsubaScene3D()` constructor.

Preview the scene in 3D with (you should see [this](https://i.imgur.com/PWg0xCU.png)):

``` bash
python load_mitsubaScene3D.py
```
## Generate poses

Use `modality_list = ['shapes', 'poses', 'layout']` in `scene_obj = mitsubaScene3D()` constructor.

``` bash
python load_mitsubaScene3D.py --scene kitchen_mi --if_sample_poses --split train
```

The number of sampled frames can be set with `'sample_pose_num'` (default: 200 for train, 20 for val). 
You can set `'sample_pose_if_vis_plt': True` to see the sampled pose from bird's-eye view.

Fine-grained hyper-parameters of the sampled cameras can be set, by copying a section `cam_params_dict = {...}` from *confs/indoor_synthetic.conf* to *kitchen_mi.conf* and modifying the entries (e.g. `heightMin/Max` for camera height, `distRays...` for controlling the distance between a camera and the closest object in the scene).

Set `'if_ceiling': False, 'if_walls': False` in `visualizer_3D_o3d.run_o3d(shapes_params={...})` to hide walls and ceiling for better seeing the poses.

You should get a visualization of the scene and sampled poses like [this](https://i.imgur.com/H4sT9UN.png) (press `[` key to switch to orthographic projection view).

Sampled poses and params will be dumped to *data/indoor_synthetic/kitchen_mi/cam.txt* and *data/indoor_synthetic/kitchen_mi/cam_params_dict.txt*. In *cam.txt*, poses are saved in OpenRooms format (see notes starting with `OpenRooms convention` in *lib/class_mitsubaScene3D.py*).

[Optionally], add `--eval_scene` to visualize view coverage map.

New files generated:

<!-- - data/indoor_synthetic/kitchen_mi/train
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
└── data/indoor_synthetic/kitchen_mi/train
    ├── cam.txt # poses in OpenRooms format
    ├── cam_params_dict.txt # params for generating poses
    ├── train.npy # poses in angles for Blender
    └── tmp_sample_poses_rendering # temporary files in sampling poses
        ├── vis_sampled_poses.png # bird's-eye view of sampled poses
        ├── depth_*.png # low-res depth map for sampled poses
        ├── normal_*.png # low-res normal map for sampled poses
        └── valid_normal_mask_*.png # low-res mask for valid normal map (i.e. cameras which are not inside objects) for sampled poses
```


## Render all modalities
Adjust `spp` properly (e.g. 32 for fast rendering, 4096).

Set `mitsubaScene3D(modality_list=['poses'])` in *load_mitsubaScene3D.py*.

``` bash
### HDR images with Mitsuba

We choose to use Mitsuba to render HDR images (instead of Blender) because of some known issues with Blender rendering (@Liwen).

``` bash
python load_mitsubaScene3D.py --scene kitchen_mi --render_2d --renderer mi
```

New files generated:

<!-- - data/indoor_synthetic/kitchen_mi/train
  - Image
    - %03d_0001.exr # HDR images (0-based index)
    - %03d_0001.png # SDR images (gamma=2.2) -->

```
.
└── data/indoor_synthetic/kitchen_mi/train
    └── Image
        ├── %03d_0001.exr # HDR images (0-based index)
        └── %03d_0001.png # SDR images (gamma=2.2)
```

### Other modalities with Blender

First create a Blender scene file (`test.blend`) from the Mitsuba scene file (`test.xml`). See notes at the beginning of `class renderer_blender_mitsubaScene_3D`.

Set desired modalities to render in `renderer_blender_mitsubaScene_3D(modality_list=[...])`.

``` bash
python load_mitsubaScene3D.py --scene kitchen_mi --render_2d --renderer blender
```

New files generated:

<!-- - data/indoor_synthetic/kitchen_mi/train
  - Depth
  - Normal
  - DiffCol # diffuse albedo
  - Roughness
  - Emit # emission
  - IndexMA # [TODO][???]
  - LightingEnvmap-8x16x256x512 # per-pixel envmaps of each frame; e.g. for each frame, 8x16 envmaps, each of resolution 256x512 -->

```
.
└── data/indoor_synthetic/kitchen_mi/train
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
python load_mitsubaScene3D.py --scene kitchen_mi --vis_2d_plt
```

To visualize other modalities ([demo](https://i.imgur.com/24i0yjA.png)), set `mitsubaScene3D(modality_list` and `visualizer_scene_2D(modality_list_vis` to desired modalities, then run the same command.

# Other datasets
### i2-sdf
Synthetic dataset from [I^2-SDF](https://jingsenzhu.github.io/i2-sdf/). Datasets can be downloaded from the project page (currently only 2 scenes; convention is explained [here](https://github.com/jingsenzhu/i2-sdf/blob/main/DATA_CONVENTION.md)). [demo](images/demo_i2sdf.png)

- data/i2-sdf-dataset
  - scan332_bedroom_relight_0/
    - depth
    - normal
    - material
      - %04d_kd.exr # diffuse albedo
      - %04d_ks.exr # specular albedo
      - %04d_roughness.exr # roughness
    - image
    - hdr
    - mask
    - light_mask
    - cameras.npz # similar format to propossed MonoSDF poses

``` bash
python load_i2sdfScene3D.py --vis_3d_o3d True --vis_2d_plt False
```

# TODO
- [ ] Blender: how to NOT render HDR images?
- [ ] better if_autoscale_scene with Mustafa's method (using Colmap points?)
- [ ] deal with extra_transform
- [ ] fix bpy.ops.import_scene.mitsuba() error
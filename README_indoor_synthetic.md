# Generating Indoor Synthetic dataset from scratch

## Prepare scnee files
Start with a scene downloaded from https://benedikt-bitterli.me/resources/, organized as below:

- data
  - indoor_synthetic
    - kitchen_mi
      - models
      - textures
      - test.xml

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
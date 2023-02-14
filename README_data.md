<!--ts-->
- [More datasets](#more-datasets)
  - [Freeviewpoint](#freeviewpoint)
  - [Matterport3D](#matterport3d)

<!-- Created by https://github.com/ekalinin/github-markdown-toc -->
<!-- Added by: jerrypiglet, at: Mon Feb 13 02:05:56 PST 2023 -->

<!--te-->

# More datasets

- data
  - Matterport3D
    - 17DRP5sb8fy (house id)
        - undistorted_normal_images
        - sens
        - matterport_depth_images
        - undistorted_depth_images
        - matterport_color_images
        - poisson_meshes
        - matterport_hdr_images
        - undistorted_color_images
        - matterport_camera_poses
        - image_overlap_data
        - region_segmentations
        - house_segmentations
        - matterport_skybox_images
        - matterport_mesh
        - undistorted_camera_parameters
        - cameras
  - free-viewpoint
      - asianRoom1
          - images
          - meshes
          - cameras
          - lightings
          - testPath.lookat
          - mi_seg_emitter
          - scene_metadata_jpg.txt
          - scene_metadata.txt
          - scale.txt
      - asianRoom2
      - sofa91
      - Hall
      - Salon2
      - Kitchen

## Freeviewpoint

From Philip et al.'21, *Free-viewpoint Indoor Neural Relighting from Multi-view Stereo*. [[Project]](https://repo-sam.inria.fr/fungraph/deep-indoor-relight/#code [[Code]](https://gitlab.inria.fr/sibr/projects/indoor_relighting)

``` bash
python test_class_freeviewpointScene3D.py --vis_2d_plt True
```
- `--if_convert_poses True` to dump poses in Openrooms format (cam.txt and K_list.txt)
- `--if_dump_shape True` to dump FIXED shape (recon.ply to recon_fixed.obj, and additional hull shape fo recon_hull.obj)

![](images/demo_freeviewpoint_o3d.png)
![](images/demo_freeviewpoint_plt_2d.png)

Holes in the scene are fixed by adding a convex hull mesh to the original mesh. See demo [here](images/demo_freeviewpoint_fix_holes.png).

TODO:
- [ ] add option to resize all frames to same dimension
- [x] fix holes on the wall
  
## Matterport3D

From Chang et al.'17, *Matterport3D: Learning from RGB-D Data in Indoor ...* [[Project]](https://niessner.github.io/Matterport/) [[Code]](https://github.com/niessner/Matterport) [[Browse]](https://aspis.cmpt.sfu.ca/scene-toolkit/scans/matterport3d/houses) [[Data organization]](https://github.com/niessner/Matterport/blob/master/data_organization.md)

![](images/demo_matterport_plt_2d.png)

TODO:
- [ ] undistord HDR images
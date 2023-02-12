# More datasets
## Freeviewpoint scenes

From Philip et al.'21, *Free-viewpoint Indoor Neural Relighting from Multi-view Stereo*. [page](https://repo-sam.inria.fr/fungraph/deep-indoor-relight/#code)

``` bash
python test_class_freeviewpointScene3D.py --vis_2d_plt True
```
- `--if_convert_poses True` to dump poses in Openrooms format (cam.txt and K_list.txt)
- `--if_dump_shape True` to dump FIXED shape (recon.ply to recon_fixed.obj, and additional hull shape fo recon_hull.obj)

![](images/demo_freeviewpoint_o3d.png)
![](images/demo_freeviewpoint_plt_2d.png)

Holes in the scene are fixed by adding a convex hull mesh to the original mesh. See demo [here](images/demo_freeviewpoint_fix_holes.png).

TODO:
- [] add option to resize all frames to same dimension
- [] fix holes on the wall
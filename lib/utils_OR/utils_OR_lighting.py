import numpy as np

def downsample_lighting_envmap(lighting_envmap, downsize_ratio: int=8, lighting_scale: float=1.):
    H_grid, W_grid, _, h, w = lighting_envmap.shape
    xx, yy = np.meshgrid(np.arange(0, H_grid, downsize_ratio), np.arange(0, W_grid, downsize_ratio))
    _ = lighting_envmap[xx.T, yy.T, :, :, :] # (30, 40, 3, 8, 16)
    H_grid_after, W_grid_after = _.shape[:2]
    lighting_envmap_downsampled = np.zeros((H_grid_after, W_grid_after, 3, h+1, w+1), dtype=_.dtype) + 1.
    lighting_envmap_downsampled[:, :, :, :h, :w] = _ * lighting_scale
    lighting_envmap_downsampled = lighting_envmap_downsampled.transpose(0, 3, 1, 4, 2).reshape(H_grid*(h+1)//downsize_ratio, W_grid*(w+1)//downsize_ratio, 3)

    return lighting_envmap_downsampled


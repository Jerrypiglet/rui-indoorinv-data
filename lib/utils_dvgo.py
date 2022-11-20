'''
adapted from dvgo.py -> get_rays_of_a_view() -> get_rays()
'''

import numpy as np

def get_rays_np(H, W, K, c2w, inverse_y: bool=True, if_normalize_d: bool=True):
    '''
    inverse_y: camera axis (z) pointing forward
    '''
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    if inverse_y:
        dirs = np.stack([(i-K[0][2])/K[0][0], (j-K[1][2])/K[1][1], np.ones_like(i)], -1)
    else:
        dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    if if_normalize_d:
        rays_d = rays_d / (np.linalg.norm(rays_d, axis=2, keepdims=True)+1e-6)
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,3], np.shape(rays_d))

    if inverse_y:
        dir_center = np.array([0., 0., 1.]).reshape((3, 1))
    else:
        dir_center = np.array([0., 0., -1.]).reshape((3, 1))
    ray_d_center = c2w[:3, :3] @ dir_center

    return rays_o, rays_d, ray_d_center

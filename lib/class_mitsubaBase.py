import numpy as np
np.set_printoptions(suppress=True)

from tqdm import tqdm
from collections import defaultdict
import torch
# Import the library using the alias "mi"
import mitsuba as mi
# Set the variant of the renderer
# from lib.global_vars import mi_variant
# mi.set_variant(mi_variant)

from lib.utils_dvgo import get_rays_np
from lib.utils_misc import green

class mitsubaBase():
    '''
    A class used to visualize/render Mitsuba scene in XML format
    '''
    def __init__(
        self, 
        device: str='', 
    ): 
        self.device = device

    def to_d(self, x: np.ndarray):
        if 'mps' in self.device: # Mitsuba RuntimeError: Cannot pack tensors on mps:0
            return x
        return torch.from_numpy(x).to(self.device)

    def get_cam_rays_list(self, H, W, K, pose_list):
        cam_rays_list = []
        for pose in pose_list:
            rays_o, rays_d, ray_d_center = get_rays_np(H, W, K, pose, inverse_y=True)
            cam_rays_list.append((rays_o, rays_d, ray_d_center))
        return cam_rays_list

    def mi_sample_rays_pts(self, cam_rays_list):
        '''
        sample per-pixel rays in NeRF/DVGO setting
        -> populate: 
            - self.mi_pts_list: [(H, W, 3), ], (-1. 1.)
            - self.mi_depth_list: [(H, W), ], (-1. 1.)
        [!] note:
            - in both self.mi_pts_list and self.mi_depth_list, np.inf values exist for pixels of infinite depth
        '''
        if self.pts_from['mi']:
            return

        self.mi_rays_ret_list = []
        self.mi_depth_list = []
        self.mi_invalid_depth_mask_list = []
        self.mi_normal_list = [] # in local OpenGL coords
        self.mi_normal_global_list = []
        self.mi_pts_list = []

        print(green('[mi_sample_rays_pts] for %d frames...'%len(cam_rays_list)))

        for frame_idx, (rays_o, rays_d, ray_d_center) in tqdm(enumerate(cam_rays_list)):
            rays_o_flatten, rays_d_flatten = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)

            xs_mi = mi.Point3f(self.to_d(rays_o_flatten))
            ds_mi = mi.Vector3f(self.to_d(rays_d_flatten))
            # ray origin, direction, t_max
            rays_mi = mi.Ray3f(xs_mi, ds_mi)
            ret = self.mi_scene.ray_intersect(rays_mi) # [mitsuba.Scene.ray_intersect] https://mitsuba.readthedocs.io/en/stable/src/api_reference.html?highlight=write_ply#mitsuba.Scene.ray_intersect
            # returned structure contains intersection location, nomral, ray step, ... # [mitsuba.SurfaceInteraction3f] https://mitsuba.readthedocs.io/en/stable/src/api_reference.html#mitsuba.SurfaceInteraction3f
            # positions = mi2torch(ret.p.torch())
            self.mi_rays_ret_list.append(ret)

            # rays_v_flatten = ret.p.numpy() - rays_o_flatten
            rays_v_flatten = ret.t.numpy()[:, np.newaxis] * rays_d_flatten
            mi_depth = np.sum(rays_v_flatten.reshape(self.H, self.W, 3) * ray_d_center.reshape(1, 1, 3), axis=-1)
            invalid_depth_mask = np.logical_or(np.isnan(mi_depth), np.isinf(mi_depth))
            self.mi_invalid_depth_mask_list.append(invalid_depth_mask)
            mi_depth[invalid_depth_mask] = np.inf
            self.mi_depth_list.append(mi_depth)

            mi_normal_global = ret.n.numpy().reshape(self.H, self.W, 3)
            # FLIP inverted normals!
            normals_flip_mask = np.logical_and(np.sum(rays_d * mi_normal_global, axis=-1) > 0, np.any(mi_normal_global != np.inf, axis=-1))
            mi_normal_global[normals_flip_mask] = -mi_normal_global[normals_flip_mask]
            mi_normal_global[invalid_depth_mask, :] = np.inf
            self.mi_normal_global_list.append(mi_normal_global)

            mi_normal_cam_opencv = mi_normal_global @ self.pose_list[frame_idx][:3, :3]
            mi_normal_cam_opengl = np.stack([mi_normal_cam_opencv[:, :, 0], -mi_normal_cam_opencv[:, :, 1], -mi_normal_cam_opencv[:, :, 2]], axis=-1) # transform normals from OpenGL convention (right-up-backward) to OpenCV (right-down-forward)
            mi_normal_cam_opengl[invalid_depth_mask, :] = np.inf
            self.mi_normal_list.append(mi_normal_cam_opengl)

            mi_pts = ret.p.numpy()
            # mi_pts = ret.t.numpy()[:, np.newaxis] * rays_d_flatten + rays_o_flatten # should be the same as above
            assert sum(ret.t.numpy()!=np.inf) > 1, 'no rays hit any surface!'
            assert np.amax(np.abs((mi_pts - ret.p.numpy())[ret.t.numpy()!=np.inf, :])) < 1e-3 # except in window areas
            mi_pts = mi_pts.reshape(self.H, self.W, 3)
            mi_pts[invalid_depth_mask, :] = np.inf

            self.mi_pts_list.append(mi_pts)

    def mi_get_segs(self, if_also_dump_xml_with_lit_area_lights_only=True):
        '''
        images/demo_mitsuba_ret_seg_2D.png; 
        Update:
            switched to use mitsuba.SurfaceInteraction3f properties to determine area emitter masks: no need for another scene with area lights only
        '''
        if not self.pts_from['mi']:
            self.mi_sample_rays_pts(self.cam_rays_list)

        self.mi_seg_dict_of_lists = defaultdict(list)
        assert len(self.mi_rays_ret_list) == len(self.mi_invalid_depth_mask_list)

        for frame_idx, ret in tqdm(enumerate(self.mi_rays_ret_list)):
            mi_seg_env = self.mi_invalid_depth_mask_list[frame_idx]
            self.mi_seg_dict_of_lists['env'].append(mi_seg_env) # shine-through area of windows

            mi_seg_area = np.array([[s.emitter() is not None for s in ret.shape]]).reshape(self.H, self.W) # [class mitsuba.ShapePtr] https://mitsuba.readthedocs.io/en/stable/src/api_reference.html#mitsuba.ShapePtr
            self.mi_seg_dict_of_lists['area'].append(mi_seg_area) # lit-up lamps

            mi_seg_obj = np.logical_and(np.logical_not(mi_seg_area), np.logical_not(mi_seg_env))
            self.mi_seg_dict_of_lists['obj'].append(mi_seg_obj) # non-emitter objects

    # def mi_get_segs_(self, if_also_dump_xml_with_lit_area_lights_only=True):
    #     '''
    #     images/demo_mitsuba_ret_seg_2D.png
    #     '''
    #     self.mi_seg_dict_of_lists = defaultdict(list)

    #     for frame_idx, mi_depth in enumerate(self.mi_depth_list):
    #         # self.mi_seg_dict_of_lists['area'].append(seg_area)
    #         mi_seg_env = self.mi_invalid_depth_mask_list[frame_idx]
    #         self.mi_seg_dict_of_lists['env'].append(mi_seg_env) # shine-through area of windows

    #         if if_also_dump_xml_with_lit_area_lights_only:
    #             rays_o, rays_d, ray_d_center = self.cam_rays_list[frame_idx]
    #             rays_o_flatten, rays_d_flatten = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)
    #             rays_mi = mi.Ray3f(mi.Point3f(self.to_d(rays_o_flatten)), mi.Vector3f(self.to_d(rays_d_flatten)))
    #             ret = self.mi_scene_lit_up_area_lights_only.ray_intersect(rays_mi)
    #             ret_t = ret.t.numpy().reshape(self.H, self.W)
    #             invalid_depth_mask = np.logical_or(np.isnan(ret_t), np.isinf(ret_t))
    #             mi_seg_area = np.logical_not(invalid_depth_mask)
    #             self.mi_seg_dict_of_lists['area'].append(mi_seg_area) # lit-up lamps

    #             mi_seg_obj = np.logical_and(np.logical_not(mi_seg_area), np.logical_not(mi_seg_env))
    #             self.mi_seg_dict_of_lists['obj'].append(mi_seg_obj) # non-emitter objects
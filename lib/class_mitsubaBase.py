import numpy as np
np.set_printoptions(suppress=True)

from tqdm import tqdm
from collections import defaultdict
import torch
import time
from pathlib import Path
import imageio
# Import the library using the alias "mi"
import mitsuba as mi
# Set the variant of the renderer
# from lib.global_vars import mi_variant
# mi.set_variant(mi_variant)

from lib.utils_io import load_img
from lib.utils_dvgo import get_rays_np
from lib.utils_misc import green, green_text, blue_text, get_list_of_keys, white_blue, yellow
from lib.utils_OR.utils_OR_lighting import convert_lighting_axis_local_to_global_np, get_lighting_envmap_dirs_global

class mitsubaBase():
    '''
    Base class used to load/visualize/render Mitsuba scene from XML file
    '''
    def __init__(
        self, 
        device: str='', 
        if_debug_info: bool=False, 
    ): 
        self.device = device
        self.if_debug_info = if_debug_info

    def to_d(self, x: np.ndarray):
        if 'mps' in self.device: # Mitsuba RuntimeError: Cannot pack tensors on mps:0
            return x
        return torch.from_numpy(x).to(self.device)

    def get_cam_rays_list(self, H, W, K, pose_list):
        cam_rays_list = []
        for _, pose in enumerate(pose_list):
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
        self.mi_rays_t_list = []

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
            rays_t = ret.t.numpy()
            self.mi_rays_t_list.append(rays_t)

            rays_v_flatten = rays_t[:, np.newaxis] * rays_d_flatten
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

        print(green_text('DONE. [mi_sample_rays_pts] for %d frames...'%len(cam_rays_list)))

    def mi_get_segs(self, if_also_dump_xml_with_lit_area_lights_only=True, if_dump=True):
        '''
        images/demo_mitsuba_ret_seg_2D.png; 
        Update:
            switched to use mitsuba.SurfaceInteraction3f properties to determine area emitter masks: no need for another scene with area lights only
        '''
        if not self.pts_from['mi']:
            self.mi_sample_rays_pts(self.cam_rays_list)

        self.mi_seg_dict_of_lists = defaultdict(list)
        assert len(self.mi_rays_ret_list) == len(self.mi_invalid_depth_mask_list)

        print(green('[mi_get_segs] for %d frames...'%len(self.mi_rays_ret_list)))

        for frame_idx, ret in tqdm(enumerate(self.mi_rays_ret_list)):
            ts = time.time()
            mi_seg_env = self.mi_invalid_depth_mask_list[frame_idx]
            self.mi_seg_dict_of_lists['env'].append(mi_seg_env) # shine-through area of windows
            
            # [class mitsuba.ShapePtr] https://mitsuba.readthedocs.io/en/stable/src/api_reference.html#mitsuba.ShapePtr
            # slow...
            mi_seg_area_file_folder = self.scene_rendering_path / 'mi_seg_emitter'
            mi_seg_area_file_folder.mkdir(parents=True, exist_ok=True)
            mi_seg_area_file_path = mi_seg_area_file_folder / ('mi_seg_emitter_%d.png'%(self.frame_id_list[frame_idx]))
            if_get_from_scratch = False
            if mi_seg_area_file_path.exists():
                mi_seg_area = load_img(mi_seg_area_file_path, (self.im_H_load, self.im_W_load), ext='png', target_HW=self.im_target_HW, if_attempt_load=True)/255.
                if mi_seg_area is None:
                    if_get_from_scratch = True
                    mi_seg_area_file_path.unlink()

            if if_get_from_scratch:
                mi_seg_area = np.array([[s is not None and s.emitter() is not None for s in ret.shape]]).reshape(self.H, self.W)
                imageio.imwrite(str(mi_seg_area_file_path), (mi_seg_area*255.).astype(np.uint8))
                print(green_text('[mi_get_segs] mi_seg_area -> %s'%str(mi_seg_area_file_path)))

            self.mi_seg_dict_of_lists['area'].append(mi_seg_area) # lit-up lamps

            mi_seg_obj = np.logical_and(np.logical_not(mi_seg_area), np.logical_not(mi_seg_env))
            self.mi_seg_dict_of_lists['obj'].append(mi_seg_obj) # non-emitter objects

        print(green_text('DONE. [mi_get_segs] for %d frames...'%len(self.mi_rays_ret_list)))

    def _fuse_3D_geometry(self, dump_path: Path=Path(''), subsample_rate_pts: int=1, subsample_HW_rates: tuple=(1, 1), if_use_mi_geometry: bool=False, if_lighting=False):
        '''
        fuse depth maps (and RGB, normals) into point clouds in global coordinates of OpenCV convention

        optionally dump pcd and cams to pickles

        Args:
            subsample_rate_pts: int, sample 1/subsample_rate_pts of points to dump
            if_use_mi_geometry: True: use geometrt from Mistuba if possible

        Returns:
            - fused geometry as dict
            - all camera poses
        '''
        assert self.if_has_poses and self.if_has_im_sdr
        if not if_use_mi_geometry:
            assert self.if_has_depth_normal

        print(white_blue('[mitsubaBase] fuse_3D_geometry '), yellow('[use Mitsuba: %s]'%str(if_use_mi_geometry)), 'for %d frames... subsample_rate_pts: %d, subsample_HW_rates: (%d, %d)'%(len(self.frame_id_list), subsample_rate_pts, subsample_HW_rates[0], subsample_HW_rates[1]))

        X_global_list = []
        rgb_global_list = []
        normal_global_list = []
        X_flatten_mask_list = []

        H_color, W_color = self.H, self.W
        for frame_idx in tqdm(range(len(self.frame_id_list))):
            t = self.pose_list[frame_idx][:3, -1].reshape((3, 1))
            R = self.pose_list[frame_idx][:3, :3]

            if if_use_mi_geometry:
                X_cam_ = (np.linalg.inv(R) @ (self.mi_pts_list[frame_idx].reshape(-1, 3).T - t)).T.reshape(H_color, W_color, 3)
                x_, y_, z_ = np.split(X_cam_, 3, axis=-1)
            else:
                uu, vv = np.meshgrid(range(W_color), range(H_color))
                x_ = (uu - self.K[0][2]) * self.depth_list[frame_idx] / self.K[0][0]
                y_ = (vv - self.K[1][2]) * self.depth_list[frame_idx] / self.K[1][1]
                z_ = self.depth_list[frame_idx]

            if if_use_mi_geometry:
                seg_dict_of_lists = self.mi_seg_dict_of_lists
            else:
                seg_dict_of_lists = self.seg_dict_of_lists
            if if_lighting:
                obj_mask = seg_dict_of_lists['obj'][frame_idx]
            else:
                obj_mask = seg_dict_of_lists['obj'][frame_idx] + seg_dict_of_lists['area'][frame_idx] # geometry is defined for objects + emitters
            assert obj_mask.shape[:2] == (H_color, W_color)

            if subsample_HW_rates != (1, 1):
                x_ = x_[::subsample_HW_rates[0], ::subsample_HW_rates[1]]
                y_ = y_[::subsample_HW_rates[0], ::subsample_HW_rates[1]]
                z_ = z_[::subsample_HW_rates[0], ::subsample_HW_rates[1]]
                H_color, W_color = H_color//subsample_HW_rates[0], W_color//subsample_HW_rates[1]
                obj_mask = obj_mask[::subsample_HW_rates[0], ::subsample_HW_rates[1]]
                
            z_ = z_.flatten()
            # X_flatten_mask = np.logical_and(np.logical_and(z_ > 0, obj_mask.flatten() > 0), ~np.isinf(z_))
            X_flatten_mask = np.logical_and(~np.isinf(z_), obj_mask.flatten() > 0)
            
            z_ = z_[X_flatten_mask]
            x_ = x_.flatten()[X_flatten_mask]
            y_ = y_.flatten()[X_flatten_mask]
            if self.if_debug_info:
                print('Valid pixels percentage: %.4f'%(sum(X_flatten_mask)/float(H_color*W_color)))
            X_flatten_mask_list.append(X_flatten_mask)

            X_ = np.stack([x_, y_, z_], axis=-1)

            X_global = (R @ X_.T + t).T
            X_global_list.append(X_global)

            rgb_global = self.im_sdr_list[frame_idx]
            if subsample_HW_rates != ():
                rgb_global = rgb_global[::subsample_HW_rates[0], ::subsample_HW_rates[1]]
            rgb_global = rgb_global.reshape(-1, 3)[X_flatten_mask]
            rgb_global_list.append(rgb_global)
            
            if if_use_mi_geometry:
                normal = self.mi_normal_list[frame_idx]
            else:
                normal = self.normal_list[frame_idx]
            if subsample_HW_rates != ():
                normal = normal[::subsample_HW_rates[0], ::subsample_HW_rates[1]]
            normal = normal.reshape(-1, 3)[X_flatten_mask]
            normal = np.stack([normal[:, 0], -normal[:, 1], -normal[:, 2]], axis=-1) # transform normals from OpenGL convention (right-up-backward) to OpenCV (right-down-forward)
            normal_global = (R @ normal.T).T
            normal_global_list.append(normal_global)

        print(blue_text('[openroomsScene] DONE. fuse_3D_geometry'))

        X_global = np.vstack(X_global_list)[::subsample_rate_pts]
        rgb_global = np.vstack(rgb_global_list)[::subsample_rate_pts]
        normal_global = np.vstack(normal_global_list)[::subsample_rate_pts]

        assert X_global.shape[0] == rgb_global.shape[0] == normal_global.shape[0]

        geo_fused_dict = {'X': X_global, 'rgb': rgb_global, 'normal': normal_global}

        return geo_fused_dict, X_flatten_mask_list

    def _fuse_3D_lighting(self, lighting_source: str, subsample_rate_pts: int=1, subsample_rate_wi: int=1, if_use_mi_geometry: bool=False, if_use_loaded_envmap_position: bool=False):
        '''
        fuse dense lighting (using corresponding surface geometry)

        Args:
            subsample_rate_pts: int, sample 1/subsample_rate_pts of points to dump

        Returns:
            - fused lighting and their associated pcd as dict
        '''

        print(white_blue('[mitsubaBase] fuse_3D_lighting [%s] for %d frames... subsample_rate_pts: %d'%(lighting_source, len(self.frame_id_list), subsample_rate_pts)))

        if if_use_mi_geometry:
            assert self.if_has_mitsuba_all
            normal_list = self.mi_normal_list
        else:
            assert self.if_has_depth_normal
            normal_list = self.normal_list

        assert lighting_source in ['lighting_SG', 'lighting_envmap'] # not supporting 'lighting_sampled' yet

        geo_fused_dict, X_lighting_flatten_mask_list = self._fuse_3D_geometry(subsample_rate_pts=subsample_rate_pts, subsample_HW_rates=self.im_lighting_HW_ratios, if_use_mi_geometry=if_use_mi_geometry, if_lighting=True)
        X_global_lighting, normal_global_lighting = geo_fused_dict['X'], geo_fused_dict['normal']

        if lighting_source == 'lighting_SG':
            assert self.if_has_lighting_SG and self.if_has_poses
        if lighting_source == 'lighting_envmap':
            assert self.if_has_lighting_envmap and self.if_has_poses

        axis_global_list = []
        weight_list = []
        lamb_list = []

        for frame_idx in tqdm(range(len(self.frame_id_list))):
            print(blue_text('[mitsubaBase] fuse_3D_lighting [%s] for frame %d...'%(lighting_source, frame_idx)))
            if lighting_source == 'lighting_SG':
                wi_num = self.lighting_params_dict['SG_params']
                if self.if_convert_lighting_SG_to_global:
                    lighting_global = self.lighting_SG_global_list[frame_idx] # (120, 160, 12(SG_num), 7)
                else:
                    lighting_global = np.concatenate(
                        (convert_lighting_axis_local_to_global_np(self.lighting_SG_local_list[frame_idx][:, :, :, :3], self.pose_list[frame_idx], normal_list[frame_idx]), 
                        self.lighting_SG_local_list[frame_idx][:, :, :, 3:]), axis=3) # (120, 160, 12(SG_num), 7); axis, lamb, weight: 3, 1, 3
                axis_np_global = lighting_global[:, :, :, :3].reshape(-1, wi_num, 3)
                weight_np = lighting_global[:, :, :, 4:].reshape(-1, wi_num, 3)

            if lighting_source == 'lighting_envmap':
                env_row, env_col, env_height, env_width = get_list_of_keys(self.lighting_params_dict, ['env_row', 'env_col', 'env_height', 'env_width'], [int, int, int, int])
                if if_use_loaded_envmap_position:
                    assert if_use_mi_geometry, 'not ready for non-mitsubaScene'
                    axis_np_global = self.process_loaded_envmap_axis_2d_for_frame(frame_idx).reshape((env_row*env_col, env_height*env_width, 3))
                else:
                    print('[_fuse_3D_lighting->get_lighting_envmap_dirs_global] This might be slow...')
                    axis_np_global = get_lighting_envmap_dirs_global(self.pose_list[frame_idx], normal_list[frame_idx], env_height, env_width) # (HW, wi, 3)
                weight_np = self.lighting_envmap_list[frame_idx].transpose(0, 1, 3, 4, 2).reshape(env_row*env_col, env_height*env_width, 3)

            if lighting_source == 'lighting_SG':
                lamb_np = lighting_global[:, :, :, 3:4].reshape(-1, wi_num, 1)

            X_flatten_mask = X_lighting_flatten_mask_list[frame_idx]
            axis_np_global = axis_np_global[X_flatten_mask]
            weight_np = weight_np[X_flatten_mask]
            if lighting_source == 'lighting_SG':
                lamb_np = lamb_np[X_flatten_mask]

            axis_global_list.append(axis_np_global)
            weight_list.append(weight_np)
            if lighting_source == 'lighting_SG':
                lamb_list.append(lamb_np)

        print(blue_text('[mitsubaBase] DONE. fuse_3D_lighting'))

        axis_global = np.vstack(axis_global_list)[::subsample_rate_pts, ::subsample_rate_wi]
        weight = np.vstack(weight_list)[::subsample_rate_pts, ::subsample_rate_wi]
        if lighting_source == 'lighting_SG':
            lamb_global = np.vstack(lamb_list)[::subsample_rate_pts, ::subsample_rate_wi]
        assert X_global_lighting.shape[0] == axis_global.shape[0]

        lighting_SG_fused_dict = {
            'pts_global_lighting': X_global_lighting, 'normal_global_lighting': normal_global_lighting,
            'axis': axis_global, 'weight': weight,}
        if lighting_source == 'lighting_SG':
            lighting_SG_fused_dict.update({'lamb': lamb_global, })

        return lighting_SG_fused_dict

    def process_loaded_envmap_axis_2d_for_frame(self, frame_idx):
        assert hasattr(self, 'lighting_envmap_position_list')
        env_height, env_width = get_list_of_keys(self.lighting_params_dict, ['env_height', 'env_width'], [int, int])
        lighting_envmap_position = self.lighting_envmap_position_list[frame_idx] # (env_row, env_col, 3, env_height, env_width)
        lighting_envmap_position = lighting_envmap_position.transpose(0, 1, 3, 4, 2)
        lighting_envmap_o = self.mi_pts_list[frame_idx][::self.im_lighting_HW_ratios[0], ::self.im_lighting_HW_ratios[1]][:, :, np.newaxis, np.newaxis]
        lighting_envmap_o = np.tile(lighting_envmap_o, (1, 1, env_height, env_width, 1))
        assert lighting_envmap_o.shape == lighting_envmap_position.shape
        axis_np_global = lighting_envmap_position - lighting_envmap_o
        axis_np_global = axis_np_global / (np.linalg.norm(axis_np_global, axis=-1, keepdims=True)+1e-6)
        return axis_np_global
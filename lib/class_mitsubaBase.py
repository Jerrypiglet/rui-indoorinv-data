import numpy as np
np.set_printoptions(suppress=True)

from tqdm import tqdm
from collections import defaultdict
import torch
import time
from pathlib import Path
import imageio
import json
import shutil
import trimesh
import cv2
# Import the library using the alias "mi"
import mitsuba as mi
from lib.utils_io import load_img, resize_intrinsics
from lib.utils_OR.utils_OR_cam import R_t_to_origin_lookatvector_up, read_cam_params_OR, dump_cam_params_OR
from lib.utils_dvgo import get_rays_np
from lib.utils_misc import green, white_red, green_text, yellow, yellow_text, white_blue, blue_text, red, vis_disp_colormap
from lib.utils_OR.utils_OR_lighting import convert_lighting_axis_local_to_global_np, get_lighting_envmap_dirs_global
from lib.utils_OR.utils_OR_cam import origin_lookat_up_to_R_t

from lib.utils_monosdf_scene import load_shape_dict_from_shape_file, load_monosdf_scale_offset

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

        # self.if_loaded_colors = False
        self.if_loaded_shapes = False
        self.if_loaded_layout = False

        self.extra_transform = None

    def to_d(self, x: np.ndarray):
        if 'mps' in self.device: # Mitsuba RuntimeError: Cannot pack tensors on mps:0
            return x
        return torch.from_numpy(x).to(self.device)

    def get_cam_rays_list(self, H_list: list, W_list: list, K_list: list, pose_list: list, convention: str='opencv'):
        assert convention in ['opengl', 'opencv']
        cam_rays_list = []
        if not isinstance(K_list, list): K_list = [K_list] * len(pose_list)
        if not isinstance(H_list, list): H_list = [H_list] * len(pose_list)
        if not isinstance(W_list, list): W_list = [W_list] * len(pose_list)
        assert len(K_list) == len(pose_list) == len(H_list) == len(W_list)
        for _, (H, W, pose, K) in enumerate(zip(H_list, W_list, pose_list, K_list)):
            rays_o, rays_d, ray_d_center = get_rays_np(H, W, K, pose, inverse_y=(convention=='opencv'))
            cam_rays_list.append((rays_o, rays_d, ray_d_center))
        return cam_rays_list

    def mi_sample_rays_pts(
        self, 
        cam_rays_list, 
        if_force: bool=False,
        ):
        '''
        sample per-pixel rays in NeRF/DVGO setting
        -> populate: 
            - self.mi_pts_list: [(H, W, 3), ], (-1. 1.)
            - self.mi_depth_list: [(H, W), ], (-1. 1.)
        [!] note:
            - in both self.mi_pts_list and self.mi_depth_list, np.inf values exist for pixels of infinite depth
        '''
        if self.pts_from['mi'] and not if_force:
            print(green('[mi_sample_rays_pts] already populated. skip.'))
            return

        self.mi_rays_ret_list = []
        self.mi_rays_t_list = []

        self.mi_depth_list = []
        self.mi_invalid_depth_mask_list = []
        self.mi_normal_opengl_list = [] # in local OpenGL coords
        self.mi_normal_opencv_list = []
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
            mi_depth = np.sum(rays_v_flatten.reshape(self._H(frame_idx), self._W(frame_idx), 3) * ray_d_center.reshape(1, 1, 3), axis=-1)
            invalid_depth_mask = np.logical_or(np.isnan(mi_depth), np.isinf(mi_depth))
            self.mi_invalid_depth_mask_list.append(invalid_depth_mask)
            mi_depth[invalid_depth_mask] = 0.
            self.mi_depth_list.append(mi_depth)

            mi_normal_global = ret.n.numpy().reshape(self._H(frame_idx), self._W(frame_idx), 3)
            # FLIP inverted normals!
            normals_flip_mask = np.logical_and(np.sum(rays_d * mi_normal_global, axis=-1) > 0, np.any(mi_normal_global != np.inf, axis=-1))
            mi_normal_global[normals_flip_mask] = -mi_normal_global[normals_flip_mask]
            mi_normal_global[invalid_depth_mask, :] = 0.
            self.mi_normal_global_list.append(mi_normal_global)

            mi_normal_cam_opencv = mi_normal_global @ self.pose_list[frame_idx][:3, :3]
            self.mi_normal_opencv_list.append(mi_normal_cam_opencv)
            mi_normal_cam_opengl = np.stack([mi_normal_cam_opencv[:, :, 0], -mi_normal_cam_opencv[:, :, 1], -mi_normal_cam_opencv[:, :, 2]], axis=-1) # transform normals from OpenGL convention (right-up-backward) to OpenCV (right-down-forward)
            mi_normal_cam_opengl[invalid_depth_mask, :] = 0.
            self.mi_normal_opengl_list.append(mi_normal_cam_opengl)

            mi_pts = ret.p.numpy()
            # mi_pts = ret.t.numpy()[:, np.newaxis] * rays_d_flatten + rays_o_flatten # should be the same as above
            if np.all(ret.t.numpy()==np.inf):
                print(white_red('no rays hit any surface! (frame_id %d, idx %d)'%(self.frame_id_list[frame_idx], frame_idx)))
            # assert np.amax(np.abs((mi_pts - ret.p.numpy())[ret.t.numpy()!=np.inf, :])) < 1e-3 # except in window areas
            mi_pts = mi_pts.reshape(self._H(frame_idx), self._W(frame_idx), 3)
            mi_pts[invalid_depth_mask, :] = 0.

            self.mi_pts_list.append(mi_pts)

        print(green_text('DONE. [mi_sample_rays_pts] for %d frames...'%len(cam_rays_list)))

    def load_monosdf_scene(self):
        shape_file = Path(self.monosdf_shape_dict['shape_file'])
        if_shape_normalized = self.monosdf_shape_dict['_shape_normalized'] == 'normalized'
        (self.monosdf_scale, self.monosdf_offset), self.monosdf_scale_mat = load_monosdf_scale_offset(Path(self.monosdf_shape_dict['camera_file']))
        # self.mi_scene = mi.load_file(str(self.xml_file))
        '''
        [!!!] transform to XML scene coords (scale & location) so that ray intersection for GT geometry does not have to adapt to ESTIMATED geometry
        '''
        shape_id_dict = {
            'type': shape_file.suffix[1:],
            'filename': str(shape_file), 
            # 'to_world': mi.ScalarTransform4f.scale([1./scale]*3).translate((-offset).flatten().tolist()),
            }
        if if_shape_normalized:
            # un-normalize to regular Mitsuba scene space
            shape_id_dict['to_world'] = mi.ScalarTransform4f.translate((-self.monosdf_offset).flatten().tolist()).scale([1./self.monosdf_scale]*3)
            
        self.mi_scene = mi.load_dict({
            'type': 'scene',
            'shape_id': shape_id_dict, 
        })

    def load_monosdf_shape(self, shape_params_dict: dict):
        '''
        load a single shape estimated from MonoSDF: images/demo_shapes_monosdf.png
        '''
        if_shape_normalized = self.monosdf_shape_dict['_shape_normalized'] == 'normalized'
        if if_shape_normalized:
            scale_offset_tuple, _ = load_monosdf_scale_offset(Path(self.monosdf_shape_dict['camera_file']))
        else:
            scale_offset_tuple = ()
        monosdf_shape_dict = load_shape_dict_from_shape_file(Path(self.monosdf_shape_dict['shape_file']), shape_params_dict, scale_offset_tuple)
        self.append_shape(monosdf_shape_dict)

    def append_shape(self, shape_dict):
        self.vertices_list.append(shape_dict['vertices'])
        self.faces_list.append(shape_dict['faces'])
        self.bverts_list.append(shape_dict['bverts'])
        self.bfaces_list.append(shape_dict['bfaces'])
        self.ids_list.append(shape_dict['_id'])
        
        self.shape_list_valid.append(shape_dict['shape_dict'])

        self.xyz_max = np.maximum(np.amax(shape_dict['vertices'], axis=0), self.xyz_max)
        self.xyz_min = np.minimum(np.amin(shape_dict['vertices'], axis=0), self.xyz_min)


    def mi_get_segs(self, if_also_dump_xml_with_lit_area_lights_only=True, if_dump=True, if_seg_emitter=True):
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
            if_get_from_scratch = True
            if mi_seg_area_file_path.exists():
                expected_shape = self.im_HW_load_list[frame_idx] if hasattr(self, 'im_HW_load_list') else self.im_HW_load
                mi_seg_area = load_img(mi_seg_area_file_path, expected_shape=expected_shape, ext='png', target_HW=self.im_HW_target, if_attempt_load=True)
                if mi_seg_area is None:
                    mi_seg_area_file_path.unlink()
                else:
                    print(yellow_text('loading mi_seg_emitter from'), '%s'%str(mi_seg_area_file_folder))
                    if_get_from_scratch = False
                    mi_seg_area = (mi_seg_area / 255.).astype(bool)

            if if_get_from_scratch and if_seg_emitter:
                mi_seg_area = np.array([[s is not None and s.emitter() is not None for s in ret.shape]]).reshape(self._H(frame_idx), self._W(frame_idx))
                imageio.imwrite(str(mi_seg_area_file_path), (mi_seg_area*255.).astype(np.uint8))
                print(green_text('[mi_get_segs] mi_seg_area -> %s'%str(mi_seg_area_file_path)))
    
            if not if_seg_emitter:
                mi_seg_area = np.zeros_like(mi_seg_env, dtype=bool)
            self.mi_seg_dict_of_lists['area'].append(mi_seg_area) # lit-up lamps

            mi_seg_obj = np.logical_and(np.logical_not(mi_seg_area), np.logical_not(mi_seg_env))
            self.mi_seg_dict_of_lists['obj'].append(mi_seg_obj) # non-emitter objects

        print(green_text('DONE. [mi_get_segs] for %d frames...'%len(self.mi_rays_ret_list)))

    def sample_poses(self, sample_pose_num: int, extra_transform: np.ndarray=None, invalid_normal_thres=-1):
        '''
        sample and write poses to OpenRooms convention (e.g. pose_format == 'OpenRooms': cam.txt)
        
        invalid_normal_threshold: if pixels of invalid normals exceed this thres, discard pose
        '''
        from lib.utils_mitsubaScene_sample_poses import mitsubaScene_sample_poses_one_scene
        assert self.axis_up == 'y+', 'not supporting other axes for now'
        if not self.if_loaded_layout: self.load_layout()

        lverts = self.layout_box_3d_transformed
        boxes = [[bverts, bfaces, shape['id']] for bverts, bfaces, shape in zip(self.bverts_list, self.bfaces_list, self.shape_list_valid) \
            if (not shape['is_layout'] and np.all(np.amax(bverts, axis=0)-np.amin(bverts, axis=0)>1e-2))] # discard thin objects
        cads = [[vertices, faces] for vertices, faces, shape in zip(self.vertices_list, self.faces_list, self.shape_list_valid) if not shape['is_layout']]

        assert sample_pose_num is not None
        assert sample_pose_num > 0
        self.cam_params_dict['samplePoint'] = sample_pose_num

        for _tmp_folder in ['mi_seg_emitter']:
            tmp_folder = self.scene_rendering_path / _tmp_folder
            if tmp_folder.exists():
                shutil.rmtree(str(tmp_folder))
        
        origin_lookat_up_list = mitsubaScene_sample_poses_one_scene(
            mitsubaScene=self, 
            scene_dict={
                'lverts': lverts, 
                'boxes': boxes, 
                'cads': cads, 
            }, 
            cam_params_dict=self.cam_params_dict, 
            path_dict={},
        ) # [pointLoc; target; up]

        pose_list = []
        origin_lookatvector_up_list = []
        for cam_param in origin_lookat_up_list:
            origin, lookat, up = np.split(cam_param.T, 3, axis=1)
            (R, t), lookatvector = origin_lookat_up_to_R_t(origin, lookat, up)
            pose_list.append(np.hstack((R, t)))
            origin_lookatvector_up_list.append((origin.reshape((3, 1)), lookatvector.reshape((3, 1)), up.reshape((3, 1))))

        # self.pose_list = pose_list[:sample_pose_num]
        # return

        H, W = self.im_H_load//4, self.im_W_load//4
        scale_factor = [t / s for t, s in zip((H, W), (self.im_H_load, self.im_W_load))]
        K = resize_intrinsics(self.K, scale_factor)
        tmp_cam_rays_list = self.get_cam_rays_list(H, W, [K]*len(pose_list), pose_list)
        normal_costs = []
        depth_costs = []
        normal_list = []
        valid_normal_mask_list = []
        depth_list = []
        print(white_blue('Rendering depth/normals for %d candidate poses... could be slow...'%len(tmp_cam_rays_list)))
        for _, (rays_o, rays_d, ray_d_center) in tqdm(enumerate(tmp_cam_rays_list)):
            rays_o_flatten, rays_d_flatten = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)

            xs_mi = mi.Point3f(self.to_d(rays_o_flatten))
            ds_mi = mi.Vector3f(self.to_d(rays_d_flatten))
            rays_mi = mi.Ray3f(xs_mi, ds_mi)
            ret = self.mi_scene.ray_intersect(rays_mi) # https://mitsuba.readthedocs.io/en/stable/src/api_reference.html?highlight=write_ply#mitsuba.Scene.ray_intersect
            rays_v_flatten = ret.t.numpy()[:, np.newaxis] * rays_d_flatten

            mi_depth = np.sum(rays_v_flatten.reshape(H, W, 3) * ray_d_center.reshape(1, 1, 3), axis=-1)
            invalid_depth_mask = np.logical_or(np.isnan(mi_depth), np.isinf(mi_depth))
            mi_depth[invalid_depth_mask] = 0
            depth_list.append(mi_depth)

            mi_normal = ret.n.numpy().reshape(H, W, 3)
            invalid_normal_mask = np.abs(np.linalg.norm(mi_normal, axis=2) - np.ones((H, W), dtype=np.float32)) > 1e-4
            invalid_normal_mask == np.logical_or(invalid_normal_mask, invalid_depth_mask)
            mi_normal[invalid_depth_mask, :] = 0
            normal_list.append(mi_normal)
            valid_normal_mask_list.append(~invalid_normal_mask)

            mi_normal = mi_normal.astype(np.float32)
            if np.sum(invalid_normal_mask) > (H*W*0.1):
                ncost = -100000. # <10% pixels with valid normals
            else:    
                mi_normal_gradx = np.abs(mi_normal[:, 1:] - mi_normal[:, 0:-1])[~invalid_normal_mask[:, 1:]]
                mi_normal_grady = np.abs(mi_normal[1:, :] - mi_normal[0:-1, :])[~invalid_normal_mask[1:, :]]
                ncost = (np.mean(mi_normal_gradx) + np.mean(mi_normal_grady)) / 2.
                if np.isnan(ncost): import ipdb; ipdb.set_trace()
                assert not np.isnan(ncost)
        
            # dcost = np.mean(np.log(mi_depth + 1)[~invalid_depth_mask])
            # assert not np.isnan(dcost)

            normal_costs.append(ncost)
            # depth_costs.append(dcost)

        normal_costs = np.array(normal_costs, dtype=np.float32)
        # depth_costs = np.array(depth_costs, dtype=np.float32)
        # normal_costs = (normal_costs - normal_costs.min()) \
        #         / (normal_costs.max() - normal_costs.min())
        # depth_costs = (depth_costs - depth_costs.min()) \
        #         / (depth_costs.max() - depth_costs.min())
        # totalCosts = normal_costs + 0.3 * depth_costs
        totalCosts = normal_costs
        camIndex = np.argsort(totalCosts)[::-1][:sample_pose_num]

        tmp_rendering_path = Path(self.scene_rendering_path) / 'tmp_sample_poses_rendering'
        if tmp_rendering_path.exists(): shutil.rmtree(str(tmp_rendering_path), ignore_errors=True)
        tmp_rendering_path.mkdir(parents=True, exist_ok=True)
        print(blue_text('Dumping tmp normal and depth by Mitsuba: %s')%str(tmp_rendering_path))
        for _, i in enumerate(camIndex):
            imageio.imwrite(str(tmp_rendering_path / ('normal_%04d.png'%_)), (np.clip((normal_list[i] + 1.)/2., 0., 1.)*255.).astype(np.uint8))
            imageio.imwrite(str(tmp_rendering_path / ('valid_normal_mask_%04d.png'%_)), (valid_normal_mask_list[i].astype(np.float32)*255.).astype(np.uint8))
            imageio.imwrite(str(tmp_rendering_path / ('depth_%04d.png'%_)), (np.clip(depth_list[i] / np.amax(depth_list[i]+1e-6), 0., 1.)*255.).astype(np.uint8))
            print(_, i, 'min depth:', np.amin(depth_list[i][depth_list[i]>0]), 'cost:', totalCosts[i])
        print(blue_text('DONE.'))
        # print(normal_costs[camIndex])

        self.pose_list = [pose_list[i] for i in camIndex]
        self.origin_lookatvector_up_list = [origin_lookatvector_up_list[i] for i in camIndex]
        self.origin_lookat_up_list = [origin_lookat_up_list[i] for i in camIndex]
        self.frame_id_list = list(range(len(self.pose_list)))

        print(blue_text('Sampled '), white_blue(str(len(self.pose_list))), blue_text('poses.'))

        dump_cam_params_OR(pose_file_root=self.pose_file.parent, origin_lookat_up_mtx_list=self.origin_lookat_up_list, cam_params_dict=self.cam_params_dict, extra_transform=extra_transform)

    def load_meta_json_pose(self, pose_file):
        assert Path(pose_file).exists(), 'Pose file not found at %s! Check if exist, or if the correct pose format was chosen in key \'pose_file\' of scene_obj.'%str(pose_file)
        with open(pose_file, 'r') as f:
            meta = json.load(f)
        Rt_c2w_b_list = []
        for idx in range(len(meta['frames'])):
            pose = np.array(meta['frames'][idx]['transform_matrix'])[:3, :4].astype(np.float32)
            R_, t_ = np.split(pose, (3,), axis=1)
            R_ = R_ / np.linalg.norm(R_, axis=1, keepdims=True) # somehow R_ was mistakenly scaled by scale_m2b; need to recover to det(R)=1
            Rt_c2w_b_list.append((R_, t_))
        return meta, Rt_c2w_b_list

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
                normal = self.mi_normal_opengl_list[frame_idx]
            else:
                normal = self.normal_list[frame_idx]
            if subsample_HW_rates != ():
                normal = normal[::subsample_HW_rates[0], ::subsample_HW_rates[1]]
            normal = normal.reshape(-1, 3)[X_flatten_mask]
            normal = np.stack([normal[:, 0], -normal[:, 1], -normal[:, 2]], axis=-1) # transform normals from OpenGL convention (right-up-backward) to OpenCV (right-down-forward)
            normal_global = (R @ normal.T).T
            normal_global_list.append(normal_global)

        print(blue_text('[%s] DONE. fuse_3D_geometry'%self.parent_class_name))

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
            normal_list = self.mi_normal_opengl_list
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

        print(blue_text('[%s] DONE. fuse_3D_lighting'%self.parent_class_name))

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

    def load_colors(self):
        '''
        load mapping from obj cat id to RGB
        '''
        self.if_loaded_colors = False
        return

    def _prepare_shapes(self):
        '''
        base function: prepare for shape lists/dicts
        '''
        self.shape_list_valid = []
        self.vertices_list = []
        self.faces_list = []
        self.ids_list = []
        self.bverts_list = []
        self.bfaces_list = []

        self.window_list = []
        self.lamp_list = []
        self.xyz_max = np.zeros(3,)-np.inf
        self.xyz_min = np.zeros(3,)+np.inf

    def export_poses_cam_txt(self, export_folder: Path, cam_params_dict={}, frame_num_all=-1):
        print(white_blue('[%s] convert poses to OpenRooms format')%self.parent_class_name)
        T_list_ = [(None, '')]
        if self.extra_transform is not None:
            T_list_.append((self.extra_transform_inv, '_extra_transform'))
            # R = R_.T
            # t = -R_.T @ t_

            # RR = np.array([[1., 0., 0.], [0., -1., 0.], [0., 0., -1]])
            # tt = np.zeros((3, 1))
            # from lib.utils_OR.utils_OR_cam import R_t_to_origin_lookatvector_up
            # (origin, lookatvector, up) = R_t_to_origin_lookatvector_up(RR, tt)
            # print((origin, lookatvector, up))

        for T_, appendix in T_list_:
            if T_ is not None:
                origin_lookat_up_mtx_list = [np.hstack((T_@_[0], T_@_[1]+T_@_[0], T_@_[2])).T for _ in self.origin_lookatvector_up_list]
            else:
                origin_lookat_up_mtx_list = [np.hstack((_[0], _[1]+_[0], _[2])).T for _ in self.origin_lookatvector_up_list]
            Rt_list = []
            # T_opengl = np.array([[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]], dtype=np.float32) # OpenGL -> OpenCV
            for Rt in self.pose_list:
                R, t = Rt[:3, :3], Rt[:3, 3:4]
                if T_ is not None:
                    # R = R @ T_
                    R = T_ @ R
                    t = T_ @ t
                    # R = R @ T_opengl @ T_
                    # t = T_ @ T_opengl @ t
                    # R = R.T
                    # t = -R @ t
                    # print('+++++++++', R, t)

                Rt_list.append((R, t))

            if appendix != '':
                # debug: two prints should agree
                (origin, lookatvector, up) = R_t_to_origin_lookatvector_up(Rt_list[0][0], Rt_list[0][1])
                print((origin.flatten(), lookatvector.flatten(), up.flatten()))
                print((T_@self.origin_lookatvector_up_list[0][0]).flatten(), (T_@self.origin_lookatvector_up_list[0][1]).flatten(), (T_@self.origin_lookatvector_up_list[0][2]).flatten())
            dump_cam_params_OR(
                pose_file_root=export_folder, 
                origin_lookat_up_mtx_list=origin_lookat_up_mtx_list, Rt_list=Rt_list, 
                cam_params_dict=cam_params_dict, K_list=self.K_list, frame_num_all=frame_num_all, appendix=appendix)

    def dump_mi_meshes(self, mi_scene, mesh_dump_root: Path):
        '''
        dump mi scene objects as separate objects
        '''
        if mesh_dump_root.exists(): shutil.rmtree(str(mesh_dump_root))
        mesh_dump_root.mkdir(parents=True, exist_ok=True)

        for shape_idx, shape, in enumerate(mi_scene.shapes()):
            if not isinstance(shape, mi.llvm_ad_rgb.Mesh): continue
            shape.write_ply(str(mesh_dump_root / ('%06d.ply'%shape_idx)))
        print(blue_text('Scene shapes dumped to: %s')%str(mesh_dump_root))

    def export_scene(self, modality_list=[], appendix='', split='', if_force=False):
        '''
        export scene to mitsubaScene data structure + monosdf inputs
        '''
        scene_export_path = self.rendering_root / 'scene_export' / (self.scene_name + appendix)
        if split != '':
            scene_export_path = scene_export_path / split
        if scene_export_path.exists():
            if if_force:
                if_reexport = 'Y'
                print(red('scene export path exists:%s . FORCE overwritten.'%str(scene_export_path)))
            else:
                if_reexport = input(red("scene export path exists:%s . Re-export? [y/n]"%str(scene_export_path)))

            if if_reexport in ['y', 'Y']:
                if not if_force:
                    if_delete = input(red("Delete? [y/n]"))
                else:
                    if_delete = 'Y'
                if if_delete in ['y', 'Y']:
                    shutil.rmtree(str(scene_export_path), ignore_errors=True)
            else:
                print(red('Aborted export.'))
                return
                
        scene_export_path.mkdir(parents=True, exist_ok=True)

        for modality in modality_list:
            # assert modality in self.modality_list, 'modality %s not in %s'%(modality, self.modality_list)

            if modality == 'poses':
                self.export_poses_cam_txt(scene_export_path, cam_params_dict=self.cam_params_dict, frame_num_all=self.frame_num_all)

                '''
                cameras for MonoSDF
                '''
                cameras = {}
                scale_mat_path = self.rendering_root / 'scene_export' / (self.scene_name + appendix) / 'scale_mat.npy'
                if split != 'val':
                    poses = [np.vstack((pose, np.array([0., 0., 0., 1.], dtype=np.float32).reshape((1, 4)))) for pose in self.pose_list]
                    poses = np.array(poses)
                    assert poses.shape[1:] == (4, 4)
                    min_vertices = poses[:, :3, 3].min(axis=0)
                    max_vertices = poses[:, :3, 3].max(axis=0)
                    center = (min_vertices + max_vertices) / 2.
                    scale = 2. / (np.max(max_vertices - min_vertices) + 3.)
                    print('[pose normalization to unit cube] --center, scale--', center, scale)
                    # we should normalized to unit cube
                    scale_mat = np.eye(4).astype(np.float32)
                    scale_mat[:3, 3] = -center
                    scale_mat[:3 ] *= scale 
                    scale_mat = np.linalg.inv(scale_mat)
                    np.save(str(scale_mat_path), {'scale_mat': scale_mat, 'center': center, 'scale': scale})
                else:
                    assert scale_mat_path.exists()
                    scale_mat_dict = np.load(str(scale_mat_path), allow_pickle=True).item()
                    scale_mat = scale_mat_dict['scale_mat']
                    center = scale_mat_dict['center']
                    scale = scale_mat_dict['scale']

                cameras['center'] = center
                cameras['scale'] = scale
                for _, pose in enumerate(self.pose_list):
                    if hasattr(self, 'K'):
                        K = self.K 
                    else:
                        assert hasattr(self, 'K_list')
                        assert len(self.K_list) == len(self.pose_list)
                        K = self.K_list[_] # (3, 3)
                        assert K.shape == (3, 3)
                    K = np.hstack((K, np.array([0., 0., 0.], dtype=np.float32).reshape((3, 1))))
                    K = np.vstack((K, np.array([0., 0., 0., 1.], dtype=np.float32).reshape((1, 4))))
                    pose = np.vstack((pose, np.array([0., 0., 0., 1.], dtype=np.float32).reshape((1, 4))))
                    pose = K @ np.linalg.inv(pose)
                    cameras['scale_mat_%d'%_] = scale_mat
                    cameras['world_mat_%d'%_] = pose
                    # cameras['split_%d'%_] = 'train' if _ < self.frame_num-10 else 'val' # 10 frames for val
                    # cameras['frame_id_%d'%_] = idx if idx < len(mitsuba_scene_dict['train'].pose_list) else idx-len(mitsuba_scene_dict['train'].pose_list)
            
                np.savez(str(scene_export_path / 'cameras.npz'), **cameras)

            if modality == 'im_hdr':
                (scene_export_path / 'Image').mkdir(parents=True, exist_ok=True)
                for _, frame_id in enumerate(self.frame_id_list):
                    im_hdr_export_path = scene_export_path / 'Image' / ('%03d_0001.exr'%_)
                    hdr_scale = self.hdr_scale_list[_]
                    cv2.imwrite(str(im_hdr_export_path), hdr_scale * self.im_hdr_list[_][:, :, [2, 1, 0]])
                    print(blue_text('HDR image %s exported to: %s'%(frame_id, str(im_hdr_export_path))))
            
            if modality == 'im_sdr':
                (scene_export_path / 'Image').mkdir(parents=True, exist_ok=True)
                for _, frame_id in enumerate(self.frame_id_list):
                    im_sdr_export_path = scene_export_path / 'Image' / ('%03d_0001.png'%_)
                    cv2.imwrite(str(im_sdr_export_path), (np.clip(self.im_sdr_list[_][:, :, [2, 1, 0]], 0., 1.)*255.).astype(np.uint8))
                    print(blue_text('SDR image %s exported to: %s'%(frame_id, str(im_sdr_export_path))))
                    print(self.modality_file_list_dict['im_sdr'][_])
            
            if modality == 'mi_normal':
                (scene_export_path / 'MiNormalGlobal').mkdir(parents=True, exist_ok=True)
                assert self.pts_from['mi']
                for _, frame_id in enumerate(self.frame_id_list):
                    mi_normal_export_path = scene_export_path / 'MiNormalGlobal' / ('%03d_0001.png'%_)
                    cv2.imwrite(str(mi_normal_export_path), (np.clip(self.mi_normal_global_list[_][:, :, [2, 1, 0]]/2.+0.5, 0., 1.)*255.).astype(np.uint8))
                    print(blue_text('Mitsuba normal (global) %s exported to: %s'%(frame_id, str(mi_normal_export_path))))
            
            if modality == 'mi_depth':
                (scene_export_path / 'MiDepth').mkdir(parents=True, exist_ok=True)
                assert self.pts_from['mi']
                for _, frame_id in enumerate(self.frame_id_list):
                    mi_depth_vis_export_path = scene_export_path / 'MiDepth' / ('%03d_0001.png'%_)
                    mi_depth = self.mi_depth_list[_].squeeze()
                    depth_normalized, depth_min_and_scale = vis_disp_colormap(mi_depth, normalize=True, valid_mask=~self.mi_invalid_depth_mask_list[_])
                    cv2.imwrite(str(mi_depth_vis_export_path), depth_normalized)
                    print(blue_text('Mitsuba depth (vis) %s exported to: %s'%(frame_id, str(mi_depth_vis_export_path))))
                    mi_depth_npy_export_path = scene_export_path / 'MiDepth' / ('%03d_0001.npy'%_)
                    np.save(str(mi_depth_npy_export_path), mi_depth)
                    print(blue_text('Mitsuba depth (npy) %s exported to: %s'%(frame_id, str(mi_depth_npy_export_path))))

            if modality == 'im_mask':
                (scene_export_path / 'ImMask').mkdir(parents=True, exist_ok=True)
                assert self.pts_from['mi']
                if hasattr(self, 'im_mask_list'):
                    assert len(self.im_mask_list) == len(self.mi_invalid_depth_mask_list)
                for _, frame_id in enumerate(self.frame_id_list):
                    im_mask_export_path = scene_export_path / 'ImMask' / ('%03d_0001.png'%_)
                    mi_invalid_depth_mask = self.mi_invalid_depth_mask_list[_]
                    if hasattr(self, 'im_mask_list'):
                        im_mask = self.im_mask_list[_]
                        assert mi_invalid_depth_mask.shape == im_mask.shape, 'invalid depth mask shape %s not equal to im_mask shape %s'%(mi_invalid_depth_mask.shape, im_mask.shape)
                        im_mask = np.logical_and(im_mask, ~mi_invalid_depth_mask)
                        print('Exporting im_mask from invalid depth mask && loaded im_mask')
                    else:
                        im_mask = ~mi_invalid_depth_mask
                        print('Exporting im_mask from invalid depth mask only')
                    cv2.imwrite(str(im_mask_export_path), (im_mask*255).astype(np.uint8))
                    print(blue_text('Mask image (for valid depths) %s exported to: %s'%(frame_id, str(im_mask_export_path))))
            
            if modality == 'shapes': 
                assert self.if_loaded_shapes, 'shapes not loaded'
                T_list_ = [(None, '')]
                if self.extra_transform is not None:
                    T_list_.append((self.extra_transform_inv, '_extra_transform'))

                for T_, appendix in T_list_:
                    shape_list = []
                    shape_export_path = scene_export_path / ('scene%s.obj'%appendix)
                    for vertices, faces in zip(self.vertices_list, self.faces_list):
                        if T_ is not None:
                            shape_list.append(trimesh.Trimesh((T_ @ vertices.T).T, faces-1))
                        else:
                            shape_list.append(trimesh.Trimesh(vertices, faces-1))
                    shape_tri_mesh = trimesh.util.concatenate(shape_list)
                    shape_tri_mesh.export(str(shape_export_path))
                    if not shape_tri_mesh.is_watertight:
                        trimesh.repair.fill_holes(shape_tri_mesh)
                        shape_tri_mesh_convex = trimesh.convex.convex_hull(shape_tri_mesh)
                        shape_tri_mesh_convex.export(str(shape_export_path.parent / ('%s_hull%s.obj'%(shape_export_path.stem, appendix))))
                        shape_tri_mesh_fixed = trimesh.util.concatenate([shape_tri_mesh, shape_tri_mesh_convex])
                        shape_tri_mesh_fixed.export(str(shape_export_path.parent / ('%s_fixed%s.obj'%(shape_export_path.stem, appendix))))
                        print(yellow('Mesh is not watertight. Filled holes and added convex hull: -> %s%s.obj, %s_hull%s.obj, %s_fixed%s.obj'%(shape_export_path.name, appendix, shape_export_path.name, appendix, shape_export_path.name, appendix)))

            

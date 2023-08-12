import numpy as np
np.set_printoptions(suppress=True)
import pyhocon
from tqdm import tqdm
from collections import defaultdict
import torch
import time
import trimesh
from pathlib import Path
import imageio
import json
import shutil
import open3d as o3d
# Import the library using the alias "mi"
import mitsuba as mi
from lib.utils_io import load_img, resize_intrinsics, center_crop
from lib.utils_OR.utils_OR_cam import R_t_to_origin_lookatvector_up_opencv, dump_cam_params_OR, convert_OR_poses_to_blender_npy, dump_blender_npy_to_json
from lib.utils_dvgo import get_rays_np
from lib.utils_misc import get_list_of_keys, green, white_red, green_text, yellow, yellow_text, white_blue, blue_text, red, vis_disp_colormap
from lib.utils_misc import get_device
from lib.utils_OR.utils_OR_lighting import convert_lighting_axis_local_to_global_np, get_lighting_envmap_dirs_global
from lib.utils_OR.utils_OR_cam import origin_lookat_up_to_R_t

from lib.utils_monosdf_scene import dump_shape_dict_to_shape_file, load_shape_dict_from_shape_file, load_monosdf_scale_offset

from .class_scene2DBase import scene2DBase

class mitsubaBase(scene2DBase):
    '''
    Base class used to load/visualize/render Mitsuba scene from XML file
    '''
    def __init__(
        self, 
        CONF: pyhocon.config_tree.ConfigTree,  
        parent_class_name: str, # e.g. mitsubaScene3D, openroomsScene3D
        root_path_dict: dict, 
        modality_list: list, 
        host: str='', 
        device_id: int=-1, 
        if_debug_info: bool=False, 
    ): 
        
        scene2DBase.__init__(
            self, 
            CONF=CONF,
            parent_class_name=parent_class_name, 
            root_path_dict=root_path_dict, 
            modality_list=modality_list, 
            if_debug_info=if_debug_info, 
            )

        self.host = host
        self.device = get_device(self.host, device_id)
        variant = self.CONF.mi_params_dict.get('variant', '')
        from lib.global_vars import mi_variant_dict
        mi.set_variant(variant if variant != '' else mi_variant_dict[self.host])

        # self.device = device
        self.if_debug_info = if_debug_info

        # self.if_loaded_colors = False
        self.if_loaded_shapes = False
        self.if_loaded_layout = False
        self.if_has_ceilling_floor = False
        self.if_loaded_tsdf = False

        # self.extra_transform = None
        # self.extra_transform_inv = None
        # self.extra_transform_homo = None
        self.if_center_offset = True # pixel centers are 0.5, 1.5, ..., H-1+0.5

        self.shape_file_path = None
        ''''
        flags to set
        '''
        self.if_scale_scene = False # if scale scale scene with self.scene_path / 'scale.txt'
        self.if_autoscale_scene = False # if auto-scale scene as did in MonoSDF (by translating/scaling all camera centers to fit in a unit box ([-1, 1]))
        
        self.mi_scene_from = None
        
        self.pcd_color = None
        self.pts_from = {'mi': False, 'depth': False}
        self.seg_from = {'mi': False, 'seg': False}
        
        self.axis_up = get_list_of_keys(self.CONF.scene_params_dict, ['axis_up'], [str])[0]
        assert self.axis_up in ['x+', 'y+', 'z+', 'x-', 'y-', 'z-']
        self.ceiling_loc = None
        self.floor_loc = None
        
        # self.extra_transform = self.CONF.scene_params_dict.get('extra_transform', None)
        # if self.extra_transform is not None:
        #     # self.extra_transform = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.float32) # y=z, z=x, x=y
        #     self.extra_transform_inv = self.extra_transform.T
        #     self.extra_transform_homo = np.eye(4, dtype=np.float32)
        #     self.extra_transform_homo[:3, :3] = self.extra_transform
        self._R = np.eye(3, dtype=np.float32)
        self._t = np.zeros((3, 1), dtype=np.float32)
        self._s = np.ones((3, 1), dtype=np.float32)
        
        self.has_shape_file = False
        if 'shape_file' in self.CONF.scene_params_dict and self.CONF.scene_params_dict.shape_file != '':
            if len(str(self.CONF.scene_params_dict.shape_file).split('/')) == 1:
                self.shape_file_path = self.scene_path / self.CONF.scene_params_dict.shape_file # self.CONF.scene_params_dict.shape_file is a single file name
            else:
                self.shape_file_path = self.dataset_root / self.CONF.scene_params_dict.shape_file # self.CONF.scene_params_dict.shape_file is a full path
            assert self.shape_file_path.exists(), 'shape file does not exist: %s'%str(self.shape_file_path)
            self.has_shape_file = True

        self.has_tsdf_file = False
        if 'tsdf_file' in self.CONF.shape_params_dict and self.CONF.shape_params_dict.tsdf_file != '':
            if len(str(self.CONF.shape_params_dict.tsdf_file).split('/')) == 1:
                self.tsdf_file_path = self.scene_path / self.CONF.shape_params_dict.tsdf_file # self.CONF.scene_params_dict.tsdf_file is a single file name
            else:
                self.tsdf_file_path = Path(self.CONF.shape_params_dict.tsdf_file) # self.CONF.scene_params_dict.tsdf_file is a full path
            # assert self.tsdf_file_path.exists(), 'shape file does not exist: %s'%str(self.tsdf_file_path)
            self.tsdf_file_path.parent.mkdir(parents=True, exist_ok=True)
            self.has_tsdf_file = True
            
        if self.CONF.shape_params_dict.get('force_regenerate_tsdf', False) and self.tsdf_file_path.exists():
            print(yellow('Removed existing tsdf file due to CONF.shape_params_dict[\'force_regenerate_tsdf\']=True: %s'%str(self.tsdf_file_path)))
            self.tsdf_file_path.unlink()

            
    def to_d(self, x: np.ndarray):
        if 'mps' in self.device: # Mitsuba RuntimeError: Cannot pack tensors on mps:0
            return x
        return torch.from_numpy(x).to(self.device)
    
    @property
    def _T(self):
        return (self._R, self._t, self._s) # rotation, translation, scale
    
    @property
    def _if_T(self):
        return not(np.allclose(self._R, np.eye(3, dtype=np.float32)) and np.allclose(self._t, np.zeros((3, 1), dtype=np.float32)) and np.allclose(self._s, np.ones((3, 1), dtype=np.float32)))
        
    def apply_T(self, X: np.ndarray, _list=['R']):
        '''
        x: (N, 3)
        '''
        assert [_ in ['R', 't', 's'] for _ in _list]
        _R, _t, _s = self._T
        assert len(X.shape) == 2 and X.shape[1] == 3
        
        if 'R' in _list:
            X = _R @ X.T
        if 't' in _list:
            X = X + _t    
        if 's' in _list:
            X = _s * X
        
        return X.T

    @property
    def _T_homo(self):
        _R, _t, _s = self._T
        return np.vstack((np.hstack((_s*_R, _s*_t)), np.array([0., 0., 0., 1.])))
    
    def compose_T(self, R: np.ndarray=np.eye(3, dtype=np.float32), t: np.ndarray=np.zeros((3, 1), dtype=np.float32), s: np.ndarray=np.ones((3, 1), dtype=np.float32)):
        '''
        s2(R2(s1(R1x+t1)+t2)) -> s12(R12x+t12)
        '''
        assert False, 'not implemented because not needed, for now'
    #     '''
    #     s(R(_s(_Rx+_t))+t) => 
    #     '''
    #     pass

    @property
    def if_has_mitsuba_scene(self):
        return True

    @property
    def if_has_mitsuba_rays_pts(self):
        return self.CONF.mi_params_dict['if_sample_rays_pts']

    @property
    def if_has_mitsuba_segs(self):
        return self.CONF.mi_params_dict['if_get_segs']

    @property
    def if_has_mitsuba_all(self):
        return all([self.if_has_mitsuba_scene, self.if_has_mitsuba_rays_pts, self.if_has_mitsuba_segs, ])

    @property
    def if_has_tsdf(self): # objs + emitters
        return all([_ in self.modality_list for _ in ['tsdf']])
    
    @property
    def if_has_shapes(self): 
        return all([_ in self.modality_list for _ in ['shapes']])

    @property
    def if_has_pcd(self):
        return 'shapes' in self.modality_list and self.scene_params_dict.get('pcd_file', '') != ''

    @property
    def if_has_layout(self):
        return all([_ in self.modality_list for _ in ['layout']])
    
    def load_mi_scene_from_shape(self, input_extra_transform_homo: bool=None, shape_file_path: Path=None):
        if shape_file_path is None:
            shape_file_path = self.shape_file_path

        assert shape_file_path.exists()
            
        print(yellow('[%s] load_mi_scene from [file]'%self.__class__.__name__) + str(shape_file_path))
        self.shape_id_dict = {
            'type': shape_file_path.suffix[1:],
            'filename': str(shape_file_path), 
            }
        
        _T = np.eye(4, dtype=np.float32)
        if input_extra_transform_homo is not None:
            _T = input_extra_transform_homo @ _T
            
        if self._if_T and not self.CONF.scene_params_dict.get('if_reorient_y_up_skip_shape', False):
            _T = self._T_homo @ _T
                
        if not np.allclose(_T, np.eye(4, dtype=np.float32)):
            self.shape_id_dict['to_world'] = mi.ScalarTransform4f(_T)        
        
        self.mi_scene = mi.load_dict({
            'type': 'scene',
            'shape_id': self.shape_id_dict, 
        })


    def process_mi_scene(self, if_postprocess_mi_frames=True, if_seg_emitter=False, force=False):
        '''
        debug_render_test_image: render test image
        debug_dump_mesh: dump all shapes into meshes
        if_postprocess_mi_frames: for each frame, sample rays and generate segmentation maps
        '''
        
        debug_render_test_image = self.CONF.mi_params_dict.get('debug_render_test_image', False)
        if debug_render_test_image:
            '''
            images/demo_mitsuba_render.png
            '''
            test_rendering_path = self.PATH_HOME / 'mitsuba' / 'tmp_render.exr'
            print(blue_text('Rendering... test frame by Mitsuba: %s')%str(test_rendering_path))
            if self.mi_scene.integrator() is None:
                print(yellow('No integrator found in the scene. Skipped: debug_render_test_image'))
            else:
                image = mi.render(self.mi_scene, spp=16)
                mi.util.write_bitmap(str(test_rendering_path), image)
                print(blue_text('DONE.'))

        debug_dump_mesh = self.CONF.mi_params_dict.get('debug_dump_mesh', False)
        if debug_dump_mesh:
            '''
            images/demo_mitsuba_dump_meshes.png
            '''
            mesh_dump_root = self.PATH_HOME / 'mitsuba' / 'meshes_dump'
            self.dump_mi_meshes(self.mi_scene, mesh_dump_root)

        if if_postprocess_mi_frames:
            if_sample_rays_pts = self.CONF.mi_params_dict.get('if_sample_rays_pts', True)
            if if_sample_rays_pts:
                print(green_text('Sampling rays...'))
                self.mi_sample_rays_pts(self.cam_rays_list, if_force=force)
                self.pts_from['mi'] = True
            
            if_get_segs = self.CONF.mi_params_dict.get('if_get_segs', True)
            if if_get_segs:
                assert if_sample_rays_pts
                self.mi_get_segs(if_seg_emitter=if_seg_emitter)
                self.seg_from['mi'] = True

    def get_cam_rays_list(self, H_list: list, W_list: list, K_list: list, pose_list: list, convention: str='opencv'):
        assert convention in ['opengl', 'opencv']
        cam_rays_list = []
        if not isinstance(K_list, list): K_list = [K_list] * len(pose_list)
        if not isinstance(H_list, list): H_list = [H_list] * len(pose_list)
        if not isinstance(W_list, list): W_list = [W_list] * len(pose_list)
        assert len(K_list) == len(pose_list) == len(H_list) == len(W_list)
        for _, (H, W, pose, K) in enumerate(zip(H_list, W_list, pose_list, K_list)):
            rays_o, rays_d, ray_d_center = get_rays_np(H, W, K, pose, inverse_y=(convention=='opencv'), if_center_offset=self.if_center_offset)
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
            mi_depth = np.sum(rays_v_flatten.reshape(self._H(frame_idx), self._W(frame_idx), 3) * ray_d_center.reshape(1, 1, 3), axis=-1).astype(np.float32)
            invalid_depth_mask = np.logical_or(np.isnan(mi_depth), np.isinf(mi_depth))
            self.mi_invalid_depth_mask_list.append(invalid_depth_mask)
            mi_depth[invalid_depth_mask] = 0.
            self.mi_depth_list.append(mi_depth)

            mi_normal_global = ret.n.numpy().reshape(self._H(frame_idx), self._W(frame_idx), 3).astype(np.float32)
            # FLIP inverted normals!
            normals_flip_mask = np.logical_and(np.sum(rays_d * mi_normal_global, axis=-1) > 0, np.any(mi_normal_global != np.inf, axis=-1))
            if np.sum(normals_flip_mask) > 0:
                mi_normal_global[normals_flip_mask] = -mi_normal_global[normals_flip_mask]
                print(green_text('[mi_sample_rays_pts] %d normals flipped!'%np.sum(normals_flip_mask)))
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

    # def load_monosdf_scene(self):
    #     shape_file = Path(self.monosdf_shape_dict['shape_file'])
    #     if_shape_normalized = self.monosdf_shape_dict['_shape_normalized'] == 'normalized'
    #     (self.monosdf_scale, self.monosdf_offset), self.monosdf_scale_mat = load_monosdf_scale_offset(Path(self.monosdf_shape_dict['camera_file']))
    #     # self.mi_scene = mi.load_file(str(self.xml_file))
    #     '''
    #     [!!!] transform to XML scene coords (scale & location) so that ray intersection for GT geometry does not have to adapt to ESTIMATED geometry
    #     '''
    #     self.shape_id_dict = {
    #         'type': shape_file.suffix[1:],
    #         'filename': str(shape_file), 
    #         # 'to_world': mi.ScalarTransform4f.scale([1./scale]*3).translate((-offset).flatten().tolist()),
    #         }
    #     if if_shape_normalized:
    #         # un-normalize to regular Mitsuba scene space
    #         self.shape_id_dict['to_world'] = mi.ScalarTransform4f.translate((-self.monosdf_offset).flatten().tolist()).scale([1./self.monosdf_scale]*3)
            
    #     self.mi_scene = mi.load_dict({
    #         'type': 'scene',
    #         'shape_id': self.shape_id_dict, 
    #     })

    # def load_monosdf_shape(self, shape_params_dict: dict):
    #     '''
    #     load a single shape estimated from MonoSDF: images/demo_shapes_monosdf.png
    #     '''
    #     if_shape_normalized = self.monosdf_shape_dict['_shape_normalized'] == 'normalized'
    #     if if_shape_normalized:
    #         scale_offset_tuple, _ = load_monosdf_scale_offset(Path(self.monosdf_shape_dict['camera_file']))
    #     else:
    #         scale_offset_tuple = ()
    #     monosdf_shape_dict = load_shape_dict_from_shape_file(Path(self.monosdf_shape_dict['shape_file']), shape_params_dict, scale_offset_tuple)
    #     self.append_shape(monosdf_shape_dict)

    def append_shape(self, shape_dict):
        self.vertices_list.append(shape_dict['vertices'])
        self.faces_list.append(shape_dict['faces'])
        self.bverts_list.append(shape_dict['bverts'])
        self.bfaces_list.append(shape_dict['bfaces'])
        self.shape_ids_list.append(shape_dict['_id'])
        
        self.shape_list_valid.append(shape_dict['shape_dict'])

        self.xyz_max = np.maximum(np.amax(shape_dict['vertices'], axis=0), self.xyz_max)
        self.xyz_min = np.minimum(np.amin(shape_dict['vertices'], axis=0), self.xyz_min)


    def mi_get_segs(self, if_seg_emitter=True):
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
            mi_seg_area_file_folder = self.scene_rendering_path_list[frame_idx] / 'mi_seg_emitter'
            mi_seg_area_file_folder.mkdir(parents=True, exist_ok=True)
            mi_seg_area_file_path = mi_seg_area_file_folder / ('mi_seg_emitter_%d.png'%(self.frame_id_list[frame_idx]))
            if_get_from_scratch = True
            if mi_seg_area_file_path.exists():
                expected_shape = self.im_HW_load_list[frame_idx] if hasattr(self, 'im_HW_load_list') else self.im_HW_load
                mi_seg_area = load_img(mi_seg_area_file_path, expected_shape=expected_shape, ext='png', target_HW=self.im_HW_target, if_attempt_load=True)
                if mi_seg_area is None:
                    mi_seg_area_file_path.unlink()
                else:
                    print(yellow_text('loading mi_seg_emitter from '), '%s'%str(mi_seg_area_file_folder))
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

    def sample_poses(self, sample_pose_num: int, extra_transform: np.ndarray=None, invalid_normal_thres=-1, if_dump=True):
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
        self.CONF.cam_params_dict['samplePoint'] = sample_pose_num

        for _tmp_folder in ['mi_seg_emitter']:
            tmp_folder = self.scene_rendering_path / _tmp_folder
            if tmp_folder.exists():
                shutil.rmtree(str(tmp_folder), ignore_errors=True)

        tmp_rendering_path = Path(self.scene_rendering_path) / 'tmp_sample_poses_rendering'
        if tmp_rendering_path.exists(): shutil.rmtree(str(tmp_rendering_path), ignore_errors=True)
        tmp_rendering_path.mkdir(parents=True, exist_ok=True)
                
        origin_lookat_up_list = mitsubaScene_sample_poses_one_scene(
            mitsubaScene=self, 
            scene_dict={
                'lverts': lverts, 
                'boxes': boxes, 
                'cads': cads, 
            }, 
            cam_params_dict=self.CONF.cam_params_dict, 
            tmp_rendering_path=tmp_rendering_path, 
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

        if if_dump:
            dump_cam_params_OR(pose_file_root=self.pose_file_path_root, origin_lookat_up_mtx_list=self.origin_lookat_up_list, cam_params_dict=self.CONF.cam_params_dict, extra_transform=extra_transform)
            
            # Dump pose file in Blender .npy files
            npy_path = self.pose_file_path_root / ('%s.npy'%self.split)
            print('-', self.pose_list[0])
            blender_poses = convert_OR_poses_to_blender_npy(pose_list=self.pose_list, export_path=npy_path)
            
            # Dump pose file in .json format
            # json_path = pose_file_root / 'transforms.json'
            # # sampled poses should have the same K for simplicity
            # f_x = self._K()[0][0]
            # f_y = self._K()[1][1]
            # camera_angle_x = 2 * np.arctan(0.5 * self._W() / f_x)
            # camera_angle_y = 2 * np.arctan(0.5 * self._H() / f_y)
            # dump_blender_npy_to_json(blender_poses=blender_poses, export_path=json_path, camera_angle_x=camera_angle_x, camera_angle_y=camera_angle_y)
            
            print(white_blue('Dumped sampled poses (cam.txt) to') + str(self.pose_file_path_root))

    def load_meta_json_pose(self, pose_file):
        assert Path(pose_file).exists(), str(pose_file)
        # if not Path(pose_file).exists():
            # return None, []
        with open(pose_file, 'r') as f:
            meta = json.load(f)
        Rt_c2w_b_list = []
        for idx in range(len(meta['frames'])):
            pose = np.array(meta['frames'][idx]['transform_matrix'])[:3, :4].astype(np.float32)
            R_, t_ = np.split(pose, (3,), axis=1)
            R_ = R_ / np.linalg.norm(R_, axis=1, keepdims=True) # somehow R_ was mistakenly scaled by scale_m2b; need to recover to det(R)=1
            Rt_c2w_b_list.append((R_, t_))
        return meta, Rt_c2w_b_list

    def load_tsdf(self, if_use_mi_geometry: bool=True, force=False):
        '''
        get scene geometry in tsdf volume via fusing from depth maps: 
            images/demo_tsdf.png
            https://i.imgur.com/r6TET8K.jpg
        '''
        if self.if_loaded_tsdf and not force: return
        
        force_fuse = self.CONF.shape_params_dict.get('if_force_fuse_tsdf', False)
        
        if self.has_tsdf_file and self.tsdf_file_path.exists() and not force_fuse:
            print(white_blue('[%s] Loading tsdf from '%self.__class__.__name__)+str(self.tsdf_file_path))
            tsdf_mesh = trimesh.load_mesh(str(self.tsdf_file_path), process=False)
            self.tsdf_fused_dict = {'vertices': np.array(tsdf_mesh.vertices), 'faces': np.array(tsdf_mesh.faces)}
            if len(tsdf_mesh.visual.vertex_colors) > 0:
                self.tsdf_fused_dict.update({'colors': np.array(tsdf_mesh.visual.vertex_colors)[:, :3].astype(np.float32)/255.})
        else:
            assert hasattr(self, 'tsdf_file_path')
            if force_fuse:
                print(yellow('[%s] force_fuse TSDF. skip loading tsdf from file.'%self.__class__.__name__))
            self.tsdf_fused_dict = self._fuse_tsdf(if_use_mi_geometry=if_use_mi_geometry, dump_path=self.tsdf_file_path)
        
        if self._if_T: # reorient
            if not self.CONF.scene_params_dict.get('if_reorient_y_up_skip_shape', False):
                self.tsdf_fused_dict.update({'vertices': self.apply_T(self.tsdf_fused_dict['vertices'], ['R', 't', 's'])})
    
        self.if_loaded_tsdf = True
        print(blue_text('[%s] DONE. load_tsdf. vertices: %d, faces: %d'%(self.__class__.__name__, self.tsdf_fused_dict['vertices'].shape[0], self.tsdf_fused_dict['faces'].shape[0])))

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
    
    def _fuse_tsdf(self, subsample_rate_pts: int=1, subsample_HW_rates: tuple=(1, 1), if_use_mi_geometry: bool=True, dump_path: Path=None):
        '''
        fuse TSDF volume from depth maps (adapted from the code by Bohan Yu (MILO paper)):
            images/demo_tsdf.png
            https://i.imgur.com/r6TET8K.jpg
        '''
        assert self.if_has_poses
        assert self.if_has_im_sdr, 'Need to load im_sdr_list for TSDF fusion'
        if not if_use_mi_geometry:
            assert self.if_has_depth_normal
        
        print(white_blue('[mitsubaBase] fuse_tsdf '), yellow('[use Mitsuba: %s]'%str(if_use_mi_geometry)), \
            'for %d frames... H %d W %d, subsample_rate_pts: %d, subsample_HW_rates: (%d, %d)'%(len(self.frame_id_list), self._H(), self._W(), subsample_rate_pts, subsample_HW_rates[0], subsample_HW_rates[1]))

        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=8.0 / 512.0,
            # voxel_length=20.0 / 512.0,
            sdf_trunc=0.05,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
            volume_unit_resolution=16,
            depth_sampling_stride=1
        )
            # voxel_length=5.0 / 512.0,
            # sdf_trunc=0.2,
            # color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        # )
        intrinsic = o3d.camera.PinholeCameraIntrinsic(self._W(), self._H(), self._K()[0][0], self._K()[1][1], self._K()[0][2], self._K()[1][2])
        poses = []
        T_opengl_opencv = np.array([[-1., 0., 0.], [0., -1., 0.], [0., 0., 1.]], dtype=np.float32) # flip x, y: Liwen's new pose (left-up-forward) -> OpenCV (right-down-forward)
        
        for p in range(5):
            for frame_idx, frame_id in tqdm(enumerate(self.frame_id_list)):
            # for i, frame in enumerate(meta['frames']):
                input_image = self.im_sdr_list[frame_idx]
                input_depth = self.mi_depth_list[frame_idx] if if_use_mi_geometry else self.depth_list[frame_idx]
                if hasattr(self, 'im_mask_list'):
                    depth_mask = self.im_mask_list[frame_idx]
                    input_depth[~depth_mask] = np.inf
                rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    o3d.geometry.Image(np.clip(input_image[::-1, ::-1] * 255, 0, 255).astype(np.uint8)),
                    o3d.geometry.Image(input_depth[::-1, ::-1].copy()),
                    depth_scale=1.0,
                    depth_trunc=10.0,
                    convert_rgb_to_intensity=False)
                extrinsic = np.vstack((np.hstack([self.pose_list[frame_idx][:3, :3] @ T_opengl_opencv, self.pose_list[frame_idx][:3, 3:4]]), np.array([0, 0, 0, 1])))
                # print(f"extrinsic {extrinsic}")
                volume.integrate(rgbd_image, intrinsic, np.linalg.inv(extrinsic))
                if p == 0:
                    curr_pose = np.concatenate([
                        np.dot(extrinsic, np.array([0.0, 0.0, 0.0, 1.0]))[:3],
                        np.dot(extrinsic, np.array([0.0, 0.0, 1.0, 1.0]))[:3],
                        np.dot(extrinsic, np.array([0.0, 1.0, 0.0, 0.0]))[:3],
                    ]).astype(np.float32)
                    # print(f"curr_pose {curr_pose}")
                    poses.append(curr_pose)
        # np.save(os.path.join(dataset_path, "poses.npy"), poses)
        tsdf_mesh_o3d = volume.extract_triangle_mesh()
        if dump_path is not None:
            o3d.io.write_triangle_mesh(str(dump_path), tsdf_mesh_o3d, False, True)
            print(f"Fused TSDF dumped mesh to {dump_path}")
        else:
            print(f"Dump path is None, not dumping TSDF mesh")
            
        print(blue_text('[%s] DONE. fuse_tsdf'%self.parent_class_name))

        tsdf_fused_dict = {'tsdf_mesh_o3d': tsdf_mesh_o3d, 'vertices': np.asarray(tsdf_mesh_o3d.vertices), 'faces': np.asarray(tsdf_mesh_o3d.triangles)}
        if tsdf_mesh_o3d.has_vertex_colors:
            tsdf_fused_dict.update({'colors': np.asarray(tsdf_mesh_o3d.vertex_colors)}) # (N, 3), [0., 1.]
        
        return tsdf_fused_dict

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

    def _init_shape_vars(self):
        '''
        base function: prepare for shape lists/dicts
        '''
        self.shape_list_valid = []
        self.vertices_list = []
        self.faces_list = []
        self.shape_ids_list = []
        self.bverts_list = []
        self.bfaces_list = []

        self.window_list = []
        self.lamp_list = []
        self.xyz_max = np.zeros(3,)-np.inf
        self.xyz_min = np.zeros(3,)+np.inf
    
    def load_single_shape(self, shape_file_path: Path=None, shape_params_dict={}, extra_transform=None, force=False):
        '''
        load and visualize shapes (objs/furniture **& emitters**) in 3D & 2D: 
        '''
        if self.if_loaded_shapes and not force: return
        
        if shape_file_path is None:
            shape_file_path = self.shape_file_path
        
        print(white_blue('[%s] load_single_shape for scene...'%self.parent_class_name), yellow('single'))

        mitsubaBase._init_shape_vars(self)

        scale_offset = () if not self.if_scale_scene else (self.scene_scale, 0.)
        shape_dict = load_shape_dict_from_shape_file(shape_file_path, shape_params_dict=shape_params_dict, scale_offset=scale_offset, extra_transform=extra_transform)
        # , scale_offset=(9.1, 0.)) # read scale.txt and resize room to metric scale in meters
        self.append_shape(shape_dict)

        self.if_loaded_shapes = True
        self.shape_ids_list = [0]
        
        print(blue_text('[%s] DONE. load_single_shape.'%(self.parent_class_name)))

        if shape_params_dict.get('if_dump_shape', False):
            dump_shape_dict_to_shape_file(shape_dict, shape_file_path)

    def export_poses_cam_txt(self, export_folder: Path, cam_params_dict={}, frame_num_all=-1):
        print(white_blue('[%s] convert poses to OpenRooms format')%self.parent_class_name)
        T_list_ = [(None, '')]
        # if self.extra_transform is not None:
        #     T_list_.append((self.extra_transform_inv, '_extra_transform'))

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
                (origin, lookatvector, up) = R_t_to_origin_lookatvector_up_opencv(Rt_list[0][0], Rt_list[0][1])
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
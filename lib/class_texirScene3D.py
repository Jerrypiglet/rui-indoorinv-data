from pathlib import Path, PosixPath
import matplotlib.pyplot as plt
import numpy as np
import trimesh
np.set_printoptions(suppress=True)

from tqdm import tqdm
import scipy
import shutil
from lib.global_vars import mi_variant_dict
import random
random.seed(0)
from lib.utils_OR.utils_OR_cam import R_t_to_origin_lookatvector_up_yUP, origin_lookat_up_to_R_t, read_cam_params_OR, normalize_v
import json
from lib.utils_io import load_matrix, load_img, convert_write_png
# from collections import defaultdict
# import trimesh
import imageio
import string
# Import the library using the alias "mi"
import mitsuba as mi

from lib.utils_misc import blue_text, yellow, get_list_of_keys, white_blue, magenta
from lib.utils_io import load_matrix, resize_intrinsics

# from .class_openroomsScene2D import openroomsScene2D
from .class_mitsubaBase import mitsubaBase
from .class_scene2DBase import scene2DBase

from lib.utils_OR.utils_OR_mesh import minimum_bounding_rectangle, sample_mesh, simplify_mesh
from lib.utils_OR.utils_OR_xml import get_XML_root
from lib.utils_OR.utils_OR_mesh import loadMesh, computeBox, get_rectangle_mesh
from lib.utils_misc import get_device

from .class_scene2DBase import scene2DBase

class texirScene3D(mitsubaBase, scene2DBase):
    '''
    A class used to visualize/render texir scenes captured by Mustafa
    '''
    def __init__(
        self, 
        root_path_dict: dict, 
        scene_params_dict: dict, 
        modality_list: list, 
        modality_filename_dict: dict, 
        im_params_dict: dict={'im_H_load': 480, 'im_W_load': 640, 'im_H_resize': 480, 'im_W_resize': 640, 'spp': 1024}, 
        cam_params_dict: dict={}, 
        BRDF_params_dict: dict={}, 
        lighting_params_dict: dict={}, # params to load & convert lighting SG & envmap to 
        shape_params_dict: dict={'if_load_mesh': True}, 
        emitter_params_dict: dict={},
        mi_params_dict: dict={'if_sample_rays_pts': True}, 
        if_debug_info: bool=False, 
        host: str='', 
        device_id: int=-1, 
    ):
        scene2DBase.__init__(
            self, 
            parent_class_name=str(self.__class__.__name__), 
            root_path_dict=root_path_dict, 
            scene_params_dict=scene_params_dict, 
            modality_list=modality_list, 
            modality_filename_dict=modality_filename_dict, 
            im_params_dict=im_params_dict, 
            cam_params_dict=cam_params_dict, 
            BRDF_params_dict=BRDF_params_dict, 
            lighting_params_dict=lighting_params_dict, 
            if_debug_info=if_debug_info, 
            )
        
        self.host = host
        self.device = get_device(self.host, device_id)
        mitsubaBase.__init__(
            self, 
            device = self.device, 
        )

        self.scene_name, self.frame_id_list_input, self.axis_up = get_list_of_keys(scene_params_dict, ['scene_name', 'frame_id_list', 'axis_up'], [str, list, str])
        self.invalid_frame_id_list = scene_params_dict.get('invalid_frame_id_list', [])
        self.frame_id_list_input = [_ for _ in self.frame_id_list_input if _ not in self.invalid_frame_id_list]
        
        self.indexing_based = scene_params_dict.get('indexing_based', 0)
        
        self.extra_transform = self.scene_params_dict.get('extra_transform', None)
        if self.extra_transform is not None:
            self.extra_transform_inv = self.extra_transform.T
            self.extra_transform_homo = np.eye(4, dtype=np.float32)
            self.extra_transform_homo[:3, :3] = self.extra_transform
            
        self.if_autoscale_scene = False

        self.scene_path = self.rendering_root / self.scene_name
        self.scene_rendering_path = self.rendering_root / self.scene_name
        self.scene_rendering_path.mkdir(parents=True, exist_ok=True)
        self.scene_name_full = self.scene_name # e.g. 'main_xml_scene0008_00_more'

        self.pose_format, pose_file = scene_params_dict['pose_file']
        assert self.pose_format in ['json', 'bundle'], 'Unsupported pose file: '+self.pose_file
        self.pose_file = self.scene_path / pose_file
        
        self.shape_file = ''
        if 'shape_file' in scene_params_dict:
            self.shape_file = self.scene_path / scene_params_dict['shape_file']

        self.shape_params_dict = shape_params_dict
        self.mi_params_dict = mi_params_dict
        variant = mi_params_dict.get('variant', '')
        mi.set_variant(variant if variant != '' else mi_variant_dict[self.host])

        self.pcd_color = None
        self.if_loaded_pcd = False
        self.near = cam_params_dict.get('near', 0.1)
        self.far = cam_params_dict.get('far', 10.)

        self.load_poses()
        self.scene_rendering_path_list = [self.scene_rendering_path] * len(self.frame_id_list)
        ''''
        flags to set
        '''
        self.pts_from = {'mi': False, 'depth': False}
        self.seg_from = {'mi': False, 'seg': False}
        
        '''
        load everything
        '''

        self.load_mi_scene(self.mi_params_dict, monosdf_scale_tuple=())
        self.load_modalities()

        if hasattr(self, 'pose_list'): 
            self.get_cam_rays(self.cam_params_dict)
        self.process_mi_scene(self.mi_params_dict, if_postprocess_mi_frames=hasattr(self, 'pose_list'))

    @property
    def frame_num(self):
        return len(self.frame_id_list)

    @property
    def frame_num_all(self):
        return len(self.frame_id_list)

    @property
    def valid_modalities(self):
        return [
            'poses', 
            'shapes', 
            'im_hdr', 'im_sdr', 
            ]

    @property
    def if_has_poses(self):
        return hasattr(self, 'pose_list')

    @property
    def if_has_shapes(self): # objs + emitters
        return 'shapes' in self.modality_list and self.scene_params_dict.get('shape_file', '') != ''

    @property
    def if_has_pcd(self):
        return 'shapes' in self.modality_list and self.scene_params_dict.get('pcd_file', '') != ''

    @property
    def if_has_mitsuba_scene(self):
        return True

    @property
    def if_has_mitsuba_rays_pts(self):
        return self.mi_params_dict['if_sample_rays_pts']

    @property
    def if_has_mitsuba_segs(self):
        return self.mi_params_dict['if_get_segs']

    @property
    def if_has_seg(self):
        return False, 'Segs not saved to labels. Use mi_seg_area, mi_seg_env, mi_seg_obj instead.'
        # return all([_ in self.modality_list for _ in ['seg']])

    @property
    def if_has_mitsuba_all(self):
        return all([self.if_has_mitsuba_scene, self.if_has_mitsuba_rays_pts, self.if_has_mitsuba_segs, ])

    @property
    def if_has_colors(self): # no semantic label colors
        return False

    def load_modalities(self):
        for _ in self.modality_list:
            result_ = scene2DBase.load_modality_(self, _)
            if not (result_ == False):
                continue
            if _ == 'shapes': self.load_shapes(self.shape_params_dict) # shapes of 1(i.e. furniture) + emitters

    def get_modality(self, modality, source: str='GT'):
        _ = scene2DBase.get_modality_(self, modality, source)
        if _ is not None:
            return _

        if 'mi_' in modality:
            assert self.pts_from['mi']

        if modality == 'mi_depth': 
            return self.mi_depth_list
        elif modality == 'mi_normal': 
            return self.mi_normal_global_list
        elif modality in ['mi_seg_area', 'mi_seg_env', 'mi_seg_obj']:
            seg_key = modality.split('_')[-1] 
            return self.mi_seg_dict_of_lists[seg_key] # Set scene_obj->mi_params_dict={'if_get_segs': True
        else:
            assert False, 'Unsupported modality: ' + modality

    def load_mi_scene(self, mi_params_dict={}, monosdf_scale_tuple=(), extra_transform_homo=None):
        '''
        load scene representation into Mitsuba 3
        '''
        if self.shape_file == '':
            print(yellow('No shape file specified. Skip loading MI scene.'))
            return
        print(yellow('Loading MI scene from shape file: ' + str(self.shape_file)))
        shape_file = Path(self.shape_file)
        assert shape_file.exists(), 'Shape file not found: ' + str(shape_file)
        shape_id_dict = {
            'type': shape_file.suffix[1:],
            'filename': str(shape_file), 
            }
        _T = np.eye(4, dtype=np.float32)
        if self.extra_transform is not None:
            # shape_id_dict['to_world'] = mi.ScalarTransform4f(self.extra_transform_homo)
            _T = self.extra_transform_homo @ _T
        if monosdf_scale_tuple != ():
            # assert self.extra_transform is None
            center, scale = monosdf_scale_tuple
            scale_mat = np.eye(4).astype(np.float32)
            scale_mat[:3, 3] = -center
            scale_mat[:3 ] *= scale 
            _T = scale_mat @ _T
            # shape_id_dict['to_world'] = mi.ScalarTransform4f(scale_mat)
        if extra_transform_homo is not None:
            # shape_id_dict['to_world'] = mi.ScalarTransform4f(self.extra_transform_homo)
            _T = extra_transform_homo @ _T
            
        if not np.allclose(_T, np.eye(4, dtype=np.float32)):
            shape_id_dict['to_world'] = mi.ScalarTransform4f(_T)

        self.mi_scene = mi.load_dict({
            'type': 'scene',
            'shape_id': shape_id_dict, 
        })

    def process_mi_scene(self, mi_params_dict={}, if_postprocess_mi_frames=True, force=False):
        debug_render_test_image = mi_params_dict.get('debug_render_test_image', False)
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

        debug_dump_mesh = mi_params_dict.get('debug_dump_mesh', False)
        if debug_dump_mesh:
            '''
            images/demo_mitsuba_dump_meshes.png
            '''
            mesh_dump_root = self.PATH_HOME / 'mitsuba' / 'meshes_dump'
            self.dump_mi_meshes(self.mi_scene, mesh_dump_root)

        if if_postprocess_mi_frames:
            if_sample_rays_pts = mi_params_dict.get('if_sample_rays_pts', True)
            if if_sample_rays_pts:
                self.mi_sample_rays_pts(self.cam_rays_list, if_force=force)
                self.pts_from['mi'] = True
            
            if_get_segs = mi_params_dict.get('if_get_segs', True)
            if if_get_segs:
                assert if_sample_rays_pts
                self.mi_get_segs(if_also_dump_xml_with_lit_area_lights_only=True)
                self.seg_from['mi'] = True
                
    def load_poses(self):
        print(white_blue('[%s] load_poses from %s'%(self.parent_class_name, str(self.pose_file))))

        self.pose_list = []
        self.K_list = []
        self.origin_lookatvector_up_list = []
        
        assert self.pose_format == 'json'
        meta_file_path = self.scene_path / 'transforms.json'
        assert meta_file_path.exists(), 'No meta file found: ' + str(meta_file_path)
        with open(str(meta_file_path), 'r') as f:
            meta = json.load(f)
            
        self.frame_id_list = []
        for idx in range(len(meta['frames'])):
            file_path = meta['frames'][idx]['file_path']
            frame_id = int(file_path.split('/')[-1].split('.')[0].split('_')[0])
            self.frame_id_list.append(frame_id)
        print('self.frame_id_list ALL in meta file:', self.frame_id_list)
            
        # fl_x, fl_y, cx, cy, w, h, camera_model = get_list_of_keys(meta, ['fl_x', 'fl_y', 'cx', 'cy', 'w', 'h', 'camera_model'], [float, float, float, float, int, int, str])
        # assert camera_model == 'OPENCV'
        f_xy = 0.5*self.im_W_load/np.tan(0.5*meta['camera_angle_x']) # original focal length - x
        K = np.array([[f_xy, 0, self.im_W_load/2.], [0, f_xy, self.im_H_load/2.], [0, 0, 1]], dtype=np.float32)
        
        if self.im_W_load != self.W or self.im_H_load != self.H:
            scale_factor = [t / s for t, s in zip((self.H, self.W), self.im_HW_load)]
            K = resize_intrinsics(K, scale_factor)

        self.K_list = [K] * len(self.frame_id_list)
        
        for frame_idx in range(len(meta['frames'])):
            c2w = np.array(meta['frames'][frame_idx]['transform_matrix']).astype(np.float32)
            
            R_, t_ = np.split(c2w[:3], (3,), axis=1)
            R = R_; t = t_
            R = R @ np.array([[-1., 0., 0.], [0., -1., 0.], [0., 0., 1.]], dtype=np.float32) # Fixed some rushed sh*t [TODO]
            if self.extra_transform is not None:
                assert self.extra_transform.shape == (3, 3) # [TODO] support 4x4
                R = self.extra_transform[:3, :3] @ R
                t = self.extra_transform[:3, :3] @ t
            self.pose_list.append(np.hstack((R, t)))

            assert np.isclose(np.linalg.det(R), 1.0), 'R is not a rotation matrix'
            
            origin = t
            lookatvector = R @ np.array([[0.], [0.], [1.]], dtype=np.float32)
            up = R @ np.array([[0.], [-1.], [0.]], dtype=np.float32)
            self.origin_lookatvector_up_list.append((origin.reshape((3, 1)), lookatvector.reshape((3, 1)), up.reshape((3, 1))))

        assert len(set(self.frame_id_list)) == len(self.frame_id_list), 'frame_id_list is not unique'
        
        self.frame_id_list_input_all = self.frame_id_list.copy()
        
        if self.frame_id_list_input != []:
            '''
            select a subset of poses
            '''
            assert all([frame_id in self.frame_id_list for frame_id in self.frame_id_list_input])
            self.pose_list = [self.pose_list[self.frame_id_list.index(frame_id)] for frame_id in self.frame_id_list_input]
            self.origin_lookatvector_up_list = [self.origin_lookatvector_up_list[self.frame_id_list.index(frame_id)] for frame_id in self.frame_id_list_input]
            self.K_list = [self.K_list[self.frame_id_list.index(frame_id)] for frame_id in self.frame_id_list_input]
            self.frame_id_list = self.frame_id_list_input
            
        print('frame_id_list:', self.frame_id_list)
        # print(self.pose_list)

        print(blue_text('[%s] DONE. load_poses (%d poses)'%(self.__class__.__name__, len(self.pose_list))))
            
    def get_cam_rays(self, cam_params_dict={}, force=False):
        if hasattr(self, 'cam_rays_list') and not force:  return
        self.cam_rays_list = self.get_cam_rays_list(self.H, self.W, self.K_list, self.pose_list, convention='opencv')

    def load_shapes(self, shape_params_dict={}):
        '''
        load and visualize shapes (objs/furniture **& emitters**) / point cloud (pcd)
        '''
        
        print(white_blue('[%s] load_shapes for scene...')%self.__class__.__name__)
        
        if self.if_loaded_shapes: 
            print('already loaded shapes. skip.')
            return
        mitsubaBase._prepare_shapes(self)
        assert self.shape_file.exists(), 'No shape file found: ' + str(self.shape_file)
        self.load_single_shape(shape_params_dict, extra_transform=self.extra_transform)
            
        self.if_loaded_shapes = True
        print(blue_text('[%s] DONE. load_shapes'%(self.__class__.__name__)))
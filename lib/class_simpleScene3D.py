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
from lib.utils_OR.utils_OR_cam import origin_lookat_up_to_R_t, read_cam_params_OR, normalize_v
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

class simpleScene3D(mitsubaBase, scene2DBase):
    '''
    A class used to visualize/render vanilla Mitsuba scenes (OpenCV cameras from cam.txt)
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

        self.scene_path = self.rendering_root / self.scene_name
        self.scene_rendering_path = self.rendering_root / self.scene_name
        self.scene_rendering_path.mkdir(parents=True, exist_ok=True)
        self.scene_name_full = self.scene_name # e.g. 'main_xml_scene0008_00_more'

        self.pose_format, pose_file = scene_params_dict['pose_file']
        assert self.pose_format in ['OpenRooms'], 'Unsupported pose file: '+self.pose_file
        self.pose_file = self.scene_path / pose_file

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
        self.pts_from = {'mi': False}
        self.seg_from = {'mi': False}
        
        '''
        load everything
        '''

        self.load_mi_scene(self.mi_params_dict)
        self.load_modalities()

        if hasattr(self, 'pose_list'): 
            self.get_cam_rays(self.cam_params_dict)
        if hasattr(self, 'mi_scene'):
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

    def load_mi_scene(self, mi_params_dict={}):
        '''
        load scene representation into Mitsuba 3
        '''
        if self.scene_params_dict.get('shape_file', '') == '':
            print(yellow('No shape file specified. Skip loading MI scene.'))
            return
        print(yellow('Loading MI scene from shape file: ' + str(self.scene_params_dict['shape_file'])))
        shape_file = Path(self.scene_params_dict['shape_file'])
        shape_id_dict = {
            'type': shape_file.suffix[1:],
            'filename': str(shape_file), 
            }
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
        
        assert self.pose_format == 'OpenRooms'
        assert self.pose_file.exists(), 'No meta file found: ' + str(self.pose_file)
        
        '''
        OpenRooms convention (matrices containing origin, lookat, up); The camera coordinates is in OpenCV convention (right-down-forward).
        '''
        cam_params = read_cam_params_OR(self.pose_file)
        self.frame_id_list = list(range(len(cam_params)))
        self.frame_id_list = [_ for _ in self.frame_id_list if _ not in self.invalid_frame_id_list]
        assert all([cam_param.shape == (3, 3) for cam_param in cam_params])

        self.pose_list
        self.origin_lookatvector_up_list
        
        for idx in self.frame_id_list:
            cam_param = cam_params[idx]
            origin, lookat, up = np.split(cam_param.T, 3, axis=1)
            (R, t), lookatvector = origin_lookat_up_to_R_t(origin, lookat, up)
            self.pose_list.append(np.hstack((R, t)))
            self.origin_lookatvector_up_list.append((origin.reshape((3, 1)), lookatvector.reshape((3, 1)), up.reshape((3, 1))))
                
        assert len(set(self.frame_id_list)) == len(self.frame_id_list), 'frame_id_list is not unique'
        
        K_list_file = self.scene_rendering_path / 'K_list.txt'
        assert K_list_file.exists(), str(K_list_file)
        
        with open(str(K_list_file), 'r') as K_in:
            K_list_data = K_in.read().splitlines()
        cam_num = int(K_list_data[0])
        assert cam_num == len(self.frame_id_list)
        self.K_list = np.array([x.split(' ') for x in K_list_data[1:]]).astype(np.float32)
        if not np.any(self.K_list): return []
        assert self.K_list.shape[0] == cam_num * 3
        self.K_list = np.split(self.K_list, cam_num, axis=0) # [[origin, lookat, up], ...]
        
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

        print(blue_text('[%s] DONE. load_poses (%d poses)'%(self.__class__.__name__, len(self.pose_list))))
            
    def get_cam_rays(self, cam_params_dict={}, force=False):
        if hasattr(self, 'cam_rays_list') and not force:  return
        self.cam_rays_list = self.get_cam_rays_list(self.H, self.W, self.K_list, self.pose_list, convention='opencv')

    def load_shapes(self, shape_params_dict={}):
        '''
        load and visualize shapes (objs/furniture **& emitters**) / point cloud (pcd)
        '''
        
        print(white_blue('[%s] load_shapes for scene...')%self.__class__.__name__)
        
        if self.scene_params_dict.get('shape_file', '') != '':
            if self.if_loaded_shapes: 
                print('already loaded shapes. skip.')
                return
            mitsubaBase._prepare_shapes(self)
            self.shape_file = self.scene_params_dict['shape_file']
            self.load_single_shape(shape_params_dict)
                
            self.if_loaded_shapes = True
            print(blue_text('[%s] DONE. load_shapes'%(self.__class__.__name__)))

        elif self.scene_params_dict.get('pcd_file', '') != '':
            if self.if_loaded_pcd: return
            pcd_file = self.scene_path / self.scene_params_dict['pcd_file']
            assert pcd_file.exists(), 'No pcd file found: ' + str(pcd_file)
            pcd_trimesh = trimesh.load_mesh(str(pcd_file), process=False)
            self.pcd = np.array(pcd_trimesh.vertices)

            self.xyz_max = np.amax(self.pcd, axis=0)
            self.xyz_min = np.amin(self.pcd, axis=0)
            self.if_loaded_pcd = True
            
            print(blue_text('[%s] DONE. load_pcd: %d points'%(self.__class__.__name__, self.pcd.shape[0])))
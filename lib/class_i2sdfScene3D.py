from pathlib import Path, PosixPath
import matplotlib.pyplot as plt
import numpy as np
import trimesh
np.set_printoptions(suppress=True)
import pyhocon
from tqdm import tqdm
import scipy
import random
random.seed(0)
from lib.utils_OR.utils_OR_cam import R_t_to_origin_lookatvector_up_opencv
from lib.utils_io import load_matrix, load_img
import mitsuba as mi

from lib.utils_misc import blue_text, yellow, get_list_of_keys, white_blue, magenta, red, get_device, check_nd_array_list_identical, white_red
from lib.utils_io import load_matrix, resize_intrinsics
from lib.utils_OR.utils_OR_xml import xml_rotation_to_matrix_homo

# from .class_openroomsScene2D import openroomsScene2D
from .class_mitsubaBase import mitsubaBase
from .class_scene2DBase import scene2DBase

from lib.utils_OR.utils_OR_mesh import sample_mesh, simplify_mesh
from lib.utils_OR.utils_OR_xml import get_XML_root
from lib.utils_OR.utils_OR_mesh import computeBox, get_rectangle_mesh
from lib.utils_from_monosdf import rend_util

from .class_scene2DBase import scene2DBase

class i2sdfScene3D(mitsubaBase, scene2DBase):
    '''
    A class used to load scenes from I2-SDF (https://github.com/jingsenzhu/i2-sdf/blob/main/DATA_CONVENTION.md)
    '''
    def __init__(
        self, 
        CONF: pyhocon.config_tree.ConfigTree,  
        root_path_dict: dict, 
        modality_list: list, 
        if_debug_info: bool=False, 
        host: str='', 
        device_id: int=-1, 
    ):
        
        mitsubaBase.__init__(
            self, 
            CONF=CONF, 
            host=host, 
            device_id=device_id, 
            parent_class_name=str(self.__class__.__name__), 
            root_path_dict=root_path_dict, 
            modality_list=modality_list, 
            if_debug_info=if_debug_info, 
        )

        '''
        scene params and frames
        '''
        
        self.frame_id_list_input = get_list_of_keys(self.CONF.scene_params_dict, ['frame_id_list'], [list])[0]
        assert self.split in ['train', 'val']
        self.invalid_frame_id_list = self.CONF.scene_params_dict.get('invalid_frame_id_list', [])
        self.frame_id_list_input = [_ for _ in self.frame_id_list_input if _ not in self.invalid_frame_id_list]
        
        self.near = self.CONF.cam_params_dict.get('near', 0.1)
        self.far = self.CONF.cam_params_dict.get('far', 3.)

        '''
        paths for: intrinsics, xml, pose, shape
        '''
        self.pose_format, pose_file = self.CONF.scene_params_dict['pose_file'].split('-')
        assert self.pose_format in ['npz'], 'Unsupported pose file: '+pose_file
        self.pose_file_path = self.scene_path / pose_file

        '''
        load everything
        '''

        self.load_mi_scene()

        if 'poses' in self.modality_list:
            self.load_poses() # attempt to generate poses indicated in self.CONF.cam_params_dict
            
        if hasattr(self, 'pose_list'): 
            self.get_cam_rays()
        if hasattr(self, 'mi_scene'):
            self.process_mi_scene(if_postprocess_mi_frames=hasattr(self, 'pose_list'), if_seg_emitter=False) # we have no idea of the emitter due to no XML scene file

        self.load_modalities()

    @property
    def frame_num(self):
        return len(self.frame_id_list)
    
    @property
    def frame_num_all(self):
        return len(self.frame_id_list)
    
    @property
    def scene_path(self):
        scene_path = self.dataset_root / self.scene_name if self.split == 'train' else self.dataset_root / self.scene_name / 'val'
        assert scene_path.exists(), 'scene path does not exist: %s'%str(scene_path)
        return scene_path

    @property
    def scene_rendering_path(self):
        return self.scene_path
    
    @property
    def scene_rendering_path_list(self):
        return [self.scene_path] * self.frame_num
    
    @property
    def if_has_pcd(self):
        # return 'shapes' in self.modality_list and self.CONF.scene_params_dict.get('pcd_file', '') != ''
        # [TODO] coming up once provided
        return False

    @property
    def valid_modalities(self):
        return [
            'im_hdr', 'im_sdr', 
            'poses', 
            'kd', 
            'ks', 
            'roughness', 
            'depth', 
            'normal', 
            'im_mask', 
            'tsdf', 
            'shapes', # from tsdf shape
            'layout', # from tsdf shape
            # 'emission', 
            ]


    @property
    def if_has_emission(self):
        # return hasattr(self, 'emission_list')
        return False

    def load_modalities(self):
        for _ in self.modality_list:
            result_ = scene2DBase.load_modality_(self, _)
            if not (result_ == False):
                continue
            if _ == 'depth': self.load_depth()
            if _ == 'normal': self.load_normal()
            if _ == 'ks': self.load_albedo('ks')
            if _ == 'kd': self.load_albedo('kd')
            if _ == 'roughness': self.load_roughness()
            if _ == 'shapes': self.load_shapes()
            if _ == 'tsdf': self.load_tsdf(if_use_mi_geometry=False)
            if _ == 'layout': self.load_layout()

    def get_modality(self, modality, source: str='GT'):

        _ = scene2DBase.get_modality_(self, modality, source)
        if _ is not None:
            return _

        if modality == 'kd': 
            return self.kd_list
        elif modality == 'ks': 
            return self.ks_list
        elif modality == 'mi_depth': 
            return self.mi_depth_list
        elif modality == 'mi_normal': 
            return self.mi_normal_global_list
        elif modality in ['mi_seg_area', 'mi_seg_env', 'mi_seg_obj']:
            seg_key = modality.split('_')[-1] 
            return self.mi_seg_dict_of_lists[seg_key] # Set scene_obj->self.CONF.mi_params_dict={'if_get_segs': True
        else:
            assert False, 'Unsupported modality: ' + modality

    def load_poses(self):
        '''
        saved poses in a similar fashion to MonoSDF
        https://github.com/jingsenzhu/i2-sdf/blob/main/DATA_CONVENTION.md#camera-information
        '''
        print(white_blue('[%s] load_poses from '%self.__class__.__name__)+str(self.pose_file_path))

        self.pose_list = []
        # self.K_list = []
        self.origin_lookatvector_up_list = []
        
        assert self.pose_format == 'npz'
        # self.pose_file_path = self.scene_path / 'transforms.json'
        assert self.pose_file_path.exists(), 'No meta file found: ' + str(self.pose_file_path)
        
        camera_dict = np.load(self.pose_file_path)
        self.frame_id_list = [int(_.replace('world_mat_', '')) for _ in camera_dict.files]
            
        if self.invalid_frame_id_list != []:
            _N = len(self.frame_id_list)
            self.frame_id_list = [x for x in self.frame_id_list if x not in self.invalid_frame_id_list]
            # print('Invalid frame id list: %s'%str(self.invalid_frame_id_list)
            print(magenta('FIRSTLY, removed %d invalid frames with invalid_frame_id_list'%(_N - len(self.frame_id_list))))
            
        for frame_idx, frame_id in enumerate(self.frame_id_list):
            print('frame_idx: %d, frame_id: %d'%(frame_idx, frame_id))
            
        world_mats = [camera_dict['world_mat_%d'%frame_id].astype(np.float32) for frame_id in self.frame_id_list]
        scale_mats = [None] * len(world_mats) # [TODO] acquire this!
        assert len(world_mats) == len(self.frame_id_list)
        
        self.K_list = []
        self.pose_list = []
        self.origin_lookatvector_up_list = []

        for scale_mat, world_mat in zip(scale_mats, world_mats):
            P = world_mat
            P = P[:3, :4]
            intrinsics, pose = rend_util.load_K_Rt_from_P(None, P)
            assert pose.shape in ((4, 4), (3, 4))
            assert intrinsics.shape in ((4, 4), (3, 4), (3, 3))
            
            assert abs(intrinsics[0][2]*2 - self.im_W_load) < 1., 'intrinsics->H/2. (%.2f) does not match self.im_W_load: (%d); resize intrinsics needed?'%(intrinsics[0][2]*2, self.im_W_load)
            assert abs(intrinsics[1][2]*2 - self.im_H_load) < 1., 'intrinsics->W/2. (%.2f) does not match self.im_H_load: (%d); resize intrinsics needed?'%(intrinsics[1][2]*2, self.im_H_load)
            
            self.K_list.append(intrinsics.astype(np.float32)[:3, :3])
            self.pose_list.append(pose.astype(np.float32)[:3, :4])

            R = pose[:3, :3].astype(np.float32)
            t = pose[:3, 3:4].astype(np.float32)
            assert np.abs(np.linalg.det(R) - 1.) < 1e-5

            (origin, lookatvector, up) = R_t_to_origin_lookatvector_up_opencv(R, t)
            self.origin_lookatvector_up_list.append((origin.reshape((3, 1)), lookatvector.reshape((3, 1)), up.reshape((3, 1))))

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
        assert check_nd_array_list_identical(self.K_list)
        self.K = self.K_list[0]

        print(blue_text('[%s] DONE. load_poses (%d poses)'%(self.__class__.__name__, len(self.pose_list))))
        
        if self._if_T: # reorient
            self.pose_list = [np.hstack((self._R @ pose[:3, :3], self.apply_T(pose[:3, 3:4].T, ['R', 't', 's']).T)) for pose in self.pose_list]
            self.origin_lookatvector_up_list = [(self._R @ origin, self.apply_T(lookatvector.T, ['R', 't', 's']).T, self._R @ up) \
                for (origin, lookatvector, up) in self.origin_lookatvector_up_list]
            
    def load_mi_scene(self, input_extra_transform_homo=None, prioritize_load_tsdf_shape: bool=True):
        '''
        load scene representation into Mitsuba 3
        '''
        
        shape_file_path = ''
        if prioritize_load_tsdf_shape and self.tsdf_file_path.exists():
            shape_file_path = self.tsdf_file_path
        elif self.has_shape_file:
            shape_file_path = self.shape_file_path

        if shape_file_path != '':
            self.load_mi_scene_from_shape(input_extra_transform_homo=input_extra_transform_homo, shape_file_path=shape_file_path)
        else:
            # xml file always exists for Mitsuba scenes
            # self.mi_scene = mi.load_file(str(self.xml_file_path))
            print(white_red('No shape file specified/found (e.g. if you have just dumped the TSDF shape, simply run again). Skip loading MI scene.'))
            return

    def load_shapes(self, prioritize_load_tsdf_shape: bool=True):
        '''
        load and visualize shapes (from MonoSDF optimized shapes, or from TSDF shape fused from depth images)
        '''
        
        print(white_blue('[%s] load_shapes for scene...')%self.__class__.__name__)
        
        if self.if_loaded_shapes: 
            print('already loaded shapes. skip.')
            return
        
        shape_file_path = ''
        if prioritize_load_tsdf_shape and self.tsdf_file_path.exists():
            shape_file_path = self.tsdf_file_path
        else:
            assert self.has_shape_file
            shape_file_path = self.shape_file_path
            
        mitsubaBase._init_shape_vars(self)
        self.load_single_shape(shape_file_path, shape_params_dict=self.CONF.shape_params_dict)
            
        self.if_loaded_shapes = True
        print(blue_text('[%s] DONE. load_shapes'%(self.__class__.__name__)))
        
        if prioritize_load_tsdf_shape: # manually turn off loaded tsdf to avoid duplicates
            self.if_loaded_tsdf = False
            self.tsdf_fused_dict = {}

        if self._if_T: # reorient
            if not self.CONF.scene_params_dict.get('if_reorient_y_up_skip_shape', False):
                self.vertices_list = [self.apply_T(vertices, ['R', 't', 's']) for vertices in self.vertices_list]
                # self.vertices_list = [(self.reorient_transform @ vertices.T).T for vertices in self.vertices_list]
                self.bverts_list = [computeBox(vertices)[0] for vertices in self.vertices_list] # recompute bounding boxes

    def get_cam_rays(self):
        if hasattr(self, 'cam_rays_list'):  return
        self.cam_rays_list = self.get_cam_rays_list(self.H, self.W, [self.K]*len(self.pose_list), self.pose_list, convention='opencv')
            
    def load_albedo(self, albedo_type='kd'):
        '''
        albedo; loaded in [0., 1.] HDR
        (H, W, 3), [0., 1.]
        '''
        if hasattr(self, '%s_list'%albedo_type): return
        assert albedo_type in ('kd', 'ks')

        print(white_blue('[%s] load_albedo (%s) for %d frames...'%(self.__class__.__name__, albedo_type, len(self.frame_id_list))))

        albedo_file_list = [self.scene_rendering_path_list[frame_idx] / (self.CONF.modality_filename_dict[albedo_type]%frame_id) for frame_idx, frame_id in enumerate(self.frame_id_list)]
        expected_shape_list = [self.im_HW_load_list[_]+(3,) for _ in self.frame_id_list] if hasattr(self, 'im_HW_load_list') else [self.im_HW_load+(3,)]*self.frame_num
        albedo_list = [load_img(albedo_file, expected_shape=__, ext='exr', target_HW=self.im_HW_target).astype(np.float32) for albedo_file, __ in zip(albedo_file_list, expected_shape_list)]
        if albedo_type == 'kd':
            self.kd_list = albedo_list
        else:
            self.ks_list = albedo_list
        
        print(blue_text('[%s] DONE. load_albedo (%s)'%(self.__class__.__name__, albedo_type)))

    def load_depth(self):
        '''
        depth;
        (H, W), ideally in [0., inf]
        '''
        if hasattr(self, 'depth_list'): return

        print(white_blue('[%s] load_depth for %d frames...'%(self.__class__.__name__, len(self.frame_id_list))))

        self.depth_file_list = [self.scene_rendering_path_list[frame_idx] / (self.CONF.modality_filename_dict['depth']%frame_id) for frame_idx, frame_id in enumerate(self.frame_id_list)]
        expected_shape_list = [self.im_HW_load_list[_]+(3,) for _ in self.frame_id_list] if hasattr(self, 'im_HW_load_list') else [self.im_HW_load+(3,)]*self.frame_num
        self.depth_list = [load_img(depth_file, expected_shape=__, ext='exr', target_HW=self.im_HW_target).astype(np.float32)[:, :, 0] for depth_file, __ in zip(self.depth_file_list, expected_shape_list)] # -> [-1., 1.], pointing inward (i.e. notebooks/images/openrooms_normals.jpg)

        print(blue_text('[%s] DONE. load_depth'%self.__class__.__name__))

        self.pts_from['depth'] = True

    def load_normal(self):
        '''
        normal, in camera coordinates (OpenGL convention: right-up-backward);
        (H, W, 3), [-1., 1.]
        [!!!] provided normals are not acrually fully normalized... see pics below
        '''
        if hasattr(self, 'normal_list'): return

        print(white_blue('[%s] load_normal for %d frames...'%(self.__class__.__name__, len(self.frame_id_list))))

        self.normal_file_list = [self.scene_rendering_path_list[frame_idx] / (self.CONF.modality_filename_dict['normal']%frame_id) for frame_idx, frame_id in enumerate(self.frame_id_list)]
        expected_shape_list = [self.im_HW_load_list[_]+(3,) for _ in self.frame_id_list] if hasattr(self, 'im_HW_load_list') else [self.im_HW_load+(3,)]*self.frame_num
        self.normal_list = [load_img(normal_file, expected_shape=__, ext='exr', target_HW=self.im_HW_target).astype(np.float32) for normal_file, __ in zip(self.normal_file_list, expected_shape_list)] # -> [-1., 1.], pointing inward (i.e. notebooks/images/openrooms_normals.jpg)
        
        '''
        mask out areas with invalid normals: e.g. normal norm map (61 from bedroom): ![](https://i.imgur.com/hVhluye.png), ![](https://i.imgur.com/Uk9RvaQ.png)
        '''
        self.im_mask_list_extra = [np.abs(np.linalg.norm(normal, axis=-1)-1)<1e-3 for normal in self.normal_list]
        if hasattr(self, 'im_mask_list'):
            self.im_mask_list  = [np.logical_and(mask, mask_extra) for mask, mask_extra in zip(self.im_mask_list, self.im_mask_list_extra)]
            
        self.normal_list = [normal / (np.linalg.norm(normal, axis=-1, keepdims=True)+1e-6) for normal in self.normal_list]
        
        print(blue_text('[%s] DONE. load_normal'%self.__class__.__name__))
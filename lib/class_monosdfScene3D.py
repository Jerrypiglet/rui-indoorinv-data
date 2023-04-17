from pathlib import Path, PosixPath
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(suppress=True)

from tqdm import tqdm
import scipy
import pyhocon
import shutil
from lib.global_vars import mi_variant_dict
import random
random.seed(0)
from lib.utils_OR.utils_OR_cam import R_t_to_origin_lookatvector_up_opencv, read_cam_params_OR, normalize_v
from lib.utils_io import load_img

import mitsuba as mi
from lib.utils_misc import blue_text, yellow, get_list_of_keys, white_blue, white_red

# from .class_openroomsScene2D import openroomsScene2D
from lib.class_mitsubaBase import mitsubaBase
# from lib.class_scene2DBase import scene2DBase

from lib.utils_from_monosdf import rend_util
from lib.utils_OR.utils_OR_mesh import computeBox

class monosdfScene3D(mitsubaBase):
    '''
    A class used to visualize/render scenes from MonoSDF preprocessed dataset format (e.g. scannet)
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

        pose_file = self.CONF.scene_params_dict['pose_file']
        assert Path(pose_file).suffix == '.npz', 'Unsupported pose file format: '+str(pose_file)
        self.pose_file_path = self.scene_path / pose_file
        assert self.pose_file_path.exists(), 'Pose file not exist: %s'%str(self.pose_file_path)
        
        self.scale_mat_path = self.scene_path / self.CONF.scene_params_dict['scale_mat_file']
        assert self.scale_mat_path.exists(), 'Scale mat file not exist: %s'%str(self.scale_mat_path)
        
        assert self.shape_file_path.exists(), 'Shape file not exist: %s'%str(self.shape_file_path)
        self.shape_if_normalized = False # False: scale poses back to original scale; True: keep poses in normalized scale
        self.if_normalize_shape_depth_from_pose =  self.CONF.scene_params_dict.get('if_normalize_shape_depth_from_pose', False)
            
        self.near = self.CONF.cam_params_dict.get('near', 0.1)
        self.far = self.CONF.cam_params_dict.get('far', 10.)

        '''
        load everything
        '''
        self.load_poses()

        self.load_modalities()
        # self.est = {}

        self.load_mi_scene()
        if hasattr(self, 'pose_list'): 
            self.get_cam_rays()
        if hasattr(self, 'mi_scene'):
            self.process_mi_scene(if_postprocess_mi_frames=hasattr(self, 'pose_list'), if_seg_emitter=False)

    @property
    def valid_modalities(self):
        return [
            # 'im_hdr', 
            'im_sdr', 
            # 'albedo', 
            # 'roughness', 
            'depth', 
            'normal', 
            # 'emission', 
            # 'layout', 
            'shapes', 
            # 'lighting_envmap', 
            ]

    @property
    def frame_num(self):
        return len(self.frame_id_list)

    @property
    def scene_rendering_path(self):
        return self.scene_path

    @property
    def scene_rendering_path_list(self):
        return [self.scene_path] * self.frame_num

    @property
    def if_has_poses(self):
        return hasattr(self, 'pose_list')

    @property
    def if_has_seg(self):
        return False, 'Segs not saved to labels. Use mi_seg_area, mi_seg_env, mi_seg_obj instead.'
        # return all([_ in self.modality_list for _ in ['seg']])

    @property
    def if_has_colors(self): # no semantic label colors
        return False

    def load_modalities(self):
        for _ in self.modality_list:
            result_ = mitsubaBase.load_modality_(self, _)
            if not (result_ == False):
                continue

            # if _ == 'emission': self.load_emission()
            # if _ == 'layout': self.load_layout()
            if _ == 'shapes': self.load_shapes() # shapes of 1(i.e. furniture) + emitters
            if _ == 'depth':
                import ipdb; ipdb.set_trace()

    def get_modality(self, modality, source: str='GT'):

        _ = mitsubaBase.get_modality_(self, modality, source)
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
            assert self.pts_from['mi'], 'Did you set mi_params_dict[if_get_segs]=True?'
            return self.mi_seg_dict_of_lists[seg_key]
        # elif modality == 'emission': 
        #     return self.emission_list
        else:
            assert False, 'Unsupported modality: ' + modality

    def load_mi_scene(self, input_extra_transform_homo=None):
        '''
        load scene representation into Mitsuba 3
        '''

        if self.has_shape_file:
            self.load_mi_scene_from_shape(input_extra_transform_homo=input_extra_transform_homo)
        else:
            # xml file always exists for Mitsuba scenes
            # self.mi_scene = mi.load_file(str(self.xml_file_path))
            print(white_red('No shape file specified/found. Skip loading MI scene.'))
            return

    def load_poses(self):
        '''
        pose_list: list of pose matrices (**camera-to-world** transformation), each (3, 4): [R|t] (OpenCV convention: right-down-forward)
        '''
        # self.load_intrinsics()
        if hasattr(self, 'pose_list'): return

        print(white_blue('[%s] load_poses from '%self.__class__.__name__) + str(self.pose_file_path))
         
        '''
        MonoSDF convention
        '''
        center_crop_type = self.CONF.cam_params_dict.get('center_crop_type', 'no_crop')
        camera_dict = np.load(self.pose_file_path)
        self.frame_id_list = [int(_.replace('world_mat_', '')) for _ in camera_dict.files if 'world_mat_' in _]
        
        world_mats = [camera_dict['world_mat_%d' % frame_id].astype(np.float32) for frame_id in self.frame_id_list]
        scale_mats = [camera_dict['scale_mat_%d' % frame_id].astype(np.float32) for frame_id in self.frame_id_list]
        
        scale_mat_dict = np.load(str(self.scale_mat_path), allow_pickle=True).item()
        scale_mat = scale_mat_dict['scale_mat']
        center = scale_mat_dict['center'] # (3,)
        scale = scale_mat_dict['scale'] # ()
        self._t = -center.reshape(3, 1) # (3, 1)
        self._s = scale
        
        self.K_list = []
        self.pose_list = []
        self.origin_lookatvector_up_list = []

        for scale_mat, world_mat in zip(scale_mats, world_mats):
            '''
            ipdb> scale_mat
            array([[3.0752, 0.    , 0.    , 4.2513],
                [0.    , 3.0752, 0.    , 2.3006],
                [0.    , 0.    , 3.0752, 1.1594],
                [0.    , 0.    , 0.    , 1.    ]], dtype=float32)
                
            ipdb> world_mat
            array([[-370.474 , -480.1891, -106.3876, 3152.1213],
                [ -62.6545,  105.3606, -603.4276,  888.8303],
                [   0.5237,   -0.6943,   -0.4936,   -0.127 ],
                [   0.    ,    0.    ,    0.    ,    1.    ]], dtype=float32)
            '''
            if self.shape_if_normalized or self.if_normalize_shape_depth_from_pose:
                P = world_mat @ scale_mat
            else:
                P = world_mat
            P = P[:3, :4]
            intrinsics, pose = rend_util.load_K_Rt_from_P(None, P)
            assert pose.shape in ((4, 4), (3, 4))
            assert intrinsics.shape in ((4, 4), (3, 4), (3, 3))

            # because we do resize and center crop 384x384 when using omnidata model, we need to adjust the camera intrinsic accordingly
            if center_crop_type == 'center_crop_for_replica':
                scale = 384 / 680
                offset = (1200 - 680 ) * 0.5
                intrinsics[0, 2] -= offset
                intrinsics[:2, :] *= scale
            elif center_crop_type == 'center_crop_for_tnt':
                scale = 384 / 540
                offset = (960 - 540) * 0.5
                intrinsics[0, 2] -= offset
                intrinsics[:2, :] *= scale
            elif center_crop_type == 'center_crop_for_dtu':
                scale = 384 / 1200
                offset = (1600 - 1200) * 0.5
                intrinsics[0, 2] -= offset
                intrinsics[:2, :] *= scale
            elif center_crop_type == 'padded_for_dtu':
                scale = 384 / 1200
                offset = 0
                intrinsics[0, 2] -= offset
                intrinsics[:2, :] *= scale
            elif center_crop_type == 'no_crop':  # for scannet dataset, we already adjust the camera intrinsic duing preprocessing so nothing to be done here
                pass
            else:
                raise NotImplementedError

            assert abs(intrinsics[0][2]*2 - self.im_W_load) < 1., 'intrinsics->H/2. (%.2f) does not match self.im_W_load: (%d); resize intrinsics needed?'%(intrinsics[0][2]*2, self.im_W_load)
            assert abs(intrinsics[1][2]*2 - self.im_H_load) < 1., 'intrinsics->W/2. (%.2f) does not match self.im_H_load: (%d); resize intrinsics needed?'%(intrinsics[1][2]*2, self.im_H_load)
            
            self.K_list.append(intrinsics.astype(np.float32))
            self.pose_list.append(pose.astype(np.float32))

            R = pose[:3, :3].astype(np.float32)
            t = pose[:3, 3:4].astype(np.float32)
            assert np.abs(np.linalg.det(R) - 1.) < 1e-5

            (origin, lookatvector, up) = R_t_to_origin_lookatvector_up_opencv(R, t)
            
            self.origin_lookatvector_up_list.append((origin.reshape((3, 1)), lookatvector.reshape((3, 1)), up.reshape((3, 1))))

        print(blue_text('[%s] DONE. load_poses'%self.parent_class_name))

    def get_cam_rays(self):
        self.cam_rays_list = self.get_cam_rays_list(self.H, self.W, self.K_list, self.pose_list, convention='opencv')

    def get_room_center_pose(self):
        '''
        generate a single camera, centered at room center and with identity rotation
        '''
        if not self.if_loaded_layout:
            self.load_layout()
        self.pose_list = [np.hstack((
            np.eye(3, dtype=np.float32), ((self.xyz_max+self.xyz_min)/2.).reshape(3, 1)
            ))]

    def load_depth(self):
        '''
        depth;
        (H, W), ideally in [0., inf]
        '''
        if hasattr(self, 'depth_list'): return

        print(white_blue('[%s] load_depth for %d frames...'%(self.parent_class_name, len(self.frame_id_list))))

        self.depth_file_list = [self.scene_rendering_path / (self.CONF.modality_filename_dict['depth']%i) for i in self.frame_id_list]
        self.depth_list = [load_img(depth_file, (self.im_H_load, self.im_W_load), ext='npy', target_HW=self.im_HW_target).astype(np.float32)[:, :] for depth_file in self.depth_file_list] # -> [-1., 1.], pointing inward (i.e. notebooks/images/openrooms_normals.jpg)

        if self.if_normalize_shape_depth_from_pose:
            self.depth_list = [depth * self._s for depth in self.depth_list]

        print(blue_text('[%s] DONE. load_depth')%self.parent_class_name)

        self.pts_from['depth'] = True

    def load_normal(self):
        '''
        normal, in camera coordinates (OpenCV convention: right-down-forward) [!!!] Different from mitsubaScene or openroomsScene;
        (3, H, W), [0., 1.] -> (H, W, 3), [-1., 1.]
        '''
        if hasattr(self, 'normal_list'): return

        print(white_blue('[%s] load_normal for %d frames...'%(self.parent_class_name, len(self.frame_id_list))))

        self.normal_file_list = [self.scene_rendering_path / (self.CONF.modality_filename_dict['normal']%i) for i in self.frame_id_list]
        self.normal_list = [load_img(normal_file, (self.im_H_load, self.im_W_load), ext='npy', target_HW=self.im_HW_target, npy_if_channel_first=True).astype(np.float32) for normal_file in self.normal_file_list]
        self.normal_list = [normal * 2. - 1. for normal in self.normal_list]
        
        print(blue_text('[%s] DONE. load_normal'%self.parent_class_name))

    def load_shapes(self):
        '''
        load and visualize shapes (objs/furniture **& emitters**) / point cloud (pcd)
        '''
        
        print(white_blue('[%s] load_shapes for scene...')%self.__class__.__name__)
        
        if self.has_shape_file:
            if self.if_loaded_shapes: 
                print('already loaded shapes. skip.')
                return
            mitsubaBase._init_shape_vars(self)
            # self.shape_file_path= self.CONF.scene_params_dict['shape_file']
            self.load_single_shape(shape_params_dict=self.CONF.shape_params_dict)
                
            self.if_loaded_shapes = True
            print(blue_text('[%s] DONE. load_shapes'%(self.__class__.__name__)))
            
            if self.if_normalize_shape_depth_from_pose:
                print(yellow('[%s] normalize shape from loaded center/scale...'%self.__class__.__name__))
                assert self._if_T, 'no extra transform has been found for the scene! Did you load center/scale?'
                self.vertices_list = [self.apply_T(vertices, ['R', 't', 's']) for vertices in self.vertices_list]
                self.bverts_list = [computeBox(vertices)[0] for vertices in self.vertices_list] # recompute bounding boxes

        else:
            print(white_red('[%s] No shape file found at: %s. skipped.'%(self.__class__.__name__, self.shape_file_path)))
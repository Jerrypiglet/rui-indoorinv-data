from abc import abstractmethod
from pathlib import Path, PosixPath
import matplotlib.pyplot as plt
# import imageio.v2 as imageio
import numpy as np
import cv2
import scipy.ndimage as ndimage
np.set_printoptions(suppress=True)
import os
import glob
from collections import defaultdict
from lib.utils_io import load_matrix, load_img, resize_img, load_HDR, scale_HDR, load_binary, load_h5, load_envmap, resize_intrinsics
from lib.utils_matseg import get_map_aggre_map
from lib.utils_misc import blue_text, get_list_of_keys, green, yellow, white_blue, red, check_list_of_tensors_size
from lib.utils_openrooms import load_OR_public_poses_to_Rt
from lib.utils_OR.utils_OR_lighting import convert_lighting_axis_local_to_global_np, convert_SG_angles_to_axis_local_np
from lib.utils_rendering_openrooms import renderingLayer
from tqdm import tqdm
import pickle
from lib.utils_io import load_matrix, load_img, convert_write_png


class scene2DBase():
    '''
    Base class used to load 2D per-pixel modalities
    '''
    def __init__(
        self, 
        parent_class_name: str, # e.g. mitsubaScene3D, openroomsScene3D
        root_path_dict: dict, 
        scene_params_dict: dict, 
        modality_list: list, 
        modality_filename_dict: dict, 
        im_params_dict: dict={'im_H_load': 480, 'im_W_load': 640, 'im_H_resize': 480, 'im_W_resize': 640, 'spp': 1024}, 
        BRDF_params_dict: dict={}, 
        lighting_params_dict: dict={'env_row': 120, 'env_col': 160, 'SG_num': 12, 'env_height': 16, 'env_width': 32}, # params to load & convert lighting SG & envmap to 
        # cam_params_dict: dict={'near': 0.1, 'far': 10.}, 
        # shape_params_dict: dict={'if_load_mesh': True}, 
        # emitter_params_dict: dict={'N_ambient_rep': '3SG-SkyGrd'},
        # mi_params_dict: dict={'if_sample_rays_pts': True, 'if_sample_poses': False}, 
        if_debug_info: bool=False, 
        # host: str='', 
    ):

        self.if_save_storage = scene_params_dict.get('if_save_storage', False) # set to True to enable removing duplicated renderer files (e.g. only one copy of geometry files in main, or emitter files only in main and mainDiffMat)
        self.if_debug_info = if_debug_info
        self.parent_class_name = parent_class_name

        self.root_path_dict = root_path_dict
        self.PATH_HOME, self.rendering_root, self.xml_scene_root = get_list_of_keys(
            self.root_path_dict, 
            ['PATH_HOME', 'rendering_root', 'xml_scene_root'], 
            [PosixPath, PosixPath, PosixPath]
            )

        # im params
        self.im_params_dict = im_params_dict

        self.im_H_load, self.im_W_load, self.im_H_resize, self.im_W_resize = get_list_of_keys(im_params_dict, ['im_H_load', 'im_W_load', 'im_H_resize', 'im_W_resize'])
        self.if_resize_im = (self.im_H_load, self.im_W_load) != (self.im_H_resize, self.im_W_resize) # resize modalities (exclusing lighting)
        self.im_target_HW = () if not self.if_resize_im else (self.im_H_resize, self.im_W_resize)
        self.H, self.W = self.im_H_resize, self.im_W_resize

        # lighting params
        self.lighting_params_dict = lighting_params_dict

        # dict for estimations
        self.est = {}

        # set up modalities to load
        self.modality_list = self.check_and_sort_modalities(list(set(modality_list)))

        self.modality_filename_dict = modality_filename_dict
        for modality in modality_filename_dict.keys():
            assert modality in self.valid_modalities, 'Invalid modality: %s in modality_filename_dict!'%modality

    @property
    @abstractmethod
    def valid_modalities(self):
        ...

    @property
    @abstractmethod
    def num_frames(self):
        ...

    def check_and_sort_modalities(self, modalitiy_list):
        modalitiy_list_new = [_ for _ in self.valid_modalities if _ in modalitiy_list]
        for _ in modalitiy_list_new:
            assert _ in self.valid_modalities, 'Invalid modality: %s'%_
        return modalitiy_list_new

    def get_modality_(self, modality, source: str='GT'):
        assert source in ['GT', 'EST']
        if modality == 'im_sdr': 
            return self.im_sdr_list
        elif modality == 'im_hdr': 
            return self.im_hdr_list
        elif modality == 'poses': 
            return self.pose_list
        elif modality == 'albedo': 
            return self.albedo_list
        elif modality == 'roughness': 
            return self.roughness_list
        elif modality == 'depth': 
            return self.depth_list
        elif modality == 'normal': 
            return self.normal_list
        elif modality == 'lighting_envmap': 
            return self.lighting_envmap_list if source=='GT' else self.est[modality]
        else:
            return None
            # assert False, 'Unsupported modality: ' + modality

    def load_modality_(self, modality):

        if modality == 'im_sdr': self.load_im_sdr(); return True
        if modality == 'im_hdr': self.load_im_hdr(); return True
        if modality == 'poses': self.load_poses(); return True
        if modality == 'albedo': self.load_albedo(); return True
        if modality == 'roughness': self.load_roughness(); return True
        if modality == 'depth': self.load_depth(); return True
        if modality == 'normal': self.load_normal(); return True
        if modality == 'lighting_envmap': self.load_lighting_envmap(); return True

        return False

    def add_modality(self, x, modality: str, source: str='GT'):
        assert source in ['GT', 'EST']
        assert modality in self.valid_modalities
        if source == 'EST':
            self.est[modality] = x
            if modality in self.modality_list:
                assert type(x)==type(self.get_modality(modality, 'GT'))
                if isinstance(x, list):
                    assert len(x) == len(self.get_modality(modality, 'GT'))
        elif source == 'GT':
            setattr(self, modality, x)
            if self.get_modality(modality, 'EST') is not None:
                assert type(x)==type(self.get_modality(modality, 'EST'))
                if isinstance(x, list):
                    assert len(x) == len(self.get_modality(modality, 'EST'))

    @property
    def if_has_im_sdr(self):
        return hasattr(self, 'im_sdr_list')

    @property
    def if_has_im_hdr(self):
        return hasattr(self, 'im_hdr_list')

    @property
    def if_has_lighting_envmap(self):
        return hasattr(self, 'lighting_envmap_list')

    @property
    def if_has_depth_normal(self):
        return all([_ in self.modality_list for _ in ['depth', 'normal']])

    @property
    def if_has_BRDF(self):
        return all([_ in self.modality_list for _ in ['albedo', 'roughness']])

    def load_im_sdr(self):
        '''
        load im in SDR; RGB, (H, W, 3), [0., 1.]
        '''
        print(white_blue('[%s] load_im_sdr')%self.parent_class_name)

        filename = self.modality_filename_dict['im_sdr']
        im_sdr_ext = filename.split('.')[-1]

        self.im_sdr_file_list = [self.scene_rendering_path / (filename%frame_id) for frame_id in self.frame_id_list]
        self.im_sdr_list = [load_img(_, expected_shape=(self.im_H_load, self.im_W_load, 3), ext=im_sdr_ext, target_HW=self.im_target_HW)/255. for _ in self.im_sdr_file_list]

        print(blue_text('[%s] DONE. load_im_sdr')%self.parent_class_name)

    def load_im_hdr(self):
        '''
        load im in HDR; RGB, (H, W, 3), [0., 1.]
        '''
        print(white_blue('[%s] load_im_hdr'%self.parent_class_name))

        filename = self.modality_filename_dict['im_hdr']
        im_hdr_ext = filename.split('.')[-1]
        im_sdr_ext = self.modality_filename_dict['im_sdr'].split('.')[-1]

        self.im_hdr_file_list = [self.scene_rendering_path / (filename%frame_id) for frame_id in self.frame_id_list]
        self.im_hdr_list = [load_img(_, expected_shape=(self.im_H_load, self.im_W_load, 3), ext=im_hdr_ext, target_HW=self.im_target_HW) for _ in self.im_hdr_file_list]
        self.hdr_scale_list = [1.] * len(self.im_hdr_list)

        for im_hdr_file, im_hdr in zip(self.im_hdr_file_list, self.im_hdr_list):
            im_sdr_file = Path(str(im_hdr_file).replace(im_hdr_ext, im_sdr_ext))
            if not im_sdr_file.exists():
                print(yellow('[%s] load_im_hdr: converting HDR to SDR and write to disk'))
                print('-> %s'%str(im_sdr_file))
                convert_write_png(hdr_image_path=str(im_hdr_file), png_image_path=str(im_sdr_file), if_mask=False, scale=1.)

        print(blue_text('[%s] DONE. load_im_hdr'%self.parent_class_name))


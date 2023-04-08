from abc import abstractmethod
from pathlib import PosixPath
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(suppress=True)
from lib.utils_io import load_img
from lib.utils_misc import blue_text, get_list_of_keys, green, yellow, white_blue, red, check_list_of_tensors_size
from lib.utils_io import load_img, convert_write_png
from lib.utils_OR.utils_OR_mesh import minimum_bounding_rectangle

class scene2DBase():
    '''
    Base class used to load 2D per-pixel modalities
    '''
    def __init__(
        self, 
        parent_class_name: str, # e.g. mitsubaScene3D, openroomsScene3D
        root_path_dict: dict, 
        modality_list: list, 
        if_debug_info: bool=False, 
        # host: str='', 
    ):

        # if 'frame_id_list' not in self.CONF.scene_params_dict:
        #     self.CONF.scene_params_dict['frame_id_list'] = []
        self.if_save_storage = self.CONF.scene_params_dict.get('if_save_storage', False) # set to True to enable removing duplicated renderer files (e.g. only one copy of geometry files in main, or emitter files only in main and mainDiffMat)
        self.if_debug_info = if_debug_info
        self.parent_class_name = parent_class_name

        self.if_loaded_colors = False
        # self.if_loaded_shapes = False
        # self.if_loaded_layout = False

        self.root_path_dict = root_path_dict
        self.PATH_HOME, self.dataset_root = get_list_of_keys(self.root_path_dict, ['PATH_HOME', 'dataset_root'], [PosixPath, PosixPath])

        # im params

        self.if_allow_crop = self.CONF.im_params_dict.get('if_allow_crop', False) # crop if loaded image is larger than target size
        if self.CONF.im_params_dict != {} and all([_ in self.CONF.im_params_dict for _ in ['im_H_load', 'im_W_load', 'im_H_resize', 'im_W_resize']]):
            self.im_H_load, self.im_W_load, self.im_H_resize, self.im_W_resize = get_list_of_keys(self.CONF.im_params_dict, ['im_H_load', 'im_W_load', 'im_H_resize', 'im_W_resize'])
            self.if_resize_im = (self.im_H_load, self.im_W_load) != (self.im_H_resize, self.im_W_resize) # resize modalities (exclusing lighting)
            self.H, self.W = self.im_H_resize, self.im_W_resize
            self.im_HW_load = (self.im_H_load, self.im_W_load)
            self.im_HW_target = () if not self.if_resize_im else (self.im_H_resize, self.im_W_resize)
        else:
            self.im_HW_load = ()
            self.im_HW_target = ()

        # dict for estimations
        self.est = {}

        # set up modalities to load
        self.modality_list = modality_list
        if self.CONF.cam_params_dict.get('if_sample_poses', False): self.modality_list.append('poses')
        self.modality_list = self.check_and_sort_modalities(list(set(self.modality_list)))

        # self.CONF.modality_filename_dict = self.CONF.modality_filename_dict
        self.modality_ext_dict = {}
        self.modality_folder_dict = {}
        for modality, filename in self.CONF.modality_filename_dict.items():
            # assert modality in self.valid_modalities, 'Invalid key [%s] in self.CONF.modality_filename_dict: NOT in self.valid_modalities!'%modality
            self.modality_ext_dict[modality] = filename.split('.')[-1] if isinstance(filename, str) else filename[-1]
            self.modality_folder_dict[modality] = filename.split('/')[0] if isinstance(filename, str) else filename[0]
        self.modality_file_list_dict = {}

    @property
    @abstractmethod
    def valid_modalities(self):
        ...

    @property
    @abstractmethod
    def frame_num(self):
        ...

    def _K(self, frame_idx: int=None):
        if hasattr(self, 'K'):
            return self.K
        elif hasattr(self, 'K_list'):
            assert frame_idx is not None, 'frame_idx is None!'
            return self.K_list[frame_idx]
        else:
            raise ValueError('No intrinsics found for %s'%self.parent_class_name)

    def _H(self, frame_idx: int=None):
        if hasattr(self, 'H'):
            return self.H
        elif hasattr(self, 'H_list'):
            assert frame_idx is not None, 'frame_idx is None!'
            return self.H_list[frame_idx]
        else:
            raise ValueError('No im H found for %s'%self.parent_class_name)

    def _W(self, frame_idx: int=None):
        if hasattr(self, 'W'):
            return self.W
        elif hasattr(self, 'W_list'):
            assert frame_idx is not None, 'frame_idx is None!'
            return self.W_list[frame_idx]
        else:
            raise ValueError('No im W found for %s'%self.parent_class_name)

    @property
    def if_has_global_HW(self):
        return hasattr(self, 'im_H') and hasattr(self, 'im_W')
    
    @property
    def if_has_poses(self):
        return hasattr(self, 'pose_list')
    
    @property
    def if_has_colors(self): # no semantic label colors
        return False
    
    @property
    def pose_file_root(self):
        return self.pose_file_path.parent if hasattr(self, 'pose_file') else self.pose_file_path_list[0].parent

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

        if_allow_crop = self.CONF.im_params_dict.get('if_allow_crop', False)
        if not 'im_sdr' in self.modality_file_list_dict:
            filename = self.CONF.modality_filename_dict['im_sdr']
            self.modality_file_list_dict['im_sdr'] = [self.scene_rendering_path_list[frame_idx] / (filename%frame_id) for frame_idx, frame_id in enumerate(self.frame_id_list)]

        if 'im_H_load_sdr' in self.CONF.im_params_dict:
            # separate H, W for loading SDR images
            expected_shape_list = [(self.CONF.im_params_dict['im_H_load_sdr'], self.CONF.im_params_dict['im_W_load_sdr'], 3,)]*self.frame_num
        else:
            expected_shape_list = [self.im_HW_load_list[_]+(3,) for _ in list(range(self.frame_num))] if hasattr(self, 'im_HW_load_list') else [self.im_HW_load+(3,)]*self.frame_num
        self.im_sdr_list = [load_img(_, expected_shape=__, ext=self.modality_ext_dict['im_sdr'], target_HW=self.im_HW_target, if_allow_crop=if_allow_crop)/255. for _, __ in zip(self.modality_file_list_dict['im_sdr'], expected_shape_list)]

        # print(self.modality_file_list_dict['im_sdr'])

        print(blue_text('[%s] DONE. load_im_sdr')%self.parent_class_name)

    def load_im_hdr(self):
        '''
        load im in HDR; RGB, (H, W, 3), [0., 1.]
        '''
        print(white_blue('[%s] load_im_hdr'%self.parent_class_name))

        if_allow_crop = self.CONF.im_params_dict.get('if_allow_crop', False)
        if not 'im_hdr' in self.modality_file_list_dict:
            filename = self.CONF.modality_filename_dict['im_hdr']
            self.modality_file_list_dict['im_hdr'] = [self.scene_rendering_path_list[frame_idx] / (filename%frame_id) for frame_idx, frame_id in enumerate(self.frame_id_list)]

        if 'im_H_load_hdr' in self.CONF.im_params_dict:
            # separate H, W for loading HDR images
            expected_shape_list = [(self.CONF.im_params_dict['im_H_load_hdr'], self.CONF.im_params_dict['im_W_load_hdr'], 3,)]*self.frame_num
        else:
            expected_shape_list = [self.im_HW_load_list[_]+(3,) for _ in list(range(self.frame_num))] if hasattr(self, 'im_HW_load_list') else [self.im_HW_load+(3,)]*self.frame_num
        self.im_hdr_list = [load_img(_, expected_shape=__, ext=self.modality_ext_dict['im_hdr'], target_HW=self.im_HW_target, if_allow_crop=if_allow_crop) for _, __ in zip(self.modality_file_list_dict['im_hdr'], expected_shape_list)]
        # print(self.modality_file_list_dict['im_hdr'])
        hdr_radiance_scale = self.CONF.im_params_dict.get('hdr_radiance_scale', 1.)
        self.hdr_scale_list = [hdr_radiance_scale] * len(self.im_hdr_list)

        # assert all([np.all(~np.isnan(xx)) for xx in self.im_hdr_list])
        for frame_id, xx in zip(self.frame_id_list, self.im_hdr_list):
            is_nan_im = np.any(np.isnan(xx), axis=-1)
            if np.any(is_nan_im):
                # print(frame_id)
                # print(np.vstack((np.where(is_nan_im)[0], np.where(is_nan_im)[1])))
                # import ipdb; ipdb.set_trace()
                print(yellow('[Warning] NaN in im_hdr'), 'frame_id: %d'%frame_id, 'percentage: %.4f percent'%(np.sum(is_nan_im).astype(np.float32)/np.prod(is_nan_im.shape[:2])*100.))
                xx[is_nan_im] = 0.
                print('NaN replaced by 0.')

        '''
        convert and write sdr files
        '''
        if not 'im_sdr' in self.modality_file_list_dict:
            filename = self.CONF.modality_filename_dict['im_sdr']
            self.modality_file_list_dict['im_sdr'] = [self.scene_rendering_path_list[frame_idx] / (filename%frame_id) for frame_idx, frame_id in enumerate(self.frame_id_list)]
        for frame_idx, (frame_id, im_hdr_file) in enumerate(zip(self.frame_id_list, self.modality_file_list_dict['im_hdr'])):

            # im_sdr_file = Path(str(im_hdr_file).replace(self.modality_ext_dict['im_hdr'], self.modality_ext_dict['im_sdr']))
            im_sdr_file = self.modality_file_list_dict['im_sdr'][frame_idx]

            if not im_sdr_file.exists():
                if 'sdr_radiance_scale' in self.CONF.im_params_dict:
                    sdr_radiance_scale = self.CONF.im_params_dict['sdr_radiance_scale']
                else:
                    sdr_radiance_scale = hdr_radiance_scale
                print(yellow('[%s] [load_im_hdr] converting HDR to SDR and write to disk (hdr_radiance_scale %.2f)'%(frame_idx, hdr_radiance_scale)))
                print('-> %s'%str(im_sdr_file))
                convert_write_png(hdr_image_path=str(im_hdr_file), png_image_path=str(im_sdr_file), if_mask=False, scale=sdr_radiance_scale)

        print(blue_text('[%s] DONE. load_im_hdr'%self.parent_class_name))

    def load_layout(self):
        '''
        Load and visualize layout in 3D & 2D; assuming room axis-up direction is axis-aligned, ...
        ... by projecting all points to floor plane, and vertically grow a cuboid to form a layout box.
        
        [!!!] Assumes that the scene is axis-aligned, and the layout box is aligned with the scene axis.
        
        images/demo_layout_mitsubaScene_3D_1.png
        images/demo_layout_mitsubaScene_3D_1_BEV.png # red is layout bbox
        '''

        print(white_blue('[mitsubaScene3D] load_layout for scene...'))
        if self.if_loaded_layout: return
        if not self.if_loaded_shapes: self.load_shapes(self.CONF.shape_params_dict)

        if self.CONF.shape_params_dict.get('if_layout_as_walls', False) and any([shape_dict['is_wall'] for shape_dict in self.shape_list_valid]):
            vertices_all = np.vstack([self.vertices_list[_] for _ in range(len(self.vertices_list)) if self.shape_list_valid[_]['is_wall']])
        else:
            vertices_all = np.vstack(self.vertices_list)

        if self.axis_up[0] == 'y':
            self.v_2d = vertices_all[:, [0, 2]]
        elif self.axis_up[0] == 'x':
            self.v_2d = vertices_all[:, [1, 3]]
        elif self.axis_up[0] == 'z':
            self.v_2d = vertices_all[:, [0, 1]]
            
        # finding minimum 2d bbox (rectangle) from contour
        self.layout_hull_2d, self.layout_hull_pts = minimum_bounding_rectangle(self.v_2d)
        
        layout_hull_2d_2x = np.vstack((self.layout_hull_2d, self.layout_hull_2d)) # (8, 2)
        if self.axis_up[0] == 'y':
            self.layout_box_3d_transformed = np.hstack((layout_hull_2d_2x[:, 0:1], np.vstack((np.zeros((4, 1))+self.xyz_min[1], np.zeros((4, 1))+self.xyz_max[1])), layout_hull_2d_2x[:, 1:2]))
            self.ceiling_loc = self.xyz_max[1]
            self.floor_loc = self.xyz_min[1]
        elif self.axis_up[0] == 'x':
            self.layout_box_3d_transformed = np.hstack((np.vstack((np.zeros((4, 1))+self.xyz_min[1], np.zeros((4, 1))+self.xyz_max[1])), layout_hull_2d_2x))
            self.ceiling_loc = self.xyz_max[0]
            self.floor_loc = self.xyz_min[0]
        elif self.axis_up[0] == 'z':
            # self.layout_box_3d_transformed = np.hstack((, np.vstack((np.zeros((4, 1)), np.zeros((4, 1))+room_height))))    
            self.layout_box_3d_transformed = np.hstack((layout_hull_2d_2x, np.vstack((np.zeros((4, 1))+self.xyz_min[1], np.zeros((4, 1))+self.xyz_max[1]))))
            self.ceiling_loc = self.xyz_max[2]
            self.floor_loc = self.xyz_min[2]

        print(blue_text('[%s] DONE. load_layout'%self.parent_class_name))

        self.if_loaded_layout = True
        self.if_has_ceilling_floor = True

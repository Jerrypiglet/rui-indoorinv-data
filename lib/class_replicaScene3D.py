from pathlib import Path, PosixPath
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(suppress=True)

from tqdm import tqdm
import scipy
import shutil
from lib.global_vars import mi_variant_dict
import random
random.seed(0)
from lib.utils_OR.utils_OR_cam import dump_cam_params_OR, origin_lookat_up_to_R_t, read_K_list_OR, read_cam_params_OR, normalize_v, R_t_to_origin_lookatvector_up
from lib.utils_io import load_img, load_matrix
# from collections import defaultdict
# import trimesh
# Import the library using the alias "mi"
import mitsuba as mi

from lib.utils_misc import blue_text, yellow, get_list_of_keys, white_blue, red
from lib.utils_io import load_matrix, resize_intrinsics

# from .class_openroomsScene2D import openroomsScene2D
from .class_mitsubaBase import mitsubaBase
from .class_scene2DBase import scene2DBase

from lib.utils_monosdf_scene import dump_shape_dict_to_shape_file, load_shape_dict_from_shape_file
from lib.utils_OR.utils_OR_mesh import minimum_bounding_rectangle, sample_mesh, simplify_mesh
from lib.utils_OR.utils_OR_xml import get_XML_root
from lib.utils_OR.utils_OR_mesh import loadMesh, computeBox, get_rectangle_mesh
from lib.utils_misc import get_device

from .class_scene2DBase import scene2DBase

class replicaScene3D(mitsubaBase, scene2DBase):
    '''
    A class used to visualize/render scenes from Philip et al. - 2021 - Free-viewpoint Indoor Neural Relighting...
    
    Dataset specifications: https://gitlab.inria.fr/sibr/projects/indoor_relighting/-/blob/master/README.MD#preprocessing-your-own-data
    '''
    def __init__(
        self, 
        root_path_dict: dict, 
        scene_params_dict: dict, 
        modality_list: list, 
        modality_filename_dict: dict, 
        im_params_dict: dict={}, 
        cam_params_dict: dict={}, 
        BRDF_params_dict: dict={}, 
        lighting_params_dict: dict={}, 
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

        self.scene_name, self.frame_id_list, self.intrinsics_path = get_list_of_keys(scene_params_dict, ['scene_name', 'frame_id_list', 'intrinsics_path'], [str, list, PosixPath])
        self.indexing_based = scene_params_dict.get('indexing_based', 0)

        self.axis_up_native = 'y+'
        self.axis_up = scene_params_dict.get('axis_up', self.axis_up_native) # native: 'z+
        assert self.axis_up in ['x+', 'y+', 'z+', 'x-', 'y-', 'z-']
        if self.axis_up != self.axis_up_native:
            # assert False, 'do something please: '+self.axis_up+' -> '+self.axis_up_native
            print(red('warning: '+self.axis_up+' -> '+self.axis_up_native))

        self.host = host
        self.device = get_device(self.host, device_id)

        self.scene_path = self.rendering_root / self.scene_name
        self.scene_rendering_path = self.scene_path / 'rendering'
        self.scene_name_full = self.scene_name # e.g.'asianRoom1'

        self.pose_format, pose_file, if_abs_path = scene_params_dict['pose_file']
        assert self.pose_format in ['OpenRooms'], 'Unsupported pose file: '+pose_file
        if if_abs_path:
            self.pose_file = Path(pose_file)
        else:
            self.pose_file = self.scene_path / 'cameras' / pose_file

        if not cam_params_dict.get('if_sample_poses', False):
            self.frame_num_all = len(read_cam_params_OR(self.pose_file))

        self.shape_file = self.scene_path / 'mesh_geo.ply' # export with Meshlab this new mesh, to remove colors
        self.shape_params_dict = shape_params_dict
        self.mi_params_dict = mi_params_dict
        variant = mi_params_dict.get('variant', '')
        mi.set_variant(variant if variant != '' else mi_variant_dict[self.host])

        self.extra_transform = self.scene_params_dict.get('extra_transform', None)
        if self.extra_transform is not None:
            # self.extra_transform = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.float32) # y=z, z=x, x=y
            self.extra_transform_inv = self.extra_transform.T
            self.extra_transform_homo = np.eye(4, dtype=np.float32)
            self.extra_transform_homo[:3, :3] = self.extra_transform

        self.near = cam_params_dict.get('near', 0.1)
        self.far = cam_params_dict.get('far', 10.)

        ''''
        flags to set
        '''
        self.pts_from = {'mi': False}
        self.seg_from = {'mi': False}

        '''
        load everything
        '''
        mitsubaBase.__init__(
            self, 
            device = self.device, 
        )

        self.load_mi_scene(self.mi_params_dict)
        if 'poses' in self.modality_list:
            self.load_poses(self.cam_params_dict) # attempt to generate poses indicated in cam_params_dict

        self.load_modalities()

        if hasattr(self, 'pose_list'): 
            self.get_cam_rays(self.cam_params_dict)
        self.process_mi_scene(self.mi_params_dict, if_postprocess_mi_frames=hasattr(self, 'pose_list'))

    @property
    def frame_num(self):
        return len(self.frame_id_list)
            
    @property
    def valid_modalities(self):
        return [
            'im_hdr', 'im_sdr', 'im_mask', 
            'poses', 
            'shapes', 
            'depth', 
            'mi_normal', 'mi_depth', 
            ]

    @property
    def if_has_poses(self):
        return hasattr(self, 'pose_list')

    @property
    def if_has_shapes(self):
        return all([_ in self.modality_list for _ in ['shapes']])

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
            if not (result_ == False): continue
            if _ == 'shapes': self.load_shapes(self.shape_params_dict) # shapes of 1(i.e. furniture) + emitters
            if _ == 'im_mask': self.load_im_mask()

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
        elif modality == 'im_mask': 
            return self.im_mask_list
        else:
            assert False, 'Unsupported modality: ' + modality

    def load_mi_scene(self, mi_params_dict={}):
        '''
        load scene representation into Mitsuba 3
        '''
        shape_id_dict = {
            'type': self.shape_file.suffix[1:],
            'filename': str(self.shape_file), 
            # 'to_world': mi.ScalarTransform4f.scale([1./scale]*3).translate((-offset).flatten().tolist()),
            }
        # if self.if_scale_scene:
            # shape_id_dict['to_world'] = mi.ScalarTransform4f.scale([1./self.scene_scale]*3)
        if self.extra_transform is not None:
            shape_id_dict['to_world'] = mi.ScalarTransform4f(self.extra_transform_homo)
        self.mi_scene = mi.load_dict({
            'type': 'scene',
            'shape_id': shape_id_dict, 
        })

    def process_mi_scene(self, mi_params_dict={}, if_postprocess_mi_frames=True):
        debug_render_test_image = mi_params_dict.get('debug_render_test_image', False)
        if debug_render_test_image:
            '''
            images/demo_mitsuba_render.png
            '''
            test_rendering_path = self.PATH_HOME / 'mitsuba' / 'tmp_render.exr'
            print(blue_text('Rendering... test frame by Mitsuba: %s')%str(test_rendering_path))
            image = mi.render(self.mi_scene, spp=16)
            mi.util.write_bitmap(str(test_rendering_path), image)
            print(blue_text('DONE.'))

        if if_postprocess_mi_frames:
            if_sample_rays_pts = mi_params_dict.get('if_sample_rays_pts', True)
            if if_sample_rays_pts:
                self.mi_sample_rays_pts(self.cam_rays_list)
                self.pts_from['mi'] = True
            
            if_get_segs = mi_params_dict.get('if_get_segs', True)
            if if_get_segs:
                assert if_sample_rays_pts
                self.mi_get_segs(if_also_dump_xml_with_lit_area_lights_only=True, if_seg_emitter=False)
                self.seg_from['mi'] = True

    def load_intrinsics(self):
        '''
        -> K: (3, 3)
        '''
        self.K = load_matrix(self.intrinsics_path)
        assert self.K.shape == (3, 3)
        self.im_W_load = int(self.K[0][2] * 2)
        self.im_H_load = int(self.K[1][2] * 2)

        if self.im_W_load != self.W or self.im_H_load != self.H:
            scale_factor = [t / s for t, s in zip((self.H, self.W), self.im_HW_load)]
            self.K = resize_intrinsics(self.K, scale_factor)

    def load_poses(self, cam_params_dict):
        '''
        pose_list: list of pose matrices (**camera-to-world** transformation), each (3, 4): [R|t] (OpenCV convention: right-down-forward)
        '''
        self.load_intrinsics()
        if hasattr(self, 'pose_list'): return
        if not self.if_loaded_shapes: self.load_shapes(self.shape_params_dict)
        if not hasattr(self, 'mi_scene'): self.process_mi_scene(self.mi_params_dict, if_postprocess_mi_frames=False)

        if_resample = 'n'
        if cam_params_dict.get('if_sample_poses', False):
            if_resample = 'y'
            # assert False, 'disabled; use '
            if hasattr(self, 'pose_list'):
                if_resample = input(red("pose_list loaded. Resample pose? [y/n]"))
            if self.pose_file.exists():
                # assert self.pose_format in ['json']
                try:
                    _num_poses = len(self.load_meta_json_pose(self.pose_file)[1])
                except: 
                    _num_poses = -1
                # if_resample = input(red('pose file exists: %s (%d poses). Resample pose? [y/n]'%(str(self.pose_file), len(self.load_meta_json_pose(self.pose_file)[1]))))
                if_resample = input(red('pose file exists: %s (%d poses). Resample pose? [y/n]'%(str(self.pose_file), _num_poses)))
            if not if_resample in ['N', 'n']:
                self.sample_poses(cam_params_dict.get('sample_pose_num'), self.extra_transform_inv)
                return

        print(white_blue('[%s] load_poses from %s'%(self.parent_class_name, str(self.pose_file))))

        self.pose_list = []
        self.K_list = []
        self.origin_lookatvector_up_list = []

        if self.pose_format == 'OpenRooms':
            cam_params = read_cam_params_OR(self.pose_file)
            if self.frame_id_list == []: self.frame_id_list = list(range(len(cam_params)))
            assert all([cam_param.shape == (3, 3) for cam_param in cam_params])

            for idx in self.frame_id_list:
                cam_param = cam_params[idx]
                origin, lookat, up = np.split(cam_param.T, 3, axis=1)
                (R, t), lookatvector = origin_lookat_up_to_R_t(origin, lookat, up)
                self.pose_list.append(np.hstack((R, t)))
                self.origin_lookatvector_up_list.append((origin.reshape((3, 1)), lookatvector.reshape((3, 1)), up.reshape((3, 1))))

    # def load_im_mask(self):
    #     '''
    #     load im_mask (H, W), np.bool
    #     '''
    #     print(white_blue('[%s] load_im_mask')%self.parent_class_name)

    #     filename = self.modality_filename_dict['im_mask']
    #     im_mask_ext = filename.split('.')[-1]
    #     # if_allow_crop = self.im_params_dict.get('if_allow_crop', False)
    #     # if_all_ones_masks = self.im_params_dict.get('if_all_ones_masks', False)

    #     self.im_mask_file_list = [self.scene_rendering_path / (filename%frame_id) for frame_id in self.frame_id_list]
    #     expected_shape_list = [self.im_HW_load_list[_] for _ in list(range(self.frame_num))] if hasattr(self, 'im_HW_load_list') else [self.im_HW_load]*self.frame_num
    #     # if if_all_ones_masks:
    #     #     self.im_mask_list = [np.ones(_, dtype=np.bool) for _ in expected_shape_list]
    #     # else:
    #     self.im_mask_list = [load_img(_, expected_shape=__, ext=im_mask_ext, target_HW=self.im_HW_target, if_allow_crop=if_allow_crop)/255. for _, __ in zip(self.im_mask_file_list, expected_shape_list)]
    #     self.im_mask_list = [_.astype(np.bool) for _ in self.im_mask_list]

    #     print(blue_text('[%s] DONE. load_im_mask')%self.parent_class_name)

    def get_cam_rays(self, cam_params_dict={}):
        if hasattr(self, 'cam_rays_list'):  return
        self.cam_rays_list = self.get_cam_rays_list(self.H, self.W, [self.K]*len(self.pose_list), self.pose_list, convention='opencv')

    def load_shapes(self, shape_params_dict={}):
        '''
        load and visualize shapes (objs/furniture **& emitters**) in 3D & 2D: 
        '''
        if self.if_loaded_shapes: return
        
        print(white_blue('[%s] load_shapes for scene...'%self.parent_class_name))

        mitsubaBase._prepare_shapes(self)

        scale_offset = ()
        #  if not self.if_scale_scene else (self.scene_scale, 0.)
        shape_dict = load_shape_dict_from_shape_file(self.shape_file, shape_params_dict=shape_params_dict, scale_offset=scale_offset, extra_transform=self.extra_transform)
        # , scale_offset=(9.1, 0.)) # read scale.txt and resize room to metric scale in meters
        self.append_shape(shape_dict)

        self.if_loaded_shapes = True
        
        print(blue_text('[%s] DONE. load_shapes: %d total, %d/%d windows lit, %d/%d area lights lit'%(
            self.parent_class_name, 
            len(self.shape_list_valid), 
            len([_ for _ in self.window_list if _['emitter_prop']['if_lit_up']]), len(self.window_list), 
            len([_ for _ in self.lamp_list if _['emitter_prop']['if_lit_up']]), len(self.lamp_list), 
            )))

        if shape_params_dict.get('if_dump_shape', False):
            dump_shape_dict_to_shape_file(shape_dict, self.shape_file)

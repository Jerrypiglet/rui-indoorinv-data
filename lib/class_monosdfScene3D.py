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
from lib.utils_io import read_cam_params, normalize_v
import json
from lib.utils_io import load_matrix, load_img, convert_write_png
# from collections import defaultdict
import trimesh

import string
# Import the library using the alias "mi"
import mitsuba as mi
# Set the variant of the renderer
# from lib.global_vars import mi_variant
# mi.set_variant(mi_variant)

from lib.utils_misc import blue_text, yellow, get_list_of_keys, white_blue, red

# from .class_openroomsScene2D import openroomsScene2D
from lib.class_mitsubaBase import mitsubaBase
from lib.class_scene2DBase import scene2DBase

from lib.utils_misc import get_device
from lib.utils_monosdf import rend_util
from lib.utils_monosdf_scene import load_monosdf_shape, load_monosdf_scale_offset
class monosdfScene3D(mitsubaBase, scene2DBase):
    '''
    A class used to visualize/render scenes from MonoSDF preprocessed dataset format (e.g. scannet)
    '''
    def __init__(
        self, 
        root_path_dict: dict, 
        scene_params_dict: dict, 
        modality_list: list, 
        modality_filename_dict: dict, 
        im_params_dict: dict={'im_H_load': 384, 'im_W_load': 384, 'im_H_resize': 384, 'im_W_resize': 384}, 
        # BRDF_params_dict: dict={}, 
        # lighting_params_dict: dict={'env_row': 120, 'env_col': 160, 'SG_num': 12, 'env_height': 16, 'env_width': 32}, # params to load & convert lighting SG & envmap to 
        cam_params_dict: dict={'near': 0.1, 'far': 10.}, 
        shape_params_dict: dict={'if_load_mesh': True}, 
        # emitter_params_dict: dict={'N_ambient_rep': '3SG-SkyGrd'},
        mi_params_dict: dict={'if_sample_rays_pts': True, 'if_sample_poses': False}, 
        if_debug_info: bool=False, 
        host: str='', 
    ):
        scene2DBase.__init__(
            self, 
            parent_class_name=str(self.__class__.__name__), 
            root_path_dict=root_path_dict, 
            scene_params_dict=scene_params_dict, 
            modality_list=modality_list, 
            modality_filename_dict=modality_filename_dict, 
            im_params_dict=im_params_dict, 
            # BRDF_params_dict=BRDF_params_dict, 
            # lighting_params_dict=lighting_params_dict, 
            if_debug_info=if_debug_info, 
            )

        self.scene_name, (_shape_normalized, shape_file) = get_list_of_keys(scene_params_dict, ['scene_name', 'shape_file'], [str, tuple])
        self.frame_id_list = get_list_of_keys(scene_params_dict, ['frame_id_list'], [list])[0]
        self.up_axis = get_list_of_keys(scene_params_dict, ['up_axis'], [str])[0]
        self.indexing_based = scene_params_dict.get('indexing_based', 0)
        assert self.up_axis in ['x+', 'y+', 'z+', 'x-', 'y-', 'z-']

        self.scene_path = self.rendering_root / self.scene_name
        self.scene_rendering_path = self.scene_path
        assert self.scene_rendering_path.exists()
        # self.scene_rendering_path.mkdir(parents=True, exist_ok=True)
        # self.xml_file = self.xml_scene_root / self.scene_name / self.xml_filename

        self.pose_format, pose_file = scene_params_dict['pose_file']
        assert self.pose_format in ['npz'], 'Unsupported pose file: '+pose_file
        self.pose_file = self.scene_path / pose_file
        
        self.shape_file = self.rendering_root / shape_file
        assert self.shape_file.exists(), 'Shape file not exist: %s'%str(self.shape_file)
        assert _shape_normalized in ['normalized', 'not-normalized'], 'Unsupported _shape_normalized indicator: %s'%_shape_normalized
        self.shape_if_normalized = _shape_normalized=='normalized'

        # self.im_params_dict = im_params_dict
        # self.lighting_params_dict = lighting_params_dict
        self.cam_params_dict = cam_params_dict
        self.shape_params_dict = shape_params_dict
        # self.emitter_params_dict = emitter_params_dict
        self.mi_params_dict = mi_params_dict

        self.im_H_load, self.im_W_load, self.im_H_resize, self.im_W_resize = get_list_of_keys(im_params_dict, ['im_H_load', 'im_W_load', 'im_H_resize', 'im_W_resize'])
        self.if_resize_im = (self.im_H_load, self.im_W_load) != (self.im_H_resize, self.im_W_resize) # resize modalities (exclusing lighting)
        self.im_target_HW = () if not self.if_resize_im else (self.im_H_resize, self.im_W_resize)
        self.H, self.W = self.im_H_resize, self.im_W_resize
        # self.im_lighting_HW_ratios = (self.im_H_resize // self.lighting_params_dict['env_row'], self.im_W_resize // self.lighting_params_dict['env_col'])
        # assert self.im_lighting_HW_ratios[0] > 0 and self.im_lighting_HW_ratios[1] > 0

        self.near = cam_params_dict.get('near', 0.1)
        self.far = cam_params_dict.get('far', 10.)
        # self.T_w_b2m = np.array([[1., 0., 0.], [0., 0., 1.], [0., -1., 0.]], dtype=np.float32) # Blender world to Mitsuba world; no need if load GT obj (already processed with scale and offset)

        self.host = host
        self.device = get_device(self.host)

        # self.modality_list = self.check_and_sort_modalities(list(set(modality_list)))
        # self.pcd_color = None
        # self.if_loaded_colors = False
        self.if_loaded_shapes = False
        # self.if_loaded_layout = False

        ''''
        flags to set
        '''
        self.pts_from = {'mi': False, 'depth': False}
        self.seg_from = {'mi': False, 'seg': False}

        '''
        load everything
        '''
        mitsubaBase.__init__(
            self, 
            device = self.device, 
        )

        self.load_mi_scene(self.mi_params_dict)
        self.load_poses(self.cam_params_dict)

        self.load_modalities()
        # self.est = {}

        self.get_cam_rays(self.cam_params_dict)
        self.process_mi_scene(self.mi_params_dict)

    def num_frames(self):
        return len(self.frame_id_list)

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
    def if_has_poses(self):
        return hasattr(self, 'pose_list')

    # @property
    # def if_has_emission(self):
    #     return hasattr(self, 'emission_list')

    # @property
    # def if_has_layout(self):
    #     return all([_ in self.modality_list for _ in ['layout']])


    @property
    def if_has_shapes(self): # objs + emitters
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

    @property
    def frame_num(self):
        return len(self.frame_id_list)

    def load_modalities(self):
        for _ in self.modality_list:
            result_ = scene2DBase.load_modality_(self, _)
            if not (result_ == False):
                continue

            # if _ == 'emission': self.load_emission()
            # if _ == 'layout': self.load_layout()
            if _ == 'shapes': self.load_shapes(self.shape_params_dict) # shapes of 1(i.e. furniture) + emitters
            if _ == 'depth':
                import ipdb; ipdb.set_trace()

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
            return self.mi_seg_dict_of_lists[seg_key]
        # elif modality == 'emission': 
        #     return self.emission_list
        else:
            assert False, 'Unsupported modality: ' + modality

    def load_mi_scene(self, mi_params_dict={}):
        '''
        load scene representation into Mitsuba 3
        '''
        variant = mi_params_dict.get('variant', '')
        if variant != '':
            mi.set_variant(variant)
        else:
            mi.set_variant(mi_variant_dict[self.host])

        self.mi_scene = mi.load_dict({
            'type': 'scene',
            'shape_id':{
                'type': 'ply',
                'filename': str(self.shape_file), 
            }
        })

    def process_mi_scene(self, mi_params_dict={}):
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

        debug_dump_mesh = mi_params_dict.get('debug_dump_mesh', False)
        if debug_dump_mesh:
            '''
            images/demo_mitsuba_dump_meshes.png
            '''
            mesh_dump_root = self.PATH_HOME / 'mitsuba' / 'meshes_dump'
            if mesh_dump_root.exists(): shutil.rmtree(str(mesh_dump_root))
            mesh_dump_root.mkdir()

            for shape_idx, shape, in enumerate(self.mi_scene.shapes()):
                if not isinstance(shape, mi.llvm_ad_rgb.Mesh): continue
                # print(type(shape), isinstance(shape, mi.llvm_ad_rgb.Mesh))
                shape.write_ply(str(mesh_dump_root / ('%06d.ply'%shape_idx)))

        if_sample_rays_pts = mi_params_dict.get('if_sample_rays_pts', True)
        if if_sample_rays_pts:
            self.mi_sample_rays_pts(self.cam_rays_list)
            self.pts_from['mi'] = True
        
        if_get_segs = mi_params_dict.get('if_get_segs', True)
        if if_get_segs:
            assert if_sample_rays_pts
            self.mi_get_segs(if_also_dump_xml_with_lit_area_lights_only=True)
            self.seg_from['mi'] = True

    def load_poses(self, cam_params_dict):
        '''
        pose_list: list of pose matrices (**camera-to-world** transformation), each (3, 4): [R|t] (OpenCV convention: right-down-forward)
        '''
        # self.load_intrinsics()
        if hasattr(self, 'pose_list'): return
        if self.mi_params_dict.get('if_sample_poses', False):
            assert False, 'disabled; use '
            if_resample = 'n'
            if hasattr(self, 'pose_list'):
                if_resample = input(red("pose_list loaded. Resample pose? [y/n]"))
            if self.pose_file.exists():
                if_resample = input(red("pose file exists: %s. Resample pose? [y/n]"%str(self.pose_file)))
            if if_resample in ['Y', 'y']:
                self.sample_poses(self.mi_params_dict.get('pose_sample_num'), cam_params_dict)
            else:
                print(yellow('ABORTED resample pose.'))
        else:
            if not self.pose_file.exists():
            # if not hasattr(self, 'pose_list'):
                self.get_room_center_pose()

        print(white_blue('[mitsubaScene] load_poses from %s'%str(self.pose_file)))
         
        if self.pose_format == 'npz':
            '''
            MonoSDF convention
            '''
            center_crop_type = cam_params_dict.get('center_crop_type', 'no_crop')
            camera_dict = np.load(self.pose_file)
            scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in self.frame_id_list]
            world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in self.frame_id_list]

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
                if self.shape_if_normalized:
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

                assert abs(intrinsics[0][2]*2 - self.H) < 1., 'intrinsics->H/2. (%.2f) does not match self.H: (%d); resize intrinsics needed?'%(intrinsics[0][2]*2, self.H)
                assert abs(intrinsics[1][2]*2 - self.W) < 1., 'intrinsics->W/2. (%.2f) does not match self.W: (%d); resize intrinsics needed?'%(intrinsics[1][2]*2, self.W)
                
                self.K_list.append(intrinsics.astype(np.float32))
                self.pose_list.append(pose.astype(np.float32))

                R = pose[:3, :3].astype(np.float32)
                t = pose[:3, 3:4].astype(np.float32)
                assert np.abs(np.linalg.det(R) - 1.) < 1e-5

                _, __, at_vector = np.split(R, 3, axis=-1)
                at_vector = normalize_v(at_vector)
                up = normalize_v(-__) # (3, 1)
                assert np.abs(np.sum(at_vector * up)) < 1e-3
                origin = t
                self.origin_lookatvector_up_list.append((origin.reshape((3, 1)), at_vector.reshape((3, 1)), up.reshape((3, 1))))

        print(blue_text('[%s] DONE. load_poses'%self.parent_class_name))

    def get_cam_rays(self, cam_params_dict={}):
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

        self.depth_file_list = [self.scene_rendering_path / (self.modality_filename_dict['depth']%i) for i in self.frame_id_list]
        self.depth_list = [load_img(depth_file, (self.im_H_load, self.im_W_load), ext='npy', target_HW=self.im_target_HW).astype(np.float32)[:, :] for depth_file in self.depth_file_list] # -> [-1., 1.], pointing inward (i.e. notebooks/images/openrooms_normals.jpg)

        print(blue_text('[%s] DONE. load_depth')%self.parent_class_name)

        self.pts_from['depth'] = True

    def load_normal(self):
        '''
        normal, in camera coordinates (OpenGL convention: right-up-backward);
        (H, W, 3), [-1., 1.]
        '''
        if hasattr(self, 'normal_list'): return

        print(white_blue('[%s] load_normal for %d frames...'%(self.parent_class_name, len(self.frame_id_list))))

        self.normal_file_list = [self.scene_rendering_path / (self.modality_filename_dict['normal']%i) for i in self.frame_id_list]
        # aa = np.load(str(self.normal_file_list[0]))
        # import ipdb; ipdb.set_trace()
        self.normal_list = [load_img(normal_file, (self.im_H_load, self.im_W_load, 3), ext='npy', target_HW=self.im_target_HW, npy_if_channel_first=True).astype(np.float32) for normal_file in self.normal_file_list] # -> [-1., 1.], pointing inward (i.e. notebooks/images/openrooms_normals.jpg)
        # self.normal_list = [normal / np.sqrt(np.maximum(np.sum(normal**2, axis=2, keepdims=True), 1e-5)) for normal in self.normal_list]
        self.normal_list = [normal * 2. - 1. for normal in self.normal_list]
        
        print(blue_text('[%s] DONE. load_normal'%self.parent_class_name))

    def load_shapes(self, shape_params_dict={}):
        '''
        load and visualize shapes (objs/furniture **& emitters**) in 3D & 2D: 
        '''
        if self.if_loaded_shapes: return
        
        print(white_blue('[%s] load_shapes for scene...')%self.parent_class_name)
        
        self.shape_list_valid = []
        self.vertices_list = []
        self.faces_list = []
        self.ids_list = []
        self.bverts_list = []
        self.bfaces_list = []

        self.xyz_max = np.zeros(3,)-np.inf
        self.xyz_min = np.zeros(3,)+np.inf
        
        monosdf_shape_dict = load_monosdf_shape(self.shape_file, shape_params_dict)

        self.vertices_list.append(monosdf_shape_dict['vertices'])
        self.faces_list.append(monosdf_shape_dict['faces'])
        self.bverts_list.append(monosdf_shape_dict['bverts'])
        self.bfaces_list.append(monosdf_shape_dict['bfaces'])
        self.ids_list.append(monosdf_shape_dict['_id'])
        
        self.shape_list_valid.append(monosdf_shape_dict['shape_dict'])

        self.xyz_max = np.maximum(np.amax(monosdf_shape_dict['vertices'], axis=0), self.xyz_max)
        self.xyz_min = np.minimum(np.amin(monosdf_shape_dict['vertices'], axis=0), self.xyz_min)

        self.if_loaded_shapes = True
        
        print(blue_text('[%s] DONE. load_shapes: %d total'%(
            self.parent_class_name, 
            len(self.shape_list_valid), 
            )))

    def load_colors(self):
        '''
        load mapping from obj cat id to RGB
        '''
        self.if_loaded_colors = False
        return

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
from lib.utils_OR.utils_OR_cam import R_t_to_origin_lookatvector_up, origin_lookat_up_to_R_t, read_cam_params_OR, normalize_v
import json
from lib.utils_io import load_matrix, load_img, convert_write_png
# from collections import defaultdict
# import trimesh
import imageio
import string
# Import the library using the alias "mi"
import mitsuba as mi

from lib.utils_misc import blue_text, yellow, get_list_of_keys, white_blue, red
from lib.utils_io import load_matrix, resize_intrinsics

# from .class_openroomsScene2D import openroomsScene2D
from .class_mitsubaBase import mitsubaBase
from .class_scene2DBase import scene2DBase

from lib.utils_OR.utils_OR_mesh import minimum_bounding_rectangle, sample_mesh, simplify_mesh
from lib.utils_OR.utils_OR_xml import get_XML_root
from lib.utils_OR.utils_OR_mesh import loadMesh, computeBox, get_rectangle_mesh
from lib.utils_misc import get_device

from .class_scene2DBase import scene2DBase

class mitsubaScene3D(mitsubaBase, scene2DBase):
    '''
    A class used to visualize/render Mitsuba scene in XML format
    '''
    def __init__(
        self, 
        root_path_dict: dict, 
        scene_params_dict: dict, 
        modality_list: list, 
        modality_filename_dict: dict, 
        im_params_dict: dict={'im_H_load': 480, 'im_W_load': 640, 'im_H_resize': 480, 'im_W_resize': 640, 'spp': 1024}, 
        cam_params_dict: dict={'near': 0.1, 'far': 10.}, 
        BRDF_params_dict: dict={}, 
        lighting_params_dict: dict={'env_row': 120, 'env_col': 160, 'SG_num': 12, 'env_height': 16, 'env_width': 32}, # params to load & convert lighting SG & envmap to 
        shape_params_dict: dict={'if_load_mesh': True}, 
        emitter_params_dict: dict={'N_ambient_rep': '3SG-SkyGrd'},
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
        self.xml_scene_root = get_list_of_keys(self.root_path_dict, ['xml_scene_root'], [PosixPath])[0]

        self.xml_filename, self.scene_name, self.mitsuba_version, self.intrinsics_path = get_list_of_keys(scene_params_dict, ['xml_filename', 'scene_name', 'mitsuba_version', 'intrinsics_path'], [str, str, str, PosixPath])
        self.split, self.frame_id_list = get_list_of_keys(scene_params_dict, ['split', 'frame_id_list'], [str, list])
        self.splits = self.split.split('+')
        assert all([_.split('_')[0] in ['train', 'val', 'train+val'] for _ in self.splits])
        if len(self.splits) > 1:
            print(yellow('Multiple splits: %s'%self.split))
        
        self.invalid_frame_id_list = scene_params_dict.get('invalid_frame_id_list', [])
        self.frame_id_list = [_ for _ in self.frame_id_list if _ not in self.invalid_frame_id_list]
        
        self.mitsuba_version, self.axis_up = get_list_of_keys(scene_params_dict, ['mitsuba_version', 'axis_up'], [str, str])
        self.indexing_based = scene_params_dict.get('indexing_based', 0)
        assert self.mitsuba_version in ['3.0.0', '0.6.0']
        
        assert self.axis_up in ['x+', 'y+', 'z+', 'x-', 'y-', 'z-']

        self.extra_transform = None

        self.host = host
        self.device = get_device(self.host, device_id)

        self.scene_path = self.rendering_root / self.scene_name
        self.scene_rendering_path = self.rendering_root / self.scene_name / self.split
        self.scene_rendering_path.mkdir(parents=True, exist_ok=True)
        self.scene_name_full = self.scene_name # e.g. 'main_xml_scene0008_00_more'

        self.xml_file = self.xml_scene_root / self.scene_name / self.xml_filename
        self.monosdf_shape_dict = scene_params_dict.get('monosdf_shape_dict', {})
        if '_shape_normalized' in self.monosdf_shape_dict:
            assert self.monosdf_shape_dict['_shape_normalized'] in ['normalized', 'not-normalized'], 'Unsupported _shape_normalized indicator: %s'%self.monosdf_shape_dict['_shape_normalized']

        self.pose_format, pose_file = scene_params_dict['pose_file']
        assert self.pose_format in ['OpenRooms', 'Blender', 'json'], 'Unsupported pose file: '+pose_file
        self.pose_file_list = [self.xml_scene_root / self.scene_name / split / pose_file for split in self.splits]

        # self.im_params_dict = im_params_dict
        # self.lighting_params_dict = lighting_params_dict
        self.near = cam_params_dict.get('near', 0.1)
        self.far = cam_params_dict.get('far', 10.)
        self.shape_params_dict = shape_params_dict
        self.emitter_params_dict = emitter_params_dict
        self.mi_params_dict = mi_params_dict
        variant = mi_params_dict.get('variant', '')
        mi.set_variant(variant if variant != '' else mi_variant_dict[self.host])

        self.im_lighting_HW_ratios = (self.im_H_resize // self.lighting_params_dict['env_row'], self.im_W_resize // self.lighting_params_dict['env_col'])
        assert self.im_lighting_HW_ratios[0] > 0 and self.im_lighting_HW_ratios[1] > 0


        # self.modality_list = self.check_and_sort_modalities(list(set(modality_list)))
        self.pcd_color = None

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
    def frame_num_all(self):
        return len(self.frame_id_list)

    @property
    def K_list(self):
        return [self.K] * self.frame_num

    @property
    def valid_modalities(self):
        return [
            'im_hdr', 'im_sdr', 
            'poses', 
            'albedo', 
            'roughness', 
            'depth', 
            'normal', 
            'emission', 
            'layout', 'shapes', 
            'lighting_envmap', 
            ]

    @property
    def if_has_poses(self):
        return hasattr(self, 'pose_list')

    @property
    def if_has_emission(self):
        return hasattr(self, 'emission_list')

    @property
    def if_has_layout(self):
        return all([_ in self.modality_list for _ in ['layout']])

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

    def load_modalities(self):
        for _ in self.modality_list:
            result_ = scene2DBase.load_modality_(self, _)
            if not (result_ == False):
                continue

            if _ == 'emission': self.load_emission()
            if _ == 'layout': self.load_layout(self.shape_params_dict)
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
            return self.mi_seg_dict_of_lists[seg_key] # Set scene_obj->mi_params_dict={'if_get_segs': True

        elif modality == 'emission': 
            return self.emission_list
        else:
            assert False, 'Unsupported modality: ' + modality

    def load_mi_scene(self, mi_params_dict={}):
        '''
        load scene representation into Mitsuba 3
        '''
        if self.monosdf_shape_dict != {}:
            self.load_monosdf_scene()
        else:
            self.mi_scene = mi.load_file(str(self.xml_file))
            # if_also_dump_xml_with_lit_area_lights_only = mi_params_dict.get('if_also_dump_xml_with_lit_area_lights_only', True)
            # if if_also_dump_xml_with_lit_area_lights_only:
            #     from lib.utils_mitsuba import dump_Indoor_area_lights_only_xml_for_mi
            #     xml_file_lit_up_area_lights_only = dump_Indoor_area_lights_only_xml_for_mi(str(self.xml_file))
            #     print(blue_text('XML (lit_up_area_lights_only) for Mitsuba dumped to: %s')%str(xml_file_lit_up_area_lights_only))
            #     self.mi_scene_lit_up_area_lights_only = mi.load_file(str(xml_file_lit_up_area_lights_only))

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
                self.mi_sample_rays_pts(self.cam_rays_list)
                self.pts_from['mi'] = True
            
            if_get_segs = mi_params_dict.get('if_get_segs', True)
            if if_get_segs:
                assert if_sample_rays_pts
                self.mi_get_segs(if_also_dump_xml_with_lit_area_lights_only=True)
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
            if any([pose_file.exists() for pose_file in self.pose_file_list]):
                # assert self.pose_format in ['json']
                _num_poses = sum([len(self.load_meta_json_pose(pose_file)[1]) for pose_file in self.pose_file_list])
                # if_resample = input(red('pose file exists: %s (%d poses). Resample pose? [y/n]'%(str(self.pose_file), len(self.load_meta_json_pose(self.pose_file)[1]))))
                if_resample = input(red('pose file exists: %s (%d poses). Resample pose? [y/n]'%(' + '.join([str(pose_file) for pose_file in self.pose_file_list]), _num_poses)))
            if not if_resample in ['N', 'n']:
                self.sample_poses(cam_params_dict.get('sample_pose_num'))
                return
            # else:
            #     print(yellow('ABORTED resample pose.'))
        # else:
        #     if not self.pose_file.exists():
        #     # if not hasattr(self, 'pose_list'):
        #         self.get_room_center_pose()
        self.pose_list = []
        self.origin_lookatvector_up_list = []
        frame_id_list_all = []
        self.frame_split_list = []
        self.frame_offset_list = []
        self.t_c2w_b_list, self.R_c2w_b_list = [], []
        self.scene_rendering_path_list = []
        
        for pose_file, split in zip(self.pose_file_list, self.splits):
            print(white_blue('[%s] load_poses from %s'%(self.__class__.__name__, str(pose_file))))
            
            pose_list = []
            origin_lookatvector_up_list = []

            if self.pose_format == 'OpenRooms':
                '''
                OpenRooms convention (matrices containing origin, lookat, up); The camera coordinates is in OpenCV convention (right-down-forward).
                '''
                cam_params = read_cam_params_OR(pose_file)
                # if self.frame_id_list == []: 
                frame_id_list = list(range(len(cam_params)))
                frame_id_list = [_ for _ in frame_id_list if _ not in self.invalid_frame_id_list]
                assert all([cam_param.shape == (3, 3) for cam_param in cam_params])

                for idx in frame_id_list:
                    cam_param = cam_params[idx]
                    origin, lookat, up = np.split(cam_param.T, 3, axis=1)
                    (R, t), lookatvector = origin_lookat_up_to_R_t(origin, lookat, up)
                    pose_list.append(np.hstack((R, t)))
                    origin_lookatvector_up_list.append((origin.reshape((3, 1)), lookatvector.reshape((3, 1)), up.reshape((3, 1))))

            elif self.pose_format in ['Blender', 'json']:
                '''
                Blender: 
                    Liwen's Blender convention: (N, 2, 3), [t, euler angles]
                    Blender x y z == Mitsuba x z -y; Mitsuba x y z == Blender x z -y
                Json:
                    Liwen's NeRF poses: [R, t]; processed: in comply with Liwen's IndoorDataset (https://github.com/william122742/inv-nerf/blob/bake/utils/dataset/indoor.py)
                '''
                '''
                [NOTE] scene.obj from Liwen is much smaller (s.t. scaling and translation here) compared to scene loaded from scene_v3.xml
                '''
                # self.scale_m2b = np.array([0.206, 0.206, 0.206], dtype=np.float32).reshape((3, 1))
                # self.trans_m2b = np.array([-0.074684, 0.23965, -0.30727], dtype=np.float32).reshape((3, 1))
                t_c2w_b_list, R_c2w_b_list = [], []

                if self.pose_format == 'Blender':
                    cam_params = np.load(pose_file)
                    assert all([cam_param.shape == (2, 3) for cam_param in cam_params])
                    # if self.frame_id_list == []: 
                    frame_id_list = list(range(len(cam_params)))
                    frame_id_list = [_ for _ in frame_id_list if _ not in self.invalid_frame_id_list]
                    for idx in frame_id_list:
                        R_ = scipy.spatial.transform.Rotation.from_euler('xyz', [cam_params[idx][1][0], cam_params[idx][1][1], cam_params[idx][1][2]])
                        R_c2w_b_list.append(R_.as_matrix())
                        assert np.allclose(R_.as_euler('xyz'), cam_params[idx][1])
                        t_c2w_b_list.append(cam_params[idx][0].reshape((3, 1)).astype(np.float32))
                
                elif self.pose_format == 'json':
                    self.meta, _Rt_c2w_b_list = self.load_meta_json_pose(pose_file)
                    # if self.frame_id_list == []: 
                    frame_id_list = list(range(len(_Rt_c2w_b_list)))
                    frame_id_list = [_ for _ in frame_id_list if _ not in self.invalid_frame_id_list]
                    assert max(frame_id_list) < len(_Rt_c2w_b_list)
                    R_c2w_b_list = [_Rt_c2w_b_list[_][0] for _ in frame_id_list]
                    t_c2w_b_list = [_Rt_c2w_b_list[_][1] for _ in frame_id_list]

                    f_xy = 0.5*self.W/np.tan(0.5*self.meta['camera_angle_x']) # original focal length
                    if not min(abs(self.K[0][0]-f_xy), abs(self.K[1][1]-f_xy)) < 1e-3:
                        print(self.K, f_xy)
                        import ipdb; ipdb.set_trace()
                        assert False, red('computed f_xy is different than read from intrinsics! double check your loaded intrinsics!')

                # self.T_c_b2m = np.array([[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]], dtype=np.float32) # OpenGL -> OpenCV
                # self.T_w_b2m = np.array([[1., 0., 0.], [0., 0., 1.], [0., -1., 0.]], dtype=np.float32) # Blender world to Mitsuba world; no need if load GT obj (already processed with scale and offset)
                T_ = np.array([[-1., 0., 0.], [0., -1., 0.], [0., 0., 1.]], dtype=np.float32) # flip x, y: Liwen's new pose (left-up-forward) -> OpenCV (right-down-forward)
                for R_c2w_b, t_c2w_b in zip(R_c2w_b_list, t_c2w_b_list):
                    R = R_c2w_b @ T_
                    t = t_c2w_b

                    (origin, lookatvector, up) = R_t_to_origin_lookatvector_up(R, t)

                    pose_list.append(np.hstack((R, t)))
                    origin_lookatvector_up_list.append((origin.reshape((3, 1)), lookatvector.reshape((3, 1)), up.reshape((3, 1))))
                    
            self.pose_list += pose_list
            self.origin_lookatvector_up_list += origin_lookatvector_up_list
            self.frame_offset_list += [len(frame_id_list_all)] * len(frame_id_list)
            frame_id_list_all += [len(frame_id_list_all) + frame_id for frame_id in frame_id_list]
            self.t_c2w_b_list += t_c2w_b_list
            self.R_c2w_b_list += R_c2w_b_list
            self.frame_split_list += [split] * len(frame_id_list)
            self.scene_rendering_path_list += [self.scene_rendering_path.parent / split] * len(frame_id_list)
            
            print(yellow(split), blue_text('Loaded {} poses from {}'.format(len(frame_id_list), pose_file)))

            '''
            Obselete code, when Liwen was using Blender poses...
            '''

            # for R_c2w_b, t_c2w_b in zip(self.R_c2w_b_list, self.t_c2w_b_list):
            #     assert abs(1.-np.linalg.det(R_c2w_b))<1e-6
            #     t_c2w_b = (t_c2w_b - self.trans_m2b) / self.scale_m2b
            #     t = self.T_w_b2m @ t_c2w_b # -> t_c2w_w

            #     R = self.T_w_b2m @ R_c2w_b @ self.T_c_b2m # https://i.imgur.com/nkzfvwt.png

            #     _, __, lookatvector = np.split(R, 3, axis=-1)
            #     lookatvector = normalize_v(lookatvector)
            #     up = normalize_v(-__) # (3, 1)
            #     assert np.abs(np.sum(lookatvector * up)) < 1e-3
            #     origin = t

            #     pose_list.append(np.hstack((R, t)))
            #     origin_lookatvector_up_list.append((origin.reshape((3, 1)), lookatvector.reshape((3, 1)), up.reshape((3, 1))))

        if self.frame_id_list == []: 
            self.frame_id_list = frame_id_list_all
            assert len(self.frame_id_list) == len(self.frame_offset_list)
            assert len(self.frame_id_list) == len(self.frame_split_list)
        self.frame_id_list = [frame_id-offset for (frame_id, offset) in zip(self.frame_id_list, self.frame_offset_list)]

        print(blue_text('[%s] DONE. load_poses (%d poses)'%(self.__class__.__name__, len(self.pose_list))))

    def get_cam_rays(self, cam_params_dict={}):
        if hasattr(self, 'cam_rays_list'):  return
        self.cam_rays_list = self.get_cam_rays_list(self.H, self.W, [self.K]*len(self.pose_list), self.pose_list, convention='opencv')

    def get_room_center_pose(self):
        '''
        generate a single camera, centered at room center and with identity rotation
        '''
        if not self.if_loaded_layout:
            self.load_layout()
        self.pose_list = [np.hstack((
            np.eye(3, dtype=np.float32), ((self.xyz_max+self.xyz_min)/2.).reshape(3, 1)
            ))]

    def load_emission(self):
        '''
        return emission in HDR; (H, W, 3)
        '''
        print(white_blue('[%s] load_emission for %d frames...'%(self.__class__.__name__, len(self.frame_id_list))))

        self.emission_file_list = [self.scene_rendering_path_list[frame_idx] / 'Emit' / ('%03d_0001.%s'%(frame_id, 'exr')) for frame_idx, frame_id in enumerate(self.frame_id_list)]
        self.emission_list = [load_img(_, expected_shape=self.im_HW_load+(3,), ext='exr', target_HW=self.im_HW_target) for _ in self.emission_file_list]

        print(blue_text('[%s] DONE. load_emission'%self.__class__.__name__))

    def load_albedo(self):
        '''
        albedo; loaded in [0., 1.] HDR
        (H, W, 3), [0., 1.]
        '''
        if hasattr(self, 'albedo_list'): return

        print(white_blue('[%s] load_albedo for %d frames...'%(self.__class__.__name__, len(self.frame_id_list))))

        self.albedo_file_list = [self.scene_rendering_path_list[frame_idx] / 'DiffCol' / ('%03d_0001.%s'%(frame_id, 'exr')) for frame_idx, frame_id in enumerate(self.frame_id_list)]
        expected_shape_list = [self.im_HW_load_list[_]+(3,) for _ in self.frame_id_list] if hasattr(self, 'im_HW_load_list') else [self.im_HW_load+(3,)]*self.frame_num
        self.albedo_list = [load_img(albedo_file, expected_shape=__, ext='exr', target_HW=self.im_HW_target).astype(np.float32) for albedo_file, __ in zip(self.albedo_file_list, expected_shape_list)]
        
        print(blue_text('[%s] DONE. load_albedo'%self.__class__.__name__))

    def load_roughness(self):
        '''
        roughness; smaller, the more specular;
        (H, W, 1), [0., 1.]
        '''
        if hasattr(self, 'roughness_list'): return

        print(white_blue('[%s] load_roughness for %d frames...'%(self.__class__.__name__, len(self.frame_id_list))))

        self.roughness_file_list = [self.scene_rendering_path_list[frame_idx] / 'Roughness' / ('%03d_0001.%s'%(frame_id, 'exr')) for frame_idx, frame_id in enumerate(self.frame_id_list)]
        expected_shape_list = [self.im_HW_load_list[_]+(3,) for _ in self.frame_id_list] if hasattr(self, 'im_HW_load_list') else [self.im_HW_load+(3,)]*self.frame_num
        self.roughness_list = [load_img(roughness_file, expected_shape=__, ext='exr', target_HW=self.im_HW_target)[:, :, 0:1].astype(np.float32) for roughness_file, __ in zip(self.roughness_file_list, expected_shape_list)]

        print(blue_text('[%s] DONE. load_roughness'%self.__class__.__name__))

    def load_depth(self):
        '''
        depth;
        (H, W), ideally in [0., inf]
        '''
        if hasattr(self, 'depth_list'): return

        print(white_blue('[%s] load_depth for %d frames...'%(self.__class__.__name__, len(self.frame_id_list))))

        self.depth_file_list = [self.scene_rendering_path_list[frame_idx] / 'Depth' / ('%03d_0001.%s'%(frame_id, 'exr')) for frame_idx, frame_id in enumerate(self.frame_id_list)]
        expected_shape_list = [self.im_HW_load_list[_]+(3,) for _ in self.frame_id_list] if hasattr(self, 'im_HW_load_list') else [self.im_HW_load+(3,)]*self.frame_num
        self.depth_list = [load_img(depth_file, expected_shape=__, ext='exr', target_HW=self.im_HW_target).astype(np.float32)[:, :, 0] for depth_file, __ in zip(self.depth_file_list, expected_shape_list)] # -> [-1., 1.], pointing inward (i.e. notebooks/images/openrooms_normals.jpg)

        print(blue_text('[%s] DONE. load_depth'%self.__class__.__name__))

        self.pts_from['depth'] = True

    def load_normal(self):
        '''
        normal, in camera coordinates (OpenGL convention: right-up-backward);
        (H, W, 3), [-1., 1.]
        '''
        if hasattr(self, 'normal_list'): return

        print(white_blue('[%s] load_normal for %d frames...'%(self.__class__.__name__, len(self.frame_id_list))))

        self.normal_file_list = [self.scene_rendering_path_list[frame_idx] / 'Normal' / ('%03d_0001.%s'%(frame_id, 'exr')) for frame_idx, frame_id in enumerate(self.frame_id_list)]
        expected_shape_list = [self.im_HW_load_list[_]+(3,) for _ in self.frame_id_list] if hasattr(self, 'im_HW_load_list') else [self.im_HW_load+(3,)]*self.frame_num
        self.normal_list = [load_img(normal_file, expected_shape=__, ext='exr', target_HW=self.im_HW_target).astype(np.float32) for normal_file, __ in zip(self.normal_file_list, expected_shape_list)] # -> [-1., 1.], pointing inward (i.e. notebooks/images/openrooms_normals.jpg)
        self.normal_list = [normal / np.sqrt(np.maximum(np.sum(normal**2, axis=2, keepdims=True), 1e-5)) for normal in self.normal_list]
        
        print(blue_text('[%s] DONE. load_normal'%self.__class__.__name__))

    def load_lighting_envmap(self):
        '''
        load lighting enemap and camra ray endpoint in HDR; 

        yields: 
            self.lighting_envmap_list: [(env_row, env_col, 3, env_height, env_width)]
            self.lighting_envmap_position_list: [(env_row, env_col, 3, env_height, env_width)]

        rendered with Blender: lib/class_renderer_blender_mitsubaScene_3D->renderer_blender_mitsubaScene_3D(); 
        '''
        assert False, 'no longer supported for now'
        print(white_blue('[%s] load_lighting_envmap'))

        self.lighting_envmap_list = []
        self.lighting_envmap_position_list = []

        env_row, env_col, env_height, env_width = get_list_of_keys(self.lighting_params_dict, ['env_row', 'env_col', 'env_height', 'env_width'], [int, int, int, int])
        folder_name_appendix = '-%dx%dx%dx%d'%(env_row, env_col, env_height, env_width)
        lighting_envmap_folder_path = self.scene_rendering_path / ('LightingEnvmap'+folder_name_appendix)
        assert lighting_envmap_folder_path.exists(), 'lighting envmap does not exist for: %s'%folder_name_appendix

        for i in tqdm(self.frame_id_list):
            envmap = np.zeros((env_row, env_col, 3, env_height, env_width), dtype=np.float32)
            envmap_position = np.zeros((env_row, env_col, 3, env_height, env_width), dtype=np.float32)
            for env_idx in tqdm(range(env_row*env_col)):
                lighting_envmap_file_path = lighting_envmap_folder_path / ('%03d_%03d.%s'%(frame_id, env_idx, 'exr'))
                lighting_envmap = load_img(lighting_envmap_file_path, ext='exr', target_HW=(env_height, env_width))
                envmap[env_idx//env_col, env_idx-env_col*(env_idx//env_col)] = lighting_envmap.transpose((2, 0, 1))

                lighting_envmap_position_m_file_path = lighting_envmap_folder_path / ('%03d_position_0001_%03d.%s'%(frame_id, env_idx, 'exr'))
                lighting_envmap_position_m = load_img(lighting_envmap_position_m_file_path, ext='exr', target_HW=(env_height, env_width)) # (H, W, 3), in Blender coords
                lighting_envmap_position = (lighting_envmap_position_m.reshape(-1, 3) @ (self.T_w_b2m.T)).reshape(env_height, env_width, 3)
                envmap_position[env_idx//env_col, env_idx-env_col*(env_idx//env_col)] = lighting_envmap_position.transpose((2, 0, 1))
                
            self.lighting_envmap_list.append(envmap)
            self.lighting_envmap_position_list.append(envmap_position)

        assert all([tuple(_.shape)==(env_row, env_col, 3, env_height, env_width) for _ in self.lighting_envmap_list])

        print(blue_text('[%s] DONE. load_lighting_envmap'%self.__class__.__name__))

    def load_shapes(self, shape_params_dict={}):
        '''
        load and visualize shapes (objs/furniture **& emitters**) in 3D & 2D: 
        '''
        if self.if_loaded_shapes: return
        
        mitsubaBase._prepare_shapes(self)
        
        if self.monosdf_shape_dict != {}:
            self.load_monosdf_shape(shape_params_dict=shape_params_dict)
        else:
            if_sample_pts_on_mesh = shape_params_dict.get('if_sample_pts_on_mesh', False)
            sample_mesh_ratio = shape_params_dict.get('sample_mesh_ratio', 1.)
            sample_mesh_min = shape_params_dict.get('sample_mesh_min', 100)
            sample_mesh_max = shape_params_dict.get('sample_mesh_max', 1000)

            if_simplify_mesh = shape_params_dict.get('if_simplify_mesh', False)
            simplify_mesh_ratio = shape_params_dict.get('simplify_mesh_ratio', 1.)
            simplify_mesh_min = shape_params_dict.get('simplify_mesh_min', 100)
            simplify_mesh_max = shape_params_dict.get('simplify_mesh_max', 1000)
            if_remesh = shape_params_dict.get('if_remesh', True) # False: images/demo_shapes_3D_NO_remesh.png; True: images/demo_shapes_3D_YES_remesh.png
            remesh_max_edge = shape_params_dict.get('remesh_max_edge', 0.1)

            if if_sample_pts_on_mesh:
                self.sample_pts_list = []

            print(white_blue('[mitsubaScene3D] load_shapes for scene...'))

            root = get_XML_root(self.xml_file)
            shapes = root.findall('shape')
            
            for shape in tqdm(shapes):
                random_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=5))
                if_emitter = False; if_window = False; if_area_light = False
                filename = None
                if shape.get('type') != 'obj':
                    assert shape.get('type') == 'rectangle'
                    '''
                    window as rectangle meshes: 
                        images/demo_mitsubaScene_rectangle_windows_1.png
                        images/demo_mitsubaScene_rectangle_windows_2.png
                    '''
                    transform_m = np.array(shape.findall('transform')[0].findall('matrix')[0].get('value').split(' ')).reshape(4, 4).astype(np.float32) # [[R,t], [0,0,0,1]]
                    (vertices, faces) = get_rectangle_mesh(transform_m[:3, :3], transform_m[:3, 3:4])
                    _id = 'rectangle_'+random_id
                    emitters = shape.findall('emitter')
                    if len(emitters) > 0:
                        assert len(emitters) == 1
                        emitter = emitters[0]
                        assert emitter.get('type') == 'area'
                        rgb = emitter.findall('rgb')[0]
                        assert rgb.get('name') == 'radiance'
                        radiance = np.array(rgb.get('value').split(', ')).astype(np.float32).reshape(3,)
                        if_emitter = True; if_area_light = True
                        # _id = 'emitter-' + _id
                        _id = 'emitter-' + emitter.get('id') if emitter.get('id') is not None else _id
                        emitter_prop = {'intensity': radiance, 'obj_type': 'obj', 'if_lit_up': np.amax(radiance) > 1e-3}
                else:
                    if not len(shape.findall('string')) > 0: continue
                    _id = shape.get('id')+'_'+random_id
                    # if 'wall' in _id.lower() or 'ceiling' in _id.lower():
                    #     continue
                    filename = shape.findall('string')[0]; assert filename.get('name') == 'filename'
                    obj_path = self.scene_path / filename.get('value') # [TODO] deal with transform
                    # if if_load_obj_mesh:
                    vertices, faces = loadMesh(obj_path) # based on L430 of adjustObjectPoseCorrectChairs.py; faces is 1-based!
                    assert len(shape.findall('emitter')) == 0 # [TODO] deal with object-based emitters
                    
                bverts, bfaces = computeBox(vertices)
                if shape.get('type') == 'obj':
                    '''
                    non-rectangle shape should not have very thin structures; if yes, discard
                    '''
                    # if np.any(np.amax(vertices, axis=0) - np.amin(vertices, axis=0) < 1e-2): # very thin objects (<1cm)
                    #     # import ipdb; ipdb.set_trace()
                    #     __ = np.amin(np.amax(vertices, axis=0) - np.amin(vertices, axis=0))
                    #     print(yellow('Discarded shape (%s) whose smallest shape dimension is %.4f < 0.01'%(_id, __))); continue
                    # if np.any(np.amax(bverts, axis=0) - np.amin(bverts, axis=0) < 1e-2): # very thin objects (<1cm)
                    #     # import ipdb; ipdb.set_trace()
                    #     __ = np.amin(np.amax(bverts, axis=0) - np.amin(bverts, axis=0))
                    #     print(yellow('Discarded shape (%s) whose smallest bbox dimension is %.4f < 0.01'%(_id, __))); continue
                    pass

                # --sample mesh--
                if if_sample_pts_on_mesh:
                    sample_pts, face_index = sample_mesh(vertices, faces, sample_mesh_ratio, sample_mesh_min, sample_mesh_max)
                    self.sample_pts_list.append(sample_pts)
                    # print(sample_pts.shape[0])

                # --simplify mesh--
                if if_simplify_mesh and simplify_mesh_ratio != 1.: # not simplying for mesh with very few faces
                    vertices, faces, (N_triangles, target_number_of_triangles) = simplify_mesh(vertices, faces, simplify_mesh_ratio, simplify_mesh_min, simplify_mesh_max, if_remesh=if_remesh, remesh_max_edge=remesh_max_edge, _id=_id)
                    if N_triangles != faces.shape[0]:
                        print('[%s] Mesh simplified to %d->%d triangles (target: %d).'%(_id, N_triangles, faces.shape[0], target_number_of_triangles))

                self.vertices_list.append(vertices)
                self.faces_list.append(faces)
                self.bverts_list.append(bverts)
                self.bfaces_list.append(bfaces)
                self.ids_list.append(_id)
                
                shape_dict = {
                    'filename': filename.get('value') if filename is not None else 'N/A', 
                    'if_in_emitter_dict': if_emitter, 
                    'id': _id, 
                    'random_id': random_id, 
                    # [IMPORTANT] currently relying on definition of walls and ceiling in XML file to identify those, becuase sometimes they can be complex meshes instead of thin rectangles
                    'is_wall': 'wall' in _id.lower(), 
                    'is_ceiling': 'ceiling' in _id.lower(), 
                    'is_layout': 'wall' in _id.lower() or 'ceiling' in _id.lower() or 'floor' in _id.lower(), 
                }
                if shape_dict['is_wall']: 
                    print('++++ is_wall:', _id, shape_dict['filename'])
                if if_emitter:
                    shape_dict.update({'emitter_prop': emitter_prop})
                    print('**** if_emitter:', _id, shape_dict['filename'], shape_dict['emitter_prop']['intensity'])
                if if_area_light:
                    # self.lamp_list.append((shape_dict, vertices, faces))
                    self.lamp_list.append(
                        {'emitter_prop': shape_dict['emitter_prop'], 'vertices': vertices, 'faces': faces, 'id': _id, 'random_id': random_id}
                    )

                self.shape_list_valid.append(shape_dict)

                self.xyz_max = np.maximum(np.amax(vertices, axis=0), self.xyz_max)
                self.xyz_min = np.minimum(np.amin(vertices, axis=0), self.xyz_min)


        self.if_loaded_shapes = True
        
        print(blue_text('[%s] DONE. load_shapes: %d total, %d/%d windows lit, %d/%d area lights lit'%(
            self.__class__.__name__, 
            len(self.shape_list_valid), 
            len([_ for _ in self.window_list if _['emitter_prop']['if_lit_up']]), len(self.window_list), 
            len([_ for _ in self.lamp_list if _['emitter_prop']['if_lit_up']]), len(self.lamp_list), 
            )))

    def get_envmap_axes(self):
        from utils_OR.utils_OR_lighting import convert_lighting_axis_local_to_global_np
        assert self.if_has_mitsuba_all
        normal_list = self.mi_normal_list
        # resample_ratio = self.H // self.lighting_params_dict['env_row']
        # assert resample_ratio == self.W // self.lighting_params_dict['env_col']
        # assert resample_ratio > 0

        lighting_local_xyz = np.tile(np.eye(3, dtype=np.float32)[np.newaxis, np.newaxis, ...], (self.H, self.W, 1, 1))
        lighting_global_xyz_list, lighting_global_pts_list = [], []
        for _idx in range(len(self.frame_id_list)):
            lighting_global_xyz = convert_lighting_axis_local_to_global_np(lighting_local_xyz, self.pose_list[_idx], normal_list[_idx])[::self.im_lighting_HW_ratios[0], ::self.im_lighting_HW_ratios[1]]
            lighting_global_pts = np.tile(np.expand_dims(self.mi_pts_list[_idx], 2), (1, 1, 3, 1))[::self.im_lighting_HW_ratios[0], ::self.im_lighting_HW_ratios[1]]
            assert lighting_global_xyz.shape == lighting_global_pts.shape == (self.lighting_params_dict['env_row'], self.lighting_params_dict['env_col'], 3, 3)
            lighting_global_xyz_list.append(lighting_global_xyz)
            lighting_global_pts_list.append(lighting_global_pts)
        return lighting_global_xyz_list, lighting_global_pts_list
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
# import trimesh

import string
# Import the library using the alias "mi"
import mitsuba as mi
# Set the variant of the renderer
# from lib.global_vars import mi_variant
# mi.set_variant(mi_variant)

from lib.utils_misc import blue_text, yellow, get_list_of_keys, white_blue, red
from lib.utils_io import load_matrix, resize_intrinsics

# from .class_openroomsScene2D import openroomsScene2D
from .class_mitsubaBase import mitsubaBase
from .class_scene2DBase import scene2DBase
from .class_monosdfScene3D import load_monosdf_scale_offset, load_monosdf_shape

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
        BRDF_params_dict: dict={}, 
        lighting_params_dict: dict={'env_row': 120, 'env_col': 160, 'SG_num': 12, 'env_height': 16, 'env_width': 32}, # params to load & convert lighting SG & envmap to 
        cam_params_dict: dict={'near': 0.1, 'far': 10.}, 
        shape_params_dict: dict={'if_load_mesh': True}, 
        emitter_params_dict: dict={'N_ambient_rep': '3SG-SkyGrd'},
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
            BRDF_params_dict=BRDF_params_dict, 
            lighting_params_dict=lighting_params_dict, 
            if_debug_info=if_debug_info, 
            )
        # self.root_path_dict = root_path_dict
        # self.PATH_HOME, self.rendering_root, self.xml_scene_root = get_list_of_keys(
        #     self.root_path_dict, 
        #     ['PATH_HOME', 'rendering_root', 'xml_scene_root'], 
        #     [PosixPath, PosixPath, PosixPath]
        #     )

        self.xml_filename, self.scene_name, self.mitsuba_version, self.intrinsics_path = get_list_of_keys(scene_params_dict, ['xml_filename', 'scene_name', 'mitsuba_version', 'intrinsics_path'], [str, str, str, PosixPath])
        self.split, self.frame_id_list = get_list_of_keys(scene_params_dict, ['split', 'frame_id_list'], [str, list])
        self.mitsuba_version, self.up_axis = get_list_of_keys(scene_params_dict, ['mitsuba_version', 'up_axis'], [str, str])
        self.indexing_based = scene_params_dict.get('indexing_based', 0)
        assert self.mitsuba_version in ['3.0.0', '0.6.0']
        assert self.split in ['train', 'val']
        assert self.up_axis in ['x+', 'y+', 'z+', 'x-', 'y-', 'z-']

        self.scene_path = self.rendering_root / self.scene_name
        self.scene_rendering_path = self.rendering_root / self.scene_name / self.split
        self.scene_rendering_path.mkdir(parents=True, exist_ok=True)

        self.xml_file = self.xml_scene_root / self.scene_name / self.xml_filename
        self.monosdf_shape_dict = scene_params_dict.get('monosdf_shape_dict', {})[0]

        self.pose_format, pose_file = scene_params_dict['pose_file']
        assert self.pose_format in ['OpenRooms', 'Blender', 'json'], 'Unsupported pose file: '+pose_file
        self.pose_file = self.xml_scene_root / self.scene_name / self.split / pose_file

        # self.im_params_dict = im_params_dict
        # self.lighting_params_dict = lighting_params_dict
        self.cam_params_dict = cam_params_dict
        self.shape_params_dict = shape_params_dict
        self.emitter_params_dict = emitter_params_dict
        self.mi_params_dict = mi_params_dict

        # self.im_H_load, self.im_W_load, self.im_H_resize, self.im_W_resize = get_list_of_keys(im_params_dict, ['im_H_load', 'im_W_load', 'im_H_resize', 'im_W_resize'])
        # self.if_resize_im = (self.im_H_load, self.im_W_load) != (self.im_H_resize, self.im_W_resize) # resize modalities (exclusing lighting)
        # self.im_target_HW = () if not self.if_resize_im else (self.im_H_resize, self.im_W_resize)
        # self.H, self.W = self.im_H_resize, self.im_W_resize
        self.im_lighting_HW_ratios = (self.im_H_resize // self.lighting_params_dict['env_row'], self.im_W_resize // self.lighting_params_dict['env_col'])
        assert self.im_lighting_HW_ratios[0] > 0 and self.im_lighting_HW_ratios[1] > 0

        self.near = cam_params_dict.get('near', 0.1)
        self.far = cam_params_dict.get('far', 10.)

        self.host = host
        self.device = get_device(self.host)

        # self.modality_list = self.check_and_sort_modalities(list(set(modality_list)))
        self.pcd_color = None
        self.if_loaded_colors = False
        self.if_loaded_shapes = False
        self.if_loaded_layout = False

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

        self.get_cam_rays(self.cam_params_dict)
        self.process_mi_scene(self.mi_params_dict)

    def num_frames(self):
        return len(self.frame_id_list)

    @property
    def valid_modalities(self):
        return [
            'im_hdr', 'im_sdr', 
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

    @property
    def frame_num(self):
        return len(self.frame_id_list)

    def load_modalities(self):
        for _ in self.modality_list:
            result_ = scene2DBase.load_modality_(self, _)
            if not (result_ == False):
                continue

            if _ == 'emission': self.load_emission()
            if _ == 'layout': self.load_layout()
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
        elif modality == 'emission': 
            return self.emission_list
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

        if self.monosdf_shape_dict == {}:
            self.mi_scene = mi.load_file(str(self.xml_file))
            if_also_dump_xml_with_lit_area_lights_only = mi_params_dict.get('if_also_dump_xml_with_lit_area_lights_only', True)
            if if_also_dump_xml_with_lit_area_lights_only:
                from lib.utils_mitsuba import dump_Indoor_area_lights_only_xml_for_mi
                xml_file_lit_up_area_lights_only = dump_Indoor_area_lights_only_xml_for_mi(str(self.xml_file))
                print(blue_text('XML (lit_up_area_lights_only) for Mitsuba dumped to: %s')%str(xml_file_lit_up_area_lights_only))
                self.mi_scene_lit_up_area_lights_only = mi.load_file(str(xml_file_lit_up_area_lights_only))
        else:
            shape_file = self.scene_path / Path(self.monosdf_shape_dict['shape_file'])
            (scale, offset) = load_monosdf_scale_offset(self.scene_path / Path(self.monosdf_shape_dict['camera_file']))
            self.mi_scene = mi.load_dict({
                'type': 'scene',
                'shape_id':{
                    'type': shape_file.suffix[1:],
                    'filename': str(shape_file), 
                    # 'to_world': mi.ScalarTransform4f.scale([1./scale]*3).translate((-offset).flatten().tolist()),
                    'to_world': mi.ScalarTransform4f.translate((-offset).flatten().tolist()).scale([1./scale]*3), 
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

    def load_intrinsics(self):
        '''
        -> K: (3, 3)
        '''
        self.K = load_matrix(self.intrinsics_path)
        assert self.K.shape == (3, 3)
        self.im_W_load = int(self.K[0][2] * 2)
        self.im_H_load = int(self.K[1][2] * 2)

        if self.im_W_load != self.W or self.im_H_load != self.H:
            scale_factor = [t / s for t, s in zip((self.H, self.W), (self.im_H_load, self.im_W_load))]
            self.K = resize_intrinsics(self.K, scale_factor)

        if self.pose_format == 'json':
            with open(self.pose_file, 'r') as f:
                self.meta = json.load(f)
            f_xy = 0.5*self.W/np.tan(0.5*self.meta['camera_angle_x']) # original focal length
            assert min(abs(self.K[0][0]-f_xy), abs(self.K[1][1]-f_xy)) < 1e-3, 'computed f_xy is different than read from intrinsics!'

    def load_poses(self, cam_params_dict):
        '''
        pose_list: list of pose matrices (**camera-to-world** transformation), each (3, 4): [R|t] (OpenCV convention: right-down-forward)
        '''
        self.load_intrinsics()
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
         
        pose_list = []
        origin_lookatvector_up_list = []

        if self.pose_format == 'OpenRooms':
            '''
            OpenRooms convention (matrices containing origin, lookat, up); The camera coordinates is in OpenCV convention (right-down-forward).
            '''
            cam_params = read_cam_params(self.pose_file)
            assert all([cam_param.shape == (3, 3) for cam_param in cam_params])

            for cam_param in cam_params:
                origin, lookat, up = np.split(cam_param.T, 3, axis=1)
                origin = origin.flatten()
                lookat = lookat.flatten()
                up = up.flatten()
                at_vector = normalize_v(lookat - origin)
                assert np.amax(np.abs(np.dot(at_vector.flatten(), up.flatten()))) < 2e-3 # two vector should be perpendicular

                t = origin.reshape((3, 1)).astype(np.float32)
                R = np.stack((np.cross(-up, at_vector), -up, at_vector), -1).astype(np.float32)
                
                pose_list.append(np.hstack((R, t)))
                origin_lookatvector_up_list.append((origin.reshape((3, 1)), at_vector.reshape((3, 1)), up.reshape((3, 1))))

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
            self.scale_m2b = np.array([0.206, 0.206, 0.206], dtype=np.float32).reshape((3, 1))
            self.trans_m2b = np.array([-0.074684, 0.23965, -0.30727], dtype=np.float32).reshape((3, 1))
            self.t_c2w_b_list, self.R_c2w_b_list = [], []

            if self.pose_format == 'Blender':
                cam_params = np.load(self.pose_file)
                assert all([cam_param.shape == (2, 3) for cam_param in cam_params])
                for idx in self.frame_id_list:
                    R_ = scipy.spatial.transform.Rotation.from_euler('xyz', [cam_params[idx][1][0], cam_params[idx][1][1], cam_params[idx][1][2]])
                    self.R_c2w_b_list.append(R_.as_matrix())
                    assert np.allclose(R_.as_euler('xyz'), cam_params[idx][1])
                    self.t_c2w_b_list.append(cam_params[idx][0].reshape((3, 1)).astype(np.float32))
            
            elif self.pose_format == 'json':
                for idx in self.frame_id_list:
                    pose = np.array(self.meta['frames'][idx]['transform_matrix'])[:3, :4].astype(np.float32)
                    R_, t_ = np.split(pose, (3,), axis=1)
                    R_ = R_ / np.linalg.norm(R_, axis=1, keepdims=True) # somehow R_ was mistakenly scaled by self.scale_m2b; need to recover to det(R)=1
                    self.R_c2w_b_list.append(R_)
                    self.t_c2w_b_list.append(t_)

            self.T_w_b2m = np.array([[1., 0., 0.], [0., 0., 1.], [0., -1., 0.]], dtype=np.float32) # Blender world to Mitsuba world; no need if load GT obj (already processed with scale and offset)
            self.T_c_b2m = np.array([[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]], dtype=np.float32)

            for R_c2w_b, t_c2w_b in zip(self.R_c2w_b_list, self.t_c2w_b_list):
                assert abs(1.-np.linalg.det(R_c2w_b))<1e-6
                t_c2w_b = (t_c2w_b - self.trans_m2b) / self.scale_m2b
                t = self.T_w_b2m @ t_c2w_b # -> t_c2w_w

                R = self.T_w_b2m @ R_c2w_b @ self.T_c_b2m # https://i.imgur.com/nkzfvwt.png

                _, __, at_vector = np.split(R, 3, axis=-1)
                at_vector = normalize_v(at_vector)
                up = normalize_v(-__) # (3, 1)
                assert np.abs(np.sum(at_vector * up)) < 1e-3
                origin = t

                pose_list.append(np.hstack((R, t)))
                origin_lookatvector_up_list.append((origin.reshape((3, 1)), at_vector.reshape((3, 1)), up.reshape((3, 1))))

        self.pose_list = pose_list
        self.origin_lookatvector_up_list = origin_lookatvector_up_list

        print(blue_text('[mistubaScene] DONE. load_poses (%d poses)'%len(self.pose_list)))

    def get_cam_rays(self, cam_params_dict={}):
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
        print(white_blue('[mitsubaScene] load_emission for %d frames...'%len(self.frame_id_list)))

        self.emission_file_list = [self.scene_rendering_path / 'Emit' / ('%03d_0001.%s'%(i, 'exr')) for i in self.frame_id_list]
        self.emission_list = [load_img(_, expected_shape=(self.im_H_load, self.im_W_load, 3), ext='exr', target_HW=self.im_target_HW) for _ in self.emission_file_list]

        print(blue_text('[mitsubaScene] DONE. load_emission'))

    def load_albedo(self):
        '''
        albedo; loaded in [0., 1.] HDR
        (H, W, 3), [0., 1.]
        '''
        if hasattr(self, 'albedo_list'): return

        print(white_blue('[mistubaScene] load_albedo for %d frames...'%len(self.frame_id_list)))

        self.albedo_file_list = [self.scene_rendering_path / 'DiffCol' / ('%03d_0001.%s'%(i, 'exr')) for i in self.frame_id_list]
        self.albedo_list = [load_img(albedo_file, (self.im_H_load, self.im_W_load, 3), ext='exr', target_HW=self.im_target_HW).astype(np.float32) for albedo_file in self.albedo_file_list]
        
        print(blue_text('[mistubaScene] DONE. load_albedo'))

    def load_roughness(self):
        '''
        roughness; smaller, the more specular;
        (H, W, 1), [0., 1.]
        '''
        if hasattr(self, 'roughness_list'): return

        print(white_blue('[mistubaScene] load_roughness for %d frames...'%len(self.frame_id_list)))

        self.roughness_file_list = [self.scene_rendering_path / 'Roughness' / ('%03d_0001.%s'%(i, 'exr')) for i in self.frame_id_list]
        self.roughness_list = [load_img(roughness_file, (self.im_H_load, self.im_W_load, 3), ext='exr', target_HW=self.im_target_HW)[:, :, 0:1].astype(np.float32) for roughness_file in self.roughness_file_list]

        print(blue_text('[mistubaScene] DONE. load_roughness'))

    def load_depth(self):
        '''
        depth;
        (H, W), ideally in [0., inf]
        '''
        if hasattr(self, 'depth_list'): return

        print(white_blue('[mistubaScene] load_depth for %d frames...'%len(self.frame_id_list)))

        self.depth_file_list = [self.scene_rendering_path / 'Depth' / ('%03d_0001.%s'%(i, 'exr')) for i in self.frame_id_list]
        # self.depth_list = [load_binary(depth_file, (self.im_H_load, self.im_W_load), target_HW=self.im_target_HW, resize_method='area')for depth_file in self.depth_file_list] # TODO: better resize method for depth for anti-aliasing purposes and better boundaries, and also using segs?
        self.depth_list = [load_img(depth_file, (self.im_H_load, self.im_W_load, 3), ext='exr', target_HW=self.im_target_HW).astype(np.float32)[:, :, 0] for depth_file in self.depth_file_list] # -> [-1., 1.], pointing inward (i.e. notebooks/images/openrooms_normals.jpg)

        print(blue_text('[mistubaScene] DONE. load_depth'))

        self.pts_from['depth'] = True

    def load_normal(self):
        '''
        normal, in camera coordinates (OpenGL convention: right-up-backward);
        (H, W, 3), [-1., 1.]
        '''
        if hasattr(self, 'normal_list'): return

        print(white_blue('[mistubaScene] load_normal for %d frames...'%len(self.frame_id_list)))

        self.normal_file_list = [self.scene_rendering_path / 'Normal' / ('%03d_0001.%s'%(i, 'exr')) for i in self.frame_id_list]
        self.normal_list = [load_img(normal_file, (self.im_H_load, self.im_W_load, 3), ext='exr', target_HW=self.im_target_HW).astype(np.float32) for normal_file in self.normal_file_list] # -> [-1., 1.], pointing inward (i.e. notebooks/images/openrooms_normals.jpg)
        self.normal_list = [normal / np.sqrt(np.maximum(np.sum(normal**2, axis=2, keepdims=True), 1e-5)) for normal in self.normal_list]
        
        print(blue_text('[mistubaScene] DONE. load_normal'))

    def load_lighting_envmap(self):
        '''
        load lighting enemap and camra ray endpoint in HDR; 

        yields: 
            self.lighting_envmap_list: [(env_row, env_col, 3, env_height, env_width)]
            self.lighting_envmap_position_list: [(env_row, env_col, 3, env_height, env_width)]

        rendered with Blender: lib/class_renderer_blender_mitsubaScene_3D->renderer_blender_mitsubaScene_3D(); 
        '''
        print(white_blue('[mitsubaScene] load_lighting_envmap'))

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
                lighting_envmap_file_path = lighting_envmap_folder_path / ('%03d_%03d.%s'%(i, env_idx, 'exr'))
                lighting_envmap = load_img(lighting_envmap_file_path, ext='exr', target_HW=(env_height, env_width))
                envmap[env_idx//env_col, env_idx-env_col*(env_idx//env_col)] = lighting_envmap.transpose((2, 0, 1))

                lighting_envmap_position_m_file_path = lighting_envmap_folder_path / ('%03d_position_0001_%03d.%s'%(i, env_idx, 'exr'))
                lighting_envmap_position_m = load_img(lighting_envmap_position_m_file_path, ext='exr', target_HW=(env_height, env_width)) # (H, W, 3), in Blender coords
                lighting_envmap_position = (lighting_envmap_position_m.reshape(-1, 3) @ (self.T_w_b2m.T)).reshape(env_height, env_width, 3)
                envmap_position[env_idx//env_col, env_idx-env_col*(env_idx//env_col)] = lighting_envmap_position.transpose((2, 0, 1))
                
            self.lighting_envmap_list.append(envmap)
            self.lighting_envmap_position_list.append(envmap_position)

        assert all([tuple(_.shape)==(env_row, env_col, 3, env_height, env_width) for _ in self.lighting_envmap_list])

        print(blue_text('[mitsubaScene] DONE. load_lighting_envmap'))

    def load_shapes(self, shape_params_dict={}):
        '''
        load and visualize shapes (objs/furniture **& emitters**) in 3D & 2D: 
        '''
        if self.if_loaded_shapes: return
        
        self.shape_list_valid = []
        self.vertices_list = []
        self.faces_list = []
        self.ids_list = []
        self.bverts_list = []
        self.bfaces_list = []

        self.window_list = []
        self.lamp_list = []

        self.xyz_max = np.zeros(3,)-np.inf
        self.xyz_min = np.zeros(3,)+np.inf
        
        if self.monosdf_shape_dict != {}:
            '''
            load a single shape estimated from MonoSDF: images/demo_shapes_monosdf.png
            '''
            (scale, offset) = load_monosdf_scale_offset(self.scene_path / Path(self.monosdf_shape_dict['camera_file']))
            monosdf_shape_dict = load_monosdf_shape(self.scene_path / Path(self.monosdf_shape_dict['shape_file']), shape_params_dict, (scale, offset))
            self.vertices_list.append(monosdf_shape_dict['vertices'])
            self.faces_list.append(monosdf_shape_dict['faces'])
            self.bverts_list.append(monosdf_shape_dict['bverts'])
            self.bfaces_list.append(monosdf_shape_dict['bfaces'])
            self.ids_list.append(monosdf_shape_dict['_id'])
            
            self.shape_list_valid.append(monosdf_shape_dict['shape_dict'])

            self.xyz_max = np.maximum(np.amax(monosdf_shape_dict['vertices'], axis=0), self.xyz_max)
            self.xyz_min = np.minimum(np.amin(monosdf_shape_dict['vertices'], axis=0), self.xyz_min)
        else:
            if_sample_mesh = shape_params_dict.get('if_sample_mesh', False)
            sample_mesh_ratio = shape_params_dict.get('sample_mesh_ratio', 1.)
            sample_mesh_min = shape_params_dict.get('sample_mesh_min', 100)
            sample_mesh_max = shape_params_dict.get('sample_mesh_max', 1000)

            if_simplify_mesh = shape_params_dict.get('if_simplify_mesh', False)
            simplify_mesh_ratio = shape_params_dict.get('simplify_mesh_ratio', 1.)
            simplify_mesh_min = shape_params_dict.get('simplify_mesh_min', 100)
            simplify_mesh_max = shape_params_dict.get('simplify_mesh_max', 1000)
            if_remesh = shape_params_dict.get('if_remesh', True) # False: images/demo_shapes_3D_NO_remesh.png; True: images/demo_shapes_3D_YES_remesh.png
            remesh_max_edge = shape_params_dict.get('remesh_max_edge', 0.1)

            if if_sample_mesh:
                self.sample_pts_list = []

            print(white_blue('[mitsubaScene3D] load_shapes for scene...'))

            root = get_XML_root(self.xml_file)
            shapes = root.findall('shape')
            
            for shape in tqdm(shapes):
                random_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=5))
                if_emitter = False; if_window = False; if_area_light = False
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
                        _id = 'emitter-' + _id
                        emitter_prop = {'intensity': radiance, 'obj_type': 'obj', 'if_lit_up': np.amax(radiance) > 1e-3}
                else:
                    if not len(shape.findall('string')) > 0: continue
                    _id = shape.findall('ref')[0].get('id')+'_'+random_id
                    # if 'walls' in _id.lower() or 'ceiling' in _id.lower():
                    #     continue
                    filename = shape.findall('string')[0]; assert filename.get('name') == 'filename'
                    obj_path = self.scene_path / filename.get('value') # [TODO] deal with transform
                    # if if_load_obj_mesh:
                    vertices, faces = loadMesh(obj_path) # based on L430 of adjustObjectPoseCorrectChairs.py; faces is 1-based!

                    assert len(shape.findall('emitter')) == 0 # [TODO] deal with object-based emitters

                    
                # --sample mesh--
                if if_sample_mesh:
                    sample_pts, face_index = sample_mesh(vertices, faces, sample_mesh_ratio, sample_mesh_min, sample_mesh_max)
                    self.sample_pts_list.append(sample_pts)
                    # print(sample_pts.shape[0])

                # --simplify mesh--
                if if_simplify_mesh and simplify_mesh_ratio != 1.: # not simplying for mesh with very few faces
                    vertices, faces, (N_triangles, target_number_of_triangles) = simplify_mesh(vertices, faces, simplify_mesh_ratio, simplify_mesh_min, simplify_mesh_max, if_remesh=if_remesh, remesh_max_edge=remesh_max_edge, _id=_id)
                    if N_triangles != faces.shape[0]:
                        print('[%s] Mesh simplified to %d->%d triangles (target: %d).'%(_id, N_triangles, faces.shape[0], target_number_of_triangles))

                bverts, bfaces = computeBox(vertices)
                self.vertices_list.append(vertices)
                self.faces_list.append(faces)
                self.bverts_list.append(bverts)
                self.bfaces_list.append(bfaces)
                self.ids_list.append(_id)
                
                shape_dict = {
                    'filename': filename.get('value'), 
                    'if_in_emitter_dict': if_emitter, 
                    'id': _id, 
                    'random_id': random_id, 
                    # [IMPORTANT] currently relying on definition of walls and ceiling in XML file to identify those, becuase sometimes they can be complex meshes instead of thin rectangles
                    'is_wall': 'walls' in _id.lower(), 
                    'is_ceiling': 'ceiling' in _id.lower(), 
                    'is_layout': 'walls' in _id.lower() or 'ceiling' in _id.lower(), 
                }
                if if_emitter:
                    shape_dict.update({'emitter_prop': emitter_prop})
                if if_area_light:
                    self.lamp_list.append((shape_dict, vertices, faces))
                self.shape_list_valid.append(shape_dict)

                self.xyz_max = np.maximum(np.amax(vertices, axis=0), self.xyz_max)
                self.xyz_min = np.minimum(np.amin(vertices, axis=0), self.xyz_min)


        self.if_loaded_shapes = True
        
        print(blue_text('[mitsubaScene3D] DONE. load_shapes: %d total, %d/%d windows lit, %d/%d area lights lit'%(
            len(self.shape_list_valid), 
            len([_ for _ in self.window_list if _[0]['emitter_prop']['if_lit_up']]), len(self.window_list), 
            len([_ for _ in self.lamp_list if _[0]['emitter_prop']['if_lit_up']]), len(self.lamp_list), 
            )))

    def load_layout(self):
        '''
        load and visualize layout in 3D & 2D; assuming room up direction is axis-aligned
        images/demo_layout_mitsubaScene_3D_1.png
        images/demo_layout_mitsubaScene_3D_1_BEV.png # red is layout bbox
        '''

        print(white_blue('[mitsubaScene3D] load_layout for scene...'))
        if self.if_loaded_layout: return
        if not self.if_loaded_shapes: self.load_shapes(self.shape_params_dict)

        vertices_all = np.vstack(self.vertices_list)

        if self.up_axis[0] == 'y':
            self.v_2d = vertices_all[:, [0, 2]]
            # room_height = np.amax(vertices_all[:, 1]) - np.amin(vertices_all[:, 1])
        elif self.up_axis[0] == 'x':
            self.v_2d = vertices_all[:, [1, 3]]
            # room_height = np.amax(vertices_all[:, 0]) - np.amin(vertices_all[:, 0])
        elif self.up_axis[0] == 'z':
            self.v_2d = vertices_all[:, [0, 1]]
            # room_height = np.amax(vertices_all[:, 2]) - np.amin(vertices_all[:, 2])
        # finding minimum 2d bbox (rectangle) from contour
        self.layout_hull_2d = minimum_bounding_rectangle(self.v_2d)
        layout_hull_2d_2x = np.vstack((self.layout_hull_2d, self.layout_hull_2d))
        if self.up_axis[0] == 'y':
            self.layout_box_3d_transformed = np.hstack((layout_hull_2d_2x[:, 0:1], np.vstack((np.zeros((4, 1))+self.xyz_min[1], np.zeros((4, 1))+self.xyz_max[1])), layout_hull_2d_2x[:, 1:2]))
        elif self.up_axis[0] == 'x':
            assert False
        elif self.up_axis[0] == 'z':
            # self.layout_box_3d_transformed = np.hstack((, np.vstack((np.zeros((4, 1)), np.zeros((4, 1))+room_height))))    
            assert False

        print(blue_text('[mitsubaScene3D] DONE. load_layout'))

        self.if_loaded_layout = True

    def load_colors(self):
        '''
        load mapping from obj cat id to RGB
        '''
        self.if_loaded_colors = False
        return

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

    def sample_poses(self, pose_sample_num: int):
        '''
        sample and write poses to OpenRooms convention (e.g. pose_format == 'OpenRooms': cam.txt)
        '''
        assert False, 'not tested'
        from lib.utils_mitsubaScene_sample_poses import mitsubaScene_sample_poses_one_scene
        assert self.up_axis == 'y+', 'not supporting other axes for now'
        if not self.if_loaded_layout: self.load_layout()

        lverts = self.layout_box_3d_transformed
        boxes = [[bverts, bfaces] for bverts, bfaces, shape in zip(self.bverts_list, self.bfaces_list, self.shape_list_valid) if not shape['is_layout']]
        cads = [[vertices, faces] for vertices, faces, shape in zip(self.vertices_list, self.faces_list, self.shape_list_valid) if not shape['is_layout']]

        self.cam_params_dict['samplePoint'] = pose_sample_num
        origin_lookat_up_list = mitsubaScene_sample_poses_one_scene(
            scene_dict={
                'lverts': lverts, 
                'boxes': boxes, 
                'cads': cads, 
            }, 
            program_dict={}, 
            param_dict=self.cam_params_dict, 
            path_dict={},
        ) # [pointLoc; target; up]

        pose_list = []
        origin_lookatvector_up_list = []
        for cam_param in origin_lookat_up_list:
            origin, lookat, up = np.split(cam_param.T, 3, axis=1)
            origin = origin.flatten()
            lookat = lookat.flatten()
            up = up.flatten()
            at_vector = normalize_v(lookat - origin)
            assert np.amax(np.abs(np.dot(at_vector.flatten(), up.flatten()))) < 2e-3 # two vector should be perpendicular
            t = origin.reshape((3, 1)).astype(np.float32)
            R = np.stack((np.cross(-up, at_vector), -up, at_vector), -1).astype(np.float32)
            pose_list.append(np.hstack((R, t)))
            origin_lookatvector_up_list.append((origin.reshape((3, 1)), at_vector.reshape((3, 1)), up.reshape((3, 1))))

        # self.pose_list = pose_list[:pose_sample_num]
        # return

        H, W = self.im_H_load//4, self.im_W_load//4
        scale_factor = [t / s for t, s in zip((H, W), (self.im_H_load, self.im_W_load))]
        K = resize_intrinsics(self.K, scale_factor)
        tmp_cam_rays_list = self.get_cam_rays_list(H, W, K, pose_list)
        normal_costs = []
        depth_costs = []
        normal_list = []
        depth_list = []
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
            mi_normal[invalid_depth_mask, :] = 0
            normal_list.append(mi_normal)

            mi_normal = mi_normal.astype(np.float32)
            mi_normal_gradx = np.abs(mi_normal[:, 1:] - mi_normal[:, 0:-1])[~invalid_depth_mask[:, 1:]]
            mi_normal_grady = np.abs(mi_normal[1:, :] - mi_normal[0:-1, :])[~invalid_depth_mask[1:, :]]
            ncost = (np.mean(mi_normal_gradx) + np.mean(mi_normal_grady)) / 2.
        
            dcost = np.mean(np.log(mi_depth + 1)[~invalid_depth_mask])

            assert not np.isnan(ncost) and not np.isnan(dcost)
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
        camIndex = np.argsort(totalCosts)[::-1]

        tmp_rendering_path = self.PATH_HOME / 'mitsuba' / 'tmp_sample_poses_rendering'
        if tmp_rendering_path.exists(): shutil.rmtree(str(tmp_rendering_path))
        tmp_rendering_path.mkdir(parents=True, exist_ok=True)
        print(blue_text('Dumping tmp normal and depth by Mitsuba: %s')%str(tmp_rendering_path))
        for i in tqdm(camIndex):
            imageio.imwrite(str(tmp_rendering_path / ('normal_%04d.png'%i)), (np.clip((normal_list[camIndex[i]] + 1.)/2., 0., 1.)*255.).astype(np.uint8))
            imageio.imwrite(str(tmp_rendering_path / ('depth_%04d.png'%i)), (np.clip(depth_list[camIndex[i]] / np.amax(depth_list[camIndex[i]]+1e-6), 0., 1.)*255.).astype(np.uint8))
        print(blue_text('DONE.'))
        # print(normal_costs[camIndex])

        self.pose_list = [pose_list[_] for _ in camIndex[:pose_sample_num]]
        self.origin_lookatvector_up_list = [origin_lookatvector_up_list[_] for _ in camIndex[:pose_sample_num]]

        # if self.pose_file.exists():
        #     txt = input(red("pose_list loaded. Overrite cam.txt? [y/n]"))
        #     if txt in ['N', 'n']:
        #         return
    
        with open(str(self.pose_file), 'w') as camOut:
            cam_poses_write = [origin_lookat_up_list[_] for _ in camIndex[:pose_sample_num]]
            camOut.write('%d\n'%len(cam_poses_write))
            print('Final sampled camera poses: %d'%len(cam_poses_write))
            for camPose in cam_poses_write:
                for n in range(0, 3):
                    camOut.write('%.3f %.3f %.3f\n'%\
                        (camPose[n, 0], camPose[n, 1], camPose[n, 2]))
        print(blue_text('cam.txt written to %s.'%str(self.pose_file)))



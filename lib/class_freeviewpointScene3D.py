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
from lib.utils_OR.utils_OR_cam import dump_cam_params_OR, origin_lookat_up_to_R_t, read_K_list_OR, read_cam_params_OR, normalize_v, R_t_to_origin_lookatvector_up_opencv
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

class freeviewpointScene3D(mitsubaBase, scene2DBase):
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

        self.scene_name, self.frame_id_list = get_list_of_keys(scene_params_dict, ['scene_name', 'frame_id_list'], [str, list])
        self.indexing_based = scene_params_dict.get('indexing_based', 0)

        self.axis_up_native = 'z+'
        self.axis_up = scene_params_dict.get('axis_up', self.axis_up_native) # native: 'z+
        assert self.axis_up in ['x+', 'y+', 'z+', 'x-', 'y-', 'z-']
        if self.axis_up != self.axis_up_native:
            assert False, 'do something please'
        
        self.if_autoscale_scene = False
        
        self.host = host
        self.device = get_device(self.host, device_id)

        self.pose_format, pose_file = scene_params_dict['pose_file']
        assert self.pose_format in ['OpenRooms', 'bundle'], 'Unsupported pose file: '+pose_file
        self.pose_file_path = self.scene_path / 'cameras' / pose_file

        self.shape_file_path = self.scene_path / 'meshes' / 'recon.ply'
        self.shape_params_dict = shape_params_dict
        self.mi_params_dict = mi_params_dict
        variant = mi_params_dict.get('variant', '')
        mi.set_variant(variant if variant != '' else mi_variant_dict[self.host])

        self.near = cam_params_dict.get('near', 0.1)
        self.far = cam_params_dict.get('far', 10.)

        ''''
        flags to set
        '''
        self.pts_from = {'mi': False}
        self.seg_from = {'mi': False}

        self.load_meta()

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

    def load_meta(self):
        self.scene_metadata_file = self.scene_path / 'scene_metadata.txt'
        assert Path(self.scene_metadata_file).exists()
        with open(str(self.scene_metadata_file), 'r') as camIn:
            scene_metadata = camIn.read().splitlines()
        frame_info_all_list = [_ for _ in scene_metadata if ('.exr' in _) or ('.jpg' in _)] # e.g. '00000.exr 997 665'
        assert frame_info_all_list[0].startswith('00000.')
        frame_id_all_list = [int(_.split(' ')[0].split('.')[0]) for _ in frame_info_all_list]
        assert frame_id_all_list[0] == 0

        if_missing_frame = False
        if frame_id_all_list[-1] != len(frame_id_all_list) - 1:
            print(red('Warning: frame_id_all_list[-1] == len(frame_id_all_list) - 1, so some frames are missing:'))
            print(set(list(range(len(frame_id_all_list)))) - set(frame_id_all_list))
            if_missing_frame = True
        else:
            assert frame_info_all_list[-1].startswith('%05d.'%frame_id_all_list[-1])
            
        if self.frame_id_list == []:
            # self.frame_id_list = list(range(len(frame_id_all_list)))
            self.frame_id_list = frame_id_all_list
            # if if_missing_frame:
            #     frame_info_all_list = [_ for _ in frame_info_all_list if int(_.split(' ')[0].split('.')[0]) in self.frame_id_list]
        self.scene_rendering_path_list = [self.scene_rendering_path] * len(self.frame_id_list)

        frame_info_all_dict = {int(_.split(' ')[0].split('.')[0]): _ for _ in frame_info_all_list}
        
        # im_HW_load_all_list = [(int(_.split(' ')[2])+1, int(_.split(' ')[1])+1) for _ in frame_info_all_list] # +1 because im H W are mistakenly recorded here (1 less) https://gitlab.inria.fr/sibr/projects/indoor_relighting/-/blob/master/preprocess/converters/createSceneMetadataEXR.py#L21
        # import ipdb; ipdb.set_trace()
        # self.im_HW_load_list = [im_HW_load_all_list[_] for _ in self.frame_id_list]
        im_HW_load_list = [frame_info_all_dict[frame_id] for frame_id in self.frame_id_list]
        self.im_HW_load_list = [(int(_.split(' ')[2])+1, int(_.split(' ')[1])+1) for _ in im_HW_load_list]
        assert len(im_HW_load_list) == len(self.frame_id_list)
        # assert 'im_H_resize' not in self.im_params_dict and 'im_W_resize' not in self.im_params_dict
        self.H_list = [_[0] for _ in self.im_HW_load_list]
        self.W_list = [_[1] for _ in self.im_HW_load_list]

        self.if_scale_scene = self.scene_params_dict.get('if_scale_scene', True)
        if self.if_scale_scene:
            self.scene_scale_file = self.scene_path / 'scale.txt'
            assert Path(self.scene_scale_file).exists()
            with open(str(self.scene_scale_file), 'r') as f:
                self.scene_scale = float(f.read().splitlines()[0])
            
    @property
    def valid_modalities(self):
        return [
            'im_hdr', 'im_sdr', 'im_mask', 
            'poses', 
            'shapes', 
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
            if _ == 'shapes': self.load_single_shape(shape_params_dict=self.shape_params_dict) # shapes of 1(i.e. furniture) + emitters
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
        assert False, '[TODO] already exists in b mitsubaBase; check if anything is new here; otherwise remove this class'
        shape_id_dict = {
            'type': self.shape_file.suffix[1:],
            'filename': str(self.shape_file), 
            # 'to_world': mi.ScalarTransform4f.scale([1./scale]*3).translate((-offset).flatten().tolist()),
            }
        if self.if_scale_scene:
            shape_id_dict['to_world'] = mi.ScalarTransform4f.scale([1./self.scene_scale]*3)
        self.mi_scene = mi.load_dict({
            'type': 'scene',
            'shape_id': shape_id_dict, 
        })

    def process_mi_scene(self, mi_params_dict={}, if_postprocess_mi_frames=True):
        assert False, '[TODO] already exists in b mitsubaBase; check if anything is new here; otherwise remove this class'
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
                self.mi_get_segs()
                self.seg_from['mi'] = True

    def load_poses(self, cam_params_dict):
        '''
        pose_list: list of pose matrices (**camera-to-world** transformation), each (3, 4): [R|t] (OpenCV convention: right-down-forward)
        '''
        # self.load_intrinsics()
        if hasattr(self, 'pose_list'): return
        # if not self.if_loaded_shapes: self.load_shapes(self.shape_params_dict)
        # if not hasattr(self, 'mi_scene'): self.process_mi_scene(self.mi_params_dict, if_postprocess_mi_frames=False)

        '''
        bundle.out format: https://www.cs.cornell.edu/~snavely/bundler/bundler-v0.3-manual.html#S6
        '''

        print(white_blue('[%s] load_poses from %s'%(self.parent_class_name, str(self.pose_file_path))))

        self.pose_list = []
        self.K_list = []
        self.origin_lookatvector_up_list = []

        if self.pose_format == 'OpenRooms':
            cam_params = read_cam_params_OR(self.pose_file_path)
            if self.frame_id_list == []: self.frame_id_list = list(range(len(cam_params)))
            assert all([cam_param.shape == (3, 3) for cam_param in cam_params])

            for idx in self.frame_id_list:
                cam_param = cam_params[idx]
                origin, lookat, up = np.split(cam_param.T, 3, axis=1)
                (R, t), lookatvector = origin_lookat_up_to_R_t(origin, lookat, up)
                self.pose_list.append(np.hstack((R, t)))
                self.origin_lookatvector_up_list.append((origin.reshape((3, 1)), lookatvector.reshape((3, 1)), up.reshape((3, 1))))

            self.K_list = read_K_list_OR(str(self.pose_file_path.parent / 'K_list.txt'))
            assert len(self.K_list) == len(self.pose_list)

        elif self.pose_format in ['bundle']:
            with open(str(self.pose_file_path), 'r') as camIn:
                cam_data = camIn.read().splitlines()

            self.frame_num_all = int(cam_data[1].split(' ')[0])
            if self.frame_id_list == []: self.frame_id_list = list(range(self.frame_num_all))
            assert self.frame_num_all >= self.frame_num

            for frame_idx, frame_id in tqdm(enumerate(self.frame_id_list)):
                cam_lines = cam_data[(2+frame_id*5):(2+frame_id*5+5)]
                f = cam_lines[0].split(' ')[0]
                if cam_lines[0].split(' ')[1:] != ['0', '0']: # no distortion
                    import ipdb; ipdb.set_trace()
                '''
                laoded R, t are: [1] world-to-camera, [2] OpenGL convention: https://www.cs.cornell.edu/~snavely/bundler/bundler-v0.3-manual.html#S6
                '''
                R_lines = [[float(_) for _ in R_line.split(' ')] for R_line in cam_lines[1:4]]
                R_ = np.array(R_lines).reshape(3, 3)
                t_line = [float(_) for _ in cam_lines[4].split(' ')]
                t_ = np.array(t_line).reshape(3, 1)
                assert np.isclose(np.linalg.det(R_), 1.)
                R = R_.T
                t = -R_.T @ t_
                if self.if_scale_scene:
                    t = t / self.scene_scale
                R = R @ np.array([[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]], dtype=np.float32) # OpenGL -> OpenCV
                self.pose_list.append(np.hstack((R, t)))

                (origin, lookatvector, up) = R_t_to_origin_lookatvector_up_opencv(R, t)
                self.origin_lookatvector_up_list.append((origin.reshape((3, 1)), lookatvector.reshape((3, 1)), up.reshape((3, 1))))

                K = np.array([[float(f), 0, self._W(frame_idx)/2.], [0, float(f), self._H(frame_idx)/2.], [0, 0, 1]], dtype=np.float32)
                if self.im_W_load != self.W or self.im_H_load != self.H:
                    scale_factor = [t / s for t, s in zip((self.H, self.W), self.im_HW_load)]
                    K = resize_intrinsics(K, scale_factor)
                self.K_list.append(K)

            print(blue_text('[%s] DONE. load_poses (%d poses)'%(self.parent_class_name, len(self.pose_list))))

            if cam_params_dict.get('if_convert_poses', False):
                self.export_poses_cam_txt(self.pose_file_path.parent, cam_params_dict=cam_params_dict, frame_num_all=self.frame_num_all)

    def load_im_mask(self):
        '''
        load im_mask (H, W), bool
        '''
        print(white_blue('[%s] load_im_mask')%self.parent_class_name)

        filename = self.modality_filename_dict['im_mask']
        im_mask_ext = filename.split('.')[-1]
        if_allow_crop = self.im_params_dict.get('if_allow_crop', False)
        if_all_ones_masks = self.im_params_dict.get('if_all_ones_masks', False)

        self.im_mask_file_list = [self.scene_rendering_path / (filename%frame_id) for frame_id in self.frame_id_list]
        expected_shape_list = [self.im_HW_load_list[_] for _ in list(range(self.frame_num))] if hasattr(self, 'im_HW_load_list') else [self.im_HW_load]*self.frame_num
        if if_all_ones_masks:
            self.im_mask_list = [np.ones(_, dtype=bool) for _ in expected_shape_list]
        else:
            self.im_mask_list = [load_img(_, expected_shape=__, ext=im_mask_ext, target_HW=self.im_HW_target, if_allow_crop=if_allow_crop)/255. for _, __ in zip(self.im_mask_file_list, expected_shape_list)]
        self.im_mask_list = [_.astype(bool) for _ in self.im_mask_list]

        print(blue_text('[%s] DONE. load_im_mask')%self.parent_class_name)

    def get_cam_rays(self, cam_params_dict={}):
        if hasattr(self, 'cam_rays_list'):  return
        self.cam_rays_list = self.get_cam_rays_list(self.H_list, self.W_list, self.K_list, self.pose_list, convention='opencv')
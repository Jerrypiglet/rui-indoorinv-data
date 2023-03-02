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
from lib.utils_OR.utils_OR_cam import R_t_to_origin_lookatvector_up_yUP, origin_lookat_up_to_R_t, read_cam_params_OR, normalize_v
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

class realScene3D(mitsubaBase, scene2DBase):
    '''
    A class used to visualize/render real scenes captured by Mustafa
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

        self.if_rc = scene_params_dict.get('if_rc', False) # True: get results from Reality Capture; False: from Colmap and converted by Mustafa
        
        self.scene_name, self.frame_id_list_input = get_list_of_keys(scene_params_dict, ['scene_name', 'frame_id_list'], [str, list])
        self.invalid_frame_id_list = scene_params_dict.get('invalid_frame_id_list', [])
        self.frame_id_list_input = [_ for _ in self.frame_id_list_input if _ not in self.invalid_frame_id_list]
        
        self.indexing_based = scene_params_dict.get('indexing_based', 0)
        
        self.extra_transform = self.scene_params_dict.get('extra_transform', None)
        if self.extra_transform is not None:
            self.extra_transform_inv = self.extra_transform.T
            self.extra_transform_homo = np.eye(4, dtype=np.float32)
            self.extra_transform_homo[:3, :3] = self.extra_transform

        self.scene_path = self.rendering_root / self.scene_name
        self.scene_rendering_path = self.rendering_root / self.scene_name
        self.scene_rendering_path.mkdir(parents=True, exist_ok=True)
        self.scene_name_full = self.scene_name # e.g. 'main_xml_scene0008_00_more'

        self.pose_format, pose_file = scene_params_dict['pose_file']
        assert self.pose_format in ['json', 'bundle'], 'Unsupported pose file: '+self.pose_file
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
        self.pts_from = {'mi': False, 'depth': False}
        self.seg_from = {'mi': False, 'seg': False}
        
        '''
        normalize poses
        [TODO] stremaline this into loader functions
        '''   
        self.if_autoscale_scene = self.scene_params_dict['if_autoscale_scene']     
        monosdf_scale_tuple = ()
        if self.if_autoscale_scene:
            print(yellow('Autoscaling scene (following MonoSDF)...'))
            poses = [np.vstack((pose, np.array([0., 0., 0., 1.], dtype=np.float32).reshape((1, 4)))) for pose in self.pose_list]
            poses = np.array(poses)
            min_vertices = poses[:, :3, 3].min(axis=0)
            max_vertices = poses[:, :3, 3].max(axis=0)
            center = (min_vertices + max_vertices) / 2.
            scale = 2. / (np.max(max_vertices - min_vertices) + 3.)
            monosdf_scale_tuple = (center, scale)
            
            for pose in self.pose_list: # modify in place
                pose[:3, 3] = (pose[:3, 3]- center) * scale
            for (origin, lookatvector, up) in self.origin_lookatvector_up_list:
                origin = (origin - center.reshape((3, 1))) * scale

        '''
        load everything
        '''

        self.load_mi_scene(self.mi_params_dict, monosdf_scale_tuple=monosdf_scale_tuple)
        self.load_modalities()

        if hasattr(self, 'pose_list'): 
            self.get_cam_rays(self.cam_params_dict)
        self.process_mi_scene(self.mi_params_dict, if_postprocess_mi_frames=hasattr(self, 'pose_list'))

        '''
        normalize shapes pcs
        [TODO] stremaline this into loader functions
        '''   
        if self.if_autoscale_scene:
            if self.scene_params_dict.get('shape_file', '') != '':
                print('Scaling shapes...')
                self.vertices_list = [(vertices - center.reshape((1, 3))) * scale for vertices in self.vertices_list]
                self.bverts_list = [(bverts - center.reshape((1, 3))) * scale for bverts in self.bverts_list]
                self.xyz_max = (self.xyz_max - center.reshape((3,))) * scale
                self.xyz_min = (self.xyz_min - center.reshape((3,))) * scale
            elif self.scene_params_dict.get('pcd_file', '') != '':
                print('Scaling pcd...')
                self.pcd = (self.pcd - center.reshape((1, 3))) * scale
                self.xyz_max = (self.xyz_max - center.reshape((3,))) * scale
                self.xyz_min = (self.xyz_min - center.reshape((3,))) * scale
                
        if self.scene_params_dict.get('if_reorient_y_up', False):
            '''
            [TODO] better align normals to axes with clustering or PCA, than manually pick patches
            '''
            print(magenta('Re-orienting scene to y-up...'))
            
            reorient_transform_file = self.scene_path / '_T.npy'
            if reorient_transform_file.exists():
                print('Loading re-orientation from file: ', reorient_transform_file)
                self.reorient_transform = np.load(reorient_transform_file)
                assert self.reorient_transform.shape == (3, 3)
            else:
                assert hasattr(self, 'mi_normal_global_list')
                
                _normal_up_dict = self.scene_params_dict['normal_up_frame_info']
                _frame_id_up = _normal_up_dict['frame_id']
                _normal_up_hw_1 = _normal_up_dict['normal_up_hw_1']; _normal_up_hw_1 = [int(_normal_up_hw_1[0]*(self.H-1)), int(_normal_up_hw_1[1]*(self.W-1))]
                _normal_up_hw_2 = _normal_up_dict['normal_up_hw_2']; _normal_up_hw_2 = [int(_normal_up_hw_2[0]*(self.H-1)), int(_normal_up_hw_2[1]*(self.W-1))]
                _frame_idx_up = self.frame_id_list.index(_frame_id_up)
                normal_up_patch = self.mi_normal_global_list[_frame_idx_up][_normal_up_hw_1[0]:_normal_up_hw_2[0]+1, _normal_up_hw_1[1]:_normal_up_hw_2[1]+1]
                
                _normal_left_dict = self.scene_params_dict['normal_left_frame_info']
                _frame_id_left = _normal_left_dict['frame_id']
                _normal_left_hw_1 = _normal_left_dict['normal_left_hw_1']; _normal_left_hw_1 = [int(_normal_left_hw_1[0]*(self.H-1)), int(_normal_left_hw_1[1]*(self.W-1))]
                _normal_left_hw_2 = _normal_left_dict['normal_left_hw_2']; _normal_left_hw_2 = [int(_normal_left_hw_2[0]*(self.H-1)), int(_normal_left_hw_2[1]*(self.W-1))]
                _frame_idx_left = self.frame_id_list.index(_frame_id_left)
                normal_left_patch = self.mi_normal_global_list[_frame_idx_left][_normal_left_hw_1[0]:_normal_left_hw_2[0]+1, _normal_left_hw_1[1]:_normal_left_hw_2[1]+1]

                import matplotlib.pyplot as plt
                plt.figure()
                plt.subplot(2, 2, 1)
                plt.imshow(self.mi_normal_global_list[_frame_idx_up]/2+0.5)
                plt.subplot(2, 2, 2)
                plt.imshow(normal_up_patch/2+0.5)
                plt.subplot(2, 2, 3)
                plt.imshow(self.mi_normal_global_list[_frame_idx_left]/2+0.5)
                plt.subplot(2, 2, 4)
                plt.imshow(normal_left_patch/2+0.5)
                plt.show()
                y_tmp = np.median(normal_up_patch.reshape(-1, 3), axis=0).flatten()
                y_tmp = y_tmp / (np.linalg.norm(y_tmp)+1e-6)
                x_tmp = - np.median(normal_left_patch.reshape(-1, 3), axis=0).flatten()
                x_tmp = x_tmp / (np.linalg.norm(x_tmp)+1e-6)
                z_tmp = np.cross(x_tmp, y_tmp)
                x_tmp = np.cross(y_tmp, z_tmp)
                
                from scipy.spatial.transform import Rotation as R
                """get rotation matrix between two vectors using scipy"""
                # vec1 = np.reshape(normal_up_tmp, (1, -1))
                # vec2 = np.reshape(np.array([0., 1., 0.]), (1, -1))
                # r = R.align_vectors(vec2, vec1)
                # _R = r[0].as_matrix()
                
                # _axes_current = np.array([[-1., 0., 0.], [0., -1., 0.], [0., 0., 1.]]) @ np.stack((x_tmp, y_tmp, z_tmp)) # (N, 3)
                _axes_current = np.stack((x_tmp, y_tmp, z_tmp)) # (N, 3)
                _axes_target = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]) #  (N, 3)
                self.reorient_transform = np.linalg.inv(_axes_current.T) # https://github.com/nghiaho12/rigid_transform_3D/blob/master/rigid_transform_3D.py; 
                '''
                (? @ _axes_current.T).T = _axes_target
                -> ? = np.linalg.inv(_axes_current.T)
                '''

                np.save(str(self.scene_path / '_T.npy'), self.reorient_transform)

            self.vertices_list = [(self.reorient_transform @ vertices.T).T for vertices in self.vertices_list]
            self.bverts_list = [(self.reorient_transform @ bverts.T).T for bverts in self.bverts_list]
            for pose in self.pose_list: # modify in place
                pose[:3, :3] = self.reorient_transform @ pose[:3, :3]
                pose[:3, 3:4] = self.reorient_transform @ pose[:3, 3:4]
            for (origin, lookatvector, up) in self.origin_lookatvector_up_list:
                origin = self.reorient_transform @ origin
                lookatvector = self.reorient_transform @ lookatvector
                up = self.reorient_transform @ up
                
            __ = np.eye(4, dtype=np.float32); __[:3, :3] = self.reorient_transform; self.reorient_transform = __
            self.load_mi_scene(self.mi_params_dict, monosdf_scale_tuple=monosdf_scale_tuple, extra_transform_homo=self.reorient_transform)
            self.load_modalities()
            self.get_cam_rays(self.cam_params_dict, force=True)
            self.process_mi_scene(self.mi_params_dict, if_postprocess_mi_frames=hasattr(self, 'pose_list'), force=True)
            

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

    def load_mi_scene(self, mi_params_dict={}, monosdf_scale_tuple=(), extra_transform_homo=None):
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
        _T = np.eye(4, dtype=np.float32)
        if self.extra_transform is not None:
            # shape_id_dict['to_world'] = mi.ScalarTransform4f(self.extra_transform_homo)
            _T = self.extra_transform_homo @ _T
        if monosdf_scale_tuple != ():
            assert not self.extra_transform
            center, scale = monosdf_scale_tuple
            scale_mat = np.eye(4).astype(np.float32)
            scale_mat[:3, 3] = -center
            scale_mat[:3 ] *= scale 
            _T = scale_mat @ _T
            # shape_id_dict['to_world'] = mi.ScalarTransform4f(scale_mat)
        if extra_transform_homo is not None:
            # shape_id_dict['to_world'] = mi.ScalarTransform4f(self.extra_transform_homo)
            _T = extra_transform_homo @ _T
            
        if not np.allclose(_T, np.eye(4, dtype=np.float32)):
            shape_id_dict['to_world'] = mi.ScalarTransform4f(_T)

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
        
        if self.pose_format == 'json':
            meta_file_path = self.scene_path / 'transforms.json'
            assert meta_file_path.exists(), 'No meta file found: ' + str(meta_file_path)
            with open(str(meta_file_path), 'r') as f:
                meta = json.load(f)
                
            self.frame_id_list = []
            for idx in range(len(meta['frames'])):
                file_path = meta['frames'][idx]['file_path']
                frame_id = int(file_path.split('/')[-1].split('.')[0].replace('img_', ''))
                self.frame_id_list.append(frame_id)
                
            # dict_keys(['fl_x', 'fl_y', 'cx', 'cy', 'w', 'h', 'camera_model', 'frames'])
            fl_x, fl_y, cx, cy, w, h, camera_model = get_list_of_keys(meta, ['fl_x', 'fl_y', 'cx', 'cy', 'w', 'h', 'camera_model'], [float, float, float, float, int, int, str])
            assert camera_model == 'OPENCV'
            assert int(h) == self.im_params_dict['im_H_load']
            assert int(w) == self.im_params_dict['im_W_load']
            K = np.array([[fl_x, 0, cx], [0, fl_y, cy], [0, 0, 1]], dtype=np.float32)
            if self.im_W_load != self.W or self.im_H_load != self.H:
                scale_factor = [t / s for t, s in zip((self.H, self.W), self.im_HW_load)]
                K = resize_intrinsics(K, scale_factor)

            self.K_list = [K] * len(self.frame_id_list)
            
            for frame_idx in range(len(meta['frames'])):
                c2w = np.array(meta['frames'][frame_idx]['transform_matrix']).astype(np.float32)
                c2w[2, :] *= -1
                c2w = c2w[np.array([1, 0, 2, 3]), :]
                c2w[0:3, 1:3] *= -1

                R_, t_ = np.split(c2w[:3], (3,), axis=1)
                # R = R / np.linalg.norm(R, axis=1, keepdims=True) # somehow R was mistakenly scaled by scale_m2b; need to recover to det(R)=1
                R = R_; t = t_
                # R = R_.T
                # t = -R_.T @ t_
                # t = np.array([[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]], dtype=np.float32) @ t # OpenGL -> OpenCV
                # R = np.array([[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]], dtype=np.float32) @ R # OpenGL -> OpenCV
                # R = np.concatenate([R[:, 1:2], -R[:, 0:1], R[:, 2:]], 1) # [Rui!!] llff specific; done in llff dataloader
                # R = R @ np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]) # [Rui] opengl (llff) -> opencv convention
                self.pose_list.append(np.hstack((R, t)))
                assert np.isclose(np.linalg.det(R), 1.0), 'R is not a rotation matrix'
                
                origin = t
                lookatvector = R @ np.array([[0.], [0.], [1.]], dtype=np.float32)
                up = R @ np.array([[0.], [-1.], [0.]], dtype=np.float32)
                self.origin_lookatvector_up_list.append((origin.reshape((3, 1)), lookatvector.reshape((3, 1)), up.reshape((3, 1))))
                
                # (origin, lookatvector, up) = R_t_to_origin_lookatvector_up_yUP(R, t)
                # origin_lookatvector_up_list.append((origin.reshape((3, 1)), lookatvector.reshape((3, 1)), up.reshape((3, 1))))

        elif self.pose_format in ['bundle']:
            with open(str(self.pose_file), 'r') as camIn:
                cam_data = camIn.read().splitlines()
            
            with open(str(self.pose_file).replace('_bundle.out', '.csv'), 'r') as csvIn:
                csv_data = csvIn.read().splitlines()
            self.frame_id_list = [int(line.split(',')[0].replace('img_', '').replace('.png', '')) for line in csv_data[1:]]
            assert len(self.frame_id_list) == int(cam_data[1].split(' ')[0])
            
            # just double check with lst file
            with open(str(self.pose_file).replace('_bundle.out', '.lst'), 'r') as lstIn:
                lst_data = lstIn.read().splitlines()
            frame_id_list_lst = [int(line.split('\\')[-1].replace('img_', '').replace('.png', '')) for line in lst_data if line != '']
            assert self.frame_id_list == frame_id_list_lst

            for frame_idx in tqdm(range(len(self.frame_id_list))):
                cam_lines = cam_data[(2+frame_idx*5):(2+frame_idx*5+5)]
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

                # (origin, lookatvector, up) = R_t_to_origin_lookatvector_up_yUP(R, t)
                # self.origin_lookatvector_up_list.append((origin.reshape((3, 1)), lookatvector.reshape((3, 1)), up.reshape((3, 1))))
                origin = t
                lookatvector = R @ np.array([[0.], [0.], [1.]], dtype=np.float32)
                up = R @ np.array([[0.], [-1.], [0.]], dtype=np.float32)
                self.origin_lookatvector_up_list.append((origin.reshape((3, 1)), lookatvector.reshape((3, 1)), up.reshape((3, 1))))

                K = np.array([[float(f), 0, self._W(frame_idx)/2.], [0, float(f), self._H(frame_idx)/2.], [0, 0, 1]], dtype=np.float32)
                self.K_list.append(K)
                
        assert len(set(self.frame_id_list)) == len(self.frame_id_list), 'frame_id_list is not unique'
        
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
            self.load_single_shape(shape_params_dict, extra_transform=self.extra_transform)
                
            self.if_loaded_shapes = True
            print(blue_text('[%s] DONE. load_shapes'%(self.__class__.__name__)))

        elif self.scene_params_dict.get('pcd_file', '') != '':
            if self.if_loaded_pcd: return
            pcd_file = self.scene_path / self.scene_params_dict['pcd_file']
            assert pcd_file.exists(), 'No pcd file found: ' + str(pcd_file)
            pcd_trimesh = trimesh.load_mesh(str(pcd_file), process=False)
            self.pcd = np.array(pcd_trimesh.vertices)
            # import ipdb; ipdb.set_trace()
            # np.array([[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]], dtype=np.float32)

            self.xyz_max = np.amax(self.pcd, axis=0)
            self.xyz_min = np.amin(self.pcd, axis=0)
            self.if_loaded_pcd = True
            
            print(blue_text('[%s] DONE. load_pcd: %d points'%(self.__class__.__name__, self.pcd.shape[0])))
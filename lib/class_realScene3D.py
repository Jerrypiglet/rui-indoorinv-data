from pathlib import Path
import numpy as np
import trimesh
np.set_printoptions(suppress=True)
import pyhocon
from tqdm import tqdm
import random
random.seed(0)
import json
import mitsuba as mi

from lib.utils_misc import blue_text, yellow, get_list_of_keys, white_blue, magenta
from lib.utils_io import load_matrix, resize_intrinsics

# from .class_openroomsScene2D import openroomsScene2D
from .class_mitsubaBase import mitsubaBase
from .class_scene2DBase import scene2DBase

from lib.utils_OR.utils_OR_mesh import computeBox
from lib.utils_misc import get_device

from .class_scene2DBase import scene2DBase

class realScene3D(mitsubaBase, scene2DBase):
    '''
    A class used to visualize/render real scenes captured by Mustafa
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

        self.CONF = CONF

        scene2DBase.__init__(
            self, 
            parent_class_name=str(self.__class__.__name__), 
            root_path_dict=root_path_dict, 
            modality_list=modality_list, 
            if_debug_info=if_debug_info, 
            )
        
        mitsubaBase.__init__(
            self, 
            host=host, 
            device_id=device_id, 
        )

        self.scene_name, self.frame_id_list_input, self.axis_up = get_list_of_keys(self.CONF.scene_params_dict, ['scene_name', 'frame_id_list', 'axis_up'], [str, list, str])
        self.invalid_frame_id_list = self.CONF.scene_params_dict.get('invalid_frame_id_list', [])
        self.invalid_frame_idx_list = self.CONF.scene_params_dict.get('invalid_frame_idx_list', [])
        self.frame_id_list_input = [_ for _ in self.frame_id_list_input if _ not in self.invalid_frame_id_list]
        
        # self.indexing_based = self.CONF.scene_params_dict.get('indexing_based', 0)
        
        self.extra_transform = self.CONF.scene_params_dict.get('extra_transform', None)
        if self.extra_transform is not None:
            self.extra_transform_inv = self.extra_transform.T
            self.extra_transform_homo = np.eye(4, dtype=np.float32)
            self.extra_transform_homo[:3, :3] = self.extra_transform

        self.scene_path = self.dataset_root / self.scene_name
        self.scene_rendering_path = self.dataset_root / self.scene_name
        self.scene_rendering_path.mkdir(parents=True, exist_ok=True)
        self.scene_name_full = self.scene_name

        self.pose_format, pose_file = self.CONF.scene_params_dict['pose_file'].split('-')
        assert self.pose_format in ['json', 'bundle'], 'Unsupported pose file: '+self.pose_file
        self.pose_file = self.scene_path / pose_file

        if self.CONF.scene_params_dict.shape_file != '':
            if len(str(self.CONF.scene_params_dict.shape_file).split('/')) == 1:
                self.shape_file_path = self.scene_path / self.CONF.scene_params_dict.shape_file
            else:
                self.shape_file_path = self.dataset_root / self.CONF.scene_params_dict.shape_file
            assert self.shape_file_path.exists(), 'shape file does not exist (have you run Monosdf first?): %s'%str(self.shape_file_path)

        self.near = self.CONF.cam_params_dict.get('near', 0.1)
        self.far = self.CONF.cam_params_dict.get('far', 10.)

        self.load_poses()
        self.scene_rendering_path_list = [self.scene_rendering_path] * len(self.frame_id_list)
        
        '''
        load everything
        '''

        self.load_mi_scene()
        self.load_modalities()

        if hasattr(self, 'pose_list'): 
            self.get_cam_rays()
        if hasattr(self, 'mi_scene'):
            self.process_mi_scene(if_postprocess_mi_frames=hasattr(self, 'pose_list'))

        '''
        re-orient scene to be axis-aligned
        '''
        IF_SCENE_RESCALED = False
        
        self.reorient_transform = np.eye(3, dtype=np.float32)
        self.if_reorient_shape = False
        if self.CONF.scene_params_dict.get('if_reorient_y_up', False):
            '''
            [TODO] better align normals to axes with clustering or PCA, then manually pick patches
            '''
            print(magenta('Re-orienting scene with provided rotation angles...'))

            print('Calculating re-orientation from blender angles input...')
            reorient_blender_angles = self.CONF.scene_params_dict['reorient_blender_angles']
            from scipy.spatial.transform import Rotation
            reorient_blender_angles = np.array(reorient_blender_angles).reshape(3,) / 180. * np.pi
            Rs = Rotation.from_euler('xyz', reorient_blender_angles).as_matrix()
            self.reorient_transform = Rs
            
            if not self.CONF.scene_params_dict.get('if_reorient_y_up_skip_shape', False):
                self.vertices_list = [(self.reorient_transform @ vertices.T).T for vertices in self.vertices_list]
                self.bverts_list = [computeBox(vertices)[0] for vertices in self.vertices_list] # recompute bounding boxes
                self.if_reorient_shape = True
            
            self.pose_list = [np.hstack((self.reorient_transform @ pose[:3, :3], self.reorient_transform @ pose[:3, 3:4])) for pose in self.pose_list] # dont rotate translation!!
            self.origin_lookatvector_up_list = [(self.reorient_transform @ origin, self.reorient_transform @ lookatvector, self.reorient_transform @ up) \
                for (origin, lookatvector, up) in self.origin_lookatvector_up_list] # dont rotate origin!!
            IF_SCENE_RESCALED = True

        if IF_SCENE_RESCALED:
            __ = np.eye(4, dtype=np.float32); __[:3, :3] = self.reorient_transform; self.reorient_transform = __
            if not self.CONF.scene_params_dict.get('if_reorient_y_up_skip_shape', False):
                if hasattr(self, 'mi_scene'):
                    self.load_mi_scene(input_extra_transform_homo=self.reorient_transform)
            # self.load_modalities()
            self.get_cam_rays(force=True)
            if hasattr(self, 'mi_scene'):
                self.process_mi_scene(if_postprocess_mi_frames=hasattr(self, 'pose_list'), force=True)
                
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
    def if_has_shapes(self): # objs + emitters
        return 'shapes' in self.modality_list and self.CONF.scene_params_dict.get('shape_file', '') != ''

    @property
    def if_has_pcd(self):
        return 'shapes' in self.modality_list and self.CONF.scene_params_dict.get('pcd_file', '') != ''


    @property
    def if_has_seg(self):
        return False, 'Segs not saved to labels. Use mi_seg_area, mi_seg_env, mi_seg_obj instead.'
        # return all([_ in self.modality_list for _ in ['seg']])

    def load_modalities(self):
        for _ in self.modality_list:
            result_ = scene2DBase.load_modality_(self, _)
            if not (result_ == False):
                continue
            if _ == 'shapes': self.load_shapes() # shapes of 1(i.e. furniture) + emitters

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

    def load_mi_scene(self, input_extra_transform_homo=None):
        '''
        load scene representation into Mitsuba 3
        '''
        if self.shape_file_path is not None:
            print(yellow('[%s] load_mi_scene from [shape file]'%self.__class__.__name__) + str(self.shape_file_path))
            self.shape_id_dict = {
                'type': self.shape_file_path.suffix[1:],
                'filename': str(self.shape_file_path), 
                }
            
            _T = np.eye(4, dtype=np.float32)
            if self.extra_transform is not None:
                _T = self.extra_transform_homo @ _T
            if input_extra_transform_homo is not None:
                _T = input_extra_transform_homo @ _T
            if not np.allclose(_T, np.eye(4, dtype=np.float32)):
                self.shape_id_dict['to_world'] = mi.ScalarTransform4f(_T)        
            
            self.mi_scene = mi.load_dict({
                'type': 'scene',
                'shape_id': self.shape_id_dict, 
            })
        else:
            # xml file always exists for Mitsuba scenes
            # self.mi_scene = mi.load_file(str(self.xml_file_path))
            print(yellow('No shape file specified. Skip loading MI scene.'))
            return

                
    def load_poses(self):
        print(white_blue('[%s] load_poses from %s'%(self.parent_class_name, str(self.pose_file))))

        self.pose_list = []
        self.K_list = []
        self.origin_lookatvector_up_list = []
        
        if self.pose_format == 'json':
            # self.pose_file = self.scene_path / 'transforms.json'
            assert self.pose_file.exists(), 'No meta file found: ' + str(self.pose_file)
            with open(str(self.pose_file), 'r') as f:
                meta = json.load(f)
                
            self.frame_id_list = []
            for frame_idx in range(len(meta['frames'])):
                file_path = meta['frames'][frame_idx]['file_path']
                frame_id = int(file_path.split('/')[-1].split('.')[0].replace('img_', ''))
                self.frame_id_list.append(frame_id)
            if self.invalid_frame_id_list != []:
                _N = len(self.frame_id_list)
                self.frame_id_list = [x for x in self.frame_id_list if x not in self.invalid_frame_id_list]
                # print('Invalid frame id list: %s'%str(self.invalid_frame_id_list)
                print(magenta('FIRSTLY, removed %d invalid frames with invalid_frame_id_list'%(_N - len(self.frame_id_list))))
            # assert self.invalid_frame_id_list == [], 'not to complicate things'
            if self.invalid_frame_idx_list != []:
                _N = len(self.frame_id_list)
                self.frame_id_list = [x for idx, x in enumerate(self.frame_id_list) if idx not in self.invalid_frame_idx_list]
                print(magenta('THEN, BASED ON NEW IDX, Removed %d invalid frames with invalid_frame_idx_list'%(_N - len(self.frame_id_list))))
                
            for frame_idx, frame_id in enumerate(self.frame_id_list):
                print('frame_idx: %d, frame_id: %d'%(frame_idx, frame_id))
            
            # dict_keys(['fl_x', 'fl_y', 'cx', 'cy', 'w', 'h', 'camera_model', 'frames'])
            fl_x, fl_y, cx, cy, camera_model = get_list_of_keys(meta, ['fl_x', 'fl_y', 'cx', 'cy', 'camera_model'])
            w = int(meta['w']); h = int(meta['h'])
            assert camera_model == 'OPENCV'
            assert int(h) == self.CONF.im_params_dict['im_H_load']
            assert int(w) == self.CONF.im_params_dict['im_W_load']
            K = np.array([[fl_x, 0, cx], [0, fl_y, cy], [0, 0, 1]], dtype=np.float32)
            if self.im_W_load != self.W or self.im_H_load != self.H:
                scale_factor = [t / s for t, s in zip((self.H, self.W), self.im_HW_load)]
                K = resize_intrinsics(K, scale_factor)

            self.K_list = [K] * len(self.frame_id_list)
            
            for frame_idx in range(len(meta['frames'])):
                file_path = meta['frames'][frame_idx]['file_path']
                frame_id = int(file_path.split('/')[-1].split('.')[0].replace('img_', ''))
                if frame_id in self.invalid_frame_id_list:
                    continue

                c2w = np.array(meta['frames'][frame_idx]['transform_matrix']).astype(np.float32)
                c2w[2, :] *= -1
                c2w = c2w[np.array([1, 0, 2, 3]), :]
                c2w[0:3, 1:3] *= -1

                R_, t_ = np.split(c2w[:3], (3,), axis=1)
                R = R_; t = t_
                if self.extra_transform is not None:
                    assert self.extra_transform.shape == (3, 3) # [TODO] support 4x4
                    R = self.extra_transform[:3, :3] @ R
                self.pose_list.append(np.hstack((R, t)))
                assert np.isclose(np.linalg.det(R), 1.0), 'R is not a rotation matrix'
                
                origin = t
                lookatvector = R @ np.array([[0.], [0.], [1.]], dtype=np.float32)
                up = R @ np.array([[0.], [-1.], [0.]], dtype=np.float32)
                self.origin_lookatvector_up_list.append((origin.reshape((3, 1)), lookatvector.reshape((3, 1)), up.reshape((3, 1))))
                
                # (origin, lookatvector, up) = R_t_to_origin_lookatvector_up_yUP(R, t)
                # origin_lookatvector_up_list.append((origin.reshape((3, 1)), lookatvector.reshape((3, 1)), up.reshape((3, 1))))
                
            if self.invalid_frame_idx_list is not None:
                self.pose_list = [x for idx, x in enumerate(self.pose_list) if idx not in self.invalid_frame_idx_list]
                self.origin_lookatvector_up_list = [x for idx, x in enumerate(self.origin_lookatvector_up_list) if idx not in self.invalid_frame_idx_list]

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
                if self.extra_transform is not None:
                    assert self.extra_transform.shape == (3, 3) # [TODO] support 4x4
                    R = self.extra_transform[:3, :3] @ R
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
        
        self.frame_id_list_input_all = self.frame_id_list.copy()
        
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
            
    def get_cam_rays(self, force=False):
        if hasattr(self, 'cam_rays_list') and not force:  return
        self.cam_rays_list = self.get_cam_rays_list(self.H, self.W, self.K_list, self.pose_list, convention='opencv')

    def load_shapes(self):
        '''
        load and visualize shapes (objs/furniture **& emitters**) / point cloud (pcd)
        '''
        
        print(white_blue('[%s] load_shapes for scene...')%self.__class__.__name__)
        
        if self.CONF.scene_params_dict.get('shape_file', '') != '':
            if self.if_loaded_shapes: 
                print('already loaded shapes. skip.')
                return
            mitsubaBase._prepare_shapes(self)
            self.shape_file = self.CONF.scene_params_dict['shape_file']
            self.load_single_shape(self.CONF.shape_params_dict, extra_transform=self.extra_transform)
                
            self.if_loaded_shapes = True
            print(blue_text('[%s] DONE. load_shapes'%(self.__class__.__name__)))

        elif self.CONF.scene_params_dict.get('pcd_file', '') != '':
            if self.if_loaded_pcd: return
            pcd_file = self.scene_path / self.CONF.scene_params_dict['pcd_file']
            assert pcd_file.exists(), 'No pcd file found: ' + str(pcd_file)
            pcd_trimesh = trimesh.load_mesh(str(pcd_file), process=False)
            self.pcd = np.array(pcd_trimesh.vertices)
            if self.extra_transform is not None:
                assert self.extra_transform.shape == (3, 3)
                self.pcd = self.pcd @ self.extra_transform.T
            # import ipdb; ipdb.set_trace()
            # np.array([[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]], dtype=np.float32)

            self.xyz_max = np.amax(self.pcd, axis=0)
            self.xyz_min = np.amin(self.pcd, axis=0)
            self.if_loaded_pcd = True
            
            print(blue_text('[%s] DONE. load_pcd: %d points'%(self.__class__.__name__, self.pcd.shape[0])))
            
    def _get_reorient_mat_(self):
        '''
        Obsolete.
        Get reorient matrix for the scene, by picking a normal on the wall and a normal on the floor
        '''
        assert hasattr(self, 'mi_normal_global_list')
        
        _normal_up_dict = self.CONF.scene_params_dict['normal_up_frame_info']
        _frame_id_up = _normal_up_dict['frame_id']
        _normal_up_hw_1 = _normal_up_dict['normal_up_hw_1']; _normal_up_hw_1 = [int(_normal_up_hw_1[0]*(self.H-1)), int(_normal_up_hw_1[1]*(self.W-1))]
        _normal_up_hw_2 = _normal_up_dict['normal_up_hw_2']; _normal_up_hw_2 = [int(_normal_up_hw_2[0]*(self.H-1)), int(_normal_up_hw_2[1]*(self.W-1))]
        _frame_idx_up = self.frame_id_list.index(_frame_id_up)
        normal_up_patch = self.mi_normal_global_list[_frame_idx_up][_normal_up_hw_1[0]:_normal_up_hw_2[0]+1, _normal_up_hw_1[1]:_normal_up_hw_2[1]+1]
        
        _normal_left_dict = self.CONF.scene_params_dict['normal_left_frame_info']
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
        '''
        (? @ _axes_current.T).T = _axes_target
        -> ? = np.linalg.inv(_axes_current.T)
        '''
        _axes_current = np.stack((x_tmp, y_tmp, z_tmp)) # (N, 3)
        _axes_target = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]) #  (N, 3)
        reorient_transform = np.linalg.inv(_axes_current.T) # https://github.com/nghiaho12/rigid_transform_3D/blob/master/rigid_transform_3D.py; 
        
        return reorient_transform

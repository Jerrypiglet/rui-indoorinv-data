from pathlib import Path, PosixPath
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import numpy as np
import cv2
import scipy.ndimage as ndimage
np.set_printoptions(suppress=True)
import os
import glob
from collections import defaultdict
from lib.utils_io import load_matrix, load_img, resize_img, load_HDR, scale_HDR, load_binary, load_h5, load_envmap, resize_intrinsics
from lib.utils_matseg import get_map_aggre_map
from lib.utils_misc import blue_text, get_list_of_keys, green, white_blue, red, check_list_of_tensors_size
from lib.utils_openrooms import load_OR_public_poses_to_Rt
from lib.utils_OR.utils_OR_lighting import convert_lighting_axis_local_to_global_np, convert_SG_angles_to_axis_local_np
from lib.utils_rendering_openrooms import renderingLayer
from tqdm import tqdm
import pickle

class openroomsScene2D(object):
    '''
    A class used to **load** OpenRooms (public/public-re versions) scene contents (2D/2.5D per-pixel DENSE properties for inverse rendering).
    For high-level semantic properties (e.g. layout, objects, emitters, use class: openroomsScene3D)
    '''
    def __init__(
        self, 
        root_path_dict: dict, 
        scene_params_dict: dict, 
        modality_list: list, 
        im_params_dict: dict={'im_H_load': 480, 'im_W_load': 640, 'im_H_resize': 480, 'im_W_resize': 640}, 
        BRDF_params_dict: dict={}, 
        lighting_params_dict: dict={'env_row': 120, 'env_col': 160, 'SG_num': 12, 'env_height': 16, 'env_width': 32}, # params to load & convert lighting SG & envmap to 
        if_debug_info: bool=False, 
        ):

        '''
        scene properties
        - frame_id_list are integers as seen in frame file names (e.g. im_1.png -> 1)
        '''
        self.if_save_storage = scene_params_dict.get('if_save_storage', False) # set to True to enable removing duplicated renderer files (e.g. only one copy of geometry files in main, or emitter files only in main and mainDiffMat)
        self.if_debug_info = if_debug_info

        self.meta_split, self.scene_name = get_list_of_keys(scene_params_dict, ['meta_split', 'scene_name'])
        assert self.meta_split in ['main_xml', 'mainDiffMat_xml', 'mainDiffLight_xml', 'main_xml1', 'mainDiffMat_xml1', 'mainDiffLight_xml1']
        assert self.scene_name.startswith('scene')
        self.scene_name_short = '_'.join(self.scene_name.split('_')[:2]) # e.g. scene_name: scene0552_00_more, scene_name_short: scene0552_00

        self.openrooms_version = scene_params_dict.get('openrooms_version', 'public_re')
        assert self.openrooms_version in ['public_re', 'public'] # TODO 'public' is not tested yet!

        self.indexing_based = scene_params_dict.get('indexing_based', 0)
        assert self.indexing_based in [0, 1], 'indexing of frame names (indexing_based) has to be either 0-based or 1-based! got: indexing_based = %d'%self.indexing_based
        assert self.indexing_based == {'public_re': 0, 'public': 1}[self.openrooms_version]

        if scene_params_dict.get('frame_id_list', []) != []: 
            self.frame_id_list = scene_params_dict.get('frame_id_list')
        else:
            print(white_blue('[openroomsScene] frame_id_list was not provided; getting frame_id_list by digging into image list...'))
            im_hdr_list = sorted(glob.glob(str(self.scene_rendering_path / 'im_*.hdr')))
            assert len(im_hdr_list) > 0, 'no image with name: im_*.hdr is found at %s!'%str(self.scene_rendering_path)
            # self.frame_id_list = sorted([int(_.split('.')[0].replace('im_', '')) for _ in im_hdr_list])
            self.frame_id_list = sorted([int(Path(_).stem.replace('im_', '')) for _ in im_hdr_list])
            print(white_blue('... got frame_id_list: [%s]'%(', '.join([str(_) for _ in self.frame_id_list]))))
        
        self.num_frames = len(self.frame_id_list)

        '''
        paths
        '''

        self.root_path_dict = root_path_dict
        self.PATH_HOME, self.rendering_root, self.xml_scene_root, self.semantic_labels_root = get_list_of_keys(
            self.root_path_dict, 
            ['PATH_HOME', 'rendering_root', 'xml_scene_root', 'semantic_labels_root'], 
            [PosixPath, PosixPath, PosixPath, PosixPath]
            )

        self.scene_rendering_path = self.rendering_root / self.meta_split / self.scene_name
        self.scene_xml_path = self.xml_scene_root / (self.meta_split.split('_')[1]) / self.scene_name
        self.intrinsics_path = self.scene_rendering_path / 'intrinsic.txt'

        '''
        im properties
        '''

        self.im_sdr_ext = im_params_dict.get('im_sdr_ext', 'png')
        self.im_hdr_ext = im_params_dict.get('im_hdr_ext', 'hdr')
        self.if_scale_hdr = im_params_dict.get('if_scale_hdr', True) # scale HDR images with segs
        self.if_scale_hdr_per_frame = im_params_dict.get('if_scale_hdr_per_frame', False) # True: individually scale each HDR frame; False: get one global HDR scale
        self.if_clip_HDR_to_01 = im_params_dict.get('if_clip_HDR_to_01', False) # only useful when scaling HDR
        
        if_direct_lighting = im_params_dict.get('if_direct_lighting', False)
        self.im_key = {True: 'imDirect_', False: 'im_'}[if_direct_lighting]
        self.imsgEnv_key = {True: 'imsgEnvDirect_', False: 'imsgEnv_'}[if_direct_lighting]
        self.imenv_key = {True: 'imenvDirect_', False: 'imenv_'}[if_direct_lighting]

        # self.im_params_dict = im_params_dict
        self.im_H_load, self.im_W_load, self.im_H_resize, self.im_W_resize = get_list_of_keys(im_params_dict, ['im_H_load', 'im_W_load', 'im_H_resize', 'im_W_resize'])
        self.if_resize_im = (self.im_H_load, self.im_W_load) != (self.im_H_resize, self.im_W_resize) # resize modalities (exclusing lighting)
        self.im_target_HW = () if not self.if_resize_im else (self.im_H_resize, self.im_W_resize)
        self.H, self.W = self.im_H_resize, self.im_W_resize

        '''
        BRDF, lighting properties
        '''
        self.clip_roughness_min_to = BRDF_params_dict.get('clip_roughness_min_to', 0.)
        assert self.clip_roughness_min_to >= 0. and self.clip_roughness_min_to <= 1.

        self.lighting_params_dict = lighting_params_dict
        self.if_convert_lighting_SG_to_global = lighting_params_dict.get('if_convert_lighting_SG_to_global', False)
        # self.rL = renderingLayer(imWidth=self.lighting_params_dict['env_col'], imHeight=self.lighting_params_dict['env_row'], isCuda=False)
        self.im_lighting_HW_ratios = (self.im_H_resize // self.lighting_params_dict['env_row'], self.im_W_resize // self.lighting_params_dict['env_col'])

        '''
        modalities to load
        '''
        self.modality_list = self.check_and_sort_modalities(list(set(modality_list)))
        if 'im_hdr' in self.modality_list and self.if_scale_hdr:
            assert 'seg' in self.modality_list

        ''''
        flags to set
        '''
        self.pts_from = {'mi': False, 'depth': False}
        self.seg_from = {'mi': False, 'seg': False}

        '''
        load everything
        '''
        self.load_modalities()

    @property
    def valid_modalities(self):
        return [
            'im_hdr', 'im_sdr', 'poses', 
            'albedo', 'roughness', 'depth', 'normal', 
            'seg', 'seg_area', 'seg_env', 'seg_obj', 
            'lighting_SG', 'lighting_envmap', 
            'semseg', 'matseg', 
            ]

    def check_and_sort_modalities(self, modalitiy_list):
        modalitiy_list_new = [_ for _ in self.valid_modalities if _ in modalitiy_list]
        for _ in modalitiy_list_new:
            assert _ in self.valid_modalities, 'Invalid modality: %s'%_
        return modalitiy_list_new

    @property
    def if_has_poses(self):
        return all([_ in self.modality_list for _ in ['poses']])

    @property
    def if_has_im_sdr(self):
        return all([_ in self.modality_list for _ in ['im_sdr']])

    @property
    def if_has_im_hdr(self):
        return all([_ in self.modality_list for _ in ['im_hdr']])

    @property
    def if_has_hdr_scale(self):
        return all([_ in self.modality_list for _ in ['im_hdr']]) and self.if_scale_hdr

    @property
    def if_has_seg(self):
        return all([_ in self.modality_list for _ in ['seg']])

    @property
    def if_has_dense_geo(self):
        return all([_ in self.modality_list for _ in ['depth', 'normal']])

    @property
    def if_has_BRDF(self):
        return all([_ in self.modality_list for _ in ['albedo', 'roughness']])

    @property
    def if_has_lighting_envmap(self):
        return all([_ in self.modality_list for _ in ['lighting_envmap']])

    @property
    def if_has_lighting_SG(self):
        return all([_ in self.modality_list for _ in ['lighting_SG']])

    @property
    def if_has_semseg(self):
        return all([_ in self.modality_list for _ in ['semseg']])

    @property
    def frame_num(self):
        return len(self.frame_id_list)

    def get_modality(self, modality):
        if modality == 'im_sdr': 
            return self.im_sdr_list
        elif modality == 'im_hdr': 
            return self.im_hdr_list
        elif modality == 'seg': 
            return self.seg_dict_of_lists
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
        elif modality == 'lighting_SG': 
            return self.lighting_SG_local_list
        elif modality == 'lighting_envmap': 
            return self.lighting_envmap_list
        elif modality == 'semseg': 
            return self.semseg_list
        elif modality == 'matseg': 
            return self.matseg_list
        elif modality == 'seg_area': 
            return self.seg_dict_of_lists['area']
        elif modality == 'seg_env': 
            return self.seg_dict_of_lists['env']
        elif modality == 'seg_obj': 
            return self.seg_dict_of_lists['obj']
        else:
            assert False, 'Unsupported modality: ' + modality

    def load_modalities(self):
        for _ in self.modality_list:
            # assert _ in ['im_hdr', 'im_sdr', 'albedo', 'roughness', 'depth', 'normal', 'seg', 'lighting_SG', 'lighting_envmap', 'poses']
            if _ == 'im_sdr': self.load_im_sdr()
            if _ == 'seg': self.load_seg()
            if _ == 'im_hdr': self.load_im_hdr()
            if _ == 'poses': self.load_poses()
            if _ == 'albedo': self.load_albedo()
            if _ == 'roughness': self.load_roughness()
            if _ == 'depth': self.load_depth()
            if _ == 'normal': self.load_normal()
            if _ == 'lighting_SG': self.load_lighting_SG()
            if _ == 'lighting_envmap': self.load_lighting_envmap()
            if _ == 'semseg': self.load_semseg()
            if _ == 'matseg': self.load_matseg()

    def load_intrinsics(self):
        '''
        -> K: (3, 3)
        '''
        self.K = load_matrix(self.intrinsics_path)
        # self.K = load_matrix('/Users/jerrypiglet/Documents/Projects/OpenRooms_RAW_loader/data/public_re_3/main_xml/scene0008_00_more/intrinsic.txt')
        assert self.K.shape == (3, 3)
        assert self.K[0][2] == float(self.im_W_load) / 2.
        assert self.K[1][2] == float(self.im_H_load) / 2.

        if self.if_resize_im:        
            scale_factor = [t / s for t, s in zip((self.im_H_resize, self.im_W_resize), (self.im_H_load, self.im_W_load))]
            self.K = resize_intrinsics(self.K, scale_factor)
        
    def load_im_sdr(self):
        '''
        load im in SDR; RGB, (H, W, 3), [0., 1.]
        '''
        print(white_blue('[openroomsScene] load_im_sdr for %d frames...'%len(self.frame_id_list)))

        self.im_sdr_ext in ['jpg', 'png']

        self.im_sdr_file_list = [self.scene_rendering_path / ('%s%d.%s'%(self.im_key, i, self.im_sdr_ext)) for i in self.frame_id_list]
        self.im_sdr_list = [load_img(_, (self.im_H_load, self.im_W_load, 3), ext=self.im_sdr_ext, target_HW=self.im_target_HW)/255. for _ in self.im_sdr_file_list]
        # check_list_of_tensors_size(self.im_sdr_list, (self.im_H_load, self.im_W_load, 3))

        print(blue_text('[openroomsScene] DONE. load_im_sdr'))

    def load_im_hdr(self):
        '''
        load im in HDR; RGB, (H, W, 3), [0., inf]
        '''

        print(white_blue('[openroomsScene] load_im_hdr for %d frames...'%len(self.frame_id_list)))

        self.im_hdr_ext in ['hdr'] # .rgbe not supported for now
        self.im_hdr_file_list = [self.scene_rendering_path / ('%s%d.%s'%(self.im_key, i, self.im_hdr_ext)) for i in self.frame_id_list]
        self.im_hdr_list = [load_HDR(_, (self.im_H_load, self.im_W_load, 3), target_HW=self.im_target_HW) for _ in self.im_hdr_file_list]

        if self.if_scale_hdr:
            if not hasattr(self, 'seg_dict_of_lists'):
                self.load_seg()

            self.hdr_scale_list = []
            if self.if_scale_hdr_per_frame:
                hdr_scale_list_ = []
                for im_hdr, seg_ori in zip(self.im_hdr_list, self.seg_dict_of_lists['ori']):
                    hdr_scale_ = scale_HDR(im_hdr, seg_ori[..., np.newaxis], fixed_scale=True, if_return_scale_only=True)
                    hdr_scale_list_.append(hdr_scale_)
                hdr_scale_global = np.median(hdr_scale_list_)
            else:
                hdr_scale_global = None

            for _, (im_hdr, seg_ori) in enumerate(zip(self.im_hdr_list, self.seg_dict_of_lists['ori'])):
                im_hdr_scaled, hdr_scale = scale_HDR(im_hdr, seg_ori[..., np.newaxis], scale_input=hdr_scale_global, if_clip_to_01=self.if_clip_HDR_to_01)
                self.im_hdr_list[_] = im_hdr_scaled
                self.hdr_scale_list.append(hdr_scale)
              
        print(blue_text('[openroomsScene] DONE. load_im_hdr'))


    def load_seg(self):
        '''
        return 3 bool masks; (H, W), float32 0./1.
        '''
        if hasattr(self, 'seg_dict_of_lists'): return

        print(white_blue('[openroomsScene] load_seg for %d frames...'%len(self.frame_id_list)))

        self.seg_dict_of_lists = defaultdict(list)

        for i in self.frame_id_list:
            seg_path = self.scene_rendering_path / ('immask_%d.png'%i)
            seg = load_img(seg_path, (self.im_H_load, self.im_W_load, 3), target_HW=self.im_target_HW, resize_method='nearest')[:, :, 0] / 255. # [0., 1.]

            seg_area = np.logical_and(seg > 0.49, seg < 0.51).astype(np.float32)
            seg_env = (seg < 0.1).astype(np.float32)
            seg_obj = (seg > 0.9) 

            if 'lighting_SG' in self.modality_list or 'lighting_envmap' in self.modality_list:
                seg_obj = seg_obj.squeeze()
                seg_obj = ndimage.binary_erosion(seg_obj, structure=np.ones((7, 7)),
                        border_value=1)

            seg_obj = seg_obj.squeeze().astype(np.float32)

            self.seg_dict_of_lists['ori'].append(seg)
            self.seg_dict_of_lists['area'].append(seg_area)
            self.seg_dict_of_lists['env'].append(seg_env)
            self.seg_dict_of_lists['obj'].append(seg_obj)

        print(blue_text('[openroomsScene] DONE. load_seg'))

        self.seg_from['seg'] = True

    def load_transforms(self):
        # load transformations # writeShapeToXML.py L588
        transform_file = self.scene_xml_path / 'transform.dat'
        with open(str(transform_file), 'rb') as fIn:
            self.transforms = pickle.load(fIn)

    def load_poses(self):
        '''
        pose_list: list of pose matrices (**camera-to-world** transformation), each (3, 4): [R|t] (OpenCV convention: right-down-forward)
        '''

        self.load_intrinsics()

        if hasattr(self, 'pose_list'): return

        print(white_blue('[openroomsScene] load_poses for %d frames...'%len(self.frame_id_list)))

        if not hasattr(self, 'transforms'):
            self.load_transforms()

        self.pose_list, self.origin_lookatvector_up_list = load_OR_public_poses_to_Rt(self.transforms, self.scene_xml_path, self.frame_id_list, False, if_1_based=self.indexing_based==1)

        if self.if_resize_im:
            pass # IMPORTANT! do nothing; keep the 3D scene (cameras and geometry), but instead resize intrinsics to account for the smaller image
        
        print(blue_text('[openroomsScene] DONE. load_poses'))

    def load_albedo(self):
        '''
        albedo; loaded in [0., 1.] LDR, then **2.2 -> HDR;
        (H, W, 3), [0., 1.]
        '''
        if hasattr(self, 'albedo_list'): return

        print(white_blue('[openroomsScene] load_albedo for %d frames...'%len(self.frame_id_list)))

        albedo_files = [self.scene_rendering_path / ('imbaseColor_%d.png'%i) for i in self.frame_id_list]
        self.albedo_list = [load_img(albedo_file, (self.im_H_load, self.im_W_load, 3), ext='png', target_HW=self.im_target_HW).astype(np.float32)/255. for albedo_file in albedo_files]
        self.albedo_list = [albedo**2.2 for albedo in self.albedo_list]
        
        print(blue_text('[openroomsScene] DONE. load_albedo'))

    def load_roughness(self):
        '''
        roughness; smaller, the more specular;
        (H, W, 1), [0., 1.]
        '''
        if hasattr(self, 'roughness_list'): return

        print(white_blue('[openroomsScene] load_roughness for %d frames...'%len(self.frame_id_list)))

        roughness_files = [self.scene_rendering_path / ('imroughness_%d.png'%i) for i in self.frame_id_list]
        self.roughness_list = [load_img(roughness_file, (self.im_H_load, self.im_W_load, 3), ext='png', target_HW=self.im_target_HW)[:, :, 0:1].astype(np.float32)/255. for roughness_file in roughness_files]

        print(blue_text('[openroomsScene] DONE. load_roughness'))

    def load_depth(self):
        '''
        depth;
        (H, W), ideally in [0., inf]
        '''
        if hasattr(self, 'depth_list'): return

        print(white_blue('[openroomsScene] load_depth for %d frames...'%len(self.frame_id_list)))

        depth_files = [self.scene_rendering_path / ('imdepth_%d.dat'%i) for i in self.frame_id_list]
        self.depth_list = [load_binary(depth_file, (self.im_H_load, self.im_W_load), target_HW=self.im_target_HW, resize_method='area')for depth_file in depth_files] # TODO: better resize method for depth for anti-aliasing purposes and better boundaries, and also using segs?
        
        print(blue_text('[openroomsScene] DONE. load_depth'))

        self.pts_from['depth'] = True
        

    def load_normal(self):
        '''
        normal, in camera coordinates (OpenGL convention: right-up-backward);
        (H, W, 3), [-1., 1.]
        '''
        if hasattr(self, 'normal_list'): return

        print(white_blue('[openroomsScene] load_normal for %d frames...'%len(self.frame_id_list)))

        normal_files = [self.scene_rendering_path / ('imnormal_%d.png'%i) for i in self.frame_id_list]
        self.normal_list = [load_img(normal_file, (self.im_H_load, self.im_W_load, 3), ext='png', target_HW=self.im_target_HW).astype(np.float32)/255.*2.-1. for normal_file in normal_files] # -> [-1., 1.], pointing inward (i.e. notebooks/images/openrooms_normals.jpg)
        self.normal_list = [normal / np.sqrt(np.maximum(np.sum(normal**2, axis=2, keepdims=True), 1e-5)) for normal in self.normal_list]
        
        print(blue_text('[openroomsScene] DONE. load_normal'))

    def load_lighting_SG(self):
        '''
        lighting in SG;
        self.if_convert_lighting_SG_to_global = False: 
            (H', W', SG_num, 7(axis_local, lamb, weight: 3, 1, 3)), in camera-and-normal-dependent local coordinates
        self.if_convert_lighting_SG_to_global = True: 
            (H', W', SG_num, 7(axis, lamb, weight: 3, 1, 3)), in global coordinates (OpenCV convention: right-down-forward)
        '''
        if hasattr(self, 'lighting_SG_local_list'): return

        print(white_blue('[openroomsScene] load_lighting_SG for %d frames...'%len(self.frame_id_list)))
        print(red('THIS MIGHT BE SLOW...'))

        lighting_SG_files = [self.scene_rendering_path / ('%s%d.h5'%(self.imsgEnv_key, i)) for i in self.frame_id_list]

        self.lighting_SG_local_list = []

        for frame_idx, lighting_SG_file in enumerate(tqdm(lighting_SG_files)):
            lighting_SG = load_h5(lighting_SG_file)
            if 'im_hdr' in self.modality_list and self.if_scale_hdr:
                hdr_scale = self.hdr_scale_list[frame_idx]
                lighting_SG[:, :, :, 3:6] = lighting_SG[:, :, :, 3:6] * hdr_scale # (120, 160, 12(SG_num), 6); theta, phi, lamb, weight: 1, 1, 1, 3
            lighting_SG = np.concatenate(
                (convert_SG_angles_to_axis_local_np(lighting_SG[:, :, :, :2]),  # (120, 160, 12(SG_num), 7); axis_local, lamb, weight: 3, 1, 3
                lighting_SG[:, :, :, 2:]), axis=3)
            self.lighting_SG_local_list.append(lighting_SG)

        env_row, env_col = self.lighting_params_dict['env_row'], self.lighting_params_dict['env_col']
        assert all([tuple(_.shape)==(env_row, env_col, self.lighting_params_dict['SG_num'], 7) for _ in self.lighting_SG_local_list])

        if self.if_convert_lighting_SG_to_global:
            if hasattr(self, 'pose_list'): self.load_poses()

            self.lighting_SG_global_list = []
            for lighting_SG_local, pose, normal in zip(self.lighting_SG_local_list, self.pose_list, self.normal_list):
                lighting_SG_global = np.concatenate(
                    (convert_lighting_axis_local_to_global_np(lighting_SG_local[:, :, :, :3], pose, normal), 
                    lighting_SG_local[:, :, :, 3:]), axis=3) # (120, 160, 12(SG_num), 7); axis, lamb, weight: 3, 1, 3
                self.lighting_SG_global_list.append(lighting_SG_global)
            assert all([tuple(_.shape)==(env_row, env_col, self.lighting_params_dict['SG_num'], 7) for _ in self.lighting_SG_global_list])

        print(blue_text('[openroomsScene] DONE. load_lighting_SG'))


    def load_lighting_envmap(self):
        if hasattr(self, 'lighting_envmap_list'): return

        print(white_blue('[openroomsScene] load_lighting_envmap for %d frames...'%len(self.frame_id_list)))
        print(red('THIS MIGHT BE SLOW...'))

        env_height, env_width = self.lighting_params_dict['env_height'], self.lighting_params_dict['env_width']
        env_row, env_col = self.lighting_params_dict['env_row'], self.lighting_params_dict['env_col']

        # if self.openrooms_version == 'public_re' and (env_height, env_width) == (8, 16):
        #     lighting_envmap_files = [self.scene_rendering_path / ('%s8x16_%d.hdr'%(imenv_key, i)) for i in self.frame_id_list]
        # else:
        lighting_envmap_files = [self.scene_rendering_path / ('%s%d.hdr'%(self.imenv_key, i)) for i in self.frame_id_list]

        self.lighting_envmap_list = []

        for idx, lighting_envmap_file in enumerate(tqdm(lighting_envmap_files)):
            envmap = load_envmap(str(lighting_envmap_file), env_height=env_height, env_width=env_width)[0].transpose(1, 2, 0, 3, 4) # -> (120, 160, 3, 8, 16)
            if 'im_hdr' in self.modality_list and self.if_scale_hdr:
                hdr_scale = self.hdr_scale_list[idx]
                envmap = envmap * hdr_scale
    
            self.lighting_envmap_list.append(envmap)

            assert all([tuple(_.shape)==(env_row, env_col, 3, env_height, env_width) for _ in self.lighting_envmap_list])

        print(blue_text('[openroomsScene] DONE. load_lighting_envmap'))

    def load_semseg(self):
        '''
        semseg, image space
        (H, W, 3), [-1., 1.]
        '''
        if hasattr(self, 'semseg_list'): return

        print(white_blue('[openroomsScene] load_semseg for %d frames...'%len(self.frame_id_list)))

        semseg_files = [self.scene_rendering_path / ('imsemLabel_%d.npy'%i) for i in self.frame_id_list]
        self.semseg_list = [
            load_img(semseg_file, (self.im_H_load, self.im_W_load), ext='npy', target_HW=self.im_target_HW, resize_method='nearest')
            for semseg_file in semseg_files]
        
        print(blue_text('[openroomsScene] DONE. load_semseg'))

    def load_matseg(self):
        '''
        matseg, dict:
        - mat_aggre_map: (H, W), int32; [0, 1, ..., num_mat_masks], 0 for invalid region
        - num_mat_masks: == np.amax(mat_aggre_map)
        '''

        if hasattr(self, 'matseg_list'): return

        print(white_blue('[openroomsScene] load_matseg for %d frames...'%len(self.frame_id_list)))

        imcadmatobj_files = [self.scene_rendering_path / ('imcadmatobj_%d.dat'%i) for i in self.frame_id_list]
        self.imcadmatobj_list = [
            load_binary(
                imcadmatobj_file, 
                (self.im_H_load, self.im_W_load), 
                target_HW=self.im_target_HW, 
                channels=3, dtype=np.int32, resize_method='nearest'
                ) 
            for imcadmatobj_file in imcadmatobj_files]

        self.matseg_list = []
        for mask in self.imcadmatobj_list:
            mat_aggre_map, num_mat_masks = get_map_aggre_map(mask) # 0 for invalid region

            h, w = mat_aggre_map.shape
            segmentation = np.zeros([50, h, w], dtype=np.uint8)
            for i in range(num_mat_masks+1):
                if i == 0:
                    # deal with backgroud
                    seg = mat_aggre_map == 0
                    segmentation[num_mat_masks, :, :] = seg.reshape(h, w) # segmentation[num_mat_masks] for invalid mask
                else:
                    seg = mat_aggre_map == i
                    segmentation[i-1, :, :] = seg.reshape(h, w) # segmentation[0..num_mat_masks-1] for plane instances
            matseg_dict = {
                'mat_aggre_map': mat_aggre_map, # (H, W), int32; [0, 1, ..., num_mat_masks], 0 for invalid region
                'num_mat_masks': num_mat_masks,  
                'instances': segmentation, # (50, 240, 320), np.uint8; instance[0:num_mat_masks] are each for one mat part; instance[num_mat_masks] is for backround/emitters (e.g. regions of lamps/windows)
                # 'semantic': 1 - torch.FloatTensor(segmentation[num_mat_masks, :, :]).unsqueeze(0), # torch.Size([50, 240, 320]) torch.Size([1, 240, 320])
            }
            self.matseg_list.append(matseg_dict)
        
        print(blue_text('[openroomsScene] DONE. load_semseg'))


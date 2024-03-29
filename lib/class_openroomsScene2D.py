from pathlib import Path, PosixPath
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import numpy as np
import scipy.ndimage as ndimage
import pyhocon

from lib.utils_OR.utils_OR_cam import read_cam_params_OR
np.set_printoptions(suppress=True)
import os
import glob
from collections import defaultdict
from lib.utils_io import load_matrix, load_img, resize_img, load_HDR, scale_HDR, load_binary, load_h5, load_envmap, resize_intrinsics
from lib.utils_matseg import get_map_aggre_map
from lib.utils_misc import blue_text, get_list_of_keys, green, white_blue, red, yellow
from lib.utils_openrooms import load_OR_public_poses_to_Rt
from lib.utils_OR.utils_OR_lighting import convert_lighting_axis_local_to_global_np, convert_SG_angles_to_axis_local_np
from lib.utils_rendering_openrooms import renderingLayer
from tqdm import tqdm
import pickle

from .class_scene2DBase import scene2DBase

class openroomsScene2D(scene2DBase):
    '''
    A class used to **load** OpenRooms (public/public-re versions) scene contents (2D/2.5D per-pixel DENSE properties for inverse rendering).
    For high-level semantic properties (e.g. layout, objects, emitters, use class: %s3D)
    '''
    def __init__(
        self, 
        modality_list: list, 
        root_path_dict: dict, 
        CONF: pyhocon.config_tree.ConfigTree,  
        host: str='', 
        device_id: int=-1, 
        if_debug_info: bool=False, 
        if_not_load_modalities: bool=False,
        ):

        scene2DBase.__init__(
            self, 
            CONF=CONF,
            parent_class_name=str(self.__class__.__name__),  
            root_path_dict=root_path_dict, 
            modality_list=modality_list, 
            if_debug_info=if_debug_info, 
            )

        '''
        scene properties
        - frame_id_list are integers as seen in frame file names (e.g. im_1.png -> 1)
        '''

        # self.meta_split, scene_name = self.scene_name.split('-')
        assert self.meta_split in ['main_xml', 'mainDiffMat_xml', 'mainDiffLight_xml', 'main_xml1', 'mainDiffMat_xml1', 'mainDiffLight_xml1']
        # self.update_scene_name(scene_name)
        assert self.scene_name.startswith('scene')
        
        self.openrooms_version = self.CONF.scene_params_dict.get('openrooms_version', 'public_re_0203')
        assert self.openrooms_version in ['public_re_0203', 'public'] # TODO 'public' is not tested yet!

        self.indexing_based = self.CONF.scene_params_dict.get('indexing_based', 0)
        assert self.indexing_based in [0, 1], 'indexing of frame names (indexing_based) has to be either 0-based or 1-based! got: indexing_based = %d'%self.indexing_based
        assert self.indexing_based == {'public_re_0203': 0, 'public': 1}[self.openrooms_version]

        self.axis_up = get_list_of_keys(self.CONF.scene_params_dict, ['axis_up'], [str])[0]
        assert self.axis_up in ['x+', 'y+', 'z+', 'x-', 'y-', 'z-']

        if self.CONF.scene_params_dict.get('frame_id_list', []) != []: 
            self.frame_id_list = self.CONF.scene_params_dict.get('frame_id_list')
        else:
            print(white_blue('[%s] frame_id_list was not provided; getting frame_id_list by digging into image list...')%self.__class__.__name__)
            im_hdr_list = sorted(glob.glob(str(self.scene_rendering_path / 'im_*.hdr')))
            assert len(im_hdr_list) > 0, 'no image with name: im_*.hdr is found at %s!'%str(self.scene_rendering_path)
            self.frame_id_list = sorted([int(Path(_).stem.replace('im_', '')) for _ in im_hdr_list])
            print(white_blue('... got frame_id_list: [%s]'%(', '.join([str(_) for _ in self.frame_id_list]))))
        
        '''
        paths
        '''
        self.semantic_labels_root = self.root_path_dict['semantic_labels_root']

        self.xml_root = get_list_of_keys(self.root_path_dict, ['xml_root'], [PosixPath])[0]
        self.scene_xml_root = self.xml_root / (self.meta_split.split('_')[1]) / self.scene_name
        self.intrinsics_path = self.xml_root / 'intrinsic.txt'
        assert self.intrinsics_path.exists(), 'intrinsics_path does not exist: %s'%str(self.intrinsics_path)
    
        pose_file = self.CONF.scene_params_dict.get('pose_file')
        assert pose_file.split('-')[0] == "OpenRooms"
        self.pose_file_path = self.scene_xml_root / pose_file.split('-')[1]
        assert self.pose_file_path.exists(), 'pose_file_path does not exist: %s'%str(self.pose_file_path)

        '''
        im properties
        '''

        self.if_scale_hdr = self.CONF.im_params_dict.get('if_scale_hdr', True) # scale HDR images with segs
        self.if_scale_hdr_per_frame = self.CONF.im_params_dict.get('if_scale_hdr_per_frame', False) # True: individually scale each HDR frame; False: get one global HDR scale
        self.if_clip_HDR_to_01 = self.CONF.im_params_dict.get('if_clip_HDR_to_01', False) # only useful when scaling HDR
        
        if_direct_lighting = self.CONF.lighting_params_dict.get('if_direct_lighting', False)
        self.im_key = {True: 'imDirect_', False: 'im_'}[if_direct_lighting]
        self.imsgEnv_key = {True: 'imsgEnvDirect_', False: 'imsgEnv_'}[if_direct_lighting]
        self.imenv_key = {True: 'imenvDirect_', False: 'imenv_'}[if_direct_lighting]

        '''
        BRDF, lighting properties
        '''
        self.clip_roughness_min_to = self.CONF.BRDF_params_dict.get('clip_roughness_min_to', 0.)
        assert self.clip_roughness_min_to >= 0. and self.clip_roughness_min_to <= 1.

        # self.CONF.lighting_params_dict = self.CONF.lighting_params_dict
        self.if_convert_lighting_SG_to_global = self.CONF.lighting_params_dict.get('if_convert_lighting_SG_to_global', False)
        # self.rL = renderingLayer(imWidth=self.CONF.lighting_params_dict['env_col'], imHeight=self.CONF.lighting_params_dict['env_row'], isCuda=False)
        self.im_lighting_HW_ratios = (self.im_H_resize // self.CONF.lighting_params_dict['env_row'], self.im_W_resize // self.CONF.lighting_params_dict['env_col'])
        
        '''
        modalities to load
        '''
        # self.modality_list = self.check_and_sort_modalities(list(set(modality_list)))
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
        if not if_not_load_modalities:
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
    def scene_name_full(self):
        return '-'.join([self.meta_split, self.scene_name]) # e.g. 'main_xml-scene0008_00_more'

    @property
    def scene_name_short(self):
        return '_'.join(self.scene_name.split('_')[:2]) # e.g. scene_name: scene0552_00, scene_name_short: scene0552_00

    @property
    def scene_path(self):
        return self.dataset_root / self.meta_split / self.scene_name

    @property
    def scene_rendering_path(self):
        return self.scene_path

    @property
    def scene_rendering_path_list(self):
        return [self.scene_path] * self.frame_num

    @property
    def if_has_hdr_scale(self):
        return all([_ in self.modality_list for _ in ['im_hdr']]) and self.if_scale_hdr

    @property
    def if_has_poses(self):
        return all([_ in self.modality_list for _ in ['poses']])

    @property
    def if_has_seg(self):
        return all([_ in self.modality_list for _ in ['seg']])

    @property
    def if_has_lighting_SG(self):
        return all([_ in self.modality_list for _ in ['lighting_SG']])

    @property
    def if_has_semseg(self):
        return all([_ in self.modality_list for _ in ['semseg']])

    def get_modality(self, modality, source: str='GT'):
        _ = scene2DBase.get_modality_(self, modality, source)
        if _ is not None:
            return _

        if modality == 'seg': 
            return self.seg_dict_of_lists
        elif modality == 'lighting_SG': 
            return self.lighting_SG_local_list
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
            result_ = scene2DBase.load_modality_(self, _)
            if not (result_ == False):
                continue
            # if _ == 'poses': self.load_poses(self.cam_params_dict)
            if _ == 'seg': self.load_seg()
            if _ == 'lighting_SG': self.load_lighting_SG()
            if _ == 'semseg': self.load_semseg()
            if _ == 'matseg': self.load_matseg()

    def load_intrinsics(self):
        '''
        -> K: (3, 3)
        '''
        self.K = load_matrix(self.intrinsics_path)
        # self.K = load_matrix('/Users/jerrypiglet/Documents/Projects/rui-indoorinv-data/data/public_re_0203_3/main_xml/scene0008_00_more/intrinsic.txt')
        assert self.K.shape == (3, 3)
        assert self.K[0][2] == float(self.im_W_load) / 2.
        assert self.K[1][2] == float(self.im_H_load) / 2.

        if self.if_resize_im:        
            scale_factor = [t / s for t, s in zip((self.im_H_resize, self.im_W_resize), (self.im_H_load, self.im_W_load))]
            self.K = resize_intrinsics(self.K, scale_factor)
        
    def load_seg(self):
        '''
        return 3 bool masks; (H, W), float32 0./1.
        '''
        if hasattr(self, 'seg_dict_of_lists'): return

        print(white_blue('[%s] load_seg for %d frames...'%(self.__class__.__name__, len(self.frame_id_list))))

        self.seg_dict_of_lists = defaultdict(list)

        for i in self.frame_id_list:
            seg_path = self.scene_rendering_path / ('immask_%d.png'%i)
            seg = load_img(seg_path, (self.im_H_load, self.im_W_load, 3), target_HW=self.im_HW_target, resize_method='nearest')[:, :, 0] / 255. # [0., 1.]

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

        print(blue_text('[%s] DONE. load_seg'%self.__class__.__name__))

        self.seg_from['seg'] = True

    def load_transforms(self):
        # load transformations # writeShapeToXML.py L588
        transform_file = self.scene_xml_root / 'transform.dat'
        with open(str(transform_file), 'rb') as fIn:
            self.transforms = pickle.load(fIn)

    def load_poses(self, cam_params_dict={}):
        '''
        pose_list: list of pose matrices (**camera-to-world** transformation), each (3, 4): [R|t] (OpenCV convention: right-down-forward)
        '''
        self.load_intrinsics()
        if hasattr(self, 'pose_list'): return

        print(white_blue('[%s] load_poses for %d frames...'%(self.__class__.__name__, len(self.frame_id_list))))

        if not hasattr(self, 'transforms'):
            self.load_transforms()

        if not self.pose_file_path.exists():
            print(yellow('[%s] cam file not found, skipped load poses. %s'%(str(self.__class__.__name__), str(self.pose_file_path))))
            return
        
        print(blue_text('[%s] loading poses from %s'%(str(self.__class__.__name__), self.pose_file_path)))
        cam_params = read_cam_params_OR(str(self.pose_file_path))
        if self.frame_id_list == []: self.frame_id_list = list(range(len(cam_params)))

        self.pose_list, self.origin_lookatvector_up_list = load_OR_public_poses_to_Rt(cam_params, self.frame_id_list, False, if_1_based=self.indexing_based==1)

        if self.if_resize_im:
            pass # IMPORTANT! do nothing; keep the 3D scene (cameras and geometry), but instead resize intrinsics to account for the smaller image
        
        print(blue_text('[%s] DONE. load_poses (%d poses)'%(self.__class__.__name__, len(self.origin_lookatvector_up_list))))

    def load_albedo(self):
        '''
        albedo; loaded in [0., 1.] LDR, then **2.2 -> HDR;
        (H, W, 3), [0., 1.]
        '''
        if hasattr(self, 'albedo_list'): return

        print(white_blue('[%s] load_albedo for %d frames...'%(self.__class__.__name__, len(self.frame_id_list))))

        self.albedo_file_list = [self.scene_rendering_path / ('imbaseColor_%d.png'%i) for i in self.frame_id_list]
        self.albedo_list = [load_img(albedo_file, (self.im_H_load, self.im_W_load, 3), ext='png', target_HW=self.im_HW_target).astype(np.float32)/255. for albedo_file in self.albedo_file_list]
        self.albedo_list = [albedo**2.2 for albedo in self.albedo_list]
        
        print(blue_text('[%s] DONE. load_albedo'%self.__class__.__name__))

    def load_roughness(self):
        '''
        roughness; smaller, the more specular;
        (H, W, 1), [0., 1.]
        '''
        if hasattr(self, 'roughness_list'): return

        print(white_blue('[%s] load_roughness for %d frames...'%(self.__class__.__name__, len(self.frame_id_list))))

        self.roughness_file_list = [self.scene_rendering_path / ('imroughness_%d.png'%i) for i in self.frame_id_list]
        self.roughness_list = [load_img(roughness_file, (self.im_H_load, self.im_W_load, 3), ext='png', target_HW=self.im_HW_target)[:, :, 0:1].astype(np.float32)/255. for roughness_file in self.roughness_file_list]

        print(blue_text('[%s] DONE. load_roughness'%self.__class__.__name__))

    def load_depth(self):
        '''
        depth;
        (H, W), ideally in [0., inf]
        '''
        if hasattr(self, 'depth_list'): return

        print(white_blue('[%s] load_depth for %d frames...'%(self.__class__.__name__, len(self.frame_id_list))))

        self.depth_file_list = [self.scene_rendering_path / ('imdepth_%d.dat'%i) for i in self.frame_id_list]
        self.depth_list = [load_binary(depth_file, (self.im_H_load, self.im_W_load), target_HW=self.im_HW_target, resize_method='area')for depth_file in self.depth_file_list] # TODO: better resize method for depth for anti-aliasing purposes and better boundaries, and also using segs?
        
        print(blue_text('[%s] DONE. load_depth'%self.__class__.__name__))

        self.pts_from['depth'] = True

    def load_normal(self):
        '''
        normal, in camera coordinates (OpenGL convention: right-up-backward);
        (H, W, 3), [-1., 1.]
        '''
        if hasattr(self, 'normal_list'): return

        print(white_blue('[%s] load_normal for %d frames...'%(self.__class__.__name__, len(self.frame_id_list))))

        self.normal_file_list = [self.scene_rendering_path / ('imnormal_%d.png'%i) for i in self.frame_id_list]
        self.normal_list = [load_img(normal_file, (self.im_H_load, self.im_W_load, 3), ext='png', target_HW=self.im_HW_target).astype(np.float32)/255.*2.-1. for normal_file in self.normal_file_list] # -> [-1., 1.], pointing inward (i.e. notebooks/images/openrooms_normals.jpg)
        self.normal_list = [normal / np.sqrt(np.maximum(np.sum(normal**2, axis=2, keepdims=True), 1e-5)) for normal in self.normal_list]
        
        print(blue_text('[%s] DONE. load_normal'%self.__class__.__name__))

    def load_lighting_SG(self):
        '''
        lighting in SG;
        self.if_convert_lighting_SG_to_global = False: 
            (H', W', SG_num, 7(axis_local, lamb, weight: 3, 1, 3)), in camera-and-normal-dependent local coordinates
        self.if_convert_lighting_SG_to_global = True: 
            (H', W', SG_num, 7(axis, lamb, weight: 3, 1, 3)), in global coordinates (OpenCV convention: right-down-forward)
        '''
        if hasattr(self, 'lighting_SG_local_list'): return

        print(white_blue('[%s] load_lighting_SG for %d frames...'%(self.__class__.__name__, len(self.frame_id_list))))
        print(red('THIS MIGHT BE SLOW...'))

        self.lighting_SG_file_list = [self.scene_rendering_path / ('%s%d.h5'%(self.imsgEnv_key, i)) for i in self.frame_id_list]

        self.lighting_SG_local_list = []

        for frame_idx, lighting_SG_file in enumerate(tqdm(self.lighting_SG_file_list)):
            lighting_SG = load_h5(lighting_SG_file)
            # if 'im_hdr' in self.modality_list and self.if_scale_hdr:
            #     hdr_scale = self.hdr_scale_list[frame_idx]
            #     lighting_SG[:, :, :, 3:6] = lighting_SG[:, :, :, 3:6] * hdr_scale # (120, 160, 12(SG_num), 6); theta, phi, lamb, weight: 1, 1, 1, 3
            lighting_SG = np.concatenate(
                (convert_SG_angles_to_axis_local_np(lighting_SG[:, :, :, :2]),  # (120, 160, 12(SG_num), 7); axis_local, lamb, weight: 3, 1, 3
                lighting_SG[:, :, :, 2:]), axis=3)
            self.lighting_SG_local_list.append(lighting_SG)

        env_row, env_col = self.CONF.lighting_params_dict['env_row'], self.CONF.lighting_params_dict['env_col']
        assert all([tuple(_.shape)==(env_row, env_col, self.CONF.lighting_params_dict['SG_num'], 7) for _ in self.lighting_SG_local_list])

        if self.if_convert_lighting_SG_to_global:
            if hasattr(self, 'pose_list'): self.load_poses(self.cam_params_dict)

            self.lighting_SG_global_list = []
            for lighting_SG_local, pose, normal in zip(self.lighting_SG_local_list, self.pose_list, self.normal_list):
                lighting_SG_global = np.concatenate(
                    (convert_lighting_axis_local_to_global_np(lighting_SG_local[:, :, :, :3], pose, normal), 
                    lighting_SG_local[:, :, :, 3:]), axis=3) # (120, 160, 12(SG_num), 7); axis, lamb, weight: 3, 1, 3
                self.lighting_SG_global_list.append(lighting_SG_global)
            assert all([tuple(_.shape)==(env_row, env_col, self.CONF.lighting_params_dict['SG_num'], 7) for _ in self.lighting_SG_global_list])

        print(blue_text('[%s] DONE. load_lighting_SG'%self.__class__.__name__))


    def load_lighting_envmap(self):
        if hasattr(self, 'lighting_envmap_list'): return

        print(white_blue('[%s] load_lighting_envmap for %d frames...'%(self.__class__.__name__, len(self.frame_id_list))))
        print(red('THIS MIGHT BE SLOW...'))

        env_height, env_width = self.CONF.lighting_params_dict['env_height'], self.CONF.lighting_params_dict['env_width']
        env_row, env_col = self.CONF.lighting_params_dict['env_row'], self.CONF.lighting_params_dict['env_col']
        
        version_key = ''
        if (env_row, env_col, env_height, env_width) == (120, 160, 8, 16):
            version_key = '8x16_'
        elif (env_row, env_col, env_height, env_width) == (6, 8, 128, 256):
            version_key = '128x256_'
        self.lighting_envmap_file_list = [self.scene_rendering_path / ('%s%s%d.hdr'%(self.imenv_key, version_key, i)) for i in self.frame_id_list]

        self.lighting_envmap_list = []

        for idx, lighting_envmap_file in enumerate(tqdm(self.lighting_envmap_file_list)):
            envmap = load_envmap(str(lighting_envmap_file), env_height=env_height, env_width=env_width, env_row=env_row, env_col=env_col, allow_resize=False)[0].transpose(1, 2, 0, 3, 4) # -> (120, 160, 3, 8, 16)
            # if 'im_hdr' in self.modality_list and self.if_scale_hdr:
            #     hdr_scale = self.hdr_scale_list[idx]
            #     envmap = envmap * hdr_scale
            self.lighting_envmap_list.append(envmap)

        assert all([tuple(_.shape)==(env_row, env_col, 3, env_height, env_width) for _ in self.lighting_envmap_list])
        print(blue_text('[%s] DONE. load_lighting_envmap'%self.__class__.__name__))

    def load_semseg(self):
        '''
        semseg, image space
        (H, W, 3), [-1., 1.]
        '''
        if hasattr(self, 'semseg_list'): return

        print(white_blue('[%s] load_semseg for %d frames...'%(self.__class__.__name__, len(self.frame_id_list))))

        self.semseg_file_list = [self.scene_rendering_path / ('imsemLabel_%d.npy'%i) for i in self.frame_id_list]
        self.semseg_list = [
            load_img(semseg_file, (self.im_H_load, self.im_W_load), ext='npy', target_HW=self.im_HW_target, resize_method='nearest')
            for semseg_file in self.semseg_file_list]
        
        print(blue_text('[%s] DONE. load_semseg'%self.__class__.__name__))

    def load_matseg(self):
        '''
        matseg, dict:
        - mat_aggre_map: (H, W), int32; [0, 1, ..., num_mat_masks], 0 for invalid region
        - num_mat_masks: == np.amax(mat_aggre_map)
        '''

        if hasattr(self, 'matseg_list'): return

        print(white_blue('[%s] load_matseg for %d frames...'%(self.__class__.__name__, len(self.frame_id_list))))

        self.imcadmatobj_file_list = [self.scene_rendering_path / ('imcadmatobj_%d.dat'%i) for i in self.frame_id_list]
        self.imcadmatobj_list = [
            load_binary(
                imcadmatobj_file, 
                (self.im_H_load, self.im_W_load), 
                target_HW=self.im_HW_target, 
                channels=3, dtype=np.int32, resize_method='nearest'
                ) 
            for imcadmatobj_file in self.imcadmatobj_file_list]

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
        
        print(blue_text('[%s] DONE. load_matseg'%self.__class__.__name__))
from math import prod
from pathlib import Path
import numpy as np
np.set_printoptions(suppress=True)
import torch
import random
random.seed(0)
from tqdm import tqdm
import mitsuba as mi
from .class_mitsubaBase import mitsubaBase
from .class_scene2DBase import scene2DBase

from lib.global_vars import mi_variant_dict
from lib.utils_OR.utils_OR_cam import R_t_to_origin_lookatvector_up_opencv
from lib.utils_io import load_img, resize_intrinsics
from lib.utils_misc import blue_text, magenta, yellow, get_list_of_keys, white_blue, red
from lib.utils_monosdf_scene import dump_shape_dict_to_shape_file, load_shape_dict_from_shape_file
from lib.utils_misc import get_device
from lib.utils_dvgo import get_meshgrid

from .class_scene2DBase import scene2DBase

class matterportScene3D(mitsubaBase, scene2DBase):
    '''
    A class used to visualize/render scenes (room) from Chang et al.'17, Matterport3D: Learning from RGB-D Data in Indoor ...
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

        self.scene_name, self.region_id_list, self.frame_id_list = get_list_of_keys(scene_params_dict, ['scene_name', 'region_id_list', 'frame_id_list'], [str, list, list])
        self.indexing_based = scene_params_dict.get('indexing_based', 0)
        
        self.axis_up_native = 'z+'
        self.axis_up = scene_params_dict.get('axis_up', self.axis_up_native) # native: 'z+
        assert self.axis_up in ['x+', 'y+', 'z+', 'x-', 'y-', 'z-']
        if self.axis_up != self.axis_up_native:
            assert False, 'do something please'

        self.if_undist = scene_params_dict.get('if_undist', False)

        self.host = host
        self.device = get_device(self.host, device_id)

        self.scene_path = self.dataset_root / 'v1/scans' / self.scene_name
        self.scene_rendering_path = self.scene_path
        self.scene_name_full = self.scene_name # e.g.'asianRoom1'
        
        self.if_need_undist = self.im_params_dict.get('if_need_undist', False)
        self.if_need_manual_undist_flags = {_: True for _ in self.modality_list}

        # self.pose_format, pose_file = scene_params_dict['pose_file']
        # assert self.pose_format in ['OpenRooms', 'bundle'], 'Unsupported pose file: '+pose_file
        # self.pose_file_path = self.scene_path / 'cameras' / pose_file

        self.shape_file_list = [self.scene_path / 'region_segmentations' / ('region%d.ply'%region_id) for region_id in self.region_id_list]
        self.shape_params_dict = shape_params_dict
        self.mi_params_dict = mi_params_dict
        variant = mi_params_dict.get('variant', '')
        mi.set_variant(variant if variant != '' else mi_variant_dict[self.host])

        self.near = cam_params_dict.get('near', 0.1)
        self.far = cam_params_dict.get('far', 10.)
        self.if_scale_scene = False

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
        self.load_modalities()

        if hasattr(self, 'pose_list'): 
            self.get_cam_rays(self.cam_params_dict)
        self.process_mi_scene(self.mi_params_dict, if_postprocess_mi_frames=hasattr(self, 'pose_list'))

        '''
        undist images if applicable
        '''
        for modality in self.modality_list:
            if modality in ['im_sdr', 'im_hdr', 'depth', 'normal']:
                if self.if_need_manual_undist_flags[modality]:

                    assert hasattr(self, 'K_undist_params_list')
                    assert len(self.frame_id_list) == len(self.K_list) == len(getattr(self, modality+'_list')) == len(self.K_undist_params_list)
                    print(magenta('Undistorting %s...'%modality))
                    has_mask_flag = hasattr(self, 'im_undist_mask_list')
                    if not has_mask_flag:
                        self.im_undist_mask_list = []
                    modality_undist_list = []
                    for frame_id, K, im, K_undist_params in tqdm(zip(self.frame_id_list, self.K_list, getattr(self, modality+'_list'), self.K_undist_params_list)):
                        im_undist, im_undist_mask = self.undistort_single_image(im, K, K_undist_params, modality=modality)
                        modality_undist_list.append(im_undist)
                        if not has_mask_flag:
                            self.im_undist_mask_list.append(im_undist_mask)
                    setattr(self, modality+'_undist_list', modality_undist_list)

    def undistort_single_image(self, im, K, K_undist_params, modality=''):
        H, W = im.shape[:2]
        undistorted_position_x, undistorted_position_y = get_meshgrid(H, W, self.if_center_offset) # True: pixel centers are 0.5, 1.5, ..., H-0.5; False: pixel centers are 0, 1, ..., H-1
        [k1, k2, p1, p2, k3] = get_list_of_keys(K_undist_params, ['k1', 'k2', 'p1', 'p2', 'k3'], [float, float, float, float, float])
        k4 = K_undist_params.get('k4', 0.) # [TODO] double check with Matterport folks, as it is not documented here... https://github.com/niessner/Matterport/blob/master/data_organization.md#matterport_camera_intrinsics 
        fx, fy = K[0][0], K[1][1]
        cx, cy = K[0][2], K[1][2]
        nx = (undistorted_position_x - cx) / fx
        ny = (undistorted_position_y - cy) / fy
        rr = nx*nx + ny*ny
        rrrr = rr*rr
        s = 1.0 + rr*k1 + rrrr*k2 + rrrr*rr*k3 + rrrr*rrrr*k4
        nx = s*nx + p2*(rr + 2*nx*nx) + 2*p1*nx*ny
        ny = s*ny + p1*(rr + 2*ny*ny) + 2*p2*nx*ny
        distorted_position_x = nx*fx + cx
        distorted_position_y = ny*fy + cy
        
        im_torch = torch.from_numpy(im).unsqueeze(0).permute(0, 3, 1, 2).float() # (1, C, H, W)
        im_mask_torch = torch.ones(1, 1, im.shape[0], im.shape[1]).float() # (1, 1, H, W)

        uv_dist_normalized_torch = torch.from_numpy(np.stack([distorted_position_x, distorted_position_y], axis=-1)).unsqueeze(0).float() # (1, H, W, 2)
        if self.if_center_offset:
            align_corners = False # (-1, 1) corresponds to the corner points of corner pixels (0, H or W)
            # uv_undist_normalized_torch = torch.from_numpy(np.stack([undistorted_position_x, undistorted_position_y], axis=-1)).unsqueeze(0).unsqueeze(0).float() # (1, 1, H, W, 2)
            uv_dist_normalized_torch = uv_dist_normalized_torch / torch.tensor([W, H]).reshape(1, 1, 1, 2).float() * 2 - 1 # (1, H, W, 2)
        else:
            align_corners = True # (-1, 1) corresponds to the CENTER points of corner pixels (0, H-1 or W-1)
            uv_dist_normalized_torch = uv_dist_normalized_torch / torch.tensor([W-1, H-1]).reshape(1, 1, 1, 2).float() * 2 - 1 # (1, H, W, 2)

        sampled_im_torch = torch.nn.functional.grid_sample(im_torch, uv_dist_normalized_torch, mode='bilinear', align_corners=align_corners) # (1, 3, H, W)
        im_undist = sampled_im_torch.squeeze(0).permute(1, 2, 0).detach().numpy() # (H, W, C)

        sampled_im_mask_torch = torch.nn.functional.grid_sample(im_mask_torch, uv_dist_normalized_torch, mode='nearest', align_corners=align_corners) # (1, 1, H, W)
        im_undist_mask = sampled_im_mask_torch.squeeze().detach().numpy() # (H, W)
        im_undist_mask = im_undist_mask == 1.

        '''
        uncomment to debug: images/debug_im_undist.png
        '''
        # import matplotlib.pyplot as plt
        # plt.figure(figsize=(20, 10))
        # plt.subplot(2, 2, 1)
        # plt.imshow(np.clip(im**(1./2.2), 0., 1.))
        # plt.grid()
        # plt.subplot(2, 2, 2)
        # plt.imshow(np.clip(im_undist**(1./2.2), 0., 1.))
        # plt.grid()
        # plt.subplot(2, 2, 3)
        # plt.imshow(im_undist_mask.astype(np.float32))
        # plt.colorbar()
        # plt.grid()
        # plt.show()

        del uv_dist_normalized_torch, sampled_im_torch, im_torch, sampled_im_mask_torch, im_mask_torch

        return im_undist, im_undist_mask

    @property
    def frame_num(self):
        return len(self.frame_id_list)
            
    @property
    def valid_modalities(self):
        return [
            'im_hdr', 'im_sdr', 
            'depth', 
            'im_sdr_undist', 
            'depth_undist', 
            'poses', 
            'shapes', 
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
        elif modality in ['mi_normal', 'mi_normal_im_overlay']: 
            return self.mi_normal_global_list
        elif modality in ['mi_seg_area', 'mi_seg_env', 'mi_seg_obj']:
            seg_key = modality.split('_')[-1] 
            return self.mi_seg_dict_of_lists[seg_key] # Set scene_obj->mi_params_dict={'if_get_segs': True
        elif modality == 'im_mask': 
            return self.im_mask_list
        else:
            assert False, 'Unsupported modality: ' + modality

    def load_mi_scene(
        self, 
        mi_params_dict={}, 
        first_N_shapes=-1, 
        if_force: bool=False,
        ):
        '''
        load scene representation into Mitsuba 3
        '''
        if hasattr(self, 'mi_scene') and not if_force:
            print('Mitsuba scene already loaded. Skip loading. (if_force=%s)'%if_force)
            return

        scene_dict = {
            'type': 'scene',
            # 'shape_id': shape_id_dict, 
        }
        if first_N_shapes != -1:
            assert first_N_shapes <= len(self.shape_file_list)
            shape_file_list = self.shape_file_list[:first_N_shapes]
            print('Loading first %d shapes'%first_N_shapes)
        else:
            shape_file_list = self.shape_file_list
        for _, shape_file in enumerate(shape_file_list):
            shape_id_dict = {
                'type': shape_file.suffix[1:],
                'filename': str(shape_file), 
                # 'to_world': mi.ScalarTransform4f.scale([1./scale]*3).translate((-offset).flatten().tolist()),
                }
            # if self.if_scale_scene:
            #     shape_id_dict['to_world'] = mi.ScalarTransform4f.scale([1./self.scene_scale]*3)
            scene_dict[str(_)] = shape_id_dict

        self.mi_scene = mi.load_dict(scene_dict)

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

    def load_meta(self):
        '''
        load house regions file (region ~= room): https://github.com/niessner/Matterport/blob/master/data_organization.md#house_segmentations
        and frame poses along with it

        -> self.modality_file_list_dict
        -> self.K_list: intrinsics; distorted/undistorted are the same
        -> self.pose_list: poses [R|t], camera-to-world; distorted/undistorted are the same
        -> self.im_HW_load_list: image size
        '''

        self.scene_metadata_file = self.scene_path / ('house_segmentations/%s.house'%self.scene_name)
        assert Path(self.scene_metadata_file).exists(), 'No such file: %s'%str(self.scene_metadata_file)
        with open(str(self.scene_metadata_file), 'r') as camIn:
            scene_metadata = camIn.read().splitlines()

        '''
        the H (house) line...
            H name label #images #panoramas #vertices #surfaces #segments #objects #categories #regions #portals #levels  0 0 0 0 0  xlo ylo zlo xhi yhi zhi  0 0 0 0 0
        '''
        house_line = scene_metadata[1].replace('  ', ' ').replace('  ', ' ').split(' '); assert house_line[0] == 'H'
        N_images, N_panoramas, N_vertices, N_surfaces, N_segments, N_objects, N_categories, self.N_regions, _, N_levels = [int(_) for _ in house_line[3:13]]
        
        self.panorama_name_list = [] # list of str
        self.panorama_index_list = [] # list of int
        self.frame_id_list = []
        self.frame_filename_list = []
        self.frame_info_list = []
        self.im_HW_load_list = []

        self.K_list = [] # undistorted/distorted: the same
        self._K_orig_list = [] # only for validation purposes
        self.extrinsics_mat_list = [] # loaded from distorted
        self.pose_list = [] # undistorted/distorted: the same
        self.origin_lookatvector_up_list = []

        self.region_main = self.region_id_list[0] # MAIN region to load shape
        for region_id in self.region_id_list:
            assert region_id < self.N_regions, 'region_id %d >= N_regions %d'%(region_id, self.N_regions)
            print('[%s] Loading region %s from %d regions of house %s (total %d frames for the house)...'%(self.__class__.__name__, region_id, self.N_regions, self.scene_name, N_images))

            ''' 
            the R line (region for self.region_id)...
                R region_index level_index 0 0 label  px py pz  xlo ylo zlo xhi yhi zhi  height  0 0 0 0
            '''
            region_line = scene_metadata[N_levels+2+region_id].replace('  ', ' ').replace('  ', ' ').split(' '); assert region_line[0] == 'R'
            assert region_line[1] == str(region_id), 'region_id %d != region_line[1] %s'%(region_id, region_line[1])
            region_label = region_line[5]
            region_description = self.region_description_dict[region_label]
            region_height = float(region_line[15])
            assert region_height > 0.

            print('region %d: %s; height %.2f'%(region_id, region_description, region_height))

            '''
            load all panoramas and find the ones in the region
                P  f4d03f729dfc49068db327584455e975  47 4  0  2.37165 4.31886 1.49283  0 0 0 0 0 
                P name  panorama_index region_index 0  px py pz  0 0 0 0 0
            '''
            panorama_name_list = []
            panorama_index_list = []
            panorama_line_list = [_.replace('  ', ' ').replace('  ', ' ').split(' ') for _ in scene_metadata if _.startswith('P')]
            for panorama_line in panorama_line_list:
                assert panorama_line[0] == 'P'
                panorama_name = panorama_line[1]
                panorama_index = int(panorama_line[2])
                panorama_region_id = int(panorama_line[3])
                if panorama_region_id == region_id:
                    panorama_name_list.append(panorama_name)
                    panorama_index_list.append(panorama_index)
            assert panorama_name_list != [], 'no parameters for region %d'%region_id
            self.panorama_name_list += panorama_name_list
            self.panorama_index_list += panorama_index_list

            '''
            load all frames and find the ones in the region
                I image_index panorama_index name camera_index yaw_index e00 e01 e02 e03 e10 e11 e12 e13 e20 e21 e22 e23 e30 e31 e32 e33  i00 i01 i02  i10 i11 i12 i20 i21 i22  width height  px py pz  0 0 0 0 0
            '''
            frame_line_list = [_.replace('  ', ' ').replace('  ', ' ').split(' ') for _ in scene_metadata if _.startswith('I')]
            frame_id_list = []
            assert frame_line_list[0][1] == '0' # first frame is 0
            assert frame_line_list[-1][1] == str(len(frame_line_list)-1)
            for frame_line in frame_line_list:
                assert frame_line[0] == 'I'
                image_index = int(frame_line[1])
                assert image_index < N_images
                panorama_index = int(frame_line[2])
                if panorama_index in panorama_index_list:
                    frame_id_list.append(image_index)
            frame_num_all = len(frame_id_list)
            
            print('region %d: total'%region_id, white_blue(str(frame_num_all)), 'frames for the region (frame_id e.g. [%s]...)'%(', '.join([str(_) for _ in frame_id_list])))
            if self.scene_params_dict['frame_id_list'] != []:
                frame_id_list = [_ for _ in self.scene_params_dict['frame_id_list'] if _ in frame_id_list]
                print('region %d: SELECTED %d frames ([%s]...)'%(region_id, len(frame_id_list), ', '.join([str(_) for _ in frame_id_list[:3]])))

            # assert len(frame_id_list) > 0
            if len(frame_id_list) == 0:
                print(red('region %d: no frames for the region; skipping...'%region_id))
                continue
            frame_line_valid_list = [frame_line_list[_] for _ in frame_id_list]

            frame_filename_list = []
            frame_info_list = []
            im_HW_load_list = []
            # K_list = [] # undistorted/distorted: the same
            # _K_orig_list = [] # only for validation purposes
            extrinsics_mat_list = [] # loaded from distorted
            pose_list = [] # undistorted/distorted: the same
            origin_lookatvector_up_list = []

            for frame_line in frame_line_valid_list:
                assert frame_line[0] == 'I'
                assert int(frame_line[1]) in frame_id_list
                assert int(frame_line[2]) in panorama_index_list
                frame_name = frame_line[3]
                # a total of 18 images are captured for one location (3 cameras, 6 yaw angles; documentation has typo)
                camera_index = int(frame_line[4]); assert camera_index in list(range(3))
                yaw_index = int(frame_line[5]); assert yaw_index in list(range(6))
                frame_filename = frame_name + '_' + '%s' + str(camera_index) + '_' + str(yaw_index) + '.%s' # '''08115b08da534f1aafff2fa81fc73512_%s0_0.%s'; %s: 'd' for *_depth_images, 'i' for *_color_images; %s: ext
                frame_filename_list.append(frame_filename)
                frame_info_list.append((frame_name, camera_index, yaw_index))

                # distorted camera parameters
                extrinsics_mat = np.array([float(_) for _ in frame_line[6:22]]).reshape(4, 4)
                extrinsics_mat_list.append(extrinsics_mat) # [world-to-camera] inverse of Rx+t
                R_ = extrinsics_mat[:3, :3]; t_ = extrinsics_mat[:3, 3:4]
                R = R_.T
                t = -R_.T @ t_
                R = R @ np.array([[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]], dtype=np.float32) # OpenGL -> OpenCV
                pose = np.concatenate([R, t], axis=1)
                pose_list.append(pose) # [camera-to-world]
                (origin, lookatvector, up) = R_t_to_origin_lookatvector_up(R, t)
                origin_lookatvector_up_list.append((origin.reshape((3, 1)), lookatvector.reshape((3, 1)), up.reshape((3, 1))))

                width, height = int(frame_line[31]), int(frame_line[32])
                im_HW_load_list.append((height, width))

                # intrinsics: not correct!!!
                # K = np.array([float(_) for _ in frame_line[22:31]]).reshape(3, 3) # https://github.com/niessner/Matterport/blob/master/data_organization.md#matterport_camera_intrinsics
                # _K_orig_list.append(K.copy())
                # if width != self.W or height != self.H:
                #     scale_factor = [t / s for t, s in zip((self.H, self.W), (height, width))]
                #     K = resize_intrinsics(K, scale_factor)
                #     print(yellow('Resized K from %s to %s'%((height, width), (self.H, self.W))))
                # print(K)
                # K_list.append(K)

            self.frame_id_list += frame_id_list
            self.frame_filename_list += frame_filename_list
            self.frame_info_list += frame_info_list
            self.im_HW_load_list += im_HW_load_list
            # self.K_list += K_list
            # self._K_orig_list += _K_orig_list
            self.extrinsics_mat_list += extrinsics_mat_list
            self.pose_list += pose_list
            self.origin_lookatvector_up_list += origin_lookatvector_up_list

        self.frame_num_all = len(self.frame_id_list)
        assert self.frame_num_all > 0, 'no frames found for the region'
        
        assert len(list(set(self.im_HW_load_list))) == 1, 'all loaded images should ideally have the same size'
        assert self.im_HW_load == self.im_HW_load_list[0], 'loaded image size should match the specified image size'

        for modality, modality_filename in self.modality_filename_dict.items():
            modality_folder, modality_tag, modality_ext = modality_filename
            if self.if_need_undist:
                if modality in ['im_sdr', 'im_hdr', 'depth', 'normal']:
                    if modality+'_undist' in self.modality_filename_dict:
                        modality_folder_undist, modality_tag_undist, modality_ext_undist = self.modality_filename_dict[modality+'_undist']
                        print(yellow('Using undistorted %s images from %s'%(modality, modality_folder_undist)))
                        self.if_need_manual_undist_flags[modality] = False
            self.modality_file_list_dict[modality] = [self.scene_rendering_path / modality_folder / (frame_filename%(modality_tag, modality_ext)) for frame_filename in self.frame_filename_list]
        
        # import ipdb; ipdb.set_trace()
        # assert 'im_H_resize' not in self.im_params_dict and 'im_W_resize' not in self.im_params_dict
        # self.H_list = [_[0] for _ in self.im_HW_load_list]
        # self.W_list = [_[1] for _ in self.im_HW_load_list]

        self.load_compare_undist_cam_parameters() # load undistorted camera parameters; TO DOUBLE CHECK AGAINST LOADED DISTORTED CAMERA PARAMETERS
        self.load_compare_camera_files()
        print(blue_text('[%s] DONE. load_poses (%d poses)'%(self.parent_class_name, len(self.pose_list))))

        if self.cam_params_dict.get('if_convert_poses', False):
            self.export_poses_cam_txt(self.pose_file_path.parent, cam_params_dict=self.cam_params_dict, frame_num_all=self.frame_num_all)

    def load_compare_undist_cam_parameters(self):
        '''
        undistorted_camera_parameters/{scene_name}.json

        load undistorted camera parameters

        https://github.com/niessner/Matterport/blob/master/data_organization.md#undistorted_camera_parameters

        Mostly to double check against the loaded distorted camera parameters loaded in self.load_meta()

        -> loaded intrinsics are the same as distorted ones acquired in self.load_meta()->self.K_list
        -> loaded extrinsics are basically [R|t] (camera-to-world); and same as distorted ones acquired in self.load_meta()->self.pose_list
        '''
        undist_cam_params_file = self.scene_path / 'undistorted_camera_parameters' / (self.scene_name + '.conf')
        assert undist_cam_params_file.exists(), 'undistorted camera parameters file does not exist: %s'%undist_cam_params_file 
        with open(str(undist_cam_params_file), 'r') as camIn:
            undist_cam_data_all = camIn.read().splitlines()
        undist_cam_data_all = [_ for _ in undist_cam_data_all[5:] if _ != [] and _ != '']
        for _ in undist_cam_data_all:
            if not (_.startswith('intrinsics') or _.startswith('scan')):
                print(_) # debug
        assert len(undist_cam_data_all) % 7 == 0 # 7 lines per camera
        undist_cam_data_per_cam = [undist_cam_data_all[_*7:(_+1)*7] for _ in range(len(undist_cam_data_all)//7)]
        undist_cam_data_per_cam_dict = {'_'.join(_[1].split(' ')[1].split('_')[:2]): _ for _ in undist_cam_data_per_cam} # key is camera id (e.g. ee59d6b5e5da4def9fe85a8ba94ecf25_d1), value is list: (intrinsics, cam info of 6 frames of 6 yaws)

        assert len(self.frame_info_list) > 0
        for frame_id, (frame_name, camera_index, yaw_index) in enumerate(self.frame_info_list):
            undist_cam_data_per_cam = undist_cam_data_per_cam_dict['%s_d%d'%(frame_name, camera_index)]
            
            # not correct!!!
            # assert undist_cam_data_per_cam[0].startswith('intrinsics_matrix')
            # intrinsics_undist = [float(_) for _ in undist_cam_data_per_cam[0].replace('  ', ' ').split(' ')[1:]] # <fx> 0 <cx>  0 <fy> <cy> 0 0 1
            # assert len(intrinsics_undist) == 9
            # # fx, cx, fy, cy = intrinsics_undist[0], intrinsics_undist[2], intrinsics_undist[4], intrinsics_undist[5]
            # # assert np.all(np.array([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]]) == self._K_orig_list[frame_id]) # just double check to make sure: intrinsics distorted/undistorted are the same
            # if not np.allclose(np.array(intrinsics_undist), self._K_orig_list[frame_id].flatten(), atol=1e-5, rtol=1e-5):
            #     print(intrinsics_undist)
            #     print(self._K_orig_list[frame_id].flatten())
            #     import ipdb; ipdb.set_trace()

            inv_extrinsics_undist = [float(_) for _ in undist_cam_data_per_cam[1+yaw_index].replace('  ', ' ').split(' ')[3:]] # <camera-to-world-matrix>
            assert len(inv_extrinsics_undist) == 16
            inv_extrinsics_undist_mat = np.array(inv_extrinsics_undist).reshape(4, 4) # <camera-to-world-matrix>
            pose_opencv = np.hstack((self.pose_list[frame_id][:3, :3]@ np.array([[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]]), self.pose_list[frame_id][:3, 3:4]))
            if not np.allclose(inv_extrinsics_undist_mat[:3], pose_opencv, atol=1e-4, rtol=1e-4):
                print('!!!')
                print(inv_extrinsics_undist_mat[:3])
                print(pose_opencv)
                import ipdb; ipdb.set_trace()
            # assert np.allclose(inv_extrinsics_undist_mat[:3], self.pose_list[frame_id][:3])

    def load_compare_camera_files(self):
        '''
        further compare against:
            - matterport_camera_intrinsics: this is the right one...
            - matterport_camera_poses
        '''
        K_list_new = []
        K_undist_params_list = []
        for frame_id, (frame_name, camera_index, yaw_index) in enumerate(self.frame_info_list):
            matterport_camera_intrinsics_file = self.scene_path / 'matterport_camera_intrinsics' / ('%s_intrinsics_%d.txt'%(frame_name, camera_index))
            assert matterport_camera_intrinsics_file.exists(), 'matterport camera intrinsics file does not exist: %s'%matterport_camera_intrinsics_file
            with open(str(matterport_camera_intrinsics_file), 'r') as camIn:
                cam_data = camIn.read().splitlines()[0].split(' ')
            width, height, fx, fy, cx, cy, k1, k2, p1, p2, k3 = [float(_) for _ in cam_data]
            assert self.im_HW_load_list[frame_id] == (int(height), int(width))
            K = np.array([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]])
            # if not np.allclose(K, self._K_orig_list[frame_id], atol=1e-5, rtol=1e-5):
            #     print(red('!!! loaded intrinsics mismatch!'))
            #     print(cam_data[2:])
            #     print(self._K_orig_list[frame_id].flatten())

            if width != self.W or height != self.H:
                scale_factor = [t / s for t, s in zip((self.H, self.W), (height, width))]
                K = resize_intrinsics(K, scale_factor)
                print(yellow('Resized K from %s to %s'%((height, width), (self.H, self.W))))

            # print(K)
            K_list_new.append(K)
            K_undist_params_list.append(dict(k1=k1, k2=k2, p1=p1, p2=p2, k3=k3))

        self.K_list = K_list_new
        self.K_undist_params_list = K_undist_params_list

    def load_im_mask(self):
        '''
        load im_mask (H, W), bool
        '''
        print(white_blue('[%s] load_im_mask')%self.parent_class_name)

        filename = self.modality_filename_dict['im_mask']
        im_mask_ext = filename.split('.')[-1]

        self.im_mask_file_list = [self.scene_rendering_path / (filename%frame_id) for frame_id in self.frame_id_list]
        expected_shape_list = [self.im_HW_load_list[_] for _ in self.frame_id_list] if hasattr(self, 'im_HW_load_list') else [self.im_HW_load]*self.frame_num
        self.im_mask_list = [load_img(_, expected_shape=__, ext=im_mask_ext, target_HW=self.im_HW_target)/255. for _, __ in zip(self.im_mask_file_list, expected_shape_list)]
        self.im_mask_list = [_.astype(bool) for _ in self.im_mask_list]

        print(blue_text('[%s] DONE. load_im_mask')%self.parent_class_name)

    # def get_cam_rays(self, cam_params_dict={}):
    #     if hasattr(self, 'cam_rays_list'):  return
    #     self.cam_rays_list = self.get_cam_rays_list(self.H_list, self.W_list, self.K_list, self.pose_list, convention='opencv')

    def get_cam_rays(self, cam_params_dict={}):
        if hasattr(self, 'cam_rays_list'):  return
        # self.cam_rays_list = self.get_cam_rays_list(self.H, self.W, [self.K]*len(self.pose_list), self.pose_list, convention='opencv')
        self.cam_rays_list = self.get_cam_rays_list(self.H, self.W, self.K_list, self.pose_list, convention='opencv')

    def load_shapes(self, shape_params_dict={}):
        '''
        load and visualize shapes (objs/furniture **& emitters**) in 3D & 2D: 
        '''
        if self.if_loaded_shapes: return
        
        mitsubaBase._prepare_shapes(self)

        scale_offset = () if not self.if_scale_scene else (self.scene_scale, 0.)
        shape_dict = load_shape_dict_from_shape_file(self.shape_file_list, shape_params_dict=shape_params_dict, scale_offset=scale_offset)
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

    def export_scene(
        self, 
        modality_list=[], 
        if_filter_with_main_region=False, # filter out frames with too many invalid rays (i.e. outside of MAIN region)
        if_force=False,
        ):
        # find invalid frames (frames with no valid rays)
        print(white_blue('Exporting %d frames... but remove invalid frames first'%len(self.frame_id_list)))

        if if_filter_with_main_region:
            print(yellow('Filtering frames with coverage of MAIN region (room) ONLY...'))
            print(yellow('Resetting mi scene to MAIN region only'))
            self.load_mi_scene(first_N_shapes=1, if_force=True)
            print(yellow('Resampling mi scene (MAIN region only)'))
            self.mi_sample_rays_pts(self.cam_rays_list, if_force=True)

        valid_frame_idx_list = []

        valid_ratio_thres = 0.5 if not if_filter_with_main_region else 0.5

        for frame_idx, frame_id in enumerate(self.frame_id_list):
            valid_mask = ~self.mi_invalid_depth_mask_list[frame_idx]
            valid_ratio = float(np.sum(valid_mask))/np.prod(valid_mask.shape[:2])
            if valid_ratio < valid_ratio_thres:
                print(yellow('frame %d has few valid rays (ratio %.2f), skip it.'%(frame_id, valid_ratio)))
                # self.frame_id_list.remove(frame_id)
            else:
                valid_frame_idx_list.append(frame_idx)
                print(frame_idx, '->', len(valid_frame_idx_list)-1, 'num valid pixels:', np.sum(valid_mask), 'ratio: %.2f'%(valid_ratio))

        _N_frames_total = len(self.frame_id_list)

        valid_member_list = [
            'frame_id_list', 
            
            'origin_lookatvector_up_list', 
            'pose_list', 
            'cam_rays_list', 
            'K_list', 

            'hdr_scale_list', 
            'im_hdr_list', 
            'im_sdr_list', 
            'im_mask_list', 
            
            'mi_depth_list', 
            'mi_normal_global_list', 
            'mi_invalid_depth_mask_list', 
            
            'im_undist_mask_list', 
        ]
        for valid_member in valid_member_list:
            if hasattr(self, valid_member):
                assert len(getattr(self, valid_member)) == _N_frames_total, 'len(%s)=%d != %d'%(valid_member, len(getattr(self, valid_member)), _N_frames_total)
                setattr(self, valid_member, [getattr(self, valid_member)[_] for _ in valid_frame_idx_list])

        print(white_blue('> Resulted in %d -> %d frames...'%(_N_frames_total, len(self.frame_id_list))))

        appendix = ''
        if if_filter_with_main_region: appendix += '_main'
        if self.if_undist: appendix += '_undist'

        mitsubaBase.export_scene(self, modality_list=modality_list, appendix=appendix, if_force=if_force)

    @property
    def region_description_dict(self):
        return {
            'a': 'bathroom (should have a toilet and a sink)', 
            'b': 'bedroom', 
            'c': 'closet', 
            'd': 'dining room (includes “breakfast rooms” other rooms people mainly eat in)', 
            'e': 'entryway/foyer/lobby (should be the front door, not any door)', 
            'f': 'familyroom (should be a room that a family hangs out in, not any area with couches)', 
            'g': 'garage', 
            'h': 'hallway', 
            'i': 'library (should be room like a library at a university, not an individual study)', 
            'j': 'laundryroom/mudroom (place where people do laundry, etc.)', 
            'k': 'kitchen', 
            'l': 'living room (should be the main “showcase” living room in a house, not any area with couches)', 
            'm': 'meetingroom/conferenceroom', 
            'n': 'lounge (any area where people relax in comfy chairs/couches that is not the family room or living room', 
            'o': 'office (usually for an individual, or a small set of people)', 
            'p': 'porch/terrace/deck/driveway (must be outdoors on ground level)', 
            'r': 'rec/game (should have recreational objects, like pool table, etc.)', 
            's': 'stairs', 
            't': 'toilet (should be a small room with ONLY a toilet)', 
            'u': 'utilityroom/toolroom ', 
            'v': 'tv (must have theater-style seating)', 
            'w': 'workout/gym/exercise', 
            'x': 'outdoor areas containing grass, plants, bushes, trees, etc.', 
            'y': 'balcony (must be outside and must not be on ground floor)', 
            'z': 'other room (it is clearly a room, but the function is not clear)', 
            'B': 'bar', 
            'C': 'classroom', 
            'D': 'dining booth', 
            'S': 'spa/sauna', 
            'Z': 'junk (reflections of mirrors, random points floating in space, etc.)', 
            '-': 'no label ', 
        }
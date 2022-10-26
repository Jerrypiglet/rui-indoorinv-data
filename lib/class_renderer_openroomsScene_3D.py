import numpy as np
from tqdm import tqdm

import mitsuba as mi
from lib.global_vars import mi_variant
mi.set_variant(mi_variant)
import torch
import matplotlib.pyplot as plt
import time

from lib.class_openroomsScene2D import openroomsScene2D
from lib.class_openroomsScene3D import openroomsScene3D

from lib.utils_rendering_PhySG import render_with_sg
from lib.utils_misc import yellow

class renderer_openroomsScene_3D(object):
    '''
    A class for differentiable renderers of OpenRooms (public/public-re versions) scene contents.

    renderer options:
    - Zhengqin's surface renderer (Li et al., 2020, Inverse Rendering for Complex Indoor Scenes)
        - input: per-pixel lighting envmap (e.g. 8x16); or SGs (to convert to envmaps)
    - PhySG surface renderer (Zhang et al., 2021, PhySG)
        - input: per-pixel lighting SGs (without having to convert to envmaps)
    '''
    def __init__(
        self, 
        openrooms_scene, 
        renderer_option: str, 
        host: str, # machine
        pts_from: str='mi', # 'mi': ray-intersection with mitsuba scene; 'depth': backprojected from OptixRenderer renderer depth maps
    ):

        assert type(openrooms_scene) in [openroomsScene2D, openroomsScene3D], '[visualizer_openroomsScene] has to take an object of openroomsScene or openroomsScene3D!'
        self.os = openrooms_scene

        self.renderer_option = renderer_option
        assert self.renderer_option in ['ZQ', 'PhySG']
        self.get_device(host)

        self.pts_from = pts_from
        assert self.pts_from in ['mi', 'depth']
        if self.pts_from == 'mi':
            assert self.os.if_has_mitsuba_scene and self.os.pts_from['mi']
        if self.pts_from == 'depth':
            assert self.os.if_has_dense_geo and self.os.pts_from['depth']

    def get_device(self, host: str):
        assert host in ['apple', 'mm1', 'qc']
        self.device = 'cpu'
        if host == 'apple':
            if torch.backends.mps.is_built() and torch.backends.mps.is_available():
                self.device = 'mps'
        else:
            if torch.cuda.is_available():
                self.device = 'cuda'
        # self.device = 'cpu'
        if self.device == 'cpu':
            print(yellow('[WARNING] rendering could be slow because device is cpu at %s'%host))

    def render(self, frame_idx: int):
        '''
        frame_idx: 0-based indexing into all frames: [0, 1, ..., self.os.frame_num-1]

        images/demo_render_PhySG_1.png
        images/demo_render_PhySG_2.png
        '''
        if self.renderer_option == 'PhySG':
            return_dict = self.render_PhySG(frame_idx)
        if self.renderer_option == 'ZQ':
            return_dict = self.render_ZQ(frame_idx)

        rgb_hdr_marched = return_dict['rgb_marched']
        im_hdr_marched = rgb_hdr_marched.cpu().numpy().reshape(self.os.im_H_resize, self.os.im_W_resize, 3)
        im_sdr_marched = np.clip(im_hdr_marched**(1./2.2), 0., 1.)
        im_sdr = np.clip(self.os.im_hdr_list[frame_idx]**(1./2.2), 0., 1.)

        plt.figure(figsize=(15, 4))
        ax = plt.subplot(131)
        ax.set_title('im_sdr GT')
        plt.imshow(im_sdr)
        ax = plt.subplot(132)
        ax.set_title('im_sdr marched by %s'%self.renderer_option)
        plt.imshow(im_sdr_marched)
        ax = plt.subplot(133)
        ax.set_title('diff')
        mask_obj = self.os.mi_seg_dict_of_lists['obj'][frame_idx].squeeze()
        im_sdr_diff = np.clip(np.abs(np.sum(im_sdr-im_sdr_marched, -1)) * mask_obj, 0., 0.5)
        plt.imshow(im_sdr_diff)
        plt.colorbar()
        plt.show()

    def render_PhySG(self, frame_idx):
        '''
        Mostly adapted from https://github.com/Jerrypiglet/PhySG/blob/master/code/model/sg_render.py#L137
        '''

        assert self.os.if_has_lighting_SG
        assert self.os.if_convert_lighting_SG_to_global, 'need to convert to global SG axis first!'
        lighting_SG_global = torch.from_numpy(self.os.lighting_SG_global_list[frame_idx]).to(self.device) # (H, W, 12, 3+1+3)
        lighting_SG_global = lighting_SG_global.flatten(0, 1) # (N(HW), 12, 3+1+3)
        # axis_global, lamb, weight = torch.split(lighting_SG_global, [3, 1, 3], dim=2) # (N, 12, 1/3)

        t = torch.from_numpy(self.os.pose_list[frame_idx][:3, -1].reshape((3, 1))).to(self.device)
        R = torch.from_numpy(self.os.pose_list[frame_idx][:3, :3]).to(self.device)
        normal_cam_opengl = torch.from_numpy(self.os.normal_list[frame_idx]).to(self.device).flatten(0, 1) # (N, 3)
        normal_cam_opencv = torch.stack([normal_cam_opengl[:, 0], -normal_cam_opengl[:, 1], -normal_cam_opengl[:, 2]], dim=-1) # transform axis from opengl convention (right-up-backward) to opencv (right-down-forward)
        normal = normal_cam_opencv @ R.T

        albedo = torch.from_numpy(self.os.albedo_list[frame_idx]).to(self.device).flatten(0, 1) # (N, 3)
        roughness = torch.from_numpy(self.os.roughness_list[frame_idx]).to(self.device).flatten(0, 1) # (N, 1)

        _, rays_d, _ = self.os.cam_rays_list[frame_idx]
        viewdirs = -torch.from_numpy(rays_d).to(self.device).flatten(0, 1)

        tic = time.time()
        ret_dict_PhySG = render_with_sg(
            lgtSGs = lighting_SG_global, 
            specular_reflectance = torch.tensor([0.05, 0.05, 0.05]).reshape(1, 3).to(self.device),
            roughness = roughness, 
            diffuse_albedo = albedo, 
            normal = normal, 
            viewdirs = viewdirs,
            )
        print('Rendering done. Took %.2f ms.'%((time.time()-tic)*1000.))
        return_dict = {
            'viewdirs_PhySG': -viewdirs, 
            'rgb_marched': ret_dict_PhySG['sg_rgb'], 
            'rgb_specular_marched': ret_dict_PhySG['sg_specular_rgb'], 
            'rgb_diffuse_marched': ret_dict_PhySG['sg_diffuse_rgb'], 
            }
        
        return return_dict


    def render_ZQ(self, frame_idx):
        pass
import numpy as np
from tqdm import tqdm

import mitsuba as mi
# from lib.global_vars import mi_variant
# mi.set_variant(mi_variant)
import torch
import matplotlib.pyplot as plt
import time

from lib.class_openroomsScene2D import openroomsScene2D
from lib.class_openroomsScene3D import openroomsScene3D

from lib.utils_rendering_PhySG import render_with_sg
from lib.utils_rendering_ZQ import rendering_layer_per_point
from lib.utils_rendering_ZQ_emitter import rendering_layer_per_point_from_emitter
from lib.utils_misc import yellow, get_device

class renderer_openroomsScene_3D(object):
    '''
    A class for differentiable renderers of OpenRooms (public/public-re versions) scene contents.

    renderer options:
    - Zhengqin Li's surface renderer (Li et al., 2020, Inverse Rendering for Complex Indoor Scenes)
        - input: per-pixel lighting envmap (e.g. 8x16); or SGs (to convert to envmaps)
    - PhySG surface renderer (Zhang et al., 2021, PhySG)
        - input: per-pixel lighting SGs (without having to convert to envmaps)
    '''
    def __init__(
        self, 
        openrooms_scene, 
        renderer_option: str, 
        host: str, # machine
        renderer_params: dict={}, 
    ):

        assert type(openrooms_scene) in [openroomsScene2D, openroomsScene3D], '[visualizer_openroomsScene] has to take an object of openroomsScene or openroomsScene3D!'
        self.os = openrooms_scene

        self.renderer_option = renderer_option
        assert self.renderer_option in ['ZQ', 'PhySG', 'ZQ_emitter']
        get_device(host)

        if self.renderer_option == 'ZQ':
            self.render_layer_ZQ = rendering_layer_per_point(
                imWidth=self.os.W, imHeight=self.os.H, 
                env_width=self.os.lighting_params_dict['env_width'], 
                env_height=self.os.lighting_params_dict['env_height'], 
                device=self.device, 
                )
        if self.renderer_option == 'ZQ_emitter':
            self.render_layer_ZQ_from_emitter = rendering_layer_per_point_from_emitter(
                device=self.device, 
                )

        self.pts_from = renderer_params.get('pts_from', 'mi') # 'mi': ray-intersection with mitsuba scene; 'depth': backprojected from OptixRenderer renderer depth maps
        assert self.pts_from in ['mi', 'depth']
        if self.pts_from == 'mi':
            assert self.os.if_has_mitsuba_scene and self.os.pts_from['mi']
        if self.pts_from == 'depth':
            assert self.os.if_has_dense_geo and self.os.pts_from['depth']

    def render(
        self, 
        frame_idx: int, 
        if_show_rendering_plt: bool=True, 
        render_params: dict={}, 
    ):
        '''
        frame_idx: 0-based indexing into all frames: [0, 1, ..., self.os.frame_num-1]
        '''
        
        if self.renderer_option == 'PhySG':
            return_dict = self.render_PhySG(frame_idx)
        if self.renderer_option == 'ZQ':
            return_dict = self.render_ZQ(frame_idx)
        if self.renderer_option == 'ZQ_emitter':
            return_dict = self.render_ZQ_emitter(frame_idx, render_params=render_params)

        im_sdr = np.clip(self.os.im_hdr_list[frame_idx]**(1./2.2), 0., 1.)

        rgb_marched_hdr = return_dict['rgb_marched']
        im_marched_hdr = rgb_marched_hdr.cpu().numpy().reshape(self.os.H, self.os.W, 3)
        im_marched_sdr = np.clip(im_marched_hdr**(1./2.2), 0., 1.)

        rgb_marched_diffuse_hdr = return_dict['rgb_marched_diffuse']
        im_marched_diffuse_hdr = rgb_marched_diffuse_hdr.cpu().numpy().reshape(self.os.H, self.os.W, 3)
        im_marched_diffuse_sdr = np.clip(im_marched_diffuse_hdr**(1./2.2), 0., 1.)

        rgb_marched_specular_hdr = return_dict['rgb_marched_specular']
        im_marched_specular_hdr = rgb_marched_specular_hdr.cpu().numpy().reshape(self.os.H, self.os.W, 3)
        im_marched_specular_sdr = np.clip(im_marched_specular_hdr**(1./2.2), 0., 1.)

        if if_show_rendering_plt:
            plt.figure(figsize=(15, 4))
            ax = plt.subplot(231)
            ax.set_title('im_sdr GT')
            plt.imshow(im_sdr)
            ax = plt.subplot(232)
            ax.set_title('im_sdr marched by %s'%self.renderer_option)
            plt.imshow(im_marched_sdr)
            ax = plt.subplot(233)
            ax.set_title('diff')
            mask_obj = self.os.mi_seg_dict_of_lists['obj'][frame_idx].squeeze()
            im_sdr_diff = np.clip(np.abs(np.sum(im_sdr-im_marched_sdr, -1)) * mask_obj, 0., 0.5)
            plt.imshow(im_sdr_diff)
            plt.colorbar()
            ax = plt.subplot(234)
            ax.set_title('im_sdr marched-diffuse by %s'%self.renderer_option)
            plt.imshow(im_marched_diffuse_sdr)
            ax = plt.subplot(235)
            ax.set_title('im_sdr marched-specular by %s'%self.renderer_option)
            plt.imshow(im_marched_specular_sdr)
            plt.show()

        return return_dict

    def render_PhySG(self, frame_idx):
        '''
        Mostly adapted from https://github.com/Jerrypiglet/PhySG/blob/master/code/model/sg_render.py#L137

        images/demo_render_PhySG_1.png
        images/demo_render_PhySG_2.png
        images/demo_render_PhySG_Direct_1.png
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
            'rgb_marched_specular': ret_dict_PhySG['sg_specular_rgb'], 
            'rgb_marched_diffuse': ret_dict_PhySG['sg_diffuse_rgb'], 
            }
        
        return return_dict

    def render_ZQ(self, frame_idx):
        '''
        images/demo_render_ZQ_1.png
        images/demo_render_ZQ_2.png
        images/demo_render_ZQ_emitter_1.png
        '''
        assert self.os.if_has_lighting_envmap
        H, W = self.os.H, self.os.W
        N_frames = 1
        N = N_frames * H * W
        rays_uv = torch.zeros([N_frames, H, W, 2], device=self.device).long()
        uu, vv = torch.meshgrid(
            torch.linspace(0, W-1, W, device=self.device),
            torch.linspace(0, H-1, H, device=self.device))  # pytorch's meshgrid has indexing='ij'
        rays_uv[:, :, :, 0] = uu.T.unsqueeze(0).expand(N_frames, -1, -1).long()
        rays_uv[:, :, :, 1] = vv.T.unsqueeze(0).expand(N_frames, -1, -1).long()
        rays_uv = rays_uv.flatten(0, 2) # (N, 2)

        lighting_envmap_cam = torch.from_numpy(self.os.lighting_envmap_list[frame_idx]).to(self.device) # (env_row, env_col, 3, env_height, env_width)
        envmap_cam = lighting_envmap_cam.flatten(0, 1) # (N, 3, env_height, env_width)

        normal_cam_opengl = torch.from_numpy(self.os.normal_list[frame_idx]).to(self.device).flatten(0, 1) # (N, 3)
        albedo = torch.from_numpy(self.os.albedo_list[frame_idx]).to(self.device).flatten(0, 1) # (N, 3)
        roughness = torch.from_numpy(self.os.roughness_list[frame_idx]).to(self.device).flatten(0, 1) # (N, 1)

        tic = time.time()
        diffuse, specular = self.render_layer_ZQ.forwardEnv(rays_uv=rays_uv, normal=normal_cam_opengl, envmap=envmap_cam, albedo=albedo, roughness=roughness)
        rgb_marched_hdr = diffuse + specular
        print('Rendering done. Took %.2f ms.'%((time.time()-tic)*1000.))
    
        return_dict = {
            'rgb_marched': rgb_marched_hdr,
            'rgb_marched_specular': specular,
            'rgb_marched_diffuse': diffuse,
        }

        return return_dict

    def render_ZQ_emitter(
        self, 
        frame_idx, 
        render_params={}, 
    ):
        max_plate = render_params.get('max_plate', 256)
        (emitter_type, emitter_index) = render_params.get('emitter_type_index')
        '''
        [TODO] simplify lamp mesh
        '''
        albedo = torch.from_numpy(self.os.albedo_list[frame_idx]).to(self.device).flatten(0, 1) # (N, 3)
        roughness = torch.from_numpy(self.os.roughness_list[frame_idx]).to(self.device).flatten(0, 1) # (N, 1)
        # normal_cam_opengl = torch.from_numpy(self.os.mi_normal_list[frame_idx]).to(self.device).flatten(0, 1) # (N, 3)
        # normal_cam_opencv = torch.stack([normal_cam_opengl[:, 0], -normal_cam_opengl[:, 1], -normal_cam_opengl[:, 2]], axis=-1)
        # R = torch.from_numpy(self.os.pose_list[frame_idx][:3, :3]).to(self.device)
        # normal = torch.nn.functional.normalize(normal_cam_opencv @ R.T, dim=-1)
        normal = torch.from_numpy(self.os.mi_normal_global_list[frame_idx]).to(self.device).flatten(0, 1) # (N, 3)

        _, rays_d, _ = self.os.cam_rays_list[frame_idx]
        viewdirs = -torch.from_numpy(rays_d).to(self.device).flatten(0, 1)
        
        assert emitter_type == 'lamp', 'no support for windows for now'
        lamp, vertices, faces = self.os.lamp_list[emitter_index]
        intensity = lamp['emitter_prop']['intensity'] # (3,)
        center = lamp['emitter_prop']['box3D_world']['center'] # (3,)

        # >>>> sample lamp
        v1 = vertices[faces[:, 0]-1, :]
        v2 = vertices[faces[:, 1]-1, :]
        v3 = vertices[faces[:, 2]-1, :]

        lpts = 1.0 / 3.0 * (v1 + v2 + v3)
        e1 = v2 - v1
        e2 = v3 - v1
        lpts_normal = np.cross(e1, e2)

        # [DEBUG] get rid of upper faces
        # faces = faces[lpts_normal[:, 1]<0]
        # from lib.utils_OR.utils_OR_mesh import writeMesh
        # writeMesh('tmp_mesh.obj', vertices, faces)
        # v1 = vertices[faces[:, 0]-1, :]
        # v2 = vertices[faces[:, 1]-1, :]
        # v3 = vertices[faces[:, 2]-1, :]
        # lpts = 1.0 / 3.0 * (v1 + v2 + v3)
        # e1 = v2 - v1
        # e2 = v3 - v1
        # lpts_normal = np.cross(e1, e2)

        lpts_area = 0.5 * np.sqrt(np.sum(
            lpts_normal * lpts_normal, axis=1, keepdims = True))
        lpts_normal = lpts_normal / np.maximum(2 * lpts_area, 1e-6)

        center = np.mean(vertices, axis=0, keepdims = True)

        normal_flip = (np.sum(lpts_normal * (lpts - center), axis=1, keepdims=True) < 0) # [TODO] ZQ is trying to deal with concave faces. Better ideas?
        normal_flip = normal_flip.astype(np.float32)
        lpts_normal = -lpts_normal * normal_flip + (1 - normal_flip) * lpts_normal

        plate_num = lpts.shape[0]

        lpts = torch.from_numpy(lpts).to(self.device) # (M=256, 3)
        lpts_normal = torch.from_numpy(lpts_normal).to(self.device)
        lpts_area = torch.from_numpy(lpts_area).to(self.device)

        if plate_num > max_plate: # [TODO] choose fixed max_plate num of plates
            prob = float(max_plate)  / float(plate_num)
            select_ind = np.random.choice([0, 1], size=(plate_num), p=[1-prob, prob])
            select_ind = torch.from_numpy(select_ind).long().to(self.device)
            lpts = lpts[select_ind == 1]
            lpts_normal = lpts_normal[select_ind == 1]
            lpts_area = lpts_area[select_ind == 1]
            prob = float(torch.sum(select_ind))  / float(plate_num)
        else:
            prob = 1

        lpts_intensity = torch.from_numpy(intensity).to(self.device).view(1, 3)
        # <<<< sample lamp

        pts = torch.from_numpy(self.os.mi_pts_list[frame_idx]).to(self.device).flatten(0, 1) # (N, 3)
        l_dirs = lpts.unsqueeze(0) - pts.unsqueeze(1) # (N, M=256, 3)
        pts_distL2 = torch.linalg.norm(l_dirs, dim=2, keepdims=True)

        l_dirs = torch.nn.functional.normalize(l_dirs, dim=2) # (N, M=256, 3), dir: scene points to lamp

        # pts_cos = torch.sum(l_dirs * normal.unsqueeze(1), dim=2, keepdim=True) # (N, M=256, 1); == ndl [!!!!!]
        lpt_cos = torch.clamp(torch.sum(l_dirs * lpts_normal.unsqueeze(0), dim=2, keepdim=True), -1, 1) # (N, M=256, 1)
        
        pts_intensity = lpts_intensity.unsqueeze(0) * lpt_cos.abs() # (N, M=256, 3)
        # pts_intensity = lpts_intensity.unsqueeze(0) * torch.clamp(pts_cos, min=0, max=1) * lpt_cos.abs() # (N, M=256, 3)
        # pts_intensity = lpts_intensity.unsqueeze(0) * torch.clamp(pts_cos, min=0, max=1) * torch.clamp(lpt_cos, min=0, max=1) # (N, M=256, 3)
        pts_intensity_weighted = pts_intensity / (pts_distL2**2) * lpts_area.unsqueeze(0) / prob

        # >>>> compute visibility
        N_pts = normal.shape[0]
        M_lpts = l_dirs.shape[1]
        ds = l_dirs.flatten(0, 1).cpu()
        # ds = (lpts.unsqueeze(0) - pts.unsqueeze(1)).flatten(0, 1).cpu()
        ray_o = pts.unsqueeze(1).expand(-1, M_lpts, -1).cpu().numpy()
        ray_e = lpts.unsqueeze(0).expand(N_pts, -1, -1).cpu().numpy()
        # ds = (torch.from_numpy(center.reshape(1, 1, 3)) - pts.unsqueeze(1).expand(-1, M_lpts, -1).cpu()).flatten(0, 1)
        ds_mi = mi.Vector3f(ds)
        xs = pts.unsqueeze(1).expand(-1, M_lpts, -1).flatten(0, 1).cpu()
        xs_mi = mi.Point3f(xs+1e-4*ds)
        # ray origin, direction, t_max
        rays_mi = mi.Ray3f(xs_mi, ds_mi)
        ret = self.os.mi_scene.ray_intersect(rays_mi) # https://mitsuba.readthedocs.io/en/stable/src/api_reference.html?highlight=write_ply#mitsuba.Scene.ray_intersect
        ts = ret.t.torch()
        # visibility = np.logical_and(ts > 1e-3, ts < (pts_distL2.flatten().cpu()-1e-2))
        # visibility = 1. - visibility.float()
        visibility = (ts > (pts_distL2.flatten().cpu()-1e-2)).float()
        # <<<< compute visibility
# 
        pts_intensity_weighted = pts_intensity_weighted * visibility.view(N_pts, M_lpts, 1).to(self.device)

        tic = time.time()
        diffuse, specular = self.render_layer_ZQ_from_emitter.forward_rays(
            normal=normal, 
            albedo=albedo, 
            roughness=roughness, 
            v_dirs=viewdirs, # scene points to camera center (-ray_d)
            l_dirs=l_dirs, # (N, M=256, 3), dir: scene points to lamp
            pts_intensity_weighted=pts_intensity_weighted, 
            )
        rgb_marched_hdr = diffuse + specular
        print('Rendering done. Took %.2f ms.'%((time.time()-tic)*1000.))
    
        return_dict = {
            'rgb_marched': rgb_marched_hdr,
            'rgb_marched_specular': specular,
            'rgb_marched_diffuse': diffuse,
            'ray_o': ray_o, 
            'ray_e': ray_e, 
            'visibility': visibility.view(N_pts, M_lpts).cpu().numpy(), 
            'ts': ts.view(N_pts, M_lpts).cpu().numpy(), 
        }

        return return_dict

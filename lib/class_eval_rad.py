import sys
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import math

from lib.class_openroomsScene3D import openroomsScene3D
from lib.class_mitsubaScene3D import mitsubaScene3D
from lib.global_vars import mi_variant_dict
from lib.utils_OR.utils_OR_emitter import sample_mesh_emitter
from lib.utils_misc import white_blue

class evaluator_scene_rad():
    '''
    evaluator for trained NeRF (rad-MLP)
    '''
    def __init__(
        self, 
        scene_object, 
        host: str, 
        INV_NERF_ROOT: str, 
        ckpt_path: str, # relative to INV_NERF_ROOT / 'checkpoints'
        dataset_key: str, 
        split: str='', 
        rad_scale: float=1.
    ):
        sys.path.insert(0, str(INV_NERF_ROOT))
        self.INV_NERF_ROOT = Path(INV_NERF_ROOT)

        self.dataset_type = dataset_key.split('-')[0]
        if self.dataset_type == 'OR':
            assert type(scene_object) is openroomsScene3D
            from configs.rad_config_openrooms import default_options
        elif self.dataset_type == 'Indoor':
            assert type(scene_object) is mitsubaScene3D
            from configs.rad_config_indoor import default_options
        else:
            assert False, 'Unknown dataset_key: %s'%dataset_key

        ckpt_path = self.INV_NERF_ROOT / 'checkpoints' / ckpt_path

        from train_rad_rui import ModelTrainer, add_model_specific_args
        from argparse import ArgumentParser
        from configs.scene_options import scene_options

        default_options['dataset'] = scene_options[dataset_key]
        parser = ArgumentParser()
        parser = add_model_specific_args(parser, default_options)
        hparams, _ = parser.parse_known_args()
        self.device = {
            'apple': 'mps', 
            'mm1': 'cuda', 
            'qc': '', 
        }[host]

        self.model = ModelTrainer(
            hparams, 
            host=host, 
            dataset_key=dataset_key, 
            if_overfit_train=False, 
            if_seg_obj=False, 
            mitsuba_variant=mi_variant_dict[host], 
            scene_object=scene_object, 
        ).to(self.device)

        checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
        self.model.load_state_dict({k: v for k, v in checkpoint['state_dict'].items() if 'nerf.' in k})
        # print(checkpoint['state_dict'].keys())
        # print(checkpoint['state_dict']['nerf.linears.0.bias'][:2])
        # print(self.model.nerf.linears[0].bias[:2])
        self.model.eval()

        self.rad_scale = rad_scale
        self.os = self.model.scene_object

    def or2nerf_th(self, x):
        """x:Bxe"""
        ret = torch.tensor([[1,1,-1]], device=x.device)*x
        return ret[:,[0,2,1]]

    # def or2nerf_np(self, x):
    #     """x:Bxe"""
    #     # ret = np.array([[1,1,-1]], dtype=x.dtype) * x
    #     # return ret[:,[0,2,1]]
    #     return x @ np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=x.dtype)

    def to_d(self, x: np.ndarray):
        if 'mps' in self.device: # Mitsuba RuntimeError: Cannot pack tensors on mps:0
            return x
        return torch.from_numpy(x).to(self.device)

    def render_im(self, frame_id: int, if_plt: bool=False):
        '''
        render one image by querying rad-MLP: 
        public_re_3_v3pose_2048:
            images/demo_eval_radMLP_render.png
            images/demo_eval_radMLP_render_166.png
            images/demo_eval_radMLP_render_208.png
        public_re_3_v5pose_2048:
            images/demo_eval_radMLP_render_110.png

        '''
        assert self.os.if_has_mitsuba_rays_pts
        (rays_o, rays_d, ray_d_center) = self.os.cam_rays_list[frame_id]
        rays_o_np = rays_o.reshape(-1, 3)
        rays_d_np = rays_d.reshape(-1, 3)

        # rays_o_nerf = mi.Point3f(self.or2nerf_np(rays_o_th)) # concert to NeRF coordinates
        # rays_d_nerf = mi.Vector3f(self.or2nerf_np(rays_d_th))
        rays_d_nerf = self.or2nerf_th(torch.from_numpy(rays_d_np).to(self.device)) # convert to NeRF coordinates

        # rays_o_th = torch.from_numpy(rays_o).to(self.device)[::2, ::2].flatten(0, 1)
        # rays_d_th = torch.from_numpy(rays_d).to(self.device)[::2, ::2].flatten(0, 1)
        # rays_o_nerf = self.or2nerf_th(rays_o_th)
        # rays_d_nerf = self.or2nerf_th(rays_d_th)

        position, _, t_, valid = self.model.ray_intersect(rays_o_np, rays_d_np, if_mi_np=True)
        if self.dataset_type == 'Indoor':
            # Liwen's model was trained using scene.obj (smaller) instead of scene_v3.xml (bigger), between which there is scaling and translation. self.os.cam_rays_list are acquired from the scene of scene_v3.xml
            scale_m2b = torch.from_numpy(np.array([0.206,0.206,0.206], dtype=np.float32).reshape((1, 3))).to(position.device)
            trans_m2b = torch.from_numpy(np.array([-0.074684,0.23965,-0.30727], dtype=np.float32).reshape((1, 3))).to(position.device)
            position = scale_m2b * position + trans_m2b
        rgbs = self.model.nerf(position.to(self.device), rays_d_nerf)['rgb'] # queried d is incoming directions!

        if if_plt:
            plt.figure()
            ax = plt.subplot(121)
            plt.imshow(np.clip(self.model.gamma_func(rgbs).detach().cpu().reshape((self.os.H, self.os.W, 3)), 0., 1.))
            ax.set_title('[%d] rendered image from rad-MLP'%frame_id)
            ax = plt.subplot(122)
            plt.imshow(np.clip(self.model.gamma_func(self.os.im_hdr_list[frame_id]/self.os.hdr_scale_list[frame_id]*self.rad_scale), 0., 1.))
            ax.set_title('[%d] GT image; set to same scale as image loaded in rad-MLP'%frame_id)
            plt.show()

    def sample_emitter(self, emitter_params={}):
        '''
        sample emitter surface radiance from rad-MLP: images/demo_emitter_o3d_sampling.png
        '''
        max_plate = emitter_params.get('max_plate', 64)
        emitter_type_index_list = emitter_params.get('emitter_type_index_list', [])
        emitter_dict = {'lamp': self.os.lamp_list, 'window': self.os.window_list}
        emitter_rays_list = []

        for emitter_type_index in emitter_type_index_list:
            (emitter_type, emitter_index) = emitter_type_index
            for emitter_index in range(len(emitter_dict[emitter_type])):
                lpts_dict = sample_mesh_emitter(emitter_type, emitter_index=emitter_index, emitter_dict=emitter_dict, max_plate=max_plate)
                rays_o_nerf = self.or2nerf_th(torch.from_numpy(lpts_dict['lpts']).to(self.device)) # convert to NeRF coordinates
                rays_d_nerf = self.or2nerf_th(torch.from_numpy(-lpts_dict['lpts_normal']).to(self.device)) # convert to NeRF coordinates
                rgbs = self.model.nerf(rays_o_nerf, rays_d_nerf)['rgb'] # queried d is incoming directions!
                intensity = rgbs.detach().cpu().numpy() / self.rad_scale # get back to original scale, without hdr scaling
                lpts_end = lpts_dict['lpts'] + lpts_dict['lpts_normal'] * np.log(intensity.sum(-1, keepdims=True)) * 0.1
                emitter_rays_list.append((lpts_dict['lpts'], lpts_end))
                # emitter_rays = o3d.geometry.LineSet()
                # emitter_rays.points = o3d.utility.Vector3dVector(np.vstack((lpts_dict['lpts'], lpts_end)))
                # emitter_rays.colors = o3d.utility.Vector3dVector([[0.3, 0.3, 0.3]]*lpts_dict['lpts'].shape[0])
                # emitter_rays.lines = o3d.utility.Vector2iVector([[_, _+lpts_dict['lpts'].shape[0]] for _ in range(lpts_dict['lpts'].shape[0])])
                # geometry_list.append([emitter_rays, 'emitter_rays'])

        return {'emitter_rays_list': emitter_rays_list}

    def sample_lighting_envmap(self, subsample_rate_pts: int=1, if_use_mi_geometry: bool=True):
        '''
        sample non-emitter locations along envmap (hemisphere) directions radiance from rad-MLP: images/demo_envmap_o3d_sampling.png
        '''

        if if_use_mi_geometry:
            assert self.os.if_has_mitsuba_all; normal_list = self.os.mi_normal_list
        else:
            assert False, 'use mi for GT geometry which was used to train rad-MLP!'
            # assert self.os.if_has_dense_geo; normal_list = self.os.normal_list
        batch_size = self.model.hparams.batch_size * 10
        lighting_fused_list = []

        print(white_blue('[evaluator_scene_rad] sampling %s for %d frames... subsample_rate_pts: %d'%('lighting_envmap', len(self.os.frame_id_list), subsample_rate_pts)))

        for idx in tqdm(range(len(self.os.frame_id_list))):
            seg_obj = self.os.mi_seg_dict_of_lists['obj'][idx].reshape(-1) # (H, W), bool, [IMPORTANT] mask off emitter area!!

            axis_global = self.os.get_lighting_envmap_dirs_global(self.os.pose_list[idx], normal_list[idx])[seg_obj] # (HW, env_num, 3)
            axis_global_rays = axis_global.reshape(-1, 3) # (HW*env_num, 3)
            mi_pts = self.os.mi_pts_list[idx] # (H, W, 3)
            env_num = self.os.lighting_params_dict['env_height'] * self.os.lighting_params_dict['env_width']
            assert axis_global.shape[1] == env_num

            mi_pts_rays = np.repeat(np.expand_dims(mi_pts.reshape(-1, 3)[seg_obj], axis=1), env_num, axis=1).reshape(-1, 3) # (HW*env_num, 3)
            assert axis_global_rays.shape == mi_pts_rays.shape

            if subsample_rate_pts != 1:
                axis_global_rays = axis_global_rays[::subsample_rate_pts]
                mi_pts_rays = mi_pts_rays[::subsample_rate_pts]

            # batching to prevent memory overflow
            intensity_list = []
            for b_id in tqdm(range(math.ceil(axis_global_rays.shape[0]*1.0/batch_size))):
                b0 = b_id*batch_size
                b1 = min((b_id+1)*batch_size, axis_global_rays.shape[0])
                axis_global_rays_select = axis_global_rays[b0:b1]
                mi_pts_rays_select = mi_pts_rays[b0:b1]

                rays_o_nerf = self.or2nerf_th(torch.from_numpy(mi_pts_rays_select).to(self.device))
                rays_d_nerf = self.or2nerf_th(torch.from_numpy(-axis_global_rays_select).to(self.device))
                rgbs = self.model.nerf(rays_o_nerf, rays_d_nerf)['rgb']
                intensity = rgbs.detach().cpu().numpy() / self.rad_scale # get back to original scale, without hdr scaling
                # intensity = intensity * 0. + 10.
                intensity_list.append(intensity)
            
            intensity_all = np.concatenate(intensity_list)
            assert intensity_all.shape[0] == mi_pts_rays.shape[0]
            lighting_fused_dict = {'X_global_lighting': mi_pts_rays, 'axis': axis_global_rays, 'weight': intensity_all}
            lighting_fused_list.append(lighting_fused_dict)

        return {'lighting_fused_list': lighting_fused_list}

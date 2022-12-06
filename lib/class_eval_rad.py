import sys
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import math
import mitsuba as mi

from lib.class_openroomsScene3D import openroomsScene3D
from lib.class_mitsubaScene3D import mitsubaScene3D
from lib.global_vars import mi_variant_dict

from lib.utils_OR.utils_OR_emitter import sample_mesh_emitter
from lib.utils_misc import get_list_of_keys, white_blue
from lib.utils_OR.utils_OR_lighting import get_lighting_envmap_dirs_global
from lib.utils_mitsuba import get_rad_meter_sensor

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
        rad_scale: float=1., 
        spec: bool=True, 
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
        self.host = host
        self.device = {
            'apple': 'mps', 
            'mm1': 'cuda', 
            'qc': '', 
        }[self.host]

        self.model = ModelTrainer(
            hparams, 
            host=self.host, 
            dataset_key=dataset_key, 
            if_overfit_train=False, 
            if_seg_obj=False, 
            mitsuba_variant=mi_variant_dict[self.host], 
            scene_object=scene_object, 
            spec=spec, 
        ).to(self.device)

        checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
        self.model.load_state_dict({k: v for k, v in checkpoint['state_dict'].items() if 'nerf.' in k})
        # print(checkpoint['state_dict'].keys())
        # print(checkpoint['state_dict']['nerf.linears.0.bias'][:2])
        # print(self.model.nerf.linears[0].bias[:2])
        self.model.eval()

        self.rad_scale = rad_scale
        self.os = self.model.scene_object

        mi.set_variant(mi_variant_dict[self.host])

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
            position = self.process_for_Indoor(position)
        rgbs = self.model.nerf(position.to(self.device), rays_d_nerf)['rgb'] # queried d is incoming directions!

        if if_plt:
            plt.figure()
            ax = plt.subplot(121)
            plt.imshow(np.clip(self.model.gamma_func(rgbs).detach().cpu().reshape((self.os.H, self.os.W, 3)), 0., 1.))
            ax.set_title('[%d] rendered image from rad-MLP'%frame_id)
            ax = plt.subplot(122)
            plt.imshow(np.clip(self.model.gamma_func(self.os.im_hdr_list[frame_id]/self.os.hdr_scale_list[frame_id]*self.rad_scale), 0., 1.))

            # a = rgbs.detach().cpu().numpy().reshape((self.os.H, self.os.W, 3))[0:100, 50:250].reshape((-1, 3))
            # b = self.os.im_hdr_list[frame_id].reshape((self.os.H, self.os.W, 3))[0:100, 50:250].reshape((-1, 3))
            ax.set_title('[%d] GT image; set to same scale as image loaded in rad-MLP'%frame_id)
            plt.show()

    def sample_emitter(self, emitter_params={}):
        '''
        sample emitter surface radiance from rad-MLP: images/demo_emitter_o3d_sampling.png
        '''
        max_plate = emitter_params.get('max_plate', 64)
        radiance_scale = emitter_params.get('radiance_scale', 1.)
        emitter_type_index_list = emitter_params.get('emitter_type_index_list', 1.)
        emitter_dict = {'lamp': self.os.lamp_list, 'window': self.os.window_list}
        emitter_rays_list = []

        for emitter_type_index in emitter_type_index_list:
            (emitter_type, _) = emitter_type_index
            for emitter_index in range(len(emitter_dict[emitter_type])):
                lpts_dict = sample_mesh_emitter(emitter_type, emitter_index=emitter_index, emitter_dict=emitter_dict, max_plate=max_plate, if_dense_sample=True)
                rays_o_nerf = self.or2nerf_th(torch.from_numpy(lpts_dict['lpts']).to(self.device)) # convert to NeRF coordinates
                rays_d_nerf = self.or2nerf_th(torch.from_numpy(-lpts_dict['lpts_normal']).to(self.device)) # convert to NeRF coordinates
                if self.dataset_type == 'Indoor':
                    rays_o_nerf = self.process_for_Indoor(rays_o_nerf)
                rgbs = self.model.nerf(rays_o_nerf, rays_d_nerf)['rgb'] # queried d is incoming directions!
                intensity = rgbs.detach().cpu().numpy() / self.rad_scale # get back to original scale, without hdr scaling
                o_ = lpts_dict['lpts']
                d_ = lpts_dict['lpts_normal'] / np.linalg.norm(lpts_dict['lpts_normal'], axis=-1, keepdims=True)
                lpts_end = o_ + d_ * np.linalg.norm(intensity, axis=-1, keepdims=True) * radiance_scale
                print('EST intensity', intensity, np.linalg.norm(intensity, axis=-1))
                emitter_rays_list.append((lpts_dict['lpts'], lpts_end))
                # emitter_rays = o3d.geometry.LineSet()
                # emitter_rays.points = o3d.utility.Vector3dVector(np.vstack((lpts_dict['lpts'], lpts_end)))
                # emitter_rays.colors = o3d.utility.Vector3dVector([[0.3, 0.3, 0.3]]*lpts_dict['lpts'].shape[0])
                # emitter_rays.lines = o3d.utility.Vector2iVector([[_, _+lpts_dict['lpts'].shape[0]] for _ in range(lpts_dict['lpts'].shape[0])])
                # geometry_list.append([emitter_rays, 'emitter_rays'])

        return {'emitter_rays_list': emitter_rays_list}

    def sample_lighting(
        self, 
        sample_type: str='emission', 
        subsample_rate_pts: int=1, 
        if_use_mi_geometry: bool=True, 
        if_mask_off_emitters: bool=False, 
        if_vis_envmap_2d_plt: bool=False, 
        lighting_scale: float=1., 
        ):

        '''
        sample non-emitter locations along hemisphere directions for incident radiance, from rad-MLP: images/demo_envmap_o3d_sampling.png
        Args:
            sample_type: 'emission' for querying radiance emitted FROM all points; 'incident' for incident radiance TOWARDS all points
        Results:
            images/demo_eval_radMLP_rample_lighting_openrooms_1.png
        '''
        
        assert sample_type in ['emission', 'incident']
        if if_use_mi_geometry:
            assert self.os.if_has_mitsuba_all; normal_list = self.os.mi_normal_list
        else:
            assert False, 'use mi for GT geometry which was used to train rad-MLP!'
            # assert self.os.if_has_depth_normal; normal_list = self.os.normal_list
        batch_size = self.model.hparams.batch_size * 10
        lighting_fused_list = []
        lighting_envmap_list = []

        print(white_blue('[evaluator_scene_rad] sampling %s for %d frames... subsample_rate_pts: %d'%('lighting_envmap', len(self.os.frame_id_list), subsample_rate_pts)))

        for idx in tqdm(range(len(self.os.frame_id_list))):
            seg_obj = self.os.mi_seg_dict_of_lists['obj'][idx].reshape(-1) # (H, W), bool, [IMPORTANT] mask off emitter area!!
            if not if_mask_off_emitters:
                seg_obj = np.ones_like(seg_obj).astype(np.bool)

            env_height, env_width = get_list_of_keys(self.os.lighting_params_dict, ['env_height', 'env_width'], [int, int])
            '''
            [emission] directions from a surface point to hemisphere directions
            '''
            env_num = self.os.lighting_params_dict['env_height'] * self.os.lighting_params_dict['env_width']
            
            samples_d = get_lighting_envmap_dirs_global(self.os.pose_list[idx], normal_list[idx], env_height, env_width)[seg_obj] # (HW, env_num, 3)
            assert samples_d.shape[1] == env_num
            samples_d = samples_d.reshape(-1, 3) # (HW*env_num, 3)
            samples_o = self.os.mi_pts_list[idx] # (H, W, 3)
            samples_o = np.expand_dims(samples_o.reshape(-1, 3)[seg_obj], axis=1)
            # samples_o = np.broadcast_to(samples_o, (samples_o.shape[0], env_num, 3))
            # samples_o = samples_o.view(-1, 3)
            samples_o = np.repeat(samples_o, env_num, axis=1).reshape(-1, 3) # (HW*env_num, 3)
            if sample_type == 'emission':
                rays_d = samples_d
                rays_o = samples_o
            elif sample_type == 'incident':
                rays_d = -samples_d # from destination point, towards opposite direction
                # [mitsuba.Interaction3f] https://mitsuba.readthedocs.io/en/stable/src/api_reference.html#mitsuba.Interaction3f
                # [mitsuba.SurfaceInteraction3f] https://mitsuba.readthedocs.io/en/latest/src/api_reference.html#mitsuba.SurfaceInteraction3f

                ds_mi = mi.Vector3f(samples_d) # (HW*env_num, 3)
                # --- [V1] hack by adding a small offset to samples_o to reduce self-intersection: images/demo_incident_rays_V1.png
                xs_mi = mi.Point3f(samples_o + samples_d * 1e-4) # (HW*env_num, 3) [TODO] how to better handle self-intersection?
                incident_rays_mi = mi.Ray3f(xs_mi, ds_mi)

                batch_sensor_dict = {
                    'type': 'batch', 
                    'film': {
                        'type': 'hdrfilm',
                        'width': 10,
                        'height': 1,
                        'pixel_format': 'rgb',
                    },
                    }
                mi_rad_list = []
                print(samples_d.shape[0])
                for _, (d, x) in tqdm(enumerate(zip(samples_d, samples_o))):
                    sensor = get_rad_meter_sensor(x, d, spp=32)
                    # print(x.shape, d.shape, sensor)
                    # batch_sensor_dict[str(_)] = sensor
                    image = mi.render(self.os.mi_scene, sensor=sensor)
                    mi_rad_list.append(image.numpy().flatten())

                # batch_sensor = mi.load_dict(batch_sensor_dict)
                mi_lighting_envmap = np.stack(mi_rad_list).reshape((self.os.H, self.os.W, self.os.lighting_params_dict['env_height'], self.os.lighting_params_dict['env_width'], 3))
                mi_lighting_envmap = mi_lighting_envmap.transpose((0, 1, 4, 2, 3))
                from lib.utils_OR.utils_OR_lighting import downsample_lighting_envmap
                lighting_envmap_vis = np.clip(downsample_lighting_envmap(mi_lighting_envmap, lighting_scale=lighting_scale, downsize_ratio=1)**(1./2.2), 0., 1.)
                plt.figure()
                plt.imshow(lighting_envmap_vis)
                plt.show()


                # scene = mi.load_file('/Users/jerrypiglet/Documents/Projects/dvgomm1/data/indoor_synthetic/kitchen/scene_v3.xml')
                # sensor = get_rad_meter_sensor(samples_o, samples_d, spp=1)
                # image = mi.render(self.os.mi_scene, sensor=sensor)
                # image = mi.render(self.os.mi_scene, sensor=sensor)
                # image = mi.render(self.os.mi_scene, sensor=batch_sensor)
                # mi.util.write_bitmap("tmp.exr", image)
                # import ipdb; ipdb.set_trace()


                # --- [V2] try to expand mi_rays_ret to env_num rays per-point; HOW TO?
                # mi_rays_ret = self.os.mi_rays_ret_list[idx] # (HW,) # [TODO how to get mi_rays_ret_expanded?
                # mi_rays_ret_expanded = mi.SurfaceInteraction3f(
                #     # t=mi.Float(mi_rays_ret.t.numpy()[..., np.newaxis].repeat(env_num, 1).flatten()), 
                #     # time=mi_rays_ret.time, 
                #     wavelengths=mi_rays_ret.wavelengths, 
                #     ps=mi.Point3f(mi_rays_ret.p.numpy()[:, np.newaxis, :].repeat(env_num, 1).reshape(-1, 3)), 
                #     # n=mi.Normal3f(mi_rays_ret.n.numpy()[:, np.newaxis, :].repeat(env_num, 1).reshape(-1, 3)), 
                # )
                # --- [V3] expansive: re-generating camera rays where each pixel has env_num rays: images/demo_incident_rays_V3.png
                # (cam_rays_o, cam_rays_d, _) = self.os.cam_rays_list[idx]
                # cam_rays_o_flatten_expanded = cam_rays_o[:, :, np.newaxis, :].repeat(env_num, 2).reshape(-1, 3)
                # cam_rays_d_flatten_expanded = cam_rays_d[:, :, np.newaxis, :].repeat(env_num, 2).reshape(-1, 3)
                # cam_rays_xs_mi = mi.Point3f(cam_rays_o_flatten_expanded)
                # cam_rays_ds_mi = mi.Vector3f(cam_rays_d_flatten_expanded)
                # cam_rays_mi = mi.Ray3f(cam_rays_xs_mi, cam_rays_ds_mi)
                # mi_rays_ret_expanded = self.os.mi_scene.ray_intersect(cam_rays_mi) # [mitsuba.Scene.ray_intersect] https://mitsuba.readthedocs.io/en/stable/src/api_reference.html?highlight=write_ply#mitsuba.Scene.ray_intersect
                # incident_rays_mi = mi_rays_ret_expanded.spawn_ray(ds_mi)


                ret = self.os.mi_scene.ray_intersect(incident_rays_mi) # [mitsuba.Scene.ray_intersect] https://mitsuba.readthedocs.io/en/stable/src/api_reference.html?highlight=write_ply#mitsuba.Scene.ray_intersect
                rays_o = ret.p.numpy() # destination point
                # >>>> debug to look for self-intersections (e.g. images/demo_self_intersection.png); use spawn rays instead to avoid self-intersection: https://mitsuba.readthedocs.io/en/stable/src/rendering/scripting_renderer.html#Spawning-rays
                # plt.figure()
                # plt.subplot(121)
                # plt.imshow(self.os.im_sdr_list[idx])
                # plt.subplot(122)
                # plt.imshow(np.amin(ret.t.numpy().reshape((self.os.H, self.os.W, env_num)), axis=-1), cmap='jet')
                # plt.colorbar()
                # plt.title('[demo of self-intersection] distance to destination point (min among all rays')
                # plt.show()
                # <<<< debug

            assert rays_d.shape == rays_o.shape # ray d and o to query rad-MLP
            if subsample_rate_pts != 1:
                rays_d = rays_d[::subsample_rate_pts]
                rays_o = rays_o[::subsample_rate_pts]

            # batching to prevent memory overflow
            rad_list = []
            for b_id in tqdm(range(math.ceil(rays_d.shape[0]*1.0/batch_size))):
                b0 = b_id*batch_size
                b1 = min((b_id+1)*batch_size, rays_d.shape[0])
                rays_d_select = rays_d[b0:b1]
                rays_o_select = rays_o[b0:b1]
                rays_o_nerf = self.or2nerf_th(torch.from_numpy(rays_o_select).to(self.device))
                rays_d_nerf = self.or2nerf_th(torch.from_numpy(-rays_d_select).to(self.device)) # nagate to comply with rad-inv coordinates
                if self.dataset_type == 'Indoor':
                    rays_o_nerf = self.process_for_Indoor(rays_o_nerf)
                rgbs = self.model.nerf(rays_o_nerf, rays_d_nerf)['rgb']
                rad = rgbs.detach().cpu().numpy() / self.rad_scale # get back to original scale, without hdr scaling
                # rad = rad * 0. + 10.
                rad_list.append(rad)
            rad_all = np.concatenate(rad_list)
            assert rad_all.shape[0] == rays_o.shape[0]

            lighting_envmap = rad_all.reshape((self.os.H, self.os.W, self.os.lighting_params_dict['env_height'], self.os.lighting_params_dict['env_width'], 3))
            lighting_envmap = lighting_envmap.transpose((0, 1, 4, 2, 3))
            lighting_envmap_list.append(lighting_envmap)

            if if_vis_envmap_2d_plt:
                assert subsample_rate_pts == 1
                assert not if_mask_off_emitters
                plt.figure(figsize=(15, 15))
                plt.subplot(311)
                plt.imshow(self.os.im_sdr_list[idx])
                plt.subplot(312)
                rad_im = lighting_envmap[self.os.H//4, self.os.W//4*3].transpose(1, 2, 0)
                plt.imshow(np.clip((rad_im*lighting_scale)**(1./2.2), 0., 1.))
                plt.subplot(313)
                from lib.utils_OR.utils_OR_lighting import downsample_lighting_envmap
                lighting_envmap_vis = np.clip(downsample_lighting_envmap(lighting_envmap, lighting_scale=lighting_scale)**(1./2.2), 0., 1.)
                plt.imshow(lighting_envmap_vis)
                plt.show()

            if subsample_rate_pts != 1:
                samples_o = samples_o[::subsample_rate_pts]
                samples_d = samples_d[::subsample_rate_pts]

            lighting_fused_dict = {'pts_global_lighting': samples_o, 'axis': samples_d, 'weight': rad_all, 'pts_end': rays_o}
            lighting_fused_list.append(lighting_fused_dict)

        return {
            'lighting_fused_list': lighting_fused_list, 
            'lighting_envmap': lighting_envmap_list, 
            }

    def process_for_Indoor(self, position):
        assert isinstance(position, torch.Tensor)
        # Liwen's model was trained using scene.obj (smaller) instead of scene_v3.xml (bigger), between which there is scaling and translation. self.os.cam_rays_list are acquired from the scene of scene_v3.xml
        scale_m2b = torch.from_numpy(np.array([0.206,0.206,0.206], dtype=np.float32).reshape((1, 3))).to(position.device)
        trans_m2b = torch.from_numpy(np.array([-0.074684,0.23965,-0.30727], dtype=np.float32).reshape((1, 3))).to(position.device)
        position = scale_m2b * position + trans_m2b
        return position
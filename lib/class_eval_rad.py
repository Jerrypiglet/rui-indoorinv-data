import sys
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import numpy as np
import mitsuba as mi

from lib.class_openroomsScene3D import openroomsScene3D
from lib.global_vars import mi_variant_dict

class evaluator_scene_rad():
    '''
    evaluator for trained NeRF (rad-MLP)
    '''
    def __init__(
        self, 
        openrooms_scene: openroomsScene3D, 
        host: str, 
        INV_NERF_ROOT: str, 
        ckpt_path: str, # relative to INV_NERF_ROOT / 'checkpoints'
        dataset_key: str, 
    ):
        sys.path.insert(0, str(INV_NERF_ROOT))
        self.INV_NERF_ROOT = Path(INV_NERF_ROOT)
        ckpt_path = self.INV_NERF_ROOT / 'checkpoints' / ckpt_path

        from train_rad_rui import ModelTrainer, add_model_specific_args
        from argparse import Namespace, ArgumentParser
        from configs.rad_config_openrooms import default_options
        from configs.openrooms_scenes import openrooms_scenes_options

        default_options['dataset'] = openrooms_scenes_options[dataset_key]
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
            if_overfit_train=False, 
            if_seg_obj=False, 
            mitsuba_variant=mi_variant_dict[host], 
            openrooms_scene=openrooms_scene, 
        ).to(self.device)

        checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
        self.model.load_state_dict(checkpoint['state_dict'])
        # print(checkpoint['state_dict'].keys())
        # print(checkpoint['state_dict']['nerf.linears.0.bias'][:2])
        # print(self.model.nerf.linears[0].bias[:2])
        self.model.eval()

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
        render one image by querying rad-MLP: images/demo_eval_radMLP_render.png
        '''
        assert self.model.openrooms_scene.if_has_mitsuba_rays_pts
        (rays_o, rays_d, ray_d_center) = self.model.openrooms_scene.cam_rays_list[frame_id]
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
        rgbs = self.model.nerf(position.to(self.device), rays_d_nerf)['rgb'] # queried d is incoming directions!

        if if_plt:
            plt.figure()
            ax = plt.subplot(121)
            plt.imshow(np.clip(self.model.gamma_func(rgbs).detach().cpu().reshape((self.model.openrooms_scene.H, self.model.openrooms_scene.W, 3)), 0., 1.))
            ax.set_title('rendered image from rad-MLP')
            ax = plt.subplot(122)
            plt.imshow(np.clip(self.model.gamma_func(self.model.openrooms_scene.im_hdr_list[frame_id]), 0., 1.))
            ax.set_title('GT image; could be s.t. scale difference')
            plt.show()

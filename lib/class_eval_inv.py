import sys
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)

from tqdm import tqdm
import mitsuba as mi
import trimesh

from lib.class_openroomsScene3D import openroomsScene3D
from lib.class_mitsubaScene3D import mitsubaScene3D
from lib.global_vars import mi_variant_dict

from lib.utils_OR.utils_OR_emitter import sample_mesh_emitter
from lib.utils_misc import get_list_of_keys, white_blue, blue_text
from lib.utils_OR.utils_OR_lighting import get_lighting_envmap_dirs_global
from lib.utils_mitsuba import get_rad_meter_sensor

class evaluator_scene_inv():
    '''
    evaluator for trained NeRF (inv-MLP)
    '''
    def __init__(
        self, 
        scene_object, 
        host: str, 
        INV_NERF_ROOT: str, 
        ckpt_path: str, # relative to INV_NERF_ROOT / 'checkpoints'
        dataset_key: str, 
        split: str='val', 
        # rad_scale: float=1., 
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

        from train_inv_rui import ModelTrainerInv, add_model_specific_args
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

        self.model = ModelTrainerInv(
            hparams, 
            host=self.host, 
            dataset_key=dataset_key, 
            if_overfit_train=False, 
            if_seg_obj_loss=False, 
            mitsuba_variant=mi_variant_dict[self.host], 
            if_load_baked=True, 
            scene_object=scene_object, 
            spec=spec, 
        ).to(self.device)

        checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
        self.model.load_state_dict({k: v for k, v in checkpoint['state_dict'].items() if ('material.' in k or 'emission_mask.' in k)}, strict=False)
        self.model.eval()

        # self.rad_scale = rad_scale
        self.os = self.model.scene_object[split]

        mi.set_variant(mi_variant_dict[self.host])

    def or2nerf_th(self, x):
        """x:Bxe"""
        ret = torch.tensor([[1,1,-1]], device=x.device)*x
        return ret[:,[0,2,1]]

    def to_d(self, x: np.ndarray):
        if 'mps' in self.device: # Mitsuba RuntimeError: Cannot pack tensors on mps:0
            return x
        return torch.from_numpy(x).to(self.device)

    def sample_shapes(
        self, 
        sample_type: str='emission_mask', 
        shape_params={}, 
        ):
        '''
        sample shape surface for sample_type:
            - 'emission_mask': emission mask (at vectices) from inv-MLP

        args:
        - shape_params
            - radiance_scale: rescale radiance magnitude (because radiance can be large, e.g. 500, 3000)
        '''
        assert self.os.if_loaded_shapes
        assert sample_type in ['emission_mask'] #, 'albedo', 'roughness', 'metallic']

        return_dict = {}
        samples_v_dict = {}

        print(white_blue('Evlauating NeRF'), 'sample_shapes for %d shapes...'%len(self.os.ids_list))

        for shape_index, (vertices, faces, _id) in tqdm(enumerate(zip(self.os.vertices_list, self.os.faces_list, self.os.ids_list))):
            assert np.amin(faces) == 1
            if sample_type == 'emission_mask':
                positions_nerf = self.or2nerf_th(torch.from_numpy(vertices).float().to(self.device)) # convert to NeRF coordinates
                emission_mask = self.model.emission_mask(positions_nerf)
                emission_mask_np = emission_mask.detach().cpu().numpy()
                alpha_np = (1 - torch.exp(-emission_mask.relu())).detach().cpu().numpy()
                # print(torch.amax(emission_mask), torch.amin(emission_mask), torch.amax(alpha_np), torch.amin(alpha_np))
                # rads = rgbs.detach().cpu().numpy() / self.rad_scale * radiance_scale # get back to original scale, without hdr scaling
                samples_v_dict[_id] = ('emission_mask', emission_mask_np * 2.)
                # samples_v_dict[_id] = ('emission_mask', alpha_np)
                # print('>', _id, np.amax(emission_mask_np), np.amin(emission_mask_np), np.amax(alpha_np), np.amin(alpha_np), )

        return_dict.update({'samples_v_dict': samples_v_dict})
        return return_dict

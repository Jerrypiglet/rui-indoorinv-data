import sys
from pathlib import Path
import torch
import torch.nn.functional as NF
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)

from tqdm import tqdm
import mitsuba as mi
import trimesh
from copy import deepcopy
import shutil

from lib.class_openroomsScene3D import openroomsScene3D
from lib.class_mitsubaScene3D import mitsubaScene3D
from lib.global_vars import mi_variant_dict
from lib.utils_inv_nerf import mi2torch
from lib.utils_misc import get_list_of_keys, white_blue, blue_text
# from lib.utils_mitsuba import get_rad_meter_sensor

from lib.utils_OR.utils_OR_emitter import sample_mesh_emitter
from lib.utils_OR.utils_OR_lighting import get_lighting_envmap_dirs_global

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
        rad_scale: float=1., 
        spec: bool=True, 
        if_monosdf: bool=False, 
        monosdf_shape_dict: dict={}, 
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
        mi.set_variant(mi_variant_dict[self.host])

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
            if_monosdf=if_monosdf, 
            monosdf_shape_dict=monosdf_shape_dict, 
        ).to(self.device)

        checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
        self.model.load_state_dict({k: v for k, v in checkpoint['state_dict'].items() if ('material.' in k or 'emission_mask.' in k)}, strict=False)
        self.model.eval()

        self.rad_scale = rad_scale
        self.os = self.model.scene_object[split]


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
        emitter_params={}, 
        if_dump_emitter_mesh: bool=False, # dump emitter mesh and the rest of scene mesh
        if_preserve_emitter_scene_mapping: bool=False, # False: simplify emitter shape; but lost connection to original mesh
        ):
        '''
        sample shape surface for sample_type:
            - 'emission_mask': emission mask (at vectices) from inv-MLP

        args:
        - shape_params
            - radiance_scale: rescale radiance magnitude (because radiance can be large, e.g. 500, 3000)
        '''
        assert self.os.if_loaded_shapes
        assert sample_type in ['emission_mask', 'emission_mask_bin', 'albedo', 'roughness', 'metallic']

        return_dict = {}
        samples_v_dict = {}

        for shape_index, (vertices, faces, _id) in tqdm(enumerate(zip(self.os.vertices_list, self.os.faces_list, self.os.shape_ids_list))):
            print(white_blue('Evaluating inv-MLP for [%s]'%sample_type), 'sample_shapes for %d/%d shapes (%d v, %d f) ...'%(shape_index, len(self.os.shape_ids_list), vertices.shape[0], faces.shape[0]))
            assert np.amin(faces) == 1 # [!!!] faces is 1-based!
            positions_nerf = self.or2nerf_th(torch.from_numpy(vertices).float().to(self.device)) # convert to NeRF coordinates
            if sample_type in ['emission_mask', 'emission_mask_bin']:
                emission_mask = self.model.emission_mask(positions_nerf)
                emission_mask_np = emission_mask.detach().cpu().numpy()
                if sample_type == 'emission_mask':
                    samples_v_dict[_id] = ('emission_mask', emission_mask_np)
                elif sample_type == 'emission_mask_bin':
                    alpha_np = (1 - torch.exp(-emission_mask.relu())).detach().cpu().numpy()
                    emission_mask_bin = alpha_np > 0.3
                    samples_v_dict[_id] = ('emission_mask_bin', (emission_mask_bin).astype(np.float32))
                    
                    if if_dump_emitter_mesh:
                        emitters_dump_root = Path('test_files/%s/emitters'%self.os.shape_name_full)
                        if emitters_dump_root.exists(): shutil.rmtree(str(emitters_dump_root))
                        emitters_dump_root.mkdir(exist_ok=False, parents=True)

                        emitters_faces_mask = np.all(emission_mask_bin.reshape(-1)[faces-1], axis=1) # (N_total_faces,), bool
                        emitters_faces_idxes_0 = np.where(emitters_faces_mask)[0] # (N_total_emitters_faces,), int, 0-based: [0, ..., N_total_faces-1]
                        faces_emitters = faces[emitters_faces_mask] # (faces_emitters, 3), int, containing 1-based vertex indexes
                        # emitters_trimesh = trimesh.Trimesh(vertices=vertices, faces=faces_emitters-1)
                        # emitters_trimesh_path = 'test_files/tmp_emitters.obj'
                        # emitters_trimesh.export(emitters_trimesh_path)
                        # print(blue_text('Exported emitter mesh to %s (%d v, %d f)'%(emitters_trimesh_path, vertices.shape[0], faces_emitters.shape[0])))
                        # faces_non_emitters = faces[~emitters_faces_mask]
                        emitters_trimesh = trimesh.Trimesh(vertices=vertices, faces=faces_emitters-1, process=not if_preserve_emitter_scene_mapping)
                        # if not if_preserve_emitter_scene_mapping:
                        #     emitters_trimesh.remove_unreferenced_vertices() # [!!!] uncomment to simplify emitter shape; but lost connection to original mesh
                        cc = trimesh.graph.connected_components(emitters_trimesh.face_adjacency, min_len=3)

                        emitter_idx = 0
                        m_list = []
                        emitter_dict_list = []
                        for _, c_ in enumerate(cc):
                            mask = np.zeros(len(emitters_trimesh.faces), dtype=bool)
                            mask[c_] = True
                            m_ = deepcopy(emitters_trimesh)
                            m_.update_faces(mask)
                            if not if_preserve_emitter_scene_mapping:
                                m_.remove_unreferenced_vertices() # [!!!] uncomment to simplify emitter shape; but lost connection to original mesh
                            if m_.area < 0.1:
                                continue
                            trimesh.repair.fill_holes(m_)
                            if if_preserve_emitter_scene_mapping:
                                # print(m_, m_.area)
                                assert m_.vertices.shape == vertices.shape
                                assert np.allclose(np.array(m_.vertices), vertices)
                            emitter_trimesh_path = emitters_dump_root / ('emitter_%d.obj'%emitter_idx)
                            m_.export(str(emitter_trimesh_path))
                            m_list.append(m_)
                            print(blue_text('Exported emitter mesh to %s (%d v, %d f, %.2f area)'%(str(emitter_trimesh_path), m_.vertices.shape[0], m_.faces.shape[0], m_.area)))
                            emitter_idx += 1

                            '''
                            [validation check] getting face mask of single emitter, over full scene faces; should yield the same shape
                            '''
                            # assert if_preserve_emitter_scene_mapping, 'otherwise cannot find mapping between faces of single emitter to original full scene mesh'
                            single_emitter_faces_idxes_0 = np.where(mask)[0] # (N_single_emitter_faces,), int, 0-based: [0, ..., N_total_emitters_faces-1]
                            single_emitter_faces_idxes_in_full_shape_face_idxes_0 = emitters_faces_idxes_0[single_emitter_faces_idxes_0] # single emitter faces idxes among full scene face idxes
                            assert single_emitter_faces_idxes_in_full_shape_face_idxes_0.shape[0] == m_.faces.shape[0]
                            single_emitter_trimesh_re = trimesh.Trimesh(vertices=vertices, faces=faces[single_emitter_faces_idxes_in_full_shape_face_idxes_0]-1, process=not if_preserve_emitter_scene_mapping)
                            single_emitter_trimesh_re.export(str(emitter_trimesh_path).replace('.obj', '_re.obj'))
                            
                            '''
                            get emitter dict for inv-nerf path tracer
                            '''
                            is_emitter_mask = torch.zeros(faces.shape[0], dtype=torch.bool)
                            is_emitter_mask[single_emitter_faces_idxes_in_full_shape_face_idxes_0] = True

                            m_vertices_th = torch.from_numpy(m_.vertices)
                            emitter_vertices = mi2torch(m_vertices_th)[m_.faces].float() # (N_emitter_faces, 3, 3)
                            emitter_area = torch.cross(emitter_vertices[:,1]-emitter_vertices[:,0], emitter_vertices[:,2]-emitter_vertices[:,0],-1)
                            emitter_normal = NF.normalize(emitter_area,dim=-1)
                            emitter_area = emitter_area.norm(dim=-1)/2.0
                            emitter_dict = {
                                'is_emitter': is_emitter_mask, # (N_total_faces,), torch.bool
                                'emitter_vertices': emitter_vertices, # (N_emitter_faces, 3, 3), torch.float32
                                'emitter_area': emitter_area, # (N_emitter_faces,), torch.float32
                                'emitter_normal': emitter_normal, # (N_emitter_faces, 3), torch.float32
                                # 'emitter_radiance': emitter_radiance, # (N_emitter_faces, 3), torch.float32
                            }
                            emitter_dict_list.append(emitter_dict)

                        # emitters_trimesh_path = 'test_files/tmp_non_emitters.obj'
                        # emitters_trimesh.export(emitters_trimesh_path)
                        # print(blue_text('Exported non-emitter mesh to %s (%d v, %d f)'%(emitters_trimesh_path, vertices.shape[0], faces_non_emitters.shape[0])))
                        # emitters_tri_mask_path = 'test_files/tmp_emitters_faces_mask.npy'
                        # np.save(emitters_tri_mask_path, face_mask)
                        # print(blue_text('Dumped [emitter mask] to %s'%(emitters_tri_mask_path)))

                        return_dict.update(
                            {'masked_emitters_trimesh_list': m_list}
                        )

                        emitter_sample_dict = self.sample_emitters(
                            m_list, 
                            emitter_params, 
                            )
                        return_dict.update(emitter_sample_dict)

                        for emitter_idx, (m_, emitter_dict) in enumerate(zip(m_list, emitter_dict_list)):
                            emitter_dict_path = emitters_dump_root / ('emitter_%d.pth'%emitter_idx)
                            intensity_median = emitter_sample_dict['intensity_median_list'][emitter_idx]
                            '''
                            in `mi` space;
                            need to convert everything to inv-MLP (`torch`) space
                            '''
                            emitter_dict.update({
                                'emitter_radiance': torch.from_numpy(intensity_median).reshape(1, 3).expand_as(emitter_dict['emitter_normal']), # (N_emitter_faces, 3), torch.float32
                                })
                            torch.save(emitter_dict, str(emitter_dict_path))
                            print(_, m_.vertices.shape, m_.faces.shape, intensity_median)
                            print(blue_text('Dumped [emitter dict] to %s'%(emitter_dict_path)))

            elif sample_type in ['albedo', 'metallic', 'roughness']:
                mat = self.model.material(positions_nerf).sigmoid()
                albedo_np, metallic_np, roughness_np = mat[...,:3].detach().cpu().numpy(), mat[...,3:4].detach().cpu().numpy(), mat[...,4:5].detach().cpu().numpy()
                samples_v_dict[_id] = {'albedo': ('albedo', albedo_np), 'metallic': ('metallic', metallic_np), 'roughness': ('roughness', roughness_np)}[sample_type]

        return_dict.update({'samples_v_dict': samples_v_dict})
        return return_dict

    def sample_emitters(
        self, 
        masked_emitters_trimesh_list: list, 
        emitter_params={}, 
        ):
        '''
        sample emitter surface radiance from rad-MLP: images/demo_emitter_o3d_sampling.png

        args:
        - masked_emitters_trimesh_list
            - list of trimesh objects returned from self.sample_shapes
        - emitter_params
            - radiance_scale: rescale radiance magnitude (because radiance can be large, e.g. 500, 3000)
        '''
        max_plate = emitter_params.get('max_plate', 64)
        radiance_scale = emitter_params.get('radiance_scale', 1.)
        
        emitter_dict = {
            'lamp': 
                [{'vertices': np.array(_.vertices), 'faces': np.array(_.faces)+1} for _ in masked_emitters_trimesh_list], 
            'window': [], 
            } # [TODO] support other emitters!!
        emitter_rays_list = []
        intensity_median_list = []

        # for emitter_type_index in emitter_type_index_list:
            # (emitter_type, _) = emitter_type_index
        for emitter_idx in range(len(emitter_dict['lamp'])): # [TODO] support other emitters!!
            lpts_dict = sample_mesh_emitter('lamp', emitter_idx=emitter_idx, emitter_dict=emitter_dict, max_plate=max_plate, if_dense_sample=True)
            rays_o_nerf = self.or2nerf_th(torch.from_numpy(lpts_dict['lpts']).float().to(self.device)) # convert to NeRF coordinates
            rays_d_nerf = self.or2nerf_th(torch.from_numpy(-lpts_dict['lpts_normal']).float().to(self.device)) # convert to NeRF coordinates
            rgbs = self.model.nerf(rays_o_nerf, rays_d_nerf)['rgb'] # queried d is incoming directions!
            intensity = rgbs.detach().cpu().numpy() / self.rad_scale # get back to original scale, without hdr scaling
            # intensity = intensity * 0. + 5.
            intensity_median = np.median(intensity, 0)
            print(white_blue('EST intensity (median)'), intensity_median, np.linalg.norm(intensity_median))
            emitter_ray_dict = {
                'v': lpts_dict['lpts'], 
                'd': lpts_dict['lpts_normal'] / (np.linalg.norm(lpts_dict['lpts_normal'], axis=-1, keepdims=True)+1e-5), 
                'l': np.linalg.norm(intensity, axis=-1, keepdims=True) * radiance_scale, 
                # 'l_vis_scale': 1. / self.rad_scale, # bring back to original OR scale
                }
            emitter_rays_list.append(emitter_ray_dict)
            intensity_median_list.append(intensity_median)

        return {'emitter_rays_list': emitter_rays_list, 'intensity_median_list': intensity_median_list}


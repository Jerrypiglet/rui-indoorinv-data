import sys
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import numpy as np
import trimesh
np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)

from tqdm import tqdm
import mitsuba as mi
import torch

from lib.class_openroomsScene3D import openroomsScene3D
from lib.class_mitsubaScene3D import mitsubaScene3D
from lib.global_vars import mi_variant_dict

from lib.utils_OR.utils_OR_emitter import sample_mesh_emitter
from lib.utils_misc import get_list_of_keys, white_blue, blue_text, red, yellow
from lib.utils_OR.utils_OR_lighting import get_lighting_envmap_dirs_global
from lib.utils_mitsuba import get_rad_meter_sensor

class evaluator_scene_scene():
    '''
    evaluator for 
    '''
    def __init__(
        self, 
        scene_object, 
        host: str, 
    ):
        self.host = host
        self.device = {
            'apple': 'mps', 
            'mm1': 'cuda', 
        }[self.host]
        mi.set_variant(mi_variant_dict[self.host])

        self.os = scene_object

    def sample_shapes(
        self, 
        sample_type: str='vis_count', 
        hdr_radiance_scale: float=1., 
        shape_params={}, 
        if_show: bool=True,
        ):
        '''
        sample shape surface for sample_type:
            - 'vis_count': visibility under camera views; follow implementation in class visualizer_scene_3D_o3d -> get_pcd_color_fused_geo(): elif pcd_color_mode == 'mi_visibility_emitter0'...
            - 't': distance to camera 0
            - 'rgb': back-projected color from cameras

        args:
        - shape_params
        '''
        assert self.os.if_loaded_shapes, 'Shape not loaded! Required to evaluate properties of vertexs or faces.'
        assert sample_type in ['vis_count', 't', 'rgb_hdr', 'rgb_sdr', 'face_normal']

        return_dict = {}
        samples_v_dict = {}

        print(white_blue('Evaluating scene for [%s]'%sample_type), 'sample_shapes for %d shapes...'%len(self.os.ids_list))
        if sample_type in ['vis_count', 'rgb_hdr', 'rgb_sdr']:
            '''
            get visibility frustum normals and centers
            '''
            max_vis_count = 0
            vis_frustum_normals_list = []
            vis_frustum_centers_list = []
            for frame_idx, (rays_o, rays_d, _) in enumerate(self.os.cam_rays_list): # [1] make sure 'poses' in scene_obj->modality_list if not sample poses ad-hoc; [2] Set scene_obj->mi_params_dict={'if_sample_rays_pts': True

                normal_up = np.cross(rays_d[0][0], rays_d[0][-1])
                normal_down = np.cross(rays_d[-1][-1], rays_d[-1][0])
                normal_left = np.cross(rays_d[-1][0], rays_d[0][0])
                normal_right = np.cross(rays_d[0][-1], rays_d[-1][-1])
                normals = np.stack((normal_up, normal_down, normal_left, normal_right), axis=-1)
                vis_frustum_normals_list.append(normals)
                vis_frustum_centers_list.append(rays_o[0, 0].reshape(1, 3))

        for shape_idx, (vertices, faces, _id) in tqdm(enumerate(zip(self.os.vertices_list, self.os.faces_list, self.os.ids_list))):
            assert np.amin(faces) == 1
            if sample_type in ['vis_count', 'rgb_hdr', 'rgb_sdr']:
                assert self.os.if_has_poses
                assert self.os.if_has_mitsuba_scene # scene_obj->modality_list=['mi'

                if sample_type == 'vis_count':
                    vis_count = np.zeros((vertices.shape[0]), dtype=np.int64)
                if sample_type in ['rgb_hdr', 'rgb_sdr']:
                    # rgb_hdr_list = [[]] * vertices.shape[0]
                    rgb_sum = np.zeros((vertices.shape[0], 3), dtype=np.float32)
                    rgb_count = np.zeros((vertices.shape[0]), dtype=np.int64)

                print('[Shape %d] Evaluating %d frames...'%(shape_idx, self.os.frame_num))
                for frame_idx, (origin, _, _) in tqdm(enumerate(self.os.origin_lookatvector_up_list)):
                    visibility_frustum = np.all(((vertices-vis_frustum_centers_list[frame_idx]) @ vis_frustum_normals_list[frame_idx]) > 0, axis=1)
                    # visibility = visibility_frustum

                    origin = np.tile(np.array(origin).reshape((1, 3)), (vertices.shape[0], 1))
                    ds_ = vertices - origin
                    ds = ds_ / (np.linalg.norm(ds_, axis=1, keepdims=1)+1e-6)
                    ds = np.array(ds).astype(np.float32)

                    xs = np.array(origin).astype(np.float32)
                    xs_mi = mi.Point3f(xs)
                    ds_mi = mi.Vector3f(ds)
                    # ray origin, direction, t_max
                    rays_mi = mi.Ray3f(xs_mi, ds_mi)
                    ret = self.os.mi_scene.ray_intersect(rays_mi) # https://mitsuba.readthedocs.io/en/stable/src/api_reference.html?highlight=write_ply#mitsuba.Scene.ray_intersect
                    # returned structure contains intersection location, nomral, ray step, ...
                    ts = ret.t.numpy()
                    visibility = np.logical_not(ts < (np.linalg.norm(ds_, axis=1, keepdims=False)))
                    visibility = np.logical_and(visibility, visibility_frustum) # (N_vertices,)
                    
                    if sample_type == 'vis_count':
                        vis_count += visibility
                    if sample_type in ['rgb_hdr', 'rgb_sdr']:
                        x_world = ret.p.numpy()[visibility]
                        _R, _t = self.os.pose_list[frame_idx][:3, :3], self.os.pose_list[frame_idx][:3, 3:4]
                        x_cam = (x_world - _t.T) @ _R
                        uv_cam_homo = (self.os.K_list[frame_idx] @ x_cam.T).T
                        uv_cam = uv_cam_homo[:, :2] / (uv_cam_homo[:, 2:3]+1e-6) # (N_valid_vertices, 2)
                        uv_valid_mask = np.logical_and(np.logical_and(uv_cam[:, 0] >= 0, uv_cam[:, 0] < self.os.W), np.logical_and(uv_cam[:, 1] >= 0, uv_cam[:, 1] < self.os.H))

                        uv_cam_normalized = (uv_cam / np.array([[self.os.W, self.os.H]], dtype=np.float32)) * 2. - 1.
                        grid = torch.from_numpy(uv_cam_normalized).unsqueeze(0).unsqueeze(2).float()
                        if sample_type == 'rgb_hdr':
                            im_ = self.os.im_hdr_list[frame_idx]
                        elif sample_type == 'rgb_sdr':
                            im_ = self.os.im_sdr_list[frame_idx]
                        sampled_rgb_th = torch.nn.functional.grid_sample(torch.from_numpy(im_).permute(2, 0, 1).unsqueeze(0).float(), grid, align_corners=True)
                        sampled_rgb_valid = (sampled_rgb_th.squeeze().numpy().T)[uv_valid_mask]
                        valid_vertices_idx = np.where(visibility)[0][uv_valid_mask]
                        assert valid_vertices_idx.shape[0] == sampled_rgb_valid.shape[0]
                        # for _idx, vertex_idx in tqdm(enumerate(valid_vertices_idx)):
                        #     rgb_list[vertex_idx].append(sampled_rgb_valid[_idx])
                        # [rgb_list[vertex_idx].append(sampled_rgb_valid[_idx]) for _idx, vertex_idx in tqdm(enumerate(valid_vertices_idx))]
                        rgb_sum[valid_vertices_idx] += sampled_rgb_valid
                        rgb_count[valid_vertices_idx] += 1

                if sample_type == 'vis_count':
                    samples_v_dict[_id] = ('vis_count', vis_count)
                    max_vis_count = max(max_vis_count, np.amax(vis_count))
                if sample_type == 'rgb_hdr':
                    # rgb_hdr = np.zeros((vertices.shape[0], 3), dtype=np.float32)
                    # for _idx, rgb_ in tqdm(enumerate(rgb_list)):
                    #     if len(rgb_) > 0:
                    #         rgb_hdr[_idx] = np.mean(rgb_, axis=0)
                    rgb_hdr = rgb_sum / (rgb_count.reshape((-1, 1))+1e-6) * hdr_radiance_scale
                    rgb_hdr[rgb_count==0] = np.array([[1., 1., 0.]], dtype=np.float32) # yellow for unordered vertices
                    samples_v_dict[_id] = ('rgb_hdr', rgb_hdr)
                if sample_type == 'rgb_sdr':
                    rgb_sdr = rgb_sum / (rgb_count.reshape((-1, 1))+1e-6)
                    rgb_sdr[rgb_count==0] = np.array([[1., 1., 0.]], dtype=np.float32) # yellow for unordered vertices
                    samples_v_dict[_id] = ('rgb_sdr', rgb_sdr)

            elif sample_type in ['face_normal']:
                '''
                Compute face normals of all visible faces, and flip inwards normals opposite to cameras
                
                images/demo_eval_scene_shapes-face_normal.png
                '''
                assert self.os.if_has_poses
                assert self.os.if_has_mitsuba_scene # scene_obj->modality_list=['mi'

                if sample_type == 'face_normal':
                    vertex_vis_count = np.zeros((vertices.shape[0]), dtype=np.int64)
                    vertex_normals = np.zeros((vertices.shape[0], 3), dtype=np.float32)
                    face_normals_mask = np.ones((faces.shape[0]), dtype=np.uint8).astype(bool)
                    normal_rays_dict = {'v': [], 'd': [], 'l': []}
                    
                print('[Shape %d] Evaluating %d frames...'%(shape_idx, len(self.os.frame_id_list)))
                face_normals = trimesh.Trimesh(faces=faces-1, vertices=vertices, process=False, maintain_order=True).face_normals # (N_faces, 3)
                for frame_idx, ret in tqdm(enumerate(self.os.mi_rays_ret_list)):
                    if sample_type == 'face_normal':
                        assert len(self.os.faces_list) == 1, 'only works for single shape scene, so that faces in Mitsuba is the same as shapes loaded from .obj'
                        assert len(self.os.mi_scene.shapes()) == 1, 'only works for single shape scene, so that faces in Mitsuba is the same as shapes loaded from .obj'
                        assert faces.shape[0] == self.os.mi_scene.shapes()[0].face_count(), 'should load MI scene from single shape; double check if you [1] load entire scene as a shape [2] preserve the shape when loading with trimesh.load_mesh'
                        face_ids = ret.prim_index.numpy() # (HW,), 0-based
                        if np.amax(face_ids) >= faces.shape[0]:
                            print(frame_idx, np.amax(face_ids), faces.shape[0])
                            print(red('[ERROR] somehow every large face id in ray-mi intersection; skipped frame %d'%frame_idx))
                            continue
                        assert np.amax(face_ids) < faces.shape[0]
                        # _face_normals = ret.n.numpy() # (HW, 3) # [???] Mitsuba auto flip normals for single sides shapes?
                        _face_normals = face_normals[face_ids] # (HW, 3)
                        
                        _, rays_d, _ = self.os.cam_rays_list[frame_idx]
                        rays_d = -rays_d.reshape((-1, 3))
                        
                        _face_normals_mask = np.sum(_face_normals * rays_d, axis=1) > 0
                        face_normals_mask[face_ids] = np.logical_and(_face_normals_mask, face_normals_mask[face_ids])
                        # [DEBUG] output face normals: blue; inwards: red
                        # face_normals[_face_normals_mask] = np.array([[0., 0., 1.]])
                        # face_normals[~_face_normals_mask] = np.array([[1., 0., 0.]])
                        
                        vertex_ids = np.array(faces[face_ids])-1 # (N, 3), 0-based
                        _face_centers = vertices[vertex_ids].mean(axis=1) # (N, 3)
                        vertex_normals[vertex_ids] += np.repeat(np.expand_dims(_face_normals, 1), 3, axis=1)
                        # vertex_normals[vertex_ids] += np.repeat(np.expand_dims(face_normals, -1), 3, axis=2)
                        vertex_vis_count[vertex_ids.reshape(-1)] += 1

                        normal_rays_dict['v'].append(np.array(_face_centers))
                        normal_rays_dict['d'].append(np.array(_face_normals))
                        normal_rays_dict['l'].append(np.ones(_face_normals.shape[0], dtype=np.float32)*0.2)
                        
                vertex_normals = vertex_normals / (vertex_vis_count.reshape((-1, 1))+1e-6)

                samples_v_dict[_id] = ('vertex_normal', vertex_normals)
                return_dict['face_normals_flipped_mask'] = ~face_normals_mask
                return_dict['normal_rays_list'] = {'v': np.concatenate(normal_rays_dict['v'], axis=0), 'd': np.concatenate(normal_rays_dict['d'], axis=0), 'l': np.concatenate(normal_rays_dict['l'], axis=0)}
                
                # DEBUG:
                # samples_v_dict[_id] = ('vis_count', (vertex_vis_count, np.amax(vertex_vis_count)))
                
            elif sample_type == 't':
                assert self.os.if_has_poses
                assert self.os.if_has_mitsuba_scene
                assert len(self.os.origin_lookatvector_up_list) == 1
                (origin, _, _) = self.os.origin_lookatvector_up_list[0]

                origin = np.tile(np.array(origin).reshape((1, 3)), (vertices.shape[0], 1))
                ds = vertices - origin
                ds = ds / (np.linalg.norm(ds, axis=1, keepdims=1)+1e-6)

                xs = origin
                xs_mi = mi.Point3f(xs)
                ds_mi = mi.Vector3f(ds)
                # ray origin, direction, t_max
                rays_mi = mi.Ray3f(xs_mi, ds_mi)
                ret = self.os.mi_scene.ray_intersect(rays_mi) # https://mitsuba.readthedocs.io/en/stable/src/api_reference.html?highlight=write_ply#mitsuba.Scene.ray_intersect
                # returned structure contains intersection location, nomral, ray step, ...
                t = ret.t.numpy()

                samples_v_dict[_id] = ('t', (t, np.amax(t)))
            else:
                print(red('sample_type %s not implemented'%sample_type))
                raise NotImplementedError
        
        if sample_type == 'vis_count':
            assert max_vis_count > 0
            for _id, v in samples_v_dict.items():
                samples_v_dict[_id] = ('vis_count', (samples_v_dict[_id][1], max_vis_count))

        return_dict.update({'samples_v_dict': samples_v_dict})
        return return_dict

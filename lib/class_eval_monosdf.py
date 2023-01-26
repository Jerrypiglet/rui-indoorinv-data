import sys
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import math
import mitsuba as mi
import trimesh

from lib.class_openroomsScene3D import openroomsScene3D
from lib.class_mitsubaScene3D import mitsubaScene3D
from lib.global_vars import mi_variant_dict

from lib.utils_OR.utils_OR_emitter import sample_mesh_emitter
from lib.utils_misc import get_list_of_keys, white_blue, blue_text
from lib.utils_OR.utils_OR_lighting import get_lighting_envmap_dirs_global

class evaluator_scene_monosdf():
    '''
    evaluator for trained MonoSDF model
    '''
    def __init__(
        self, 
        scene_object, 
        host: str, 
        MONOSDF_ROOT: str, 
        ckpt_path: str, # relative to MONOSDF_ROOT / 'exps'
        conf_path: str, # relative to MONOSDF_ROOT / 'exps'
        # dataset_key: str, 
        # split: str='', 
        rad_scale: float=1., 
        # spec: bool=True, 
    ):
        sys.path.insert(0, str(Path(MONOSDF_ROOT) / 'code'))
        self.MONOSDF_ROOT = Path(MONOSDF_ROOT)

        from pyhocon import ConfigFactory
        import utils.general as utils_monosdf

        assert rad_scale == 1.
        ckpt_path = self.MONOSDF_ROOT / 'exps' / ckpt_path
        conf_path = self.MONOSDF_ROOT / 'exps' / conf_path

        self.conf = ConfigFactory.parse_file(str(conf_path))
        self.dataset_conf = self.conf.get_config('dataset')
        assert self.dataset_conf['if_hdr']
        self.dataset_conf['num_views'] = -1
        self.plot_conf = self.conf.get_config('plot')

        # eval_dataset = utils_monosdf.get_class(conf.get_string('train.dataset_class'))(**self.dataset_conf)
        if_pixel_train = self.conf.get_config('dataset').get('if_pixel', False)
        if_hdr = self.conf.get_config('dataset').get('if_hdr', False)

        conf_model = self.conf.get_config('model')
        self.model = utils_monosdf.get_class(self.conf.get_string('train.model_class'))(conf=conf_model, if_hdr=if_hdr)
        self.host = host
        self.device = {
            'apple': 'mps', 
            'mm1': 'cuda', 
            'qc': '', 
        }[self.host]
        self.model.to(self.device)

        saved_model_state = torch.load(ckpt_path)
        # deal with multi-gpu training model
        if list(saved_model_state["model_state_dict"].keys())[0].startswith("module."):
            saved_model_state["model_state_dict"] = {k[7:]: v for k, v in saved_model_state["model_state_dict"].items()}
        self.model.load_state_dict(saved_model_state["model_state_dict"], strict=True)

        self.model.eval()

        self.rad_scale = rad_scale
        self.os = scene_object
        assert self.os.H==320 and self.os.W==640, 'MonoSDF was trained with image res of 320x640!'

    def export_mesh(self, mesh_path: str='test_files/monosdf_mesh.ply', resolution: int=1024):
        from utils.plots import get_surface_sliding
        # exporting mesh from SDF
        print(white_blue('-> Exporting MonoSDF mesh to %s...'%mesh_path))
        with torch.no_grad():
            mesh = get_surface_sliding(
                path='', 
                epoch='', 
                sdf=lambda x: self.model.implicit_network(x)[:, 0], 
                resolution=resolution, 
                grid_boundary=self.conf.get_list('plot.grid_boundary'), 
                level=0,  
                return_mesh=True,  
                )
        mesh.export(mesh_path, 'ply')
        print(blue_text('-> Exported.'))

    def render_im(self, frame_id: int, offset_in_scan: int=0, split_n_pixels: int=1024, if_plt: bool=False):
        import utils.general as utils_monosdf
        from utils import rend_util
        import utils.plots as plots

        frame_id_monosdf = frame_id + offset_in_scan # `indice` in MonoSDF (which is conditioning on per-image features; so that you would want to input the real image id in monosdf dataset)

        # Load cameras following convert_mitsubaScene3D_to_monosdf.py and {monosdf}/code/datasets/scene_dataset.py
        K = self.os.K # (3, 3)
        K = np.hstack((K, np.array([0., 0., 0.], dtype=np.float32).reshape((3, 1))))
        K = np.vstack((K, np.array([0., 0., 0., 1.], dtype=np.float32).reshape((1, 4))))

        pose = self.os.pose_list[frame_id] # (3, 4)
        pose = np.vstack((pose, np.array([0., 0., 0., 1.], dtype=np.float32).reshape((1, 4))))

        world_mat = K @ np.linalg.inv(pose)
        scale_mat = self.os.monosdf_scale_mat

        P = world_mat @ scale_mat
        P = P[:3, :4]
        intrinsics, pose = rend_util.load_K_Rt_from_P(None, P) # (4, 4), (4, 4)
        intrinsics = torch.from_numpy(intrinsics).float().to(self.device).unsqueeze(0) # (1, 4, 4)
        pose = torch.from_numpy(pose).float().to(self.device).unsqueeze(0) # (1, 4, 4)

        # gather input dict
        uv = np.mgrid[0:self.os.H, 0:self.os.W].astype(np.int32)
        uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float().to(self.device)
        uv = uv.reshape(2, -1).transpose(1, 0).unsqueeze(0) # (1, HW, 2)

        model_input = dict(intrinsics=intrinsics, pose=pose, uv=uv)
        indices = torch.tensor(frame_id_monosdf, dtype=torch.int64).reshape(1,)

        # Forward network
        total_pixels_im = self.os.H * self.os.W
        split = utils_monosdf.split_input(model_input, total_pixels_im, n_pixels=split_n_pixels)
        res = []
        for s in tqdm(split):
            out = self.model(s, indices)
            d = {'rgb_values': out['rgb_values'].detach(),
                'normal_map': out['normal_map'].detach(),
                'depth_values': out['depth_values'].detach()}
            if 'rgb_un_values' in out:
                d['rgb_un_values'] = out['rgb_un_values'].detach()
            res.append(d)

        model_outputs = utils_monosdf.merge_output(res, total_pixels_im, 1)
        ground_truth = {
            'rgb': torch.from_numpy(self.os.im_hdr_list[frame_id].reshape(1, -1, 3)).float(), 
            'normal': torch.from_numpy(self.os.mi_normal_opencv_list[frame_id].reshape(1, -1, 3)).float(), 
            'depth': torch.from_numpy(self.os.mi_depth_list[frame_id].reshape(1, -1, 1)).float(), 
        }
        plot_data = self.get_plot_data(model_input, model_outputs, model_input['pose'], ground_truth['rgb'], ground_truth['normal'], ground_truth['depth'])

        merge_path = plots.plot(self.model.implicit_network,
                indices,
                plot_data,
                'test_files',
                0,
                [self.os.H, self.os.W],
                if_hdr=True, 
                if_tensorboard=False, 
                **self.plot_conf
                )

        print(white_blue('-> Exported rendering result to %s...'%merge_path))

    def get_plot_data(self, model_input, model_outputs, pose, rgb_gt, normal_gt, depth_gt):
        from model.loss import compute_scale_and_shift
        batch_size, num_samples, _ = rgb_gt.shape

        rgb_eval = model_outputs['rgb_values'].reshape(batch_size, num_samples, 3)
        normal_map = model_outputs['normal_map'].reshape(batch_size, num_samples, 3)
        normal_map = (normal_map + 1.) / 2.
      
        depth_map = model_outputs['depth_values'].reshape(batch_size, num_samples)
        depth_gt = depth_gt.to(depth_map.device)
        scale, shift = compute_scale_and_shift(depth_map[..., None], depth_gt, depth_gt > 0.)
        depth_map = depth_map * scale + shift
        
        # save point cloud
        # depth = depth_map.reshape(1, 1, self.img_res[0], self.img_res[1])
        # pred_points = self.get_point_cloud(depth, model_input, model_outputs)
        # gt_depth = depth_gt.reshape(1, 1, self.img_res[0], self.img_res[1])
        # gt_points = self.get_point_cloud(gt_depth, model_input, model_outputs)
        
        plot_data = {
            'rgb_gt': rgb_gt,
            'normal_gt': (normal_gt + 1.)/ 2.,
            'depth_gt': depth_gt,
            'pose': pose,
            'rgb_eval': rgb_eval,
            'normal_map': normal_map,
            'depth_map': depth_map,
            # "pred_points": pred_points,
            # "gt_points": gt_points,
        }

        return plot_data

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

    def render_im_(self, frame_id: int, if_plt: bool=False):
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

        args:
        - emitter_params
            - radiance_scale: rescale radiance magnitude (because radiance can be large, e.g. 500, 3000)
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
                # if self.dataset_type == 'Indoor':
                #     rays_o_nerf = self.process_for_Indoor(rays_o_nerf)
                rgbs = self.model.nerf(rays_o_nerf, rays_d_nerf)['rgb'] # queried d is incoming directions!
                intensity = rgbs.detach().cpu().numpy() / self.rad_scale # get back to original scale, without hdr scaling
                # intensity = intensity * 0. + 5.
                print(white_blue('EST intensity'), np.linalg.norm(intensity, axis=-1))
                emitter_rays_list.append({
                    'v': lpts_dict['lpts'], 
                    'd': lpts_dict['lpts_normal'] / (np.linalg.norm(lpts_dict['lpts_normal'], axis=-1, keepdims=True)+1e-5), 
                    'l': np.linalg.norm(intensity, axis=-1, keepdims=True) * radiance_scale
                    })

                # emitter_rays = o3d.geometry.LineSet()
                # emitter_rays.points = o3d.utility.Vector3dVector(np.vstack((lpts_dict['lpts'], lpts_end)))
                # emitter_rays.colors = o3d.utility.Vector3dVector([[0.3, 0.3, 0.3]]*lpts_dict['lpts'].shape[0])
                # emitter_rays.lines = o3d.utility.Vector2iVector([[_, _+lpts_dict['lpts'].shape[0]] for _ in range(lpts_dict['lpts'].shape[0])])
                # geometry_list.append([emitter_rays, 'emitter_rays'])

        return {'emitter_rays_list': emitter_rays_list}

    def sample_shapes(
        self, 
        sample_type: str='rad', 
        shape_params={}, 
        ):
        '''
        sample shape surface for sample_type:
            - 'rad': radiance (at vectices along vertice normals) from rad-MLP

        args:
        - shape_params
            - radiance_scale: rescale radiance magnitude (because radiance can be large, e.g. 500, 3000)
        '''
        radiance_scale = shape_params.get('radiance_scale', 1.)
        assert self.os.if_loaded_shapes
        assert sample_type in ['rad'] #, 'incident-rad']

        return_dict = {}
        samples_v_dict = {}
        if sample_type == 'rad':
            shape_rays_dict = {}

        print(white_blue('Evlauating rad-MLP for [%s]'%sample_type), 'sample_shapes for %d shapes...'%len(self.os.ids_list))

        for shape_index, (vertices, faces, _id) in tqdm(enumerate(zip(self.os.vertices_list, self.os.faces_list, self.os.ids_list))):
            assert np.amin(faces) == 1
            shape_tri_mesh = trimesh.Trimesh(vertices=vertices, faces=faces-1) # [IMPORTANT] faces-1 because Trimesh faces are 0-based

            if sample_type == 'rad':
                vertex_normals = shape_tri_mesh.vertex_normals # already normalized

                if vertex_normals.shape[0] != vertices.shape[0]: # [TODO] why?
                    # print(vertex_normals.shape[0], vertices.shape[0], faces.shape)
                    continue
                
                shape_rays_dict[_id] = {'v': vertices, 'd': vertex_normals}
                rays_o_nerf = self.or2nerf_th(torch.from_numpy(vertices).float().to(self.device)) # convert to NeRF coordinates
                rays_d_nerf = self.or2nerf_th(torch.from_numpy(-vertex_normals).float().to(self.device)) # convert to NeRF coordinates
                rgbs = self.model.nerf(rays_o_nerf, rays_d_nerf)['rgb'] # queried d is incoming directions!
                rads = rgbs.detach().cpu().numpy() / self.rad_scale * radiance_scale # get back to original scale, without hdr scaling
                samples_v_dict[_id] = ('rad', rads)

        return_dict.update({'samples_v_dict': samples_v_dict})
        if sample_type == 'rad':
            return_dict.update({'shape_rays_dict': shape_rays_dict})
        return return_dict

    def sample_lighting(
        self, 
        sample_type: str='rad', 
        subsample_rate_pts: int=1, 
        if_use_mi_geometry: bool=True, 
        if_use_loaded_envmap_position: bool=False, 
        if_mask_off_emitters: bool=False, 
        if_vis_envmap_2d_plt: bool=False, 
        lighting_scale: float=1., 
        ):

        '''
        sample non-emitter locations along hemisphere directions for incident radiance, from rad-MLP: images/demo_envmap_o3d_sampling.png
        Args:
            sample_type: 'rad' for querying radiance emitted FROM all points; 'incident-rad' for incident radiance TOWARDS all points
        Results:
            images/demo_eval_radMLP_rample_lighting_openrooms_1.png
        '''
        
        assert sample_type in ['rad', 'incident-rad']
        if if_use_mi_geometry:
            assert self.os.if_has_mitsuba_all; normal_list = self.os.mi_normal_list
        else:
            assert False, 'use mi for GT geometry which was used to train rad-MLP!'
            # assert self.os.if_has_depth_normal; normal_list = self.os.normal_list
        batch_size = self.model.hparams.batch_size * 10
        lighting_fused_list = []
        lighting_envmap_list = []

        print(white_blue('[evaluator_scene_rad] sampling %s for %d frames... subsample_rate_pts: %d'%('lighting_envmap', len(self.os.frame_id_list), subsample_rate_pts)))

        for frame_idx in tqdm(range(len(self.os.frame_id_list))):
            seg_obj = self.os.mi_seg_dict_of_lists['obj'][frame_idx] # (H, W), bool, [IMPORTANT] mask off emitter area!!
            if not if_mask_off_emitters:
                seg_obj = np.ones_like(seg_obj).astype(np.bool)

            env_height, env_width, env_row, env_col = get_list_of_keys(self.os.lighting_params_dict, ['env_height', 'env_width', 'env_row', 'env_col'], [int, int, int, int])
            downsize_ratio = self.os.lighting_params_dict.get('env_downsample_rate', 1) # over loaded envmap rows/cols
            wi_num = env_height * env_width

            '''
            [emission] directions from a surface point to hemisphere directions
            '''
            if if_use_loaded_envmap_position:
                assert if_use_mi_geometry
                samples_d = self.os.process_loaded_envmap_axis_2d_for_frame(frame_idx).reshape(env_row, env_col, env_height*env_width, 3)[::downsize_ratio, ::downsize_ratio] # (env_row//downsize_ratio, env_col//downsize_ratio, wi_num, 3)
            else:
                samples_d = get_lighting_envmap_dirs_global(self.os.pose_list[frame_idx], normal_list[frame_idx], env_height, env_width) # (HW, wi_num, 3)
                samples_d = samples_d.reshape(env_row, env_col, env_height*env_width, 3)[::downsize_ratio, ::downsize_ratio] # (env_row//downsize_ratio, env_col//downsize_ratio, wi_num, 3)
            
            seg_obj = seg_obj[::self.os.im_lighting_HW_ratios[0], ::self.os.im_lighting_HW_ratios[1]][::downsize_ratio, ::downsize_ratio]
            assert seg_obj.shape[:2] == samples_d.shape[:2]

            samples_d = samples_d[seg_obj].reshape(-1, 3) # (HW*wi_num, 3)
            samples_o = self.os.mi_pts_list[frame_idx][::self.os.im_lighting_HW_ratios[0], ::self.os.im_lighting_HW_ratios[1]][::downsize_ratio, ::downsize_ratio] # (H, W, 3)
            samples_o = np.expand_dims(samples_o[seg_obj].reshape(-1, 3), axis=1)
            # samples_o = np.broadcast_to(samples_o, (samples_o.shape[0], wi_num, 3))
            # samples_o = samples_o.view(-1, 3)
            samples_o = np.repeat(samples_o, wi_num, axis=1).reshape(-1, 3) # (HW*wi_num, 3)
            if sample_type == 'rad':
                rays_d = samples_d
                rays_o = samples_o
            elif sample_type == 'incident-rad':
                rays_d = -samples_d # from destination point, towards opposite direction
                # [mitsuba.Interaction3f] https://mitsuba.readthedocs.io/en/stable/src/api_reference.html#mitsuba.Interaction3f
                # [mitsuba.SurfaceInteraction3f] https://mitsuba.readthedocs.io/en/latest/src/api_reference.html#mitsuba.SurfaceInteraction3f

                ds_mi = mi.Vector3f(samples_d) # (HW*wi_num, 3)
                # --- [V1] hack by adding a small offset to samples_o to reduce self-intersection: images/demo_incident_rays_V1.png
                xs_mi = mi.Point3f(samples_o + samples_d * 1e-4) # (HW*wi_num, 3) [TODO] how to better handle self-intersection?
                incident_rays_mi = mi.Ray3f(xs_mi, ds_mi)

                # batch_sensor_dict = {
                #     'type': 'batch', 
                #     'film': {
                #         'type': 'hdrfilm',
                #         'width': 10,
                #         'height': 1,
                #         'pixel_format': 'rgb',
                #     },
                #     }
                # mi_rad_list = []
                # print(samples_d.shape[0])
                # for _, (d, x) in tqdm(enumerate(zip(samples_d, samples_o))):
                #     sensor = get_rad_meter_sensor(x, d, spp=32)
                #     # print(x.shape, d.shape, sensor)
                #     # batch_sensor_dict[str(_)] = sensor
                #     image = mi.render(self.os.mi_scene, sensor=sensor)
                #     mi_rad_list.append(image.numpy().flatten())

                # # batch_sensor = mi.load_dict(batch_sensor_dict)
                # mi_lighting_envmap = np.stack(mi_rad_list).reshape((self.os.H, self.os.W, env_height//downsize_ratio, env_width//downsize_ratio, 3))
                # mi_lighting_envmap = mi_lighting_envmap.transpose((0, 1, 4, 2, 3))
                # from lib.utils_OR.utils_OR_lighting import downsample_lighting_envmap
                # lighting_envmap_vis = np.clip(downsample_lighting_envmap(mi_lighting_envmap, lighting_scale=lighting_scale, downsize_ratio=1)**(1./2.2), 0., 1.)
                # plt.figure()
                # plt.imshow(lighting_envmap_vis)
                # plt.show()


                # scene = mi.load_file('/Users/jerrypiglet/Documents/Projects/dvgomm1/data/indoor_synthetic/kitchen/scene_v3.xml')
                # sensor = get_rad_meter_sensor(samples_o, samples_d, spp=1)
                # image = mi.render(self.os.mi_scene, sensor=sensor)
                # image = mi.render(self.os.mi_scene, sensor=sensor)
                # image = mi.render(self.os.mi_scene, sensor=batch_sensor)
                # mi.util.write_bitmap("tmp.exr", image)
                # import ipdb; ipdb.set_trace()


                # --- [V2] try to expand mi_rays_ret to wi_num rays per-point; HOW TO?
                # mi_rays_ret = self.os.mi_rays_ret_list[frame_idx] # (HW,) # [TODO how to get mi_rays_ret_expanded?
                # mi_rays_ret_expanded = mi.SurfaceInteraction3f(
                #     # t=mi.Float(mi_rays_ret.t.numpy()[..., np.newaxis].repeat(wi_num, 1).flatten()), 
                #     # time=mi_rays_ret.time, 
                #     wavelengths=mi_rays_ret.wavelengths, 
                #     ps=mi.Point3f(mi_rays_ret.p.numpy()[:, np.newaxis, :].repeat(wi_num, 1).reshape(-1, 3)), 
                #     # n=mi.Normal3f(mi_rays_ret.n.numpy()[:, np.newaxis, :].repeat(wi_num, 1).reshape(-1, 3)), 
                # )
                # --- [V3] expansive: re-generating camera rays where each pixel has wi_num rays: images/demo_incident_rays_V3.png
                # (cam_rays_o, cam_rays_d, _) = self.os.cam_rays_list[frame_idx]
                # cam_rays_o_flatten_expanded = cam_rays_o[:, :, np.newaxis, :].repeat(wi_num, 2).reshape(-1, 3)
                # cam_rays_d_flatten_expanded = cam_rays_d[:, :, np.newaxis, :].repeat(wi_num, 2).reshape(-1, 3)
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
                # plt.imshow(self.os.im_sdr_list[frame_idx])
                # plt.subplot(122)
                # plt.imshow(np.amin(ret.t.numpy().reshape((self.os.H, self.os.W, wi_num)), axis=-1), cmap='jet')
                # plt.colorbar()
                # plt.title('[demo of self-intersection] distance to destination point (min among all rays')
                # plt.show()
                # <<<< debug

            assert rays_d.shape == rays_o.shape # ray d and o to query rad-MLP
            if subsample_rate_pts != 1:
                rays_d = rays_d[::subsample_rate_pts]
                rays_o = rays_o[::subsample_rate_pts]

            # batching to prevent memory overflow
            batch_num = math.ceil(rays_d.shape[0]*1.0/batch_size)
            print(blue_text('Querying rad-MLP for %d batches of rays...')%batch_num)
            rad_list = []
            for b_id in tqdm(range(batch_num)):
                b0 = b_id*batch_size
                b1 = min((b_id+1)*batch_size, rays_d.shape[0])
                rays_d_select = rays_d[b0:b1]
                rays_o_select = rays_o[b0:b1]
                rays_o_nerf = self.or2nerf_th(torch.from_numpy(rays_o_select).to(self.device))
                rays_d_nerf = self.or2nerf_th(torch.from_numpy(-rays_d_select).to(self.device)) # nagate to comply with rad-inv coordinates
                # if self.dataset_type == 'Indoor':
                #     rays_o_nerf = self.process_for_Indoor(rays_o_nerf)
                rgbs = self.model.nerf(rays_o_nerf, rays_d_nerf)['rgb']
                rad = rgbs.detach().cpu().numpy() / self.rad_scale # get back to original scale, without hdr scaling
                # rad = rad * 0. + 10.
                rad_list.append(rad)
            rad_all = np.concatenate(rad_list)
            assert rad_all.shape[0] == rays_o.shape[0]

            lighting_envmap = rad_all.reshape((env_row//downsize_ratio, env_col//downsize_ratio, env_height, env_width, 3))
            lighting_envmap = lighting_envmap.transpose((0, 1, 4, 2, 3))
            lighting_envmap_list.append(lighting_envmap)

            if if_vis_envmap_2d_plt:
                assert subsample_rate_pts == 1
                assert not if_mask_off_emitters
                plt.figure(figsize=(15, 15))
                plt.subplot(311)
                plt.imshow(self.os.im_sdr_list[frame_idx])
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

    # def process_for_Indoor(self, position):
    #     assert isinstance(position, torch.Tensor)
    #     # Liwen's model was trained using scene.obj (smaller) instead of scene_v3.xml (bigger), between which there is scaling and translation. self.os.cam_rays_list are acquired from the scene of scene_v3.xml
    #     scale_m2b = torch.from_numpy(np.array([0.206,0.206,0.206], dtype=np.float32).reshape((1, 3))).to(position.device)
    #     trans_m2b = torch.from_numpy(np.array([-0.074684,0.23965,-0.30727], dtype=np.float32).reshape((1, 3))).to(position.device)
    #     position = scale_m2b * position + trans_m2b
    #     return position

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
from lib.utils_from_monosdf import rend_util

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

        # assert rad_scale == 1.
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

        self.conf_model = self.conf.get_config('model')
        self.model = utils_monosdf.get_class(self.conf.get_string('train.model_class'))(conf=self.conf_model, if_hdr=if_hdr)
        self.host = host
        self.device = {
            'apple': 'mps', 
            'mm1': 'cuda', 
            'qc': '', 
        }[self.host]
        assert self.device == 'cuda', 'MonoSDF has ops built with customized cuda kernels'
        self.model.to(self.device)

        saved_model_state = torch.load(ckpt_path)
        if list(saved_model_state["model_state_dict"].keys())[0].startswith("module."):
            saved_model_state["model_state_dict"] = {k[7:]: v for k, v in saved_model_state["model_state_dict"].items()}
        self.model.load_state_dict(saved_model_state["model_state_dict"], strict=True)

        self.model.eval()

        self.rad_scale = rad_scale
        self.os = scene_object

        assert [self.os.H, self.os.W]==self.dataset_conf['img_res'], \
                'MonoSDF was trained with image res of %s; scene_object uses %s!'%(str(self.dataset_conf['img_res']), str([self.os.H, self.os.W]))

        uv = np.mgrid[0:self.os.H, 0:self.os.W].astype(np.int32)
        uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float().to(self.device)
        self.uv = uv.reshape(2, -1).transpose(1, 0).unsqueeze(0) # (1, HW, 2)

        intrinsics, _ = self.load_K_pose(0)
        ray_dirs_tmp, _ = rend_util.get_camera_params(self.uv, torch.eye(4)[None].to(self.device), intrinsics)
        self.depth_scale = ray_dirs_tmp[0, :, 2:] # (N, 1)

    def export_mesh(self, mesh_path: str='test_files/monosdf_mesh.ply', resolution: int=1024):
        from utils.plots import get_surface_sliding
        # exporting mesh from SDF
        print(white_blue('-> Exporting MonoSDF mesh to %s ...'%mesh_path))
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

    def load_K_pose(self, frame_id: int):

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
        '''
        ↑ R in pose is the same as R in self.os.pose_list[frame_id]
        ↑ t in pose is t in self.os.pose_list[frame_id], after transformation via normalize_x: ![](https://i.imgur.com/HObmuVc.png)
        ↑ intrinsics is the same as self.os.K
        '''

        intrinsics = torch.from_numpy(intrinsics).float().to(self.device).unsqueeze(0) # (1, 4, 4)
        pose = torch.from_numpy(pose).float().to(self.device).unsqueeze(0) # (1, 4, 4)

        return intrinsics, pose

    def render_im_scratch(
        self, 
        frame_id: int, 
        offset_in_scan: int=0, 
        if_integrate: bool=False, # Truel to integrate over camera rays; False to only query single surface point per ray
        if_plt: bool=False
        ):
        '''
        rendering image using 
        [1] if_integrate=True: 
            ![](images/demo_evaluator_monosdf_render_im_scratch_integrate.png)
            ![](images/demo_evaluator_monosdf_render_im_scratch_integrate_OR.png)
            integration over samples per-ray; 
        [2] if_integrate=False: 
            ![](images/demo_evaluator_monosdf_render_im_scratch_surface.png)
            ![](images/demo_evaluator_monosdf_render_im_scratch_surface_OR.png)
            sample surface points along -cam_d dirs. 
        '''

        rays_o, rays_d, _ = self.os.cam_rays_list[frame_id]
        rays_o_, rays_d_ = self.to_d(rays_o.reshape(-1, 3)).float(), self.to_d(rays_d.reshape(-1, 3)).float()

        if not if_integrate:
            mi_pts = self.os.mi_pts_list[frame_id]
            mi_pts_ = self.to_d(mi_pts.reshape(-1, 3)).float()
            x_mono, rays_d_mono = self.normalize_x(mi_pts_, True), rays_d_
            indices = self.to_d(np.zeros((x_mono.shape[0]))).long()

            # One sample per-ray; instead of N samples like in MonoSDF
            rgbs_N = self.query_rad_scene_pts(
                x_mono, 
                -rays_d_mono, 
                indices, 
                per_split_size = 4096*4, 
                )
            rgbs_hdr_scaled = rgbs_N.reshape(self.os.H, self.os.W, 3).detach().cpu().numpy() / self.rad_scale
            rgbs_sdr = np.clip(rgbs_hdr_scaled ** (1./2.2), 0., 1.)
            depths = np.zeros((self.os.H, self.os.W))
        else:
            rays_o_mono, rays_d_mono = self.normalize_x(rays_o_, True), rays_d_
            rays_return_dict = self.query_rays(
                rays_o_mono, 
                rays_d_mono, 
                # indices, 
                per_split_size = 1024, 
            )
            rgbs_hdr_scaled = rays_return_dict['rgb_values'].reshape(self.os.H, self.os.W, 3).detach().cpu().numpy() / self.rad_scale
            rgbs_sdr = np.clip(rgbs_hdr_scaled ** (1./2.2), 0., 1.)
            depths = rays_return_dict['depth_values'].reshape(self.os.H, self.os.W).detach().cpu().numpy()

        if if_plt:
            plt.figure()
            plt.title('render_im_scratch (frame %d)'%frame_id)
            ax = plt.subplot(221)
            plt.imshow(rgbs_sdr)
            ax.set_title('monosdf rgb (SDR; scaled to GT)')
            ax = plt.subplot(222)
            plt.imshow(depths); plt.colorbar()
            ax.set_title('monosdf depth')
            ax = plt.subplot(223)
            plt.imshow(np.clip(self.os.im_hdr_list[frame_id]**(1./2.2), 0., 1.))
            ax.set_title('GT rgb (SDR)')
            ax = plt.subplot(224)
            plt.imshow(self.os.mi_depth_list[frame_id]); plt.colorbar()
            ax.set_title('GT depth')
            plt.show()
        else:
            plt.imsave('test_files/mono_render_im_scratch_im_%d.png'%frame_id, rgbs)

    def render_im(
        self, 
        frame_id: int, 
        offset_in_scan: int=0, 
        split_n_pixels: int=1024, 
        if_plt: bool=False
        ):
        '''
        rendering image using MonoSDF's eval code
        '''

        import utils.general as utils_monosdf
        import utils.plots as plots

        frame_id_monosdf = frame_id + offset_in_scan # `indice` in MonoSDF (which is conditioning on per-image features; so that you would want to input the real image id in monosdf dataset)

        intrinsics, pose = self.load_K_pose(frame_id)

        # gather input dict
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

        print(white_blue('-> Exported rendering result to %s ...'%merge_path))

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

    def query_rad_scene_pts(
        self, 
        x_mono: torch.Tensor, # (N, 3), float32
        d_mono: torch.Tensor, # (N, 3), float32
        indices: torch.Tensor, # (N,), long
        per_split_size = 4096*4*4*2
    ):
        assert len(x_mono.shape)==2, 'x_mono has to be 2D: (N, 3)'
        assert len(d_mono.shape)==2, 'd_mono has to be 2D: (N, 3)'
        assert len(indices.shape)==1, 'indices has to be 1D: (N,)'
        assert indices.shape[0]==x_mono.shape[0], 'indices size mismatch with x_mono!'

        rgbs_all = []

        for (x, d, i) in tqdm(zip(torch.split(x_mono, per_split_size), torch.split(d_mono, per_split_size), torch.split(indices, per_split_size))):
            sdf, feature_vectors, gradients = self.model.implicit_network.get_outputs(x)
            rgb_flat = self.model.rendering_network(x, gradients, d, feature_vectors, i, if_pixel_input=True)['rgb']
            rgbs_all.append(rgb_flat.detach().cpu())

        return torch.cat(rgbs_all) # (N, 3)

    def query_rays(
        self, 
        rays_o_mono: torch.Tensor, # (N, 3), float32
        rays_d_mono: torch.Tensor, # (N, 3), float32
        # indices: torch.Tensor, # (N,), long
        per_split_size = 1024, 
        if_from_one_frame=True, # True: all rays from one frame
    ):
        assert len(rays_o_mono.shape)==2, 'rays_o_mono has to be 2D: (N, 3)'
        assert len(rays_d_mono.shape)==2, 'rays_d_mono has to be 2D: (N, 3)'

        import utils.general as utils_monosdf

        res = []

        print('-- query_rays in %d batches...'%(self.depth_scale.shape[0]//per_split_size))
        if if_from_one_frame:
            depth_scale_list = torch.split(self.depth_scale, per_split_size)

        for _, (o_, d_) in enumerate(tqdm(zip(torch.split(rays_o_mono, per_split_size), torch.split(rays_d_mono, per_split_size)))):
            z_vals, z_samples_eik = self.model.ray_sampler.get_z_vals(d_, o_, self.model)
            N_samples = z_vals.shape[1]

            points = o_.unsqueeze(1) + z_vals.unsqueeze(2) * d_.unsqueeze(1) # (1024, 1, 3) + (1024, 98, 1) * (1024, 1, 3)
            points_flat = points.reshape(-1, 3) # (1024, N_samples, 3) -> (1024*N_samples, 3)

            dirs = d_.unsqueeze(1).repeat(1,N_samples,1)
            dirs_flat = dirs.reshape(-1, 3)

            sdf, feature_vectors, gradients_sdf = self.model.implicit_network.get_outputs(points_flat)
            
            # points_flat: (N_pixels*N_samples, 3)
            i_ = self.to_d(np.zeros(points_flat.shape[0])).long()
            rendering_output_dict = self.model.rendering_network(points_flat, gradients_sdf, dirs_flat, feature_vectors, i_, if_pixel_input=True)
            rgb_flat = rendering_output_dict['rgb']
            rgb = rgb_flat.reshape(-1, N_samples, 3)

            weights = self.model.volume_rendering(z_vals, sdf)

            rgb_values = torch.sum(weights.unsqueeze(-1) * rgb, 1)

            result_dict = {
                'rgb': rgb.detach().cpu(), # (N, 98, 3)
                'rgb_values': rgb_values.detach().cpu(), # (N, 3)
                'sdf': sdf.reshape(z_vals.shape).detach().cpu(), # (N, 98)
                'weights': weights.detach().cpu(), # (N, 98)
            }
            if if_from_one_frame:
                depth_scale_ =depth_scale_list[_]
                depth_values = torch.sum(weights * z_vals, 1, keepdims=True) / (weights.sum(dim=1, keepdims=True) +1e-8)
                # we should scale rendered distance to depth along z direction
                depth_values = depth_scale_ * depth_values
                result_dict.update({
                    'depth_values': depth_values.detach().cpu(), # (N, 1)
                    'z_vals': z_vals.detach().cpu(), # (N, 98)
                    'depth_vals': z_vals.detach().cpu() * depth_scale_.detach().cpu(), # (N, 98)
                })

            res.append(result_dict)

        assert sum([_['rgb'].shape[0] for _ in res]) == rays_o_mono.shape[0]
        model_outputs = utils_monosdf.merge_output(res, rays_o_mono.shape[0], 1)
        return model_outputs

    def normalize_x(self, x: torch.tensor, if_offset: bool=True):
        '''
        x: (N, 3)
        '''
        scale, offset = self.os.monosdf_scale, torch.from_numpy(self.os.monosdf_offset).to(x.device)
        if if_offset:
            return scale * (x + offset)
        else:
            x = scale * x
            x = x / (torch.linalg.norm(x, axis=1, keepdim=True)+1e-12)
            return x

    def to_d(self, x: np.ndarray):
        if 'mps' in self.device: # Mitsuba RuntimeError: Cannot pack tensors on mps:0
            return x
        return torch.from_numpy(x).to(self.device)

    def sample_shapes(
        self, 
        sample_type: str='rad', 
        shape_params={}, 
        ):
        '''
        sample shape surface for sample_type:
            - 'rad': radiance (at vectices along vertice normals) from MonoSDF

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

        print(white_blue('Evaluating MonoSDF for [%s]'%sample_type), 'sample_shapes for %d shapes...'%len(self.os.ids_list))

        for shape_index, (vertices, faces, _id) in tqdm(enumerate(zip(self.os.vertices_list, self.os.faces_list, self.os.ids_list))):
            assert np.amin(faces) == 1
            shape_tri_mesh = trimesh.Trimesh(vertices=vertices, faces=faces-1) # [IMPORTANT] faces-1 because Trimesh faces are 0-based

            if sample_type == 'rad':
                vertex_normals = shape_tri_mesh.vertex_normals # already normalized

                if vertex_normals.shape[0] != vertices.shape[0]: # [TODO] why?
                    # print(vertex_normals.shape[0], vertices.shape[0], faces.shape)
                    continue
                
                shape_rays_dict[_id] = {'v': vertices, 'd': vertex_normals}
                vertices_mono, vertex_normals_mono = self.normalize_x(self.to_d(vertices).float(), True), self.to_d(vertex_normals).float()
                indices = self.to_d(np.zeros((vertices_mono.shape[0]))).long()

                # One sample per-ray; instead of N samples like in MonoSDF
                rgbs = self.query_rad_scene_pts(
                    vertices_mono, 
                    -vertex_normals_mono, 
                    indices, 
                    per_split_size = 4096*4, 
                    )

                rads = rgbs.numpy() / self.rad_scale * radiance_scale # get back to original scale, without hdr scaling

                # sdf, feature_vectors, gradients = self.model.implicit_network.get_outputs(vertices_mono)
                # rgb_flat = self.model.rendering_network(vertices_mono, gradients, -vertex_normals_mono, feature_vectors, indices, if_pixel_input=True)
                # rads = rgb_flat.detach().cpu().numpy() / self.rad_scale * radiance_scale # get back to original scale, without hdr scaling

                samples_v_dict[_id] = ('rad', rads)

        return_dict.update({'samples_v_dict': samples_v_dict})
        if sample_type == 'rad':
            return_dict.update({'shape_rays_dict': shape_rays_dict})
        return return_dict
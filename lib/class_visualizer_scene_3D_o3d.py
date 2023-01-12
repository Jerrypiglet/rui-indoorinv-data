import numpy as np
np.random.seed(0)
from tqdm import tqdm
from math import prod
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import open3d.visualization as vis

import trimesh
# from sympy import Point3D, Line3D, Plane, sympify, Rational
# from copy import deepcopy
# import torch
from pathlib import Path
import copy

# Import the library using the alias "mi"
import mitsuba as mi
# Set the variant of the renderer
# from lib.global_vars import mi_variant
# mi.set_variant(mi_variant)

from lib.class_openroomsScene2D import openroomsScene2D
from lib.class_openroomsScene3D import openroomsScene3D
from lib.class_mitsubaScene3D import mitsubaScene3D

from lib.utils_misc import get_list_of_keys, gen_random_str, yellow, white_red
from lib.utils_o3d import text_3d, get_arrow_o3d, get_sphere, remove_walls, remove_ceiling
from lib.utils_io import load_HDR, to_nonHDR
from lib.utils_OR.utils_OR_mesh import writeMesh
from lib.utils_vis import color_map_color

aabb_01 = np.array([[0, 0, 0],
                    [0, 0, 1],
                    [0, 1, 1],
                    [0, 1, 0],
                    [1, 0, 0],
                    [1, 0, 1],
                    [1, 1, 1],
                    [1, 1, 0]])

class visualizer_scene_3D_o3d(object):
    '''
    A class used to **visualize** OpenRooms (public/public-re versions) scene contents (2D/2.5D per-pixel DENSE properties for inverse rendering + 3D semantics).
    '''
    def __init__(
        self, 
        openrooms_scene, 
        modality_list_vis: list, 
        if_debug_info: bool=False, 
    ):

        assert type(openrooms_scene) in [openroomsScene2D, openroomsScene3D, mitsubaScene3D], '[visualizer_openroomsScene] has to take an object of openroomsScene, openroomsScene3D, mitsubaScene3D!'

        self.os = openrooms_scene
        self.if_debug_info = if_debug_info

        self.modality_list_vis = list(set(modality_list_vis))
        for _ in self.modality_list_vis:
            assert _ in ['dense_geo', 'cameras', 'lighting_SG', 'lighting_envmap', 'layout', 'shapes', 'emitters', 'mi']
        if 'mi' in self.modality_list_vis:
            self.mi_pcd_color_list = None
        self.extra_geometry_list = []

    def run_demo(self, extra_geometry_list=[]):

        geoms = []

        sphere = o3d.geometry.TriangleMesh.create_sphere(1.0)
        sphere.compute_vertex_normals()
        sphere.translate(np.array([0, 0, -3.5]))
        box = o3d.geometry.TriangleMesh.create_box(2, 4, 4)
        box.translate(np.array([-1, -2, -2]))
        box.compute_triangle_normals()

        mat_sphere = vis.rendering.MaterialRecord()
        mat_sphere.shader = 'defaultLit'
        mat_sphere.base_color = [0.8, 0, 0, 1.0]

        mat_box = vis.rendering.MaterialRecord()
        # mat_box.shader = 'defaultLitTransparency'
        mat_box.shader = 'defaultLitSSR'
        mat_box.base_color = [0.467, 0.467, 0.467, 0.2]
        mat_box.base_roughness = 0.0
        mat_box.base_reflectance = 0.0
        mat_box.base_clearcoat = 1.0
        mat_box.thickness = 1.0
        mat_box.transmission = 1.0
        mat_box.absorption_distance = 10
        mat_box.absorption_color = [0.5, 0.5, 0.5]


        geoms += [{'name': 'sphere', 'geometry': sphere, 'material': mat_sphere},
                {'name': 'box', 'geometry': box, 'material': mat_box}]
        vis.draw(geoms)

    def run_o3d_shader(
        self, 
        **kwargs
    ):

        '''
        choise of shaders: https://blog.csdn.net/qq_31254435/article/details/124573045
        '''

        # self.run_demo(extra_geometry_list=self.o3d_geometry_list)
        geoms = []
        for _ in self.o3d_geometry_list:
            
            geo_name = gen_random_str(5)
            mat = vis.rendering.MaterialRecord()
            mat.shader = 'defaultUnlit'
            mat.point_size = 5.0

            if isinstance(_, list):
                geo, geo_name = _
            else:
                geo = _
            
            geo_dict = {'geometry': geo, 'name': geo_name, 'material': mat}
            # geo_dict = {'geometry': geo, 'name': geo_name}

            if 'layout' in geo_name: # room layout box mesh
                mat = vis.rendering.MaterialRecord()
                mat.shader = 'defaultLitSSR'
                # mat.shader = 'defaultUnlitTransparency'
                mat.base_color = [0.8, 0.8, 0.8, 0.5]
                geo_dict.update({'material': mat})

            if 'envmap' in geo_name: # envmap hemisphere for window emitters
                mat = vis.rendering.MaterialRecord()
                mat.shader = 'defaultLitTransparency'
                # mat.shader = 'defaultLitSSR'
                mat.base_color = [0.467, 0.467, 0.467, 0.9]
                geo_dict.update({'material': mat})
            
            if 'shape_emitter' in geo_name: # shapes for emitters, using normal shader
                mat = vis.rendering.MaterialRecord()
                mat.shader = 'normals'
                geo_dict.update({'material': mat})

            if 'shape_obj' in geo_name: # shapes for objs, using default lit shader
                mat = vis.rendering.MaterialRecord()
                mat.shader = 'defaultLit'
                geo_dict.update({'material': mat})

            geoms.append(geo_dict)
        
        vis.draw(geoms)

        # vis.draw_geometries(self.o3d_geometry_list, mesh_show_back_face=True)

    def run_o3d(
        self, 
        if_shader: bool=False, 
        **kwargs, 
    ):
        self.o3d_geometry_list = self.load_o3d_geometry_list(
            self.modality_list_vis, 
            **kwargs
            )

        if if_shader:
            self.run_o3d_shader()
        else:
            self.init_o3d_vis()
            for _ in self.o3d_geometry_list:
                if isinstance(_, list):
                    geo, geo_name = _
                else:
                    geo = _
                self.vis.add_geometry(geo)

            self.vis.run()

    def init_o3d_vis(self):
        self.vis = o3d.visualization.Visualizer()
        self.os.W = self.vis.create_window()
        self.opt = self.vis.get_render_option()
        self.opt.background_color = np.asarray([1., 1., 1.])

    def load_o3d_geometry_list(
        self, 
        modality_list: list, 
        cam_params: dict = {}, 
        dense_geo_params: dict = {}, 
        lighting_params: dict = {}, 
        layout_params: dict = {}, 
        shapes_params: dict = {}, 
        emitter_params: dict = {}, 
        mi_params: dict = {}, 
    ):
        
        o3d_geometry_list = []

        o3d_geometry_list += [o3d.geometry.TriangleMesh.create_coordinate_frame()]

        if 'cameras' in modality_list:
            o3d_geometry_list += self.collect_cameras(cam_params)

        if 'dense_geo' in modality_list:
            o3d_geometry_list += self.collect_dense_geo(
                dense_geo_params
            )
        
        if 'lighting_SG' in modality_list:
            o3d_geometry_list += self.collect_lighting_SG(
                lighting_params
            )

        if 'lighting_envmap' in modality_list:
            o3d_geometry_list += self.collect_lighting_envmap(
                lighting_params
            )

        if 'layout' in modality_list:
            o3d_geometry_list += self.collect_layout(
                layout_params
            )

        if 'shapes' in modality_list:
            o3d_geometry_list += self.collect_shapes(
                shapes_params
            )

        if 'emitters' in modality_list:
            o3d_geometry_list += self.collect_emitters(
                emitter_params
            )

        if 'mi' in modality_list:
            o3d_geometry_list += self.collect_mi(
                mi_params
            )

        o3d_geometry_list += self.extra_geometry_list

        return o3d_geometry_list

    def add_extra_geometry(self, geometry_list: list=[], if_processed_geometry_list: bool=False):
        '''
        if_processed_geometry_list: True if already processed into list of geometries
        '''
        valid_extra_geometry_list = ['rays', 'pts']
        if if_processed_geometry_list:
            self.extra_geometry_list += geometry_list
            return
        for geometry_type, geometry in geometry_list:
            assert geometry_type in valid_extra_geometry_list
            if geometry_type == 'rays':
                ray_o = geometry['ray_o'] # (N, 3)
                ray_e = geometry['ray_e'] # (N, 3)
                # visibility = geometry['visibility'].squeeze() # (N,)
                # [TODO] add options for colormap tensor (e.g. colorize according to t, or visibility)
                assert len(ray_o.shape)==len(ray_e.shape)==2
                # assert len(visibility.shape)==1
                assert ray_o.shape[0]==ray_e.shape[0]
                # ==visibility.shape[0]
                dirs = o3d.geometry.LineSet()
                dirs.points = o3d.utility.Vector3dVector(np.vstack((ray_o, ray_e)))
                # dirs.colors = o3d.utility.Vector3dVector([[1., 0., 0.] if vis == 1 else [0.8, 0.8, 0.8] for vis in visibility]) # red: visible; blue: not visible
                if 'ray_c' in geometry:
                    ray_c = geometry['ray_c']
                    assert type(ray_c)==np.ndarray and ray_c.shape in [(3,), (ray_o.shape[0], 3)]
                else:
                    ray_c = [[1., 0., 0.]] * ray_o.shape[0]
                dirs.colors = o3d.utility.Vector3dVector(ray_c)
                dirs.lines = o3d.utility.Vector2iVector([[_, _+ray_o.shape[0]] for _ in range(ray_o.shape[0])])
                self.extra_geometry_list.append(dirs)
            if geometry_type == 'pts':
                pts = geometry['pts'] # (N, 3)
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pts)
                pcd.colors = o3d.utility.Vector3dVector([[0., 0., 0.]]*pts.shape[0])
                self.extra_geometry_list.append(pcd)

    def get_pcd_color_fused_geo(self, pcd_color_mode: str):
        assert pcd_color_mode in ['rgb', 'normal', 'dist_emitter0', 'mi_visibility_emitter0']
        pcd_color = None

        if pcd_color_mode == 'rgb':
            assert self.os.if_has_im_sdr and self.os.if_has_depth_normal
            pcd_color = self.geo_fused_dict['rgb']
        
        elif pcd_color_mode == 'normal': # images/demo_pcd_color_normal.png
            assert self.os.if_has_im_sdr and self.os.if_has_depth_normal
            pcd_color = (self.geo_fused_dict['normal'] + 1.) / 2.
        
        elif pcd_color_mode == 'dist_emitter0': # images/demo_pcd_color_dist.png
            assert self.os.if_has_im_sdr and self.os.if_has_depth_normal and self.os.if_has_shapes
            emitter_0 = (self.os.lamp_list + self.os.window_list)[0]
            emitter_0_center = emitter_0['emitter_prop']['box3D_world']['center'].reshape((1, 3))
            dist_to_emitter_0 = np.linalg.norm(emitter_0_center - self.geo_fused_dict['X'], axis=1, keepdims=False)
            pcd_color = color_map_color(dist_to_emitter_0, vmin=np.amin(dist_to_emitter_0), vmax=np.amax(dist_to_emitter_0))
        
        elif pcd_color_mode == 'mi_visibility_emitter0': # images/demo_pcd_color_mi_visibility_emitter0.png
            assert self.os.if_has_im_sdr and self.os.if_has_depth_normal and self.os.if_has_shapes
            assert self.os.if_has_mitsuba_scene
            emitter_0 = (self.os.lamp_list + self.os.window_list)[0]
            emitter_0_center = emitter_0['emitter_prop']['box3D_world']['center'].reshape((1, 3))
            X_to_emitter_0 = emitter_0_center - self.geo_fused_dict['X']
            xs = self.geo_fused_dict['X']
            xs_mi = mi.Point3f(xs)
            ds = X_to_emitter_0 / (np.linalg.norm(X_to_emitter_0, axis=1, keepdims=1)+1e-6)
            ds_mi = mi.Vector3f(ds)
            # ray origin, direction, t_max
            rays_mi = mi.Ray3f(xs_mi, ds_mi)
            ret = self.os.mi_scene.ray_intersect(rays_mi) # https://mitsuba.readthedocs.io/en/stable/src/api_reference.html?highlight=write_ply#mitsuba.Scene.ray_intersect
            # returned structure contains intersection location, nomral, ray step, ...
            # positions = mi2torch(ret.p.torch())
            # normals = mi2torch(ret.n.torch())
            ts = ret.t.numpy()
            visibility = ts < np.linalg.norm(X_to_emitter_0, axis=1, keepdims=False)
            visibility = np.logical_and(ts > 1e-2, visibility)
            visibility = 1. - visibility.astype(np.float32)
            pcd_color = color_map_color(visibility)
            # pcd_color = color_map_color(ts, vmin=np.amin(ts), vmax=np.amax(ts[ts<np.inf]))
            # pcd_color = np.ones((visibility.shape[0], 3)) * 0.5

            # return xs, ds, ts, visibility

        return pcd_color
        
    def collect_cameras(self, cam_params: dict={}):
        assert self.os.if_has_poses

        if_cam_axis_only = cam_params.get('if_cam_axis_only', False)
        if_cam_traj = cam_params.get('if_cam_traj', False)
        subsample_cam_rate = cam_params.get('subsample_cam_rate', 1)
        near, far = self.os.near, self.os.far

        pose_list = self.os.pose_list
        # origin_lookatvector_up_list = self.os.origin_lookatvector_up_list
        cam_frustrm_list = []
        # cam_axes_list = []
        cam_center_list = []
        cam_traj_list = []
        # cam_o_d_list = []

        # for cam_idx, (pose, origin_lookatvector_up) in enumerate(zip(pose_list, origin_lookatvector_up_list)):
        #     origin, lookat, up = origin_lookatvector_up[0].flatten(), origin_lookatvector_up[1].flatten(), origin_lookatvector_up[2].flatten()
            
        #     cam_o = origin.flatten()
        #     cam_d = lookat.flatten()
        #     cam_d = cam_d / np.linalg.norm(cam_d)
        #     cam_o_d_list.append((cam_o, cam_d))

        #     right = np.cross(lookat, up).flatten()
        #     cam = [cam_o]
        #     frustum_scale = 0.5
        #     cam += [cam_o + cam_d * frustum_scale + up * 0.5 * frustum_scale - right * 0.5 * frustum_scale]
        #     cam += [cam_o + cam_d * frustum_scale + up * 0.5 * frustum_scale + right * 0.5 * frustum_scale]
        #     cam += [cam_o + cam_d * frustum_scale - up * 0.5 * frustum_scale - right * 0.5 * frustum_scale]
        #     cam += [cam_o + cam_d * frustum_scale - up * 0.5 * frustum_scale + right * 0.5 * frustum_scale]
        #     cam = np.array(cam)
        #     cam_list.append(cam)

        cam_list = []
        cam_axis_list = []
        for (rays_o, rays_d, _) in self.os.cam_rays_list:
            cam_o = rays_o[0,0] # (3,)
            cam_d = rays_d[[0,0,-1,-1],[0,-1,0,-1]] # get cam_d of 4 corners: (4, 3)
            cam_list.append(np.array([cam_o, *(cam_o+cam_d*max(near, far*0.05))]))

            cam_axis = cam_d.mean(0)
            cam_axis_list.append(np.array([cam_o, cam_o+cam_axis]))

        c2w_list = pose_list

        cam_axis_arrow_list = []

        for cam_idx, cam in enumerate(cam_list):
            # cam_color = [0.5, 0.5, 0.5]
            cam_color = [0., 0., 0.] # default: black
            if cam_idx == 0:
                cam_color = [0., 0., 1.] # highlight first cam frustrm with blue
            elif cam_idx == len(cam_list)-1:
                cam_color = [1., 0., 0.] # highlight last cam frustrm with red

            cam_frustrm = o3d.geometry.LineSet()
            cam_frustrm.points = o3d.utility.Vector3dVector(cam)
            if len(cam) == 5:
                cam_frustrm.colors = o3d.utility.Vector3dVector([cam_color for i in range(8)])
                cam_frustrm.lines = o3d.utility.Vector2iVector([[0,1],[0,2],[0,3],[0,4],[1,2],[2,4],[4,3],[3,1]])
            elif len(cam) == 8:
                cam_frustrm.colors = o3d.utility.Vector3dVector([cam_color for i in range(12)])
                cam_frustrm.lines = o3d.utility.Vector2iVector([
                    [0,1],[1,3],[3,2],[2,0],
                    [4,5],[5,7],[7,6],[6,4],
                    [0,4],[1,5],[3,7],[2,6],
                ])
            cam_frustrm_list.append(cam_frustrm)

            cam_center = o3d.geometry.PointCloud()
            cam_center.points = o3d.utility.Vector3dVector(np.array(cam[0]).reshape(1, 3))
            cam_center.colors = o3d.utility.Vector3dVector(np.array(cam_color).reshape(1, 3))
            cam_center_list.append(cam_center)

            cam_axis = cam_axis_list[cam_idx]
            cam_axis_arrow = o3d.geometry.LineSet()
            cam_axis_arrow.points = o3d.utility.Vector3dVector(cam_axis)
            cam_axis_arrow.colors = o3d.utility.Vector3dVector([cam_color for i in range(2)])
            cam_axis_arrow.lines = o3d.utility.Vector2iVector([[0,1]])
            cam_axis_arrow_list.append(cam_axis_arrow)
            
            if if_cam_traj and cam_idx < len(cam_list)-1:
                cam_traj = o3d.geometry.LineSet()   
                cam_traj.points = o3d.utility.Vector3dVector(np.array([cam_list[cam_idx][0], cam_list[cam_idx+1][1]]))
                cam_traj.colors = o3d.utility.Vector3dVector([cam_color for i in range(2)])
                cam_traj.lines = o3d.utility.Vector2iVector([[0,1]])
                cam_traj_list.append(cam_traj)

        # if subsample_cam_rate != 1: # subsample camera poses if too many
        #     cam_frustrm_list = cam_axes_list[::subsample_cam_rate] + [cam_axes_list[-1]]

        if if_cam_axis_only:
            geometry_list = [
                *cam_axis_arrow_list, *cam_center_list, *cam_traj_list, # pcd + cams
            ]
        else:
            geometry_list = [
                *cam_frustrm_list, *cam_center_list, *cam_traj_list, # pcd + cams
            ]

        return geometry_list

    def collect_dense_geo(self, dense_geo_params: dict={}):

        assert self.os.if_has_im_sdr and self.os.if_has_depth_normal

        geometry_list = []

        subsample_pcd_rate = dense_geo_params.get('subsample_pcd_rate', 10)
        if_ceiling = dense_geo_params.get('if_ceiling', False)
        if_walls = dense_geo_params.get('if_walls', False)
        if_normal = dense_geo_params.get('if_normal', False)
        subsample_normal_rate_x = dense_geo_params.get('subsample_normal_rate_x', 5) # subsample_normal_rate_x is multiplicative to subsample_pcd_rate

        self.geo_fused_dict, _ = self.os._fuse_3D_geometry(subsample_rate=subsample_pcd_rate)

        xyz_pcd, rgb_pcd, normal_pcd = get_list_of_keys(self.geo_fused_dict, ['X', 'rgb', 'normal'])
        # N_pcd = xyz_pcd.shape[0]

        pcd_color_mode = dense_geo_params.get('pcd_color_mode', 'rgb')
        pcd_color = self.get_pcd_color_fused_geo(pcd_color_mode)

        if pcd_color_mode == 'mi_visibility_emitter0': # show all occluded rays
            xs, ds, ts, visibility = _
            xs_end = xs + ds * ts[:, np.newaxis]

            xs = xs[visibility==0.]; xs_end = xs_end[visibility==0.]; visibility = visibility[visibility==0.]
            _subsample_rate = 100
            xs = xs[::_subsample_rate]; xs_end = xs_end[::_subsample_rate]; visibility = visibility[::_subsample_rate]

            dirs = o3d.geometry.LineSet()
            dirs.points = o3d.utility.Vector3dVector(np.vstack((xs, xs_end)))
            # dirs.colors = o3d.utility.Vector3dVector([[1., 0., 0.] if vis == 1 else [0., 0., 1.] for vis in visibility]) # red: visible; blue: not visible
            dirs.colors = o3d.utility.Vector3dVector([[1., 0., 0.] if vis == 1 else [0.8, 0.8, 0.8] for vis in visibility]) # red: visible; blue: not visible
            dirs.lines = o3d.utility.Vector2iVector([[_, _+xs.shape[0]] for _ in range(xs.shape[0])])

            geometry_list.append(dirs)

        xyz_pcd_max = np.amax(xyz_pcd, axis=0)
        xyz_pcd_min = np.amin(xyz_pcd, axis=0)

        pcd_mask = None
        if pcd_color is None:
            pcd_color = copy.deepcopy(self.pcd_color)
        assert pcd_color.shape[0] == xyz_pcd.shape[0]

        if not if_ceiling:
            xyz_pcd, pcd_color = remove_ceiling(xyz_pcd, pcd_color, if_debug_info=self.if_debug_info)
        if not if_walls:
            assert self.os.if_has_layout
            layout_bbox_3d = self.os.layout_box_3d_transformed
            xyz_pcd, pcd_color = remove_walls(layout_bbox_3d, xyz_pcd, pcd_color, if_debug_info=self.if_debug_info)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz_pcd)
        pcd.colors = o3d.utility.Vector3dVector(pcd_color)

        out_bbox_pcd = o3d.geometry.LineSet()
        out_bbox_pcd.points = o3d.utility.Vector3dVector(xyz_pcd_min + aabb_01 * (xyz_pcd_max - xyz_pcd_min))
        out_bbox_pcd.colors = o3d.utility.Vector3dVector([[1,0,0] for i in range(12)])
        out_bbox_pcd.lines = o3d.utility.Vector2iVector([[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]])

        geometry_list += [pcd]

        if if_normal:
            normal_length = np.amin(xyz_pcd_max - xyz_pcd_min) / 5.
            if pcd_mask is not None:
                normal_pcd = normal_pcd[pcd_mask]

            normal_pcd_end = xyz_pcd[::subsample_normal_rate_x] + normal_pcd[::subsample_normal_rate_x] * normal_length
            normals = o3d.geometry.LineSet()
            normals.points = o3d.utility.Vector3dVector(np.vstack((xyz_pcd[::subsample_normal_rate_x], normal_pcd_end)))
            normals.colors = o3d.utility.Vector3dVector([[0.5, 0.5, 0.5] for _ in range(normal_pcd_end.shape[0])])
            normals.lines = o3d.utility.Vector2iVector([[_, _+normal_pcd_end.shape[0]] for _ in range(normal_pcd_end.shape[0])])

            geometry_list.append(normals)

        return geometry_list

    def collect_lighting_SG(self, lighting_params: dict={}):
        '''
        the classroom scene: images/demo_lighting_SG_o3d.png
        '''
        assert self.os.if_has_lighting_SG
        if_use_mi_geometry = lighting_params.get('if_use_mi_geometry', True)

        subsample_lighting_pts_rate = lighting_params.get('subsample_lighting_pts_rate', 1)
        lighting_SG_fused_dict = self.os._fuse_3D_lighting(subsample_rate_pts=subsample_lighting_pts_rate, if_use_mi_geometry=if_use_mi_geometry, lighting_source='lighting_SG')

        geometry_list = self.process_lighting(lighting_SG_fused_dict, lighting_params=lighting_params, lighting_source='lighting_SG')
        return geometry_list

    def collect_lighting_envmap(self, lighting_params: dict={}):
        '''
        the classroom scene: images/demo_lighting_envmap_o3d.png
        '''
        assert self.os.if_has_lighting_envmap
        if_use_mi_geometry = lighting_params.get('if_use_mi_geometry', True)

        subsample_lighting_pts_rate = lighting_params.get('subsample_lighting_pts_rate', 1)
        subsample_lighting_wi_rate = lighting_params.get('subsample_lighting_wi_rate', 1)
        if_use_loaded_envmap_position = lighting_params.get('if_use_loaded_envmap_position', False)
        lighting_envmap_fused_dict = self.os._fuse_3D_lighting(
            subsample_rate_pts=subsample_lighting_pts_rate, subsample_rate_wi=subsample_lighting_wi_rate, 
            if_use_mi_geometry=if_use_mi_geometry, if_use_loaded_envmap_position=if_use_loaded_envmap_position, lighting_source='lighting_envmap')

        lighting_if_show_hemisphere = lighting_params.get('lighting_if_show_hemisphere', False)
        if lighting_if_show_hemisphere:
            lighting_envmap_fused_dict['weight'] = np.ones_like(lighting_envmap_fused_dict['weight'], dtype=np.float32)

        geometry_list = self.process_lighting(lighting_envmap_fused_dict, lighting_params=lighting_params, lighting_source='lighting_envmap', lighting_color=[1., 0., 1.]) # pink
        return geometry_list

    def process_lighting(self, lighting_fused_dict: dict, lighting_source: str, lighting_params: dict={}, lighting_color=[1., 0., 0.], if_X_multiplied: bool=False, if_use_pts_end: bool=False):
        '''
        if_X_multiplied: True if X_global_lighting is already multiplied by wi_num
        '''
        assert lighting_source in ['lighting_SG', 'lighting_envmap'] # not supporting 'lighting_sampled' yet

        geometry_list = []

        lighting_scale = lighting_params.get('lighting_scale', 1.) # if autoscale, act as extra scale
        lighting_keep_ratio = lighting_params.get('lighting_keep_ratio', 0.05)
        lighting_further_clip_ratio = lighting_params.get('lighting_further_clip_ratio', 0.1)
        lighting_autoscale = lighting_params.get('lighting_autoscale', True)

        lighting_if_show_hemisphere = lighting_params.get('lighting_if_show_hemisphere', False) # images/demo_lighting_envmap_hemisphere_axes_o3d.png
        if lighting_if_show_hemisphere:
            lighting_keep_ratio = 0.
            lighting_further_clip_ratio = 0.
            lighting_autoscale = False
            lighting_scale = 10.

            # show local xyz axis for frame 0
            from utils_OR.utils_OR_lighting import convert_lighting_axis_local_to_global_np
            _idx = 0
            assert self.os.if_has_mitsuba_all
            normal_list = self.os.mi_normal_list
            lighting_local_xyz = np.tile(np.eye(3, dtype=np.float32)[np.newaxis, np.newaxis, ...], (self.os.H, self.os.W, 1, 1))
            lighting_global_xyz = convert_lighting_axis_local_to_global_np(lighting_local_xyz, self.os.pose_list[_idx], normal_list[_idx])[::16, ::16]
            lighting_global_pts = np.tile(np.expand_dims(self.os.mi_pts_list[_idx], 2), (1, 1, 3, 1))[::16, ::16]
            assert lighting_global_xyz.shape == lighting_global_pts.shape
            for _axis_idx, _axis_color in enumerate([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]):
                lighting_axes = o3d.geometry.LineSet()
                _N_pts = prod(lighting_global_xyz.shape[:2])
                _lighting_global_pts = lighting_global_pts[:, :, _axis_idx].reshape(-1, 3)
                _lighting_global_xyz = lighting_global_xyz[:, :, _axis_idx].reshape(-1, 3)
                lighting_axes.points = o3d.utility.Vector3dVector(np.vstack((_lighting_global_pts, _lighting_global_pts+_lighting_global_xyz*lighting_scale/50.)))
                lighting_axes.colors = o3d.utility.Vector3dVector([_axis_color for _ in range(_N_pts)])
                lighting_axes.lines = o3d.utility.Vector2iVector([[_, _+_N_pts] for _ in range(_N_pts)])
                geometry_list += [lighting_axes]

        # if lighting_autoscale:
        #     lighting_scale = 1.

        # if lighting_source == 'lighting_SG':
        #     xyz_pcd, normal_pcd, axis_pcd, weight_pcd, lamb_SG_pcd = get_list_of_keys(lighting_fused_dict, ['pts_global_lighting', 'normal_global_lighting', 'axis', 'weight_SG', 'lamb_SG'])
        # if lighting_source == 'lighting_envmap':
        xyz_pcd, axis_pcd, weight_pcd = get_list_of_keys(lighting_fused_dict, ['pts_global_lighting', 'axis', 'weight']) # [TODO] lamb is not visualized for now

        wi_num = axis_pcd.shape[1] # num of rays per-point

        if not if_X_multiplied:
            xyz_pcd = np.tile(np.expand_dims(xyz_pcd, 1), (1, wi_num, 1)).reshape((-1, 3))
        length_pcd = np.linalg.norm(weight_pcd.reshape((-1, 3)), axis=1, keepdims=True) / 50.

        # keep only SGs pointing towards outside of the surface
        axis_mask = np.ones_like(length_pcd.squeeze()).astype(bool)
        if 'normal_global_lighting' in lighting_fused_dict: 
            normal_pcd = lighting_fused_dict['normal_global_lighting']
            axis_mask = np.sum(np.repeat(np.expand_dims(normal_pcd, 1), wi_num, axis=1) * axis_pcd, axis=2) > 0.
        axis_mask =  axis_mask.reshape(-1) # (N*12,)

        if lighting_keep_ratio > 0.:
            assert lighting_keep_ratio > 0. and lighting_keep_ratio <= 1.
            percentile = np.percentile(length_pcd.flatten(), 100-lighting_keep_ratio*100)
            axis_mask = np.logical_and(axis_mask, length_pcd.flatten() > percentile) # (N*12,)

        if lighting_further_clip_ratio > 0.: # within **keeped** SGs after lighting_keep_ratio
            assert lighting_further_clip_ratio > 0. and lighting_further_clip_ratio <= 1.
            percentile = np.percentile(length_pcd[axis_mask].flatten(), 100-lighting_further_clip_ratio*100)
            length_pcd[length_pcd > percentile] = percentile

        if lighting_autoscale:
            # ic(np.amax(length_pcd))
            ceiling_y, floor_y = np.amax(xyz_pcd[:, 2]), np.amin(xyz_pcd[:, 2])
            length_pcd = length_pcd / np.amax(length_pcd) * abs(ceiling_y - floor_y)
            # ic(np.amax(length_pcd))
        
        length_pcd *= lighting_scale
        axis_pcd_end = xyz_pcd + length_pcd * axis_pcd.reshape((-1, 3))
        if if_use_pts_end:
            assert 'pts_end' in lighting_fused_dict
            axis_pcd_end = lighting_fused_dict['pts_end']

        xyz_pcd, axis_pcd_end = xyz_pcd[axis_mask], axis_pcd_end[axis_mask]
        N_pts = xyz_pcd.shape[0]
        # print('Showing lighting for %d points...'%N_pts)
        
        lighting_arrows = o3d.geometry.LineSet()
        lighting_arrows.points = o3d.utility.Vector3dVector(np.vstack((xyz_pcd, axis_pcd_end)))
        lighting_arrows.colors = o3d.utility.Vector3dVector([lighting_color for _ in range(N_pts)])
        lighting_arrows.lines = o3d.utility.Vector2iVector([[_, _+N_pts] for _ in range(N_pts)])
        
        geometry_list += [lighting_arrows]

        return geometry_list

    def collect_layout(self, layout_params: dict={}):
        '''
        images/demo_layout_o3d.png
        '''
        assert self.os.if_has_layout
        return_list = []

        if hasattr(self.os, 'layout_mesh_transformed'):
            layout_mesh = self.os.layout_mesh_transformed.as_open3d
            layout_mesh.compute_vertex_normals()
            return_list.append([layout_mesh, 'layout'])

        layout_bbox_3d = self.os.layout_box_3d_transformed
        layout_bbox_pcd = o3d.geometry.LineSet()
        layout_bbox_pcd.points = o3d.utility.Vector3dVector(layout_bbox_3d)
        layout_bbox_pcd.colors = o3d.utility.Vector3dVector([[1,0,0] for i in range(12)])
        layout_bbox_pcd.lines = o3d.utility.Vector2iVector([[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]])

        return return_list + [layout_bbox_pcd]

    def collect_shapes(self, shapes_params: dict={}):
        '''
        collect shapes and bboxes for objs + emitters (shapes)

        to visualize emitter properties, go to collect_emitters

        images/demo_shapes_o3d.png
        images/demo_shapes_o3d_2_shader.png

        images/demo_shapes_emitter_o3d.png
        '''
        assert self.os.if_has_shapes

        if_obj_meshes = shapes_params.get('if_meshes', True) and self.os.shape_params_dict.get('if_load_obj_mesh', False)
        if_emitter_meshes = shapes_params.get('if_meshes', True) and self.os.shape_params_dict.get('if_load_emitter_mesh', False)
        if_ceiling = shapes_params.get('if_ceiling', False)
        if_walls = shapes_params.get('if_walls', False)

        self.os.load_colors()

        geometry_list = []

        # emitters_obj_random_id_list = [shape['random_id'] for shape in self.os.shape_list_valid if shape['if_in_emitter_dict']]

        for shape_idx, (shape_dict, vertices, faces, bverts) in tqdm(enumerate(zip(
            self.os.shape_list_valid, 
            self.os.vertices_list, 
            self.os.faces_list, 
            self.os.bverts_list, 
        ))):

            if_emitter = shape_dict['if_in_emitter_dict']
            cat_name = 'N/A'; cat_id = -1

            # if np.amax(bverts[:, 1]) <= np.amin(bverts[:, 1]):
            #     obj_color = [0., 0., 0.] # black for invalid objects
            #     cat_name = 'INVALID'
            #     import ipdb; ipdb.set_trace()
            #     continue
            # else:
            obj_path = shape_dict['filename']
            if 'uv_mapped.obj' in obj_path and not(if_ceiling and if_walls):
                continue # skipping layout as an object
            # if shape_dict['random_id'] in emitters_obj_random_id_list and not if_emitter: # SKIP emitters
            #     continue # skip shape if it is also in the list as an emitter (so that we don't create two shapes for one emitters)
            if not if_walls and shape_dict.get('is_wall', False): continue
            if not if_ceiling and shape_dict.get('is_ceiling', False): continue

            if self.os.if_loaded_colors:
                cat_id_str = str(obj_path).split('/')[-3]
                assert cat_id_str in self.os.OR_mapping_cat_str_to_id_name_dict, 'not valid cat_id_str: %s; %s'%(cat_id_str, obj_path)
                cat_id, cat_name = self.os.OR_mapping_cat_str_to_id_name_dict[cat_id_str]
                obj_color = self.os.OR_mapping_id_to_color_dict[cat_id]
                obj_color = [float(x)/255. for x in obj_color]
                linestyle = '-'
                linewidth = 1
                if if_emitter:
                    linewidth = 3
                    linestyle = '--'
                    obj_color = [1., np.random.random()*0.5, np.random.random()*0.5] # red-ish for emitters
                    '''
                    images/OR42_color_mapping_light.png
                    '''
            else:
                obj_color = [0.7, 0.7, 0.7]
                if if_emitter:
                    obj_color = [1., np.random.random()*0.5, np.random.random()*0.5] # red-ish for emitters
                    # print(yellow(str(obj_color)), shape_dict['random_id'])


            # trimesh.repair.fill_holes(shape_mesh)
            # trimesh.repair.fix_winding(shape_mesh)
            # trimesh.repair.fix_inversion(shape_mesh)
            # trimesh.repair.fix_normals(shape_mesh)
            shape_bbox = o3d.geometry.LineSet()
            shape_bbox.points = o3d.utility.Vector3dVector(bverts)
            shape_bbox.colors = o3d.utility.Vector3dVector([obj_color if not if_emitter else [0., 0., 0.] for i in range(12)]) # black for emitters
            shape_bbox.lines = o3d.utility.Vector2iVector([[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]])
            geometry_list.append(shape_bbox)

            '''
            [optional] load mashes & labels
            '''
            if_mesh = if_obj_meshes if not if_emitter else if_emitter_meshes
            if if_mesh:
                assert np.amax(faces-1) < vertices.shape[0]
                shape_mesh = trimesh.Trimesh(vertices=vertices, faces=faces-1) # [IMPORTANT] faces-1 because Trimesh faces are 0-based
                shape_mesh = shape_mesh.as_open3d

                shape_mesh.paint_uniform_color(obj_color)
                shape_mesh.compute_vertex_normals()
                shape_mesh.compute_triangle_normals()
                geometry_list.append([shape_mesh, 'shape_emitter_'+shape_dict['random_id'] if if_emitter else 'shape_obj_'+shape_dict['random_id']])

            # print('[collect_shapes] --', if_emitter, shape_idx, obj_path, cat_name, cat_id, shape_dict['random_id'])

            # shape_label = o3d.visualization.gui.Label3D([0., 0., 0.], np.mean(bverts, axis=0).reshape((3, 1)), cat_name)
            # geometry_list.append(shape_label)
            # self.vis.add_3d_label(np.mean(bverts, axis=0), cat_name)

            if_labels = shapes_params.get('if_labels', True)
            if if_labels:
                pcd_10 = text_3d(
                    cat_name, 
                    # pos=np.mean(bverts, axis=0).tolist(), 
                    # pos=[np.mean(bverts[:, 0], axis=0), np.amax(bverts[:, 1], axis=0)+0.2*(np.amax(bverts[:, 1], axis=0)-np.amin(bverts[:, 1], axis=0)), np.mean(bverts[:, 2], axis=0)], 
                    pos=[np.mean(bverts[:, 0], axis=0), np.amax(bverts[:, 1], axis=0)+0.05*(np.amax(bverts[:, 1], axis=0)-np.amin(bverts[:, 1], axis=0)), np.mean(bverts[:, 2], axis=0)], 
                    direction=(0., 0., 1), 
                    degree=270., 
                    font_size=250, density=1, text_color=tuple([int(_*255) for _ in obj_color]))
                # pcd_10 = text_3d(cat_name, pos=np.mean(bverts, axis=0).tolist(), font_size=100, density=10)
                geometry_list.append(pcd_10)

        if_dump_mesh = shapes_params.get('if_dump_mesh', False)
        if if_dump_mesh:
            num_vertices = 0
            f_list = []
            for vertices, faces in zip(self.os.vertices_list, self.os.faces_list):
                f_list.append(copy.deepcopy(faces + num_vertices))
                num_vertices += vertices.shape[0]
            f = np.array(self.os.layout_mesh_ori.faces)
            f_list.append(f+num_vertices)
            v = np.array(self.os.layout_mesh_ori.vertices)
            v_list = copy.deepcopy(self.os.vertices_list) + [v]
            writeMesh('./tmp_mesh.obj', np.vstack(v_list), np.vstack(f_list))

        if_voxel_volume = shapes_params.get('if_voxel_volume', False)
        if if_voxel_volume:
            '''
            show voxels of unit sizes along object/bbox vertices; just to show the scale of the scene: images/demo_shapes_voxel_o3d.png
            '''
            # voxel_grid = o3d.geometry.VoxelGrid.create_dense(xyz_min.reshape((3, 1)), np.ones((3, 1)), 1., xyz_max[0]-xyz_min[0], xyz_max[1]-xyz_min[1], xyz_max[2]-xyz_min[2])
            bverts_all = np.vstack(self.os.bverts_list)
            pcd_bverts_all = o3d.geometry.PointCloud()
            pcd_bverts_all.points = o3d.utility.Vector3dVector(bverts_all)
            pcd_bverts_all.colors = o3d.utility.Vector3dVector(np.random.uniform(0, 1, size=(bverts_all.shape[0], 3)))

            voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd_bverts_all, voxel_size=1.)
            geometry_list.append(voxel_grid)

        if_sampled_pts = shapes_params.get('if_sampled_pts', False)
        if if_sampled_pts and hasattr(self.os, 'sample_pts_list'):
            sample_pts = np.concatenate(self.os.sample_pts_list)
            self.add_extra_geometry([
                ('pts', {'pts': sample_pts, }),
            ]) 

        return geometry_list

    def collect_emitters(self, emitter_params: dict={}):
        '''
        emitter PHYSICAL PROPERTIES (emitter shapes visualized by self.collect_shapes)
        images/demo_emitters_o3d.png
        images/demo_envmap_o3d.png # added envmap hemisphere
        '''
        assert self.os.if_has_shapes

        if_half_envmap = emitter_params.get('if_half_envmap', True)
        scale_SG_length = emitter_params.get('scale_SG_length', 2.)

        self.os.load_colors()

        geometry_list = []

        for shape_idx, (shape_dict, bverts) in enumerate(zip(
            self.os.shape_list_valid, 
            self.os.bverts_list, 
        )):

            if not shape_dict['if_in_emitter_dict']: continue # EMITTERS only

            '''
            WINDOWS
            '''
            if shape_dict['emitter_prop']['obj_type'] == 'window': # if window
                light_center = np.mean(bverts, 0).flatten()
                label_SG_list = ['', 'Sky', 'Grd']
                for label_SG, color, scale in zip(label_SG_list, ['k', 'b', 'g'], [1*scale_SG_length, 0.5*scale_SG_length, 0.5*scale_SG_length]):
                    light_axis = np.asarray(shape_dict['emitter_prop']['axis%s_world'%label_SG]).flatten()
                
                    light_axis_world = np.asarray(light_axis).reshape(3,) # transform back to world system
                    light_axis_world = light_axis_world / np.linalg.norm(light_axis_world)
                    # light_axis_world = np.array([1., 0., 0.])
                    light_axis_end = np.asarray(light_center).reshape(3,) + light_axis_world * np.log(shape_dict['emitter_prop']['intensity'+label_SG])
                    light_axis_end = light_axis_end.flatten()

                    a_light = get_arrow_o3d(light_center, light_axis_end, scale=scale, color=color)
                    geometry_list.append(a_light)

                if if_half_envmap:
                    env_map_path = shape_dict['emitter_prop']['envMapPath']
                    im_envmap_ori = load_HDR(Path(env_map_path))
                    im_envmap_ori_SDR, im_envmap_ori_scale = to_nonHDR(im_envmap_ori)

                    sphere_envmap = get_sphere(hemisphere_normal=shape_dict['emitter_prop']['box3D_world']['zAxis'].reshape((3,)), envmap=im_envmap_ori_SDR)
                    geometry_list.append([sphere_envmap, 'envmap'])

        '''
        LAMPS samples as area lights: images/demo_emitter_o3d_sampling.png
        '''
        if_sampling_emitter = emitter_params.get('if_sampling_emitter', True)
        if if_sampling_emitter:
            max_plate = emitter_params.get('max_plate', 64)
            radiance_scale = emitter_params.get('radiance_scale', 1.)
            from lib.utils_OR.utils_OR_emitter import sample_mesh_emitter
            emitter_dict = {'lamp': self.os.lamp_list, 'window': self.os.window_list}
            for emitter_type in ['lamp']:
                for emitter_index in range(len(emitter_dict[emitter_type])):
                    lpts_dict = sample_mesh_emitter(emitter_type, emitter_index=emitter_index, emitter_dict=emitter_dict, max_plate=max_plate)
                    # for lpts, lpts_normal, lpts_intensity in zip(lpts_dict['lpts'], lpts_dict['lpts_normal'], lpts_dict['lpts_intensity']):
                    o_ = lpts_dict['lpts']
                    d_ = lpts_dict['lpts_normal'] / (np.linalg.norm(lpts_dict['lpts_normal'], axis=-1, keepdims=True)+1e-5)
                    lpts_end = o_ + d_ * np.linalg.norm(lpts_dict['lpts_intensity'], axis=-1) * radiance_scale
                    print(white_red('GT intensity'), lpts_dict['lpts_intensity'], np.linalg.norm(lpts_dict['lpts_intensity'], axis=-1))
                    emitter_rays = o3d.geometry.LineSet()
                    emitter_rays.points = o3d.utility.Vector3dVector(np.vstack((lpts_dict['lpts'], lpts_end)))
                    emitter_rays.colors = o3d.utility.Vector3dVector([[1., 0., 0.]]*lpts_dict['lpts'].shape[0]) # RED for GT
                    emitter_rays.lines = o3d.utility.Vector2iVector([[_, _+lpts_dict['lpts'].shape[0]] for _ in range(lpts_dict['lpts'].shape[0])])
                    geometry_list.append([emitter_rays, 'emitter_rays'])

        return geometry_list

    def collect_mi(self, mi_params: dict={}):
        '''
        images/demo_mi_o3d_1.png
        images/demo_mi_o3d_2.png
        '''
        assert self.os.if_has_mitsuba_scene

        geometry_list = []

        if_cam_rays = mi_params.get('if_cam_rays', True) # if show per-pixel rays
        cam_rays_if_pts = mi_params.get('cam_rays_if_pts', True) # if cam rays end in surface intersections
        if cam_rays_if_pts:
            assert self.os.if_has_mitsuba_rays_pts
        cam_rays_subsample = mi_params.get('cam_rays_subsample', 10)

        if if_cam_rays: 
            for frame_idx, (rays_o, rays_d, _) in enumerate(self.os.cam_rays_list[0:1]): # show only first frame
                rays_of_a_view = o3d.geometry.LineSet()

                if cam_rays_if_pts:
                    ret = self.os.mi_rays_ret_list[frame_idx]
                    rays_t_flatten = ret.t.numpy()[::cam_rays_subsample][:, np.newaxis]
                    rays_t_flatten[rays_t_flatten==np.inf] = 0.
                else:
                    rays_t_flatten = np.ones((rays_o.shape[0], 1), dtype=np.float32)

                rays_o_flatten, rays_d_flatten = rays_o.reshape(-1, 3)[::cam_rays_subsample], rays_d.reshape(-1, 3)[::cam_rays_subsample]

                rays_end_flatten = rays_o_flatten + rays_d_flatten * rays_t_flatten
                rays_of_a_view.points = o3d.utility.Vector3dVector(np.vstack((rays_o_flatten, rays_end_flatten)))
                rays_of_a_view.colors = o3d.utility.Vector3dVector([[0.3, 0.3, 0.3]]*rays_o_flatten.shape[0])
                rays_of_a_view.lines = o3d.utility.Vector2iVector([[_, _+rays_o_flatten.shape[0]] for _ in range(rays_o_flatten.shape[0])])
                geometry_list.append(rays_of_a_view)

                pcd_rays_end = o3d.geometry.PointCloud()
                pcd_rays_end.points = o3d.utility.Vector3dVector(rays_end_flatten)
                pcd_rays_end.colors = o3d.utility.Vector3dVector([[0., 0., 0.]]*rays_o_flatten.shape[0])

                geometry_list.append(pcd_rays_end)

        '''
        if show per-pixel pts (see: no floating points): 
        images/demo_mitsuba_ret_normals.png
        images/demo_mitsuba_ret_pts_1.png
        images/demo_mitsuba_ret_pts_2.png
        '''
        if_pts = mi_params.get('if_pts', True)
        pts_subsample = mi_params.get('pts_subsample', 10)
        if_pts_colorize_rgb = mi_params.get('if_pts_colorize_rgb', True)
        if_ceiling = mi_params.get('if_ceiling', True)
        if_walls = mi_params.get('if_walls', True)

        if_normal = mi_params.get('if_normal', True)
        normal_subsample = mi_params.get('normal_subsample', 10)
        normal_scale = mi_params.get('normal_scale', 0.2)

        if if_pts:
            for frame_idx, (mi_depth, mi_normals, mi_pts) in enumerate(zip(self.os.mi_depth_list, self.os.mi_normal_global_list, self.os.mi_pts_list)):
                # assert np.sum(mi_depth==np.inf)==0
                mi_pts_ = mi_pts[mi_depth!=np.inf, :][::pts_subsample] # [H, W, 3] -> [N', 3]
                pcd_pts = o3d.geometry.PointCloud()
                if self.mi_pcd_color_list is None:
                    if if_pts_colorize_rgb and self.os.if_has_im_sdr:
                        mi_color_ = self.os.im_sdr_list[frame_idx][mi_depth!=np.inf, :][::pts_subsample] # [H, W, 3] -> [N', 3]
                    else:
                        mi_color_ = np.array([[0.7, 0.7, 0.7]]*mi_pts_.shape[0])
                else:
                    assert isinstance(self.mi_pcd_color_list, list)
                    mi_color_ = self.mi_pcd_color_list[frame_idx][::pts_subsample]

                if not if_ceiling:
                    mi_pts_, mi_color_ = remove_ceiling(mi_pts_, mi_color_, if_debug_info=self.if_debug_info)
                if not if_walls:
                    assert self.os.if_has_layout
                    layout_bbox_3d = self.os.layout_box_3d_transformed
                    mi_pts_, mi_color_ = remove_walls(layout_bbox_3d, mi_pts_, mi_color_, if_debug_info=self.if_debug_info)

                assert mi_color_.shape[0] == mi_pts_.shape[0]
                pcd_pts.points = o3d.utility.Vector3dVector(mi_pts_)
                pcd_pts.colors = o3d.utility.Vector3dVector(mi_color_)
                geometry_list.append(pcd_pts)

                if if_normal:
                    pcd_normals = o3d.geometry.LineSet()
                    mi_pts_ = mi_pts[mi_depth!=np.inf, :][::normal_subsample] # [H, W, 3] -> [N', 3]
                    mi_normals_end = mi_pts_ + normal_scale * mi_normals[mi_depth!=np.inf, :][::normal_subsample] # [H, W, 3] -> [N', 3]
                    pcd_normals.points = o3d.utility.Vector3dVector(np.vstack((mi_pts_, mi_normals_end)))
                    pcd_normals.colors = o3d.utility.Vector3dVector([[0., 0., 0.] for _ in range(mi_normals_end.shape[0])])
                    pcd_normals.lines = o3d.utility.Vector2iVector([[_, _+mi_normals_end.shape[0]] for _ in range(mi_normals_end.shape[0])])
                    geometry_list.append(pcd_normals)

        return geometry_list

    def set_mi_pcd_color_from_input(self, input_colors_tuple: tuple=()):
        assert input_colors_tuple != ()
        color_tensor_list, color_type = input_colors_tuple
        assert len(color_tensor_list) == self.os.frame_num
        assert color_type in ['dist', 'mask']

        if color_type == 'mask':
            self.mi_pcd_color_list = [color_map_color(color_tensor) for color_tensor in color_tensor_list]
        if color_type == 'dist':
            self.mi_pcd_color_list = [color_map_color(color_tensor, vmin=np.amin(color_tensor), vmax=np.amax(color_tensor)) for color_tensor in color_tensor_list]

        assert self.mi_pcd_color_list is not None
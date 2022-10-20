import numpy as np
from tqdm import tqdm
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import open3d.visualization as vis

import trimesh
from sympy import Point3D, Line3D, Plane, sympify, Rational
from copy import deepcopy
import torch
from pathlib import Path
import copy

from lib.class_openroomsScene2D import openroomsScene2D
from lib.class_openroomsScene3D import openroomsScene3D

from lib.utils_misc import blue_text, get_list_of_keys, green, white_blue, red, check_list_of_tensors_size
from lib.utils_o3d import text_3d, get_arrow_o3d
from lib.utils_OR.utils_OR_xml import gen_random_str
from lib.utils_io import load_HDR, to_nonHDR
from lib.utils_OR.utils_OR_mesh import writeMesh


aabb_01 = np.array([[0, 0, 0],
                    [0, 0, 1],
                    [0, 1, 1],
                    [0, 1, 0],
                    [1, 0, 0],
                    [1, 0, 1],
                    [1, 1, 1],
                    [1, 1, 0]])

class visualizer_openroomsScene_o3d(object):
    '''
    A class used to **visualize** OpenRooms (public/public-re versions) scene contents (2D/2.5D per-pixel DENSE properties for inverse rendering + 3D semantics).
    '''
    def __init__(
        self, 
        openrooms_scene, 
        modality_list: list, 
    ):

        assert type(openrooms_scene) in [openroomsScene2D, openroomsScene3D], '[visualizer_openroomsScene] has to take an object of openroomsScene or openroomsScene3D!'

        self.openrooms_scene = openrooms_scene

        self.modality_list = modality_list
        for _ in self.modality_list:
            assert _ in ['dense_geo', 'cameras', 'lighting_SG', 'layout', 'shapes', 'emitters']

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


    def get_sphere(self, scale=5., resolution=200, hemisphere_normal=None, envmap=None):
        sphere = o3d.geometry.TriangleMesh.create_sphere(scale, resolution=resolution)
        sphere.compute_vertex_normals()
        sphere.compute_triangle_normals()
        sphere.normalize_normals()

        if hemisphere_normal is not None:
            assert hemisphere_normal.shape==(3,)
            triangles = np.asarray(sphere.triangles).copy()
            triangle_normals = np.asarray(sphere.triangle_normals).copy()
            vertices = np.asarray(sphere.vertices).copy()
            mask = np.sum(triangle_normals * hemisphere_normal.reshape(1, 3), axis=1) > 0
            triangles_new = triangles[mask]

            sphere = deepcopy(sphere)
            sphere.triangles = o3d.utility.Vector3iVector(triangles_new)
            sphere.vertices = o3d.utility.Vector3dVector(vertices)
            sphere.compute_vertex_normals()
            sphere.compute_triangle_normals()

            if envmap is None:
                sphere.vertex_colors = o3d.utility.Vector3dVector(np.random.rand(vertices.shape[0], 3))
            else:
                '''
                sample global envmap with sphere normals
                '''
                vertex_normals = np.asarray(sphere.vertex_normals).copy()
                # /home/ruizhu/Documents/Projects/semanticInverse/train/models_def/models_layout_emitter_lightAccu.py -> sample_envmap()
                vertex_normals_SG = np.stack((vertex_normals[:, 2], -vertex_normals[:, 0], vertex_normals[:, 1]), axis=-1)
                cos_theta = vertex_normals_SG[:, 2]
                theta_SG = np.arccos(cos_theta) # [0, pi]
                cos_phi = vertex_normals_SG[:, 0] / np.sin(theta_SG)
                sin_phi = vertex_normals_SG[:, 1] / np.sin(theta_SG)
                phi_SG = np.arctan2(sin_phi, cos_phi)
                uu_normalized = phi_SG / np.pi # pixel center when align_corners = False; -pi~pi -> -1~1; (N,)
                vv_normalized = theta_SG * 2. / np.pi -1. # pixel center when align_corners = False; 0~pi -> -1~1; (N,)
                uv_normalized = np.stack([uu_normalized, vv_normalized], axis=-1) # (N, 2)
                uv_normalized_torch = torch.from_numpy(uv_normalized).unsqueeze(0).unsqueeze(0).float() # (1, 1, N, 2)
                envmap_torch = torch.from_numpy(envmap).permute(2, 0, 1).unsqueeze(0).float() # (1, 3, H, W)
                sampled_envmap_torch = torch.nn.functional.grid_sample(envmap_torch, uv_normalized_torch, mode='bilinear', align_corners=True) # (1, 3, 1, N)
                sampled_envmap = sampled_envmap_torch.squeeze().numpy().transpose() # (N, 3)
                sphere.vertex_colors = o3d.utility.Vector3dVector(sampled_envmap) # [TODO] not sure how to set triangle colors... the Open3D documentation is pretty confusing and actually does not work... http://www.open3d.org/docs/release/python_api/open3d.t.geometry.TriangleMesh.html

        # self.vis.add_geometry(sphere)
        return sphere

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
            self.modality_list, 
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
        self.w = self.vis.create_window()
        self.opt = self.vis.get_render_option()
        self.opt.background_color = np.asarray([1., 1., 1.])

    def load_o3d_geometry_list(
        self, 
        modality_list: list, 
        cam_params: dict = {}, 
        dense_geo_params: dict = {}, 
        lighting_SG_params: dict = {}, 
        layout_params: dict = {}, 
        shapes_params: dict = {}, 
        emitters_params: dict = {}, 
    ):
        
        o3d_geometry_list = [o3d.geometry.TriangleMesh.create_coordinate_frame()]

        if 'cameras' in modality_list:
            o3d_geometry_list += self.collect_cameras(cam_params)

        if 'dense_geo' in modality_list:
            o3d_geometry_list += self.collect_dense_geo(
                dense_geo_params
            )
        
        if 'lighting_SG' in modality_list:
            '''
            the classroom scene: images/demo_lighting_SG_o3d.png
            '''
            o3d_geometry_list += self.collect_lighting_SG(
                lighting_SG_params
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
                emitters_params
            )

        return o3d_geometry_list
        
    def collect_cameras(self, cam_params: dict={}):
        assert self.openrooms_scene.if_has_cameras

        subsample_cam_rate = cam_params.get('subsample_cam_rate', 1)

        pose_list = self.openrooms_scene.pose_list
        origin_lookat_up_list = self.openrooms_scene.origin_lookat_up_list
        cam_frustrm_list = []
        cam_axes_list = []
        cam_center_list = []
        cam_o_d_list = []
        cam_list = []

        for cam_idx, (pose, origin_lookat_up) in enumerate(zip(pose_list, origin_lookat_up_list)):
            origin, lookat, up = origin_lookat_up[0].flatten(), origin_lookat_up[1].flatten(), origin_lookat_up[2].flatten()
            
            cam_o = origin.flatten()
            cam_d = lookat.flatten()
            cam_d = cam_d / np.linalg.norm(cam_d)
            cam_o_d_list.append((cam_o, cam_d))

            right = np.cross(lookat, up).flatten()
            cam = [cam_o]
            frustum_scale = 0.5
            cam += [cam_o + cam_d * frustum_scale + up * 0.5 * frustum_scale - right * 0.5 * frustum_scale]
            cam += [cam_o + cam_d * frustum_scale + up * 0.5 * frustum_scale + right * 0.5 * frustum_scale]
            cam += [cam_o + cam_d * frustum_scale - up * 0.5 * frustum_scale - right * 0.5 * frustum_scale]
            cam += [cam_o + cam_d * frustum_scale - up * 0.5 * frustum_scale + right * 0.5 * frustum_scale]
            cam = np.array(cam)
            cam_list.append(cam)

        c2w_list = pose_list

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

        if subsample_cam_rate != 1: # subsample camera poses if too many
            cam_frustrm_list = cam_axes_list[::subsample_cam_rate] + [cam_axes_list[-1]]

        geometry_list = [
            *cam_frustrm_list, *cam_center_list, # pcd + cams
        ]

        return geometry_list

    def collect_dense_geo(self, dense_geo_params: dict={}):

        assert self.openrooms_scene.if_has_im_sdr and self.openrooms_scene.if_has_dense_geo

        subsample_pcd_rate = dense_geo_params.get('subsample_pcd_rate', 10)
        if_ceiling = dense_geo_params.get('if_ceiling', False)
        if_walls = dense_geo_params.get('if_walls', False)
        if_normal = dense_geo_params.get('if_normal', False)
        subsample_normal_rate_x = dense_geo_params.get('subsample_normal_rate_x', 5) # subsample_normal_rate_x is multiplicative to subsample_pcd_rate

        geo_fused_dict, _, _ = self.openrooms_scene._fuse_3D_geometry(subsample_rate=subsample_pcd_rate)

        xyz_pcd, rgb_pcd, normal_pcd = get_list_of_keys(geo_fused_dict, ['X', 'rgb', 'normal'])
        # N_pcd = xyz_pcd.shape[0]

        xyz_pcd_max = np.amax(xyz_pcd, axis=0)
        xyz_pcd_min = np.amin(xyz_pcd, axis=0)

        pcd_mask = None

        if not if_ceiling:
            # remove ceiling points
            ceiling_y = np.amax(xyz_pcd[:, 1]) # y axis is up
            pcd_mask = xyz_pcd[:, 1] < (ceiling_y*0.95)
            xyz_pcd = xyz_pcd[pcd_mask]
            rgb_pcd = rgb_pcd[pcd_mask]
            print('Removed points close to ceiling... percentage: %.2f'%(np.sum(pcd_mask)*100./xyz_pcd.shape[0]))

        if not if_walls:
            assert self.openrooms_scene.if_has_layout
            layout_bbox_3d = self.openrooms_scene.layout_box_3d_transformed
            dists_all = np.zeros((xyz_pcd.shape[0]), dtype=np.float32) + np.inf

            for wall_v_idxes in [(4, 0, 5), (6, 5, 2), (7, 6, 3), (7, 3, 4)]:
                plane_normal = np.cross(layout_bbox_3d[wall_v_idxes[1]]-layout_bbox_3d[wall_v_idxes[0]], layout_bbox_3d[wall_v_idxes[2]]-layout_bbox_3d[wall_v_idxes[0]])
                plane_normal = plane_normal / np.linalg.norm(plane_normal)
                
                l1 = xyz_pcd - layout_bbox_3d[wall_v_idxes[0]].reshape(1, 3)
                dist_ = np.sum(l1 * plane_normal.reshape(1, 3), axis=1)
                dists_all = np.minimum(dist_, dists_all)

            layout_sides = np.vstack((layout_bbox_3d[1]-layout_bbox_3d[0], layout_bbox_3d[3]-layout_bbox_3d[0], layout_bbox_3d[4]-layout_bbox_3d[0]))
            layout_dimensions = np.linalg.norm(layout_sides, axis=1)
            print(layout_dimensions)

            pcd_mask = dists_all > np.amin(layout_dimensions)*0.05 # threshold is 5% of the shortest room dimension
            xyz_pcd = xyz_pcd[pcd_mask]
            rgb_pcd = rgb_pcd[pcd_mask]
            print('Removed points close to walls... percentage: %.2f'%(np.sum(pcd_mask)*100./xyz_pcd.shape[0]))

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz_pcd)
        pcd.colors = o3d.utility.Vector3dVector(rgb_pcd)

        out_bbox_pcd = o3d.geometry.LineSet()
        out_bbox_pcd.points = o3d.utility.Vector3dVector(xyz_pcd_min + aabb_01 * (xyz_pcd_max - xyz_pcd_min))
        out_bbox_pcd.colors = o3d.utility.Vector3dVector([[1,0,0] for i in range(12)])
        out_bbox_pcd.lines = o3d.utility.Vector2iVector([[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]])

        geometry_list = [pcd]

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

    def collect_lighting_SG(self, lighting_SG_params: dict={}):

        assert self.openrooms_scene.if_has_lighting_SG

        subsample_lighting_SG_rate = lighting_SG_params.get('subsample_lighting_SG_rate', 1)
        SG_scale = lighting_SG_params.get('SG_scale', 1.) # if autoscale, act as extra scale
        SG_keep_ratio = lighting_SG_params.get('SG_keep_ratio', 0.05)
        SG_clip_ratio = lighting_SG_params.get('SG_clip_ratio', 0.1)
        SG_autoscale = lighting_SG_params.get('SG_autoscale', True)
        # if SG_autoscale:
        #     SG_scale = 1.

        lighting_SG_fused_dict = self.openrooms_scene._fuse_3D_lighting_SG(subsample_rate=subsample_lighting_SG_rate)

        xyz_SG_pcd, normal_SG_pcd, axis_SG_pcd, weight_SG_pcd, lamb_SG_pcd = get_list_of_keys(lighting_SG_fused_dict, ['X_global_SG', 'normal_global_SG', 'axis_SG', 'weight_SG', 'lamb_SG'])


        N_pts = xyz_SG_pcd.shape[0]
        SG_num = axis_SG_pcd.shape[1]

        xyz_SG_pcd = np.tile(np.expand_dims(xyz_SG_pcd, 1), (1, SG_num, 1)).reshape((-1, 3))
        length_SG_pcd = np.linalg.norm(weight_SG_pcd.reshape((-1, 3)), axis=1, keepdims=True) / 500.
        

        # keep only SGs pointing towards outside of the surface
        axis_SG_mask = np.ones_like(length_SG_pcd.squeeze()).astype(bool)
        axis_SG_mask = np.sum(np.repeat(np.expand_dims(normal_SG_pcd, 1), SG_num, axis=1) * axis_SG_pcd, axis=2) > 0.
        axis_SG_mask =  axis_SG_mask.reshape(-1) # (N*12,)

        if SG_keep_ratio > 0.:
            assert SG_keep_ratio > 0. and SG_keep_ratio <= 1.
            percentile = np.percentile(length_SG_pcd.flatten(), 100-SG_keep_ratio*100)
            axis_SG_mask = np.logical_and(axis_SG_mask, length_SG_pcd.flatten() > percentile) # (N*12,)
        else:
            pass

        if SG_clip_ratio > 0.: # within keeped SGs
            assert SG_clip_ratio > 0. and SG_clip_ratio <= 1.
            percentile = np.percentile(length_SG_pcd[axis_SG_mask].flatten(), 100-SG_clip_ratio*100)
            length_SG_pcd[length_SG_pcd > percentile] = percentile

        if SG_autoscale:
            # ic(np.amax(length_SG_pcd))
            ceiling_y, floor_y = np.amax(xyz_SG_pcd[:, 2]), np.amin(xyz_SG_pcd[:, 2])
            length_SG_pcd = length_SG_pcd / np.amax(length_SG_pcd) * abs(ceiling_y - floor_y)
            # ic(np.amax(length_SG_pcd))
        
        length_SG_pcd *= SG_scale

        axis_SG_pcd_end = xyz_SG_pcd + length_SG_pcd * axis_SG_pcd.reshape((-1, 3))
        xyz_SG_pcd, axis_SG_pcd_end = xyz_SG_pcd[axis_SG_mask], axis_SG_pcd_end[axis_SG_mask]
        N_pts = xyz_SG_pcd.shape[0]
        print('Showing %d lighting SGs...'%N_pts)
        
        axis_SGs = o3d.geometry.LineSet()
        axis_SGs.points = o3d.utility.Vector3dVector(np.vstack((xyz_SG_pcd, axis_SG_pcd_end)))
        axis_SGs.colors = o3d.utility.Vector3dVector([[0., 0., 1.] for _ in range(N_pts)])
        axis_SGs.lines = o3d.utility.Vector2iVector([[_, _+N_pts] for _ in range(N_pts)])
        
        geometry_list = [axis_SGs]

        return geometry_list

    def collect_layout(self, layout_params: dict={}):
        '''
        images/demo_layout_o3d.png
        '''
        assert self.openrooms_scene.if_has_layout

        layout_mesh = self.openrooms_scene.layout_mesh_transformed.as_open3d
        layout_mesh.compute_vertex_normals()

        layout_bbox_3d = self.openrooms_scene.layout_box_3d_transformed
        layout_bbox_pcd = o3d.geometry.LineSet()
        layout_bbox_pcd.points = o3d.utility.Vector3dVector(layout_bbox_3d)
        layout_bbox_pcd.colors = o3d.utility.Vector3dVector([[1,0,0] for i in range(12)])
        layout_bbox_pcd.lines = o3d.utility.Vector2iVector([[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]])

        return [[layout_mesh, 'layout'], layout_bbox_pcd]

    def collect_shapes(self, shapes_params: dict={}):
        '''
        collect shapes and bboxes for objs (objs + emitters)

        to visualize emitter properties, go to collect_emitters

        images/demo_shapes_o3d.png
        images/demo_shapes_o3d_2_shader.png

        images/demo_shapes_emitter_o3d.png
        '''
        assert self.openrooms_scene.if_has_shapes

        if_obj_meshes = shapes_params.get('if_meshes', True) and self.openrooms_scene.shape_params_dict.get('if_load_obj_mesh', False)
        if_emitter_meshes = shapes_params.get('if_meshes', True) and self.openrooms_scene.shape_params_dict.get('if_load_emitter_mesh', False)
        if_labels = shapes_params.get('if_labels', True)

        self.openrooms_scene.load_colors()

        geometry_list = []

        emitters_obj_random_id_list = [shape['filename'] for shape in self.openrooms_scene.shape_list_valid if shape['if_in_emitter_dict']]

        for shape_idx, (shape, vertices, faces, bverts) in enumerate(zip(
            self.openrooms_scene.shape_list_valid, 
            self.openrooms_scene.vertices_list, 
            self.openrooms_scene.faces_list, 
            self.openrooms_scene.bverts_list, 
        )):

            if_emitter = shape['if_in_emitter_dict']

            if np.amax(bverts[:, 1]) <= np.amin(bverts[:, 1]):
                obj_color = [0., 0., 0.] # black for invalid objects
                cat_name = 'INVALID'
                continue
            else:
                obj_path = shape['filename']
                if 'uv_mapped.obj' in obj_path:
                    continue # skipping layout as an object
                if shape['random_id'] in emitters_obj_random_id_list and not if_emitter:
                    continue # skip shape if it is also in the list as an emitter (so that we don't create two shapes for one emitters)
                cat_id_str = str(obj_path).split('/')[-3]
                assert cat_id_str in self.openrooms_scene.OR_mapping_cat_str_to_id_name_dict, 'not valid cat_id_str: %s; %s'%(cat_id_str, obj_path)
                cat_id, cat_name = self.openrooms_scene.OR_mapping_cat_str_to_id_name_dict[cat_id_str]
                obj_color = self.openrooms_scene.OR_mapping_id_to_color_dict[cat_id]
                obj_color = [float(x)/255. for x in obj_color]
                linestyle = '-'
                linewidth = 1
                if if_emitter:
                    linewidth = 3
                    linestyle = '--'
                    obj_color = [0.4, 0.4, 0.4] # dark grey for emitters; colormap see 
                    '''
                    images/OR42_color_mapping_light.png
                    '''

            '''
            applicable if necessary
            '''

            # trimesh.repair.fill_holes(shape_mesh)
            # trimesh.repair.fix_winding(shape_mesh)
            # trimesh.repair.fix_inversion(shape_mesh)
            # trimesh.repair.fix_normals(shape_mesh)

            if_mesh = if_obj_meshes if not if_emitter else if_emitter_meshes

            if if_mesh:
                assert np.amax(faces-1) < vertices.shape[0]
                shape_mesh = trimesh.Trimesh(vertices=vertices, faces=faces-1) # [IMPORTANT] faces-1 because Trimesh faces are 0-based
                shape_mesh = shape_mesh.as_open3d

                N_triangles = len(shape_mesh.triangles)
                if shapes_params.get('simply_ratio', 1.) != 1.: # not simplying for mesh with very few faces
                    target_number_of_triangles = int(len(shape_mesh.triangles)*shapes_params.get('simply_ratio', 1.))
                    target_number_of_triangles = max(10000, min(50, target_number_of_triangles))
                    shape_mesh = shape_mesh.simplify_quadric_decimation(target_number_of_triangles=target_number_of_triangles)
                    print('[%s] Mesh simplified to %d->%d triangles.'%(cat_name, N_triangles, len(shape_mesh.triangles)))

                shape_mesh.paint_uniform_color(obj_color)
                shape_mesh.compute_vertex_normals()
                shape_mesh.compute_triangle_normals()
                geometry_list.append([shape_mesh, 'shape_emitter_'+shape['random_id'] if if_emitter else 'shape_obj_'+shape['random_id']])

            print('--', if_emitter, cat_name, shape_idx, cat_id, obj_path)

            # shape_label = o3d.visualization.gui.Label3D([0., 0., 0.], np.mean(bverts, axis=0).reshape((3, 1)), cat_name)
            # geometry_list.append(shape_label)
            # self.vis.add_3d_label(np.mean(bverts, axis=0), cat_name)

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

            shape_bbox = o3d.geometry.LineSet()
            shape_bbox.points = o3d.utility.Vector3dVector(bverts)
            shape_bbox.colors = o3d.utility.Vector3dVector([obj_color if not if_emitter else [0., 0., 0.] for i in range(12)]) # black for emitters
            shape_bbox.lines = o3d.utility.Vector2iVector([[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]])
            geometry_list.append(shape_bbox)

        if_dump_mesh = shapes_params.get('if_dump_mesh', False)

        if if_dump_mesh:
            num_vertices = 0
            f_list = []
            for vertices, faces in zip(self.openrooms_scene.vertices_list, self.openrooms_scene.faces_list):
                f_list.append(copy.deepcopy(faces + num_vertices))
                num_vertices += vertices.shape[0]
            f = np.array(self.openrooms_scene.layout_mesh_ori.faces)
            f_list.append(f+num_vertices)
            v = np.array(self.openrooms_scene.layout_mesh_ori.vertices)
            v_list = copy.deepcopy(self.openrooms_scene.vertices_list) + [v]
            writeMesh('./tmp_mesh.obj', np.vstack(v_list), np.vstack(f_list))

        return geometry_list

    def collect_emitters(self, emitters_params: dict={}):
        '''
        images/demo_emitters_o3d.png
        images/demo_envmap_o3d.png # added envmap hemisphere
        '''
        assert self.openrooms_scene.if_has_shapes

        if_half_envmap = emitters_params.get('if_half_envmap', True)
        scale_SG_length = emitters_params.get('scale_SG_length', 2.)

        self.openrooms_scene.load_colors()

        geometry_list = []

        for shape_idx, (shape, bverts) in enumerate(zip(
            self.openrooms_scene.shape_list_valid, 
            self.openrooms_scene.bverts_list, 
        )):

            if not shape['if_in_emitter_dict']:
                continue

            # print('--', if_emitter, shape_idx, cat_id, cat_name, obj_path)

            light_center = np.mean(bverts, 0).flatten()

            '''
            WINDOWS
            '''
            if 'axis_world' in shape['emitter_prop']: # if window
                label_SG_list = ['', 'Sky', 'Grd']
                for label_SG, color, scale in zip(label_SG_list, ['k', 'b', 'g'], [1*scale_SG_length, 0.5*scale_SG_length, 0.5*scale_SG_length]):
                    light_axis = np.asarray(shape['emitter_prop']['axis%s_world'%label_SG]).flatten()
                
                    light_axis_world = np.asarray(light_axis).reshape(3,) # transform back to world system
                    light_axis_world = light_axis_world / np.linalg.norm(light_axis_world)
                    # light_axis_world = np.array([1., 0., 0.])
                    light_axis_end = np.asarray(light_center).reshape(3,) + light_axis_world * np.log(shape['emitter_prop']['intensity'+label_SG])
                    light_axis_end = light_axis_end.flatten()

                    a_light = get_arrow_o3d(light_center, light_axis_end, scale=scale, color=color)
                    geometry_list.append(a_light)

                if if_half_envmap:
                    env_map_path = shape['emitter_prop']['envMapPath']
                    im_envmap_ori = load_HDR(Path(env_map_path))
                    im_envmap_ori_SDR, im_envmap_ori_scale = to_nonHDR(im_envmap_ori)

                    sphere_envmap = self.get_sphere(hemisphere_normal=shape['emitter_prop']['box3D_world']['zAxis'].reshape((3,)), envmap=im_envmap_ori_SDR)
                    geometry_list.append([sphere_envmap, 'envmap'])

        return geometry_list




from pathlib import Path, PosixPath
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(suppress=True)

from tqdm import tqdm
import pickle
import trimesh
import shutil
from collections import defaultdict

# Import the library using the alias "mi"
import mitsuba as mi
# Set the variant of the renderer
from lib.global_vars import mi_variant
mi.set_variant(mi_variant)

from lib.utils_io import load_HDR, to_nonHDR
from lib.utils_misc import blue_text, get_list_of_keys, white_blue, get_datetime, gen_random_str
from lib.utils_mitsuba import dump_OR_xml_for_mi

from .class_openroomsScene2D import openroomsScene2D

from lib.utils_OR.utils_OR_mesh import minimum_bounding_rectangle, mesh_to_contour, load_trimesh, remove_top_down_faces, mesh_to_skeleton, transform_v
from lib.utils_OR.utils_OR_xml import get_XML_root, parse_XML_for_shapes_global
from lib.utils_OR.utils_OR_mesh import loadMesh, computeBox, flip_ceiling_normal
from lib.utils_OR.utils_OR_transform import transform_with_transforms_xml_list
from lib.utils_OR.utils_OR_emitter import load_emitter_dat_world
from lib.utils_dvgo import get_rays_np

class openroomsScene3D(openroomsScene2D):
    '''
    A class used to visualize OpenRooms (public/public-re versions) scene contents (2D/2.5D per-pixel DENSE properties for inverse rendering).
    For high-level semantic properties (e.g. layout, objects, emitters, use class: openroomsScene3D)
    '''
    def __init__(
        self, 
        root_path_dict: dict, 
        scene_params_dict: dict, 
        modality_list: list, 
        im_params_dict: dict={'im_H_load': 480, 'im_W_load': 640, 'im_H_resize': 480, 'im_W_resize': 640}, 
        BRDF_params_dict: dict={}, 
        lighting_params_dict: dict={'env_row': 120, 'env_col': 160, 'SG_num': 12, 'env_height': 16, 'env_width': 32}, # params to load & convert lighting SG & envmap to 
        cam_params_dict: dict={'near': 0.1, 'far': 7.}, 
        shape_params_dict: dict={'if_load_mesh': True}, 
        emitter_params_dict: dict={'N_ambient_rep': '3SG-SkyGrd'},
        mi_params_dict: dict={'if_sample_rays_pts': True}, 
    ):

        for _ in modality_list:
            assert _ in super().valid_modalities + self.valid_modalities_3D, 'Invalid modality: %s'%_

        self.if_loaded_colors = False

        self.cam_params_dict = cam_params_dict
        self.shape_params_dict = shape_params_dict
        self.emitter_params_dict = emitter_params_dict
        self.mi_params_dict = mi_params_dict


        super().__init__(
            root_path_dict = root_path_dict, 
            scene_params_dict = scene_params_dict, 
            modality_list = modality_list, 
            im_params_dict = im_params_dict, 
            BRDF_params_dict = BRDF_params_dict, 
            lighting_params_dict = lighting_params_dict, 
        )

        self.shapes_root, self.layout_root, self.envmaps_root = get_list_of_keys(self.root_path_dict, ['shapes_root', 'layout_root', 'envmaps_root'], [PosixPath, PosixPath, PosixPath])
        self.xml_file = self.scene_xml_path / ('%s.xml'%self.meta_split.split('_')[0]) # load from one of [main, mainDiffLight, mainDiffMat]
        self.pcd_color = None

        '''
        load everything
        '''
        self.load_cam_rays(self.cam_params_dict)
        self.load_modalities_3D()

    @property
    def valid_modalities_3D(self):
        return ['layout', 'shapes', 'mi']


    @property
    def if_has_shapes(self): # objs + emitters
        return all([_ in self.modality_list for _ in ['shapes']])

    @property
    def if_has_mitsuba_scene(self):
        return all([_ in self.modality_list for _ in ['mi']])

    @property
    def if_has_mitsuba_rays_pts(self):
        return 'mi' in self.modality_list and self.mi_params_dict['if_sample_rays_pts']

    @property
    def if_has_colors(self):
        return self.if_loaded_colors

    def load_modalities_3D(self):
        for _ in self.modality_list:
            if _ == 'layout': self.load_layout()
            if _ == 'shapes': self.load_shapes(self.shape_params_dict) # shapes of 1(i.e. furniture) + emitters
            if _ == 'mi': self.load_mi(self.mi_params_dict)

    def load_mi(self, mi_params_dict={}):
        '''
        load scene representation into Mitsuba 3
        '''
        xml_dump_dir = self.PATH_HOME / 'mitsuba'

        if_also_dump_xml_with_lit_lamps_only = mi_params_dict.get('if_also_dump_xml_with_lit_lamps_only', True)

        self.mi_xml_dump_path = dump_OR_xml_for_mi(
            str(self.xml_file), 
            shapes_root=self.shapes_root, 
            layout_root=self.layout_root, 
            envmaps_root=self.envmaps_root, 
            xml_dump_dir=xml_dump_dir, 
            origin_lookatvector_up_tuple=self.origin_lookatvector_up_list[0], # [debug] set to any frame_idx
            if_no_emitter_shape=False, 
            if_also_dump_xml_with_lit_lamps_only=if_also_dump_xml_with_lit_lamps_only, 
            )
        print(blue_text('XML for Mitsuba dumped to: %s')%str(self.mi_xml_dump_path))

        self.mi_scene = mi.load_file(str(self.mi_xml_dump_path))
        if if_also_dump_xml_with_lit_lamps_only:
            self.mi_scene_lit_up_lamps_only = mi.load_file(str(self.mi_xml_dump_path).replace('.xml', '_lit_up_lamps_only.xml'))

        debug_dump_mesh = mi_params_dict.get('debug_dump_mesh', False)
        if debug_dump_mesh:
            '''
            images/demo_mitsuba_dump_meshes.png
            '''
            mesh_dump_root = self.PATH_HOME / 'mitsuba' / 'meshes_dump'
            if mesh_dump_root.exists():
                shutil.rmtree(str(mesh_dump_root))
            mesh_dump_root.mkdir()

            for shape_idx, shape, in enumerate(self.mi_scene.shapes()):
                shape.write_ply(str(mesh_dump_root / ('%06d.ply'%shape_idx)))

        debug_render_test_image = mi_params_dict.get('debug_render_test_image', False)
        if debug_render_test_image:
            '''
            images/demo_mitsuba_render.png
            '''
            print(blue_text('Rendering... test frame by Mitsuba: %s')%str(self.PATH_HOME / 'mitsuba' / 'tmp_render.png'))
            image = mi.render(self.mi_scene, spp=64)
            mi.util.write_bitmap(str(self.PATH_HOME / 'mitsuba' / 'tmp_render.png'), image)
            mi.util.write_bitmap(str(self.PATH_HOME / 'mitsuba' / 'tmp_render.exr'), image)
            if if_also_dump_xml_with_lit_lamps_only:
                image = mi.render(self.mi_scene_lit_up_lamps_only, spp=64)
                mi.util.write_bitmap(str(self.PATH_HOME / 'mitsuba' / 'tmp_render_lit_up_lamps_only.exr'), image)

            print(blue_text('DONE.'))

        if_sample_rays_pts = mi_params_dict.get('if_sample_rays_pts', True)
        if if_sample_rays_pts:
            self.mi_sample_rays_pts()
        
        if_get_segs = mi_params_dict.get('if_get_segs', True)
        if if_get_segs:
            assert if_sample_rays_pts
            self.mi_get_segs(if_also_dump_xml_with_lit_lamps_only=if_also_dump_xml_with_lit_lamps_only)

    def load_cam_rays(self, cam_params_dict={}):
        H, W = self.im_H_resize, self.im_W_resize
        K = self.K
        self.near = cam_params_dict.get('near', 0.1)
        self.far = cam_params_dict.get('far', 7.)
        
        self.cam_rays_list = []
        for frame_idx in range(self.num_frames):
            rays_o, rays_d, ray_d_center = get_rays_np(H, W, K, self.pose_list[frame_idx], inverse_y=True)
            self.cam_rays_list.append((rays_o, rays_d, ray_d_center))

    def mi_sample_rays_pts(self):
        '''
        sample per-pixel rays in NeRF/DVGO setting
        -> populate: 
            - self.mi_pts_list: [(H, W, 3), ], (-1. 1.)
            - self.mi_depth_list: [(H, W), ], (-1. 1.)
        [!] note:
            - in both self.mi_pts_list and self.mi_depth_list, np.inf values exist for pixels of infinite depth
        '''
        assert self.if_has_mitsuba_scene

        self.mi_rays_ret_list = []
        self.mi_depth_list = []
        self.mi_invalid_depth_mask_list = []
        self.mi_normal_global_list = []
        self.mi_pts_list = []

        for frame_idx, (rays_o, rays_d, ray_d_center) in enumerate(self.cam_rays_list):
            rays_o_flatten, rays_d_flatten = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)

            xs_mi = mi.Point3f(rays_o_flatten)
            ds_mi = mi.Vector3f(rays_d_flatten)
            # ray origin, direction, t_max
            rays_mi = mi.Ray3f(xs_mi, ds_mi)
            ret = self.mi_scene.ray_intersect(rays_mi) # https://mitsuba.readthedocs.io/en/stable/src/api_reference.html?highlight=write_ply#mitsuba.Scene.ray_intersect
            # returned structure contains intersection location, nomral, ray step, ...
            # positions = mi2torch(ret.p.torch())
            self.mi_rays_ret_list.append(ret)

            # rays_v_flatten = ret.p.numpy() - rays_o_flatten
            rays_v_flatten = ret.t.numpy()[:, np.newaxis] * rays_d_flatten
            mi_depth = np.sum(rays_v_flatten.reshape(self.im_H_resize, self.im_W_resize, 3) * ray_d_center.reshape(1, 1, 3), axis=-1)
            invalid_depth_mask = np.logical_or(np.isnan(mi_depth), np.isinf(mi_depth))
            self.mi_invalid_depth_mask_list.append(invalid_depth_mask)
            mi_depth[invalid_depth_mask] = np.inf
            self.mi_depth_list.append(mi_depth)

            mi_normals = ret.n.numpy().reshape(self.im_H_resize, self.im_W_resize, 3)
            normals_flip_mask = np.logical_and(np.sum(rays_d * mi_normals, axis=-1) > 0, np.any(mi_normals != np.inf, axis=-1))
            mi_normals[normals_flip_mask] = -mi_normals[normals_flip_mask]
            mi_normals[invalid_depth_mask, :] = np.inf
            self.mi_normal_global_list.append(mi_normals)

            # mi_pts = ret.p.numpy().reshape(self.im_H_resize, self.im_W_resize, 3)
            mi_pts = ret.t.numpy()[:, np.newaxis] * rays_d_flatten + rays_o_flatten
            assert np.amax(np.abs((mi_pts - ret.p.numpy())[ret.t.numpy()!=np.inf, :])) < 1e-3 # except in window areas
            mi_pts = mi_pts.reshape(self.im_H_resize, self.im_W_resize, 3)
            mi_pts[invalid_depth_mask, :] = np.inf

            self.mi_pts_list.append(mi_pts)

        self.pts_from['mi'] = True

    def mi_get_segs(self, if_also_dump_xml_with_lit_lamps_only=True):
        self.mi_seg_dict_of_lists = defaultdict(list)

        for frame_idx, mi_depth in enumerate(self.mi_depth_list):
            # self.mi_seg_dict_of_lists['area'].append(seg_area)
            mi_seg_env = self.mi_invalid_depth_mask_list[frame_idx]
            self.mi_seg_dict_of_lists['env'].append(mi_seg_env) # shine-through area of windows

            if if_also_dump_xml_with_lit_lamps_only:
                rays_o, rays_d, ray_d_center = self.cam_rays_list[frame_idx]
                rays_o_flatten, rays_d_flatten = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)
                rays_mi = mi.Ray3f(mi.Point3f(rays_o_flatten), mi.Vector3f(rays_d_flatten))
                ret = self.mi_scene_lit_up_lamps_only.ray_intersect(rays_mi)
                
                ret_t = ret.t.numpy().reshape(self.im_H_resize, self.im_W_resize)
                invalid_depth_mask = np.logical_or(np.isnan(ret_t), np.isinf(ret_t))
                mi_seg_area = np.logical_not(invalid_depth_mask)
                self.mi_seg_dict_of_lists['area'].append(mi_seg_area) # lit-up lamps

                mi_seg_obj = np.logical_and(np.logical_not(mi_seg_area), np.logical_not(mi_seg_env))
                self.mi_seg_dict_of_lists['obj'].append(mi_seg_obj) # non-emitter objects

        self.seg_from['mi'] = True
        
    def load_shapes(self, shape_params_dict={}):
        '''
        load and visualize shapes (objs/furniture **& emitters**) in 3D & 2D: 

        images/demo_shapes_3D.png
        images/demo_emitters_3D.png # the classroom scene

        '''
        if_load_obj_mesh = shape_params_dict.get('if_load_obj_mesh', False)
        if_load_emitter_mesh = shape_params_dict.get('if_load_emitter_mesh', False)
        print(white_blue('[openroomsScene3D] load_shapes for scene...'))

        # load emitter properties from light*.dat files of **a specific N_ambient_representation**
        self.emitter_dict_of_lists_world = load_emitter_dat_world(light_dir=self.scene_rendering_path, N_ambient_rep=self.emitter_params_dict['N_ambient_rep'], if_save_storage=self.if_save_storage)

        # load general shapes and emitters, and fuse with previous emitter properties
        # print(main_xml_file)
        root = get_XML_root(self.xml_file)

        self.shape_list_ori, self.emitter_list = parse_XML_for_shapes_global(
            root=root, 
            scene_xml_path=self.scene_xml_path, 
            root_uv_mapped=self.shapes_root, 
            root_layoutMesh=self.layout_root, 
            root_EnvDataset=self.envmaps_root, 
            if_return_emitters=True, 
            light_dat_lists=self.emitter_dict_of_lists_world)

        assert self.shape_list_ori[0]['filename'].endswith('uv_mapped.obj')
        assert self.shape_list_ori[1]['filename'].endswith('container.obj')
        assert self.emitter_list[0]['emitter_prop']['if_env'] == True # first of emitter_list is the env

        # start to load objects

        self.vertices_list = []
        self.faces_list = []
        self.ids_list = []
        self.bverts_list = []
        
        light_axis_list = []
        # self.num_vertices = 0
        obj_path_list = []
        self.shape_list_valid = []
        self.window_list = []
        self.lamp_list = []
        
        print(blue_text('[openroomsScene3D] loading %d shapes and %d emitters...'%(len(self.shape_list_ori), len(self.emitter_list[1:]))))

        self.emitter_env = self.emitter_list[0] # e.g. {'if_emitter': True, 'emitter_prop': {'emitter_type': 'envmap', 'if_env': True, 'emitter_filename': '.../EnvDataset/1611L.hdr', 'emitter_scale': 164.1757}}
        assert self.emitter_env['if_emitter']
        assert self.emitter_env['emitter_prop']['emitter_type'] == 'envmap'

        for shape_idx, shape in tqdm(enumerate(self.shape_list_ori + self.emitter_list[1:])): # self.emitter_list[0] is the envmap
            if 'container' in shape['filename']:
                continue
            
        #     if_emitter = shape['if_emitter'] and 'combined_filename' in shape['emitter_prop'] and shape_idx >= len(shape_list)
            if_emitter = shape['if_in_emitter_dict']
            if if_emitter:
                obj_path = shape['emitter_prop']['emitter_filename']
        #         obj_path = shape['filename']
            else:
                obj_path = shape['filename']

            bbox_file_path = obj_path.replace('.obj', '.pickle')
            if 'layoutMesh' in bbox_file_path:
                bbox_file_path = Path('layoutMesh') / Path(bbox_file_path).relative_to(self.root_path_dict['layout_root'])
            elif 'uv_mapped' in bbox_file_path:
                bbox_file_path = Path('uv_mapped') / Path(bbox_file_path).relative_to(self.root_path_dict['shapes_root'])
            bbox_file_path = self.root_path_dict['shape_pickles_root'] / bbox_file_path

            #  Path(bbox_file_path).exists(), 'Rerun once first with if_load_mesh=True, to dump pickle files for shapes to %s'%bbox_file_path
            
            if_load_mesh = if_load_obj_mesh if not if_emitter else if_load_emitter_mesh

            if if_load_mesh or (not Path(bbox_file_path).exists()):
                vertices, faces = loadMesh(obj_path) # based on L430 of adjustObjectPoseCorrectChairs.py
                bverts, bfaces = computeBox(vertices)
                if not Path(bbox_file_path).exists():
                    Path(bbox_file_path).parent.mkdir(parents=True, exist_ok=True)
                    with open(bbox_file_path, "wb") as f:
                        pickle.dump(dict(bverts=bverts, bfaces=bfaces), f)
            else:
                with open(bbox_file_path, "rb") as f:
                    bbox_dict = pickle.load(f)
                bverts, bfaces = bbox_dict['bverts'], bbox_dict['bfaces']

            if if_load_mesh:
                vertices_transformed, _ = transform_with_transforms_xml_list(shape['transforms_list'], vertices)
            bverts_transformed, transforms_converted_list = transform_with_transforms_xml_list(shape['transforms_list'], bverts)

            y_max = bverts_transformed[:, 1].max()
            points_2d = bverts_transformed[abs(bverts_transformed[:, 1] - y_max) < 1e-5, :]
            if points_2d.shape[0] != 4:
                assert if_load_mesh
                bverts_transformed, bfaces = computeBox(vertices_transformed) # dealing with cases like pillow, where its y axis after transformation does not align with world's (because pillows do not have to stand straight)
            
            # if not(any(ext in shape['filename'] for ext in ['window', 'door', 'lamp'])):
        
            obj_path_list.append(obj_path)
    
            if if_load_mesh:
                self.vertices_list.append(vertices_transformed)
                # self.faces_list.append(faces+self.num_vertices)
                if '/uv_mapped.obj' in shape['filename']:
                    faces = flip_ceiling_normal(faces, vertices)
                self.faces_list.append(faces)
                # self.num_vertices += vertices_transformed.shape[0]
            else:
                self.vertices_list.append(None)
                self.faces_list.append(None)
            self.bverts_list.append(bverts_transformed)
            self.ids_list.append(shape['id'])
            
            self.shape_list_valid.append(shape)



            if if_emitter:
                if shape['emitter_prop']['obj_type'] == 'window':
                    self.window_list.append(shape)
                elif shape['emitter_prop']['obj_type'] == 'obj':
                    self.lamp_list.append(shape)

        print(blue_text('[openroomsScene3D] DONE. load_shapes: %d total, %d/%d windows lit, %d/%d lamps lit'%(
            len(self.shape_list_valid), 
            len([_ for _ in self.window_list if _['emitter_prop']['if_lit_up']]), len(self.window_list), 
            len([_ for _ in self.lamp_list if _['emitter_prop']['if_lit_up']]), len(self.lamp_list), 
            )))

    def load_layout(self):
        '''
        load and visualize layout in 3D & 2D: 

        images/demo_layout_3D.png
        images/demo_layout_2D.png

        '''

        print(white_blue('[openroomsScene3D] load_layout for scene...'))

        self.layout_obj_file = self.layout_root / self.scene_name_short / 'uv_mapped.obj'
        self.layout_mesh_ori = load_trimesh(self.layout_obj_file) # returns a Trimesh object
        # mesh = mesh.dump()[0]
        self.layout_mesh = remove_top_down_faces(self.layout_mesh_ori)
        self.v = np.asarray(self.layout_mesh.vertices)
        self.e = self.layout_mesh.edges

        # simply mesh -> skeleton
        self.v_skeleton, self.e_skeleton, room_height = mesh_to_skeleton(self.layout_mesh)

        # find 2d floor contour
        self.v_2d, self.e_2d = mesh_to_contour(self.layout_mesh)
        # finding minimum 2d bbox (rectangle) from contour
        self.layout_hull_2d = minimum_bounding_rectangle(self.v_2d)

        # 2d cuvboid hull -> 3d bbox
        self.layout_box_3d = np.hstack((np.vstack((self.layout_hull_2d, self.layout_hull_2d)), np.vstack((np.zeros((4, 1)), np.zeros((4, 1))+room_height))))    

        # transform and visualize layout in scene-specific TRANSFORMED coordinates (basically transforming/normalizing all objects with a single transformation, for rendering purposes)
        if not hasattr(self, 'transforms'):
            self.load_transforms()
        T_layout = self.transforms[0]

        self.v_skeleton_transformed = transform_v(self.v_skeleton, T_layout) # to TRANSFORMED coordinates
        
        v_ori_transformed = transform_v(np.asarray(self.layout_mesh_ori.vertices), T_layout) # skeleton to TRANSFORMED coordinates
        self.layout_mesh_ori_transformed = trimesh.Trimesh(vertices=v_ori_transformed, faces=self.layout_mesh_ori.faces) # original mesh to TRANSFORMED coordinates

        v_transformed = transform_v(np.asarray(self.layout_mesh.vertices), T_layout) # skeleton to TRANSFORMED coordinates
        self.layout_mesh_transformed = trimesh.Trimesh(vertices=v_transformed, faces=self.layout_mesh.faces) # original mesh to TRANSFORMED coordinates

        self.layout_box_3d_transformed = transform_v(self.layout_box_3d, T_layout)

        print(blue_text('[openroomsScene3D] DONE. load_layout'))

    def load_colors(self):
        '''
        load mapping from obj cat id to RGB
        '''

        if self.if_has_colors:
            pass

        OR_mapping_cat_str_to_id_file = self.semantic_labels_root / 'semanticLabelName_OR42.txt'
        with open(str(OR_mapping_cat_str_to_id_file)) as f:
            mylist = f.read().splitlines() 
        
        self.OR_mapping_cat_str_to_id_name_dict = {x.split(' ')[0]: (int(x.split(' ')[1]), x.split(' ')[2]) for x in mylist} # cat id is 0-based (0 being unlabelled)!
        
        OR_mapping_id_to_color_file = self.semantic_labels_root / 'colors/OR4X_mapping_catInt_to_RGB_light.pkl'
        with (open(OR_mapping_id_to_color_file, "rb")) as f:
            OR4X_mapping_catInt_to_RGB_light = pickle.load(f)
        self.OR_mapping_id_to_color_dict = OR4X_mapping_catInt_to_RGB_light['OR42']

        self.if_loaded_colors = True
from pathlib import Path, PosixPath
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(suppress=True)

from tqdm import tqdm
import pickle
import trimesh
import shutil
from collections import defaultdict
from math import prod
from lib.global_vars import mi_variant_dict
import torch
# Import the library using the alias "mi"
import mitsuba as mi
# Set the variant of the renderer
# from lib.global_vars import mi_variant
# mi.set_variant(mi_variant)

from lib.utils_misc import blue_text, yellow, get_list_of_keys, white_blue
from lib.utils_mitsuba import dump_OR_xml_for_mi

from .class_openroomsScene2D import openroomsScene2D
from .class_mitsubaBase import mitsubaBase

from lib.utils_OR.utils_OR_mesh import minimum_bounding_rectangle, mesh_to_contour, load_trimesh, remove_top_down_faces, mesh_to_skeleton, transform_v
from lib.utils_OR.utils_OR_xml import get_XML_root, parse_XML_for_shapes_global
from lib.utils_OR.utils_OR_mesh import loadMesh, computeBox, flip_ceiling_normal
from lib.utils_OR.utils_OR_transform import transform_with_transforms_xml_list
from lib.utils_OR.utils_OR_emitter import load_emitter_dat_world
from lib.utils_OR.utils_OR_lighting import convert_lighting_axis_local_to_global_np, get_ls_np
from lib.utils_dvgo import get_rays_np
from lib.utils_misc import get_device

class openroomsScene3D(openroomsScene2D, mitsubaBase):
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
        if_debug_info: bool=False, 
        host: str='', 
    ):

        self.if_loaded_colors = False

        self.cam_params_dict = cam_params_dict
        self.shape_params_dict = shape_params_dict
        self.emitter_params_dict = emitter_params_dict
        self.mi_params_dict = mi_params_dict

        self.host = host
        self.device = get_device(self.host)

        openroomsScene2D.__init__(
            self, 
            if_debug_info = if_debug_info, 
            root_path_dict = root_path_dict, 
            scene_params_dict = scene_params_dict, 
            modality_list = list(set(modality_list)), 
            im_params_dict = im_params_dict, 
            BRDF_params_dict = BRDF_params_dict, 
            lighting_params_dict = lighting_params_dict, 
        )

        self.modality_list = self.check_and_sort_modalities(list(set(self.modality_list)))
        self.shapes_root, self.layout_root, self.envmaps_root = get_list_of_keys(self.root_path_dict, ['shapes_root', 'layout_root', 'envmaps_root'], [PosixPath, PosixPath, PosixPath])
        self.xml_file = self.scene_xml_path / ('%s.xml'%self.meta_split.split('_')[0]) # load from one of [main, mainDiffLight, mainDiffMat]
        self.pcd_color = None


        '''
        load everything
        '''
        self.load_cam_rays(self.cam_params_dict)
        if 'mi' in self.modality_list:
            mitsubaBase.__init__(
                self, 
                device = self.device, 
            )
        self.load_modalities_3D()

    @property
    def valid_modalities_3D(self):
        return ['layout', 'shapes', 'mi']

    @property
    def valid_modalities(self):
        return super().valid_modalities + self.valid_modalities_3D

    @property
    def if_has_layout(self):
        return all([_ in self.modality_list for _ in ['layout']])

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
    def if_has_mitsuba_segs(self):
        return 'mi' in self.modality_list and self.mi_params_dict['if_get_segs']

    @property
    def if_has_mitsuba_all(self):
        return all([self.if_has_mitsuba_scene, self.if_has_mitsuba_rays_pts, self.if_has_mitsuba_segs, ])

    @property
    def if_has_colors(self):
        return self.if_loaded_colors

    def load_modalities_3D(self):
        for _ in self.modality_list:
            if _ == 'layout': self.load_layout()
            if _ == 'shapes': self.load_shapes(self.shape_params_dict) # shapes of 1(i.e. furniture) + emitters
            if _ == 'mi': self.load_mi(self.mi_params_dict)

    def get_modality(self, modality):
        if modality in super().valid_modalities:
            return super(openroomsScene3D, self).get_modality(modality)

        if 'mi_' in modality:
            assert self.pts_from['mi']

        if modality == 'mi_depth': 
            return self.mi_depth_list
        elif modality == 'mi_normal': 
            return self.mi_normal_global_list
        elif modality in ['mi_seg_area', 'mi_seg_env', 'mi_seg_obj']:
            seg_key = modality.split('_')[-1] 
            return self.seg_dict_of_lists[seg_key]
        else:
            assert False, 'Unsupported modality: ' + modality

    def load_mi(self, mi_params_dict={}):
        '''
        load scene representation into Mitsuba 3
        '''
        xml_dump_dir = self.PATH_HOME / 'mitsuba'

        if_also_dump_xml_with_lit_lamps_only = mi_params_dict.get('if_also_dump_xml_with_lit_lamps_only', True)
        variant = mi_params_dict.get('variant', '')
        if variant != '':
            mi.set_variant(variant)
        else:
            mi.set_variant(mi_variant_dict[self.host])

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
            self.mi_sample_rays_pts(self.cam_rays_list)
            self.pts_from['mi'] = True

        if_get_segs = mi_params_dict.get('if_get_segs', True)
        if if_get_segs:
            assert if_sample_rays_pts
            self.mi_get_segs(if_also_dump_xml_with_lit_lamps_only=if_also_dump_xml_with_lit_lamps_only)
            self.seg_from['mi'] = True

    def load_cam_rays(self, cam_params_dict={}):
        self.near = cam_params_dict.get('near', 0.1)
        self.far = cam_params_dict.get('far', 7.)
        self.cam_rays_list = self.get_cam_rays_list(self.H, self.W, self.K, self.pose_list)

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
                    self.window_list.append((shape, vertices_transformed, faces))
                elif shape['emitter_prop']['obj_type'] == 'obj':
                    self.lamp_list.append((shape, vertices_transformed, faces))

        print(blue_text('[openroomsScene3D] DONE. load_shapes: %d total, %d/%d windows lit, %d/%d lamps lit'%(
            len(self.shape_list_valid), 
            len([_ for _ in self.window_list if _[0]['emitter_prop']['if_lit_up']]), len(self.window_list), 
            len([_ for _ in self.lamp_list if _[0]['emitter_prop']['if_lit_up']]), len(self.lamp_list), 
            )))

    def load_layout(self):
        '''
        load and visualize layout in 3D & 2D: 

        images/demo_layout_3D.png
        images/demo_layout_2D.png
        '''

        print(white_blue('[openroomsScene3D] load_layout for scene...'))

        self.layout_obj_file = self.layout_root / self.scene_name_short / 'uv_mapped.obj'
        self.layout_mesh_ori = load_trimesh(self.layout_obj_file) # returns a Trimesh object; [!!!] 0-index faces
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
        self.v_2d_transformed = transform_v(np.hstack((self.v_2d, np.zeros((6, 1), dtype=self.v_2d.dtype))), T_layout)[:, [0, 2]]
        self.layout_hull_2d_transformed = self.layout_box_3d_transformed[:4, [0, 2]]

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

    def _fuse_3D_geometry(self, dump_path: Path=Path(''), subsample_rate_pts: int=1, subsample_HW_rates: tuple=(1, 1), if_use_mi_geometry: bool=False):
        '''
        fuse depth maps (and RGB, normals) into point clouds in global coordinates of OpenCV convention

        optionally dump pcd and cams to pickles

        Args:
            subsample_rate_pts: int, sample 1/subsample_rate_pts of points to dump
            if_use_mi_geometry: True: use geometrt from Mistuba if possible

        Returns:
            - fused geometry as dict
            - all camera poses
        '''
        assert self.if_has_poses and self.if_has_im_sdr
        if not if_use_mi_geometry:
            assert self.if_has_dense_geo

        print(white_blue('[openroomsScene] fuse_3D_geometry '), yellow('[use Mitsuba: %s]'%str(if_use_mi_geometry)), 'for %d frames... subsample_rate_pts: %d, subsample_HW_rates: (%d, %d)'%(len(self.frame_id_list), subsample_rate_pts, subsample_HW_rates[0], subsample_HW_rates[1]))

        X_global_list = []
        rgb_global_list = []
        normal_global_list = []
        X_flatten_mask_list = []
        X_lighting_flatten_mask_list = []

        for frame_idx in tqdm(range(len(self.frame_id_list))):
            H_color, W_color = self.im_H_resize, self.im_W_resize
            uu, vv = np.meshgrid(range(W_color), range(H_color))
            x_ = (uu - self.K[0][2]) * self.depth_list[frame_idx] / self.K[0][0]
            y_ = (vv - self.K[1][2]) * self.depth_list[frame_idx] / self.K[1][1]
            if if_use_mi_geometry:
                z_ = self.mi_depth_list[frame_idx]
            else:
                z_ = self.depth_list[frame_idx]

            if if_use_mi_geometry:
                obj_mask = self.mi_seg_dict_of_lists['obj'][frame_idx] + self.mi_seg_dict_of_lists['area'][frame_idx] # geometry is defined for objects + emitters
            else:
                obj_mask = self.seg_dict_of_lists['obj'][frame_idx] + self.seg_dict_of_lists['area'][frame_idx] # geometry is defined for objects + emitters
            assert obj_mask.shape[:2] == (H_color, W_color)
            lighting_mask = self.seg_dict_of_lists['obj'][frame_idx] # lighting is defined for non-emitter objects only
            assert lighting_mask.shape[:2] == (H_color, W_color)

            if subsample_HW_rates != (1, 1):
                x_ = x_[::subsample_HW_rates[0], ::subsample_HW_rates[1]]
                y_ = y_[::subsample_HW_rates[0], ::subsample_HW_rates[1]]
                z_ = z_[::subsample_HW_rates[0], ::subsample_HW_rates[1]]
                H_color, W_color = H_color//subsample_HW_rates[0], W_color//subsample_HW_rates[1]
                obj_mask = obj_mask[::subsample_HW_rates[0], ::subsample_HW_rates[1]]
                lighting_mask = lighting_mask[::subsample_HW_rates[0], ::subsample_HW_rates[1]]
                
            z_ = z_.flatten()
            X_flatten_mask = np.logical_and(z_ > 0, obj_mask.flatten() > 0)
            X_lighting_flatten_mask = np.logical_and(z_ > 0, lighting_mask.flatten() > 0)
            
            z_ = z_[X_flatten_mask]
            x_ = x_.flatten()[X_flatten_mask]
            y_ = y_.flatten()[X_flatten_mask]
            if self.if_debug_info:
                print('Valid pixels percentage: %.4f'%(sum(X_flatten_mask)/float(H_color*W_color)))
            X_flatten_mask_list.append(X_flatten_mask)
            X_lighting_flatten_mask_list.append(X_lighting_flatten_mask)

            X_ = np.stack([x_, y_, z_], axis=-1)
            t = self.pose_list[frame_idx][:3, -1].reshape((3, 1))
            R = self.pose_list[frame_idx][:3, :3]

            X_global = (R @ X_.T + t).T
            X_global_list.append(X_global)

            rgb_global = self.im_sdr_list[frame_idx]
            if subsample_HW_rates != ():
                rgb_global = rgb_global[::subsample_HW_rates[0], ::subsample_HW_rates[1]]
            rgb_global = rgb_global.reshape(-1, 3)[X_flatten_mask]
            rgb_global_list.append(rgb_global)
            
            if if_use_mi_geometry:
                normal = self.mi_normal_list[frame_idx]
            else:
                normal = self.normal_list[frame_idx]
            if subsample_HW_rates != ():
                normal = normal[::subsample_HW_rates[0], ::subsample_HW_rates[1]]
            normal = normal.reshape(-1, 3)[X_flatten_mask]
            normal = np.stack([normal[:, 0], -normal[:, 1], -normal[:, 2]], axis=-1) # transform normals from OpenGL convention (right-up-backward) to OpenCV (right-down-forward)
            normal_global = (R @ normal.T).T
            normal_global_list.append(normal_global)

        print(blue_text('[openroomsScene] DONE. fuse_3D_geometry'))

        X_global = np.vstack(X_global_list)[::subsample_rate_pts]
        rgb_global = np.vstack(rgb_global_list)[::subsample_rate_pts]
        normal_global = np.vstack(normal_global_list)[::subsample_rate_pts]

        assert X_global.shape[0] == rgb_global.shape[0] == normal_global.shape[0]

        geo_fused_dict = {'X': X_global, 'rgb': rgb_global, 'normal': normal_global}

        return geo_fused_dict, X_flatten_mask_list, X_lighting_flatten_mask_list

    def get_lighting_envmap_dirs_global(self, pose, normal):
        env_height, env_width = get_list_of_keys(self.lighting_params_dict, ['env_height', 'env_width'], [int, int])
        wi_num = env_height * env_width
        ls_local = get_ls_np(env_height, env_width) # (3, 8, 16)
        lighting_axis_local = ls_local[np.newaxis, np.newaxis].transpose(0, 1, 3, 4, 2).reshape(1, 1, -1, 3) # -> (1, 1, 8*16, 3)
        axis_np_global = convert_lighting_axis_local_to_global_np(lighting_axis_local, pose, normal).reshape(-1, wi_num, 3) # (120*160, 3, 8, 16)
        return axis_np_global.astype(np.float32)

    def _fuse_3D_lighting(self, lighting_source: str, subsample_rate_pts: int=1, if_use_mi_geometry: bool=False):
        '''
        fuse dense lighting (using corresponding surface geometry)

        Args:
            subsample_rate_pts: int, sample 1/subsample_rate_pts of points to dump

        Returns:
            - fused lighting and their associated pcd as dict
        '''

        print(white_blue('[openroomsScene] fuse_3D_lighting [%s] for %d frames... subsample_rate_pts: %d'%(lighting_source, len(self.frame_id_list), subsample_rate_pts)))

        if if_use_mi_geometry:
            assert self.if_has_mitsuba_all
            normal_list = self.mi_normal_list
        else:
            assert self.if_has_dense_geo
            normal_list = self.normal_list

        assert lighting_source in ['lighting_SG', 'lighting_envmap'] # not supporting 'lighting_sampled' yet

        geo_fused_dict, _, X_lighting_flatten_mask_list = self._fuse_3D_geometry(subsample_rate_pts=subsample_rate_pts, subsample_HW_rates=self.im_lighting_HW_ratios, if_use_mi_geometry=if_use_mi_geometry)
        X_global_lighting, normal_global_lighting = geo_fused_dict['X'], geo_fused_dict['normal']

        if lighting_source == 'lighting_SG':
            assert self.if_has_lighting_SG and self.if_has_poses
        if lighting_source == 'lighting_envmap':
            assert self.if_has_lighting_envmap and self.if_has_poses

        axis_global_list = []
        weight_list = []
        lamb_list = []

        for idx in tqdm(range(len(self.frame_id_list))):
            if lighting_source == 'lighting_SG':
                wi_num = self.lighting_params_dict['SG_params']
                if self.if_convert_lighting_SG_to_global:
                    lighting_global = self.lighting_SG_global_list[idx] # (120, 160, 12(SG_num), 7)
                else:
                    lighting_global = np.concatenate(
                        (convert_lighting_axis_local_to_global_np(self.lighting_params_dict, self.lighting_SG_local_list[idx][:, :, :, :3], self.pose_list[idx], normal_list[idx]), 
                        self.lighting_SG_local_list[idx][:, :, :, 3:]), axis=3) # (120, 160, 12(SG_num), 7); axis, lamb, weight: 3, 1, 3
                axis_np_global = lighting_global[:, :, :, :3].reshape(-1, wi_num, 3)
                weight_np = lighting_global[:, :, :, 4:].reshape(-1, wi_num, 3)

            if lighting_source == 'lighting_envmap':
                axis_np_global = self.get_lighting_envmap_dirs_global(self.pose_list[idx], normal_list[idx])

            if lighting_source == 'lighting_SG':
                lamb_np = lighting_global[:, :, :, 3:4].reshape(-1, wi_num, 1)

            X_flatten_mask = X_lighting_flatten_mask_list[idx]
            axis_np_global = axis_np_global[X_flatten_mask]
            weight_np = weight_np[X_flatten_mask]
            if lighting_source == 'lighting_SG':
                lamb_np = lamb_np[X_flatten_mask]

            axis_global_list.append(axis_np_global)
            weight_list.append(weight_np)
            if lighting_source == 'lighting_SG':
                lamb_list.append(lamb_np)

        print(blue_text('[openroomsScene] DONE. fuse_3D_lighting'))

        axis_global = np.vstack(axis_global_list)[::subsample_rate_pts]
        weight = np.vstack(weight_list)[::subsample_rate_pts]
        if lighting_source == 'lighting_SG':
            lamb_global = np.vstack(lamb_list)[::subsample_rate_pts]

        assert X_global_lighting.shape[0] == axis_global.shape[0]

        lighting_SG_fused_dict = {
            'X_global_lighting': X_global_lighting, 'normal_global_lighting': normal_global_lighting,
            'axis': axis_global, 'weight': weight,}
        if lighting_source == 'lighting_SG':
            lighting_SG_fused_dict.update({'lamb': lamb_global, })

        return lighting_SG_fused_dict
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

from lib.utils_misc import blue_text, yellow, get_list_of_keys, white_blue, red
from lib.utils_mitsuba import dump_OR_xml_for_mi

from .class_openroomsScene2D import openroomsScene2D
from .class_mitsubaBase import mitsubaBase

from lib.utils_OR.utils_OR_mesh import minimum_bounding_rectangle, mesh_to_contour, load_trimesh, remove_top_down_faces, mesh_to_skeleton, transform_v, sample_mesh, simplify_mesh
from lib.utils_OR.utils_OR_xml import get_XML_root, parse_XML_for_shapes_global
from lib.utils_OR.utils_OR_mesh import loadMesh, computeBox, flip_ceiling_normal
from lib.utils_OR.utils_OR_transform import transform_with_transforms_xml_list
from lib.utils_OR.utils_OR_emitter import load_emitter_dat_world
from lib.utils_misc import get_device
from lib.utils_io import read_cam_params_OR

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
        modality_filename_dict: dict, 
        im_params_dict: dict={'im_H_load': 480, 'im_W_load': 640, 'im_H_resize': 480, 'im_W_resize': 640}, 
        cam_params_dict: dict={'near': 0.1, 'far': 10.}, 
        BRDF_params_dict: dict={}, 
        lighting_params_dict: dict={'env_row': 120, 'env_col': 160, 'SG_num': 12, 'env_height': 16, 'env_width': 32}, # params to load & convert lighting SG & envmap to 
        shape_params_dict: dict={'if_load_mesh': True}, 
        emitter_params_dict: dict={'N_ambient_rep': '3SG-SkyGrd'},
        mi_params_dict: dict={'if_sample_rays_pts': True}, 
        if_debug_info: bool=False, 
        host: str='', 
    ):
        openroomsScene2D.__init__(
            self, 
            if_debug_info = if_debug_info, 
            root_path_dict = root_path_dict, 
            scene_params_dict = scene_params_dict, 
            cam_params_dict = cam_params_dict, 
            modality_list = list(set(modality_list)), 
            modality_filename_dict=modality_filename_dict, 
            im_params_dict = im_params_dict, 
            BRDF_params_dict = BRDF_params_dict, 
            lighting_params_dict = lighting_params_dict, 
        )
        self.host = host
        self.device = get_device(self.host)

        self.shape_params_dict = shape_params_dict
        self.emitter_params_dict = emitter_params_dict
        self.mi_params_dict = mi_params_dict
        variant = mi_params_dict.get('variant', '')
        mi.set_variant(variant if variant != '' else mi_variant_dict[self.host])

        if self.cam_params_dict.get('if_sample_poses', False): self.modality_list.append('mi') # need mi scene to sample poses
        self.modality_list = self.check_and_sort_modalities(list(set(self.modality_list)))
        self.shapes_root, self.layout_root, self.envmaps_root = get_list_of_keys(self.root_path_dict, ['shapes_root', 'layout_root', 'envmaps_root'], [PosixPath, PosixPath, PosixPath])
        self.xml_file = self.scene_xml_path / ('%s.xml'%self.meta_split.split('_')[0]) # load from one of [main, mainDiffLight, mainDiffMat]
        self.monosdf_shape_dict = scene_params_dict.get('monosdf_shape_dict', {})
        if '_shape_normalized' in self.monosdf_shape_dict:
            assert self.monosdf_shape_dict['_shape_normalized'] in ['normalized', 'not-normalized'], 'Unsupported _shape_normalized indicator: %s'%_shape_normalized

        self.pcd_color = None

        '''
        load everything
        '''
        if 'poses' in self.modality_list:
            self.load_poses(self.cam_params_dict) # attempt to generate poses indicated in cam_params_dict
        if hasattr(self, 'pose_list'): 
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

    def get_modality(self, modality, source: str='GT'):
        if modality in super().valid_modalities:
            return super(openroomsScene3D, self).get_modality(modality, source=source)

        if 'mi_' in modality:
            assert self.pts_from['mi']

        if modality == 'mi_depth': 
            return self.mi_depth_list
        elif modality == 'mi_normal': 
            return self.mi_normal_global_list
        elif modality in ['mi_seg_area', 'mi_seg_env', 'mi_seg_obj']:
            seg_key = modality.split('_')[-1] 
            return self.mi_seg_dict_of_lists[seg_key]
        else:
            assert False, 'Unsupported modality: ' + modality

    def load_mi(self, mi_params_dict={}, if_postprocess_mi_frames=True):
        '''
        load scene representation into Mitsuba 3
        '''
        xml_dump_dir = self.PATH_HOME / 'mitsuba'

        if_also_dump_xml_with_lit_area_lights_only = False

        if hasattr(self, 'mi_scene'): 
            pass
        else:
            if self.monosdf_shape_dict != {}:
                self.load_monosdf_scene()
            else:
                # if_also_dump_xml_with_lit_area_lights_only = mi_params_dict.get('if_also_dump_xml_with_lit_area_lights_only', True)
                self.mi_xml_dump_path = dump_OR_xml_for_mi(
                    str(self.xml_file), 
                    shapes_root=self.shapes_root, 
                    layout_root=self.layout_root, 
                    envmaps_root=self.envmaps_root, 
                    xml_dump_dir=xml_dump_dir, 
                    # origin_lookatvector_up_tuple=self.origin_lookatvector_up_list[0], # [debug] set to any frame_idx
                    if_no_emitter_shape=False, 
                    if_also_dump_xml_with_lit_area_lights_only=if_also_dump_xml_with_lit_area_lights_only, 
                    )
                print(blue_text('XML for Mitsuba dumped to: %s')%str(self.mi_xml_dump_path))
                self.mi_scene = mi.load_file(str(self.mi_xml_dump_path))
                # if if_also_dump_xml_with_lit_area_lights_only:
                #     self.mi_scene_lit_up_area_lights_only = mi.load_file(str(self.mi_xml_dump_path).replace('.xml', '_lit_up_area_lights_only.xml'))

                debug_dump_mesh = mi_params_dict.get('debug_dump_mesh', False)
                if debug_dump_mesh:
                    '''
                    images/demo_mitsuba_dump_meshes.png
                    '''
                    mesh_dump_root = self.PATH_HOME / 'mitsuba' / 'meshes_dump'
                    if mesh_dump_root.exists():
                        shutil.rmtree(str(mesh_dump_root))
                    mesh_dump_root.mkdir(parents=True, exist_ok=True)

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
                    # if if_also_dump_xml_with_lit_area_lights_only:
                    #     image = mi.render(self.mi_scene_lit_up_area_lights_only, spp=64)
                    #     mi.util.write_bitmap(str(self.PATH_HOME / 'mitsuba' / 'tmp_render_lit_up_area_lights_only.exr'), image)

                    print(blue_text('DONE.'))

        if if_postprocess_mi_frames:
            if_sample_rays_pts = mi_params_dict.get('if_sample_rays_pts', True)
            if if_sample_rays_pts and not self.pts_from['mi'] and hasattr(self, 'cam_rays_list'):
                self.mi_sample_rays_pts(self.cam_rays_list)
                self.pts_from['mi'] = True

            if_get_segs = mi_params_dict.get('if_get_segs', True)
            if if_get_segs and not self.seg_from['mi']:
                assert if_sample_rays_pts
                self.mi_get_segs(if_also_dump_xml_with_lit_area_lights_only=if_also_dump_xml_with_lit_area_lights_only)
                self.seg_from['mi'] = True

    def load_poses(self, cam_params_dict={}):
        '''
        pose_list: list of pose matrices (**camera-to-world** transformation), each (3, 4): [R|t] (OpenCV convention: right-down-forward)
        '''
        self.load_intrinsics()
        if hasattr(self, 'pose_list'): return
        if not self.if_loaded_shapes: self.load_shapes(self.shape_params_dict)
        if not hasattr(self, 'mi_scene'): self.load_mi(self.mi_params_dict, if_postprocess_mi_frames=False)

        if_resample = 'n'
        if cam_params_dict.get('if_sample_poses', False):
            if_resample = 'y'
            if hasattr(self, 'pose_list'):
                if_resample = input(red("pose_list loaded. Resample pose? [y/n]"))
            if self.pose_file.exists():
                if_resample = input(red('pose file exists: %s (%d poses). Resample pose? [y/n]'%(str(self.pose_file), len(read_cam_params_OR(self.pose_file)))))
            if not if_resample in ['N', 'n']:
                self.sample_poses(cam_params_dict.get('sample_pose_num'))
                return
        
        if if_resample in ['N', 'n']:
            openroomsScene2D.load_poses(self, cam_params_dict)

    def load_cam_rays(self, cam_params_dict={}):
        self.near = cam_params_dict.get('near', 0.1)
        self.far = cam_params_dict.get('far', 7.)
        self.cam_rays_list = self.get_cam_rays_list(self.H, self.W, [self.K]*len(self.pose_list), self.pose_list, convention='opencv')

    def load_shapes(self, shape_params_dict={}):
        '''
        load and visualize shapes (objs/furniture **& emitters**) in 3D & 2D: 

        images/demo_shapes_3D.png
        images/demo_emitters_3D.png # the classroom scene

        '''
        if self.if_loaded_shapes: return
        if not self.if_loaded_layout: self.load_layout()

        self.vertices_list = []
        self.faces_list = []
        self.ids_list = []
        self.bverts_list = []
        self.bfaces_list = []

        self.shape_list_valid = []

        self.window_list = []
        self.lamp_list = []
        self.xyz_max = np.zeros(3,)-np.inf
        self.xyz_min = np.zeros(3,)+np.inf

        if self.monosdf_shape_dict != {}:
            self.load_monosdf_shape(shape_params_dict=shape_params_dict)
            self.shape_name_full = self.scene_name_full + '--' + Path(self.monosdf_shape_dict['shape_file']).stem
        else:
            self.shape_name_full = self.scene_name_full
            if_load_obj_mesh = shape_params_dict.get('if_load_obj_mesh', False)
            if_load_emitter_mesh = shape_params_dict.get('if_load_emitter_mesh', False)
            print(white_blue('[openroomsScene3D] load_shapes for scene...'))

            if_sample_pts_on_mesh = shape_params_dict.get('if_sample_pts_on_mesh', False)
            sample_mesh_ratio = shape_params_dict.get('sample_mesh_ratio', 1.)
            sample_mesh_min = shape_params_dict.get('sample_mesh_min', 100)
            sample_mesh_max = shape_params_dict.get('sample_mesh_max', 1000)

            if_simplify_mesh = shape_params_dict.get('if_simplify_mesh', False)
            simplify_mesh_ratio = shape_params_dict.get('simplify_mesh_ratio', 1.)
            simplify_mesh_min = shape_params_dict.get('simplify_mesh_min', 100)
            simplify_mesh_max = shape_params_dict.get('simplify_mesh_max', 1000)
            if_remesh = shape_params_dict.get('if_remesh', True) # False: images/demo_shapes_3D_NO_remesh.png; True: images/demo_shapes_3D_YES_remesh.png
            remesh_max_edge = shape_params_dict.get('remesh_max_edge', 0.1)

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

            
            if if_sample_pts_on_mesh:
                self.sample_pts_list = []

            light_axis_list = []
            # self.num_vertices = 0
            obj_path_list = []
            
            print(blue_text('[openroomsScene3D] loading %d shapes and %d emitters...'%(len(self.shape_list_ori), len(self.emitter_list[1:]))))

            self.emitter_env = self.emitter_list[0] # e.g. {'if_emitter': True, 'emitter_prop': {'emitter_type': 'envmap', 'if_env': True, 'emitter_filename': '.../EnvDataset/1611L.hdr', 'emitter_scale': 164.1757}}
            assert self.emitter_env['if_emitter']
            assert self.emitter_env['emitter_prop']['emitter_type'] == 'envmap'

            for shape_idx, shape_dict in tqdm(enumerate(self.shape_list_ori + self.emitter_list[1:])): # self.emitter_list[0] is the envmap
                if 'container' in shape_dict['filename']:
                    continue
                
                _id = shape_dict['id'] + '_' + shape_dict['random_id']
                
            #     if_emitter = shape_dict['if_emitter'] and 'combined_filename' in shape_dict['emitter_prop'] and shape_idx >= len(shape_list)
                if_emitter = shape_dict['if_in_emitter_dict']
                if if_emitter:
                    obj_path = shape_dict['emitter_prop']['emitter_filename']
            #         obj_path = shape_dict['filename']
                else:
                    obj_path = shape_dict['filename']

                bbox_file_path = obj_path.replace('.obj', '.pickle')
                if 'layoutMesh' in bbox_file_path:
                    bbox_file_path = Path('layoutMesh') / Path(bbox_file_path).relative_to(self.root_path_dict['layout_root'])
                elif 'uv_mapped' in bbox_file_path: # box mesh for the enclosure of the room (walls, ceiling and floor)
                    bbox_file_path = Path('uv_mapped') / Path(bbox_file_path).relative_to(self.root_path_dict['shapes_root'])
                bbox_file_path = self.root_path_dict['shape_pickles_root'] / bbox_file_path

                #  Path(bbox_file_path).exists(), 'Rerun once first with if_load_mesh=True, to dump pickle files for shapes to %s'%bbox_file_path
                
                if_load_mesh = if_load_obj_mesh if not if_emitter else if_load_emitter_mesh

                if if_load_mesh: # or (not Path(bbox_file_path).exists()):
                    if_convert_to_double_sided = 'uv_mapped.obj' in str(obj_path) # convert uv_mapped.obj to double sides mesh (OpenRooms only)
                    # if_convert_to_double_sided = False
                    vertices, faces = loadMesh(obj_path, if_convert_to_double_sided=if_convert_to_double_sided) # based on L430 of adjustObjectPoseCorrectChairs.py
                    bverts, bfaces = computeBox(vertices)
                    if not Path(bbox_file_path).exists():
                        Path(bbox_file_path).parent.mkdir(parents=True, exist_ok=True)
                        with open(bbox_file_path, "wb") as f:
                            pickle.dump(dict(bverts=bverts, bfaces=bfaces), f)
                    
                    # --a bunch of fixes if broken meshes; SLOW--
                    _ = trimesh.Trimesh(vertices=vertices, faces=faces-1) # [IMPORTANT] faces-1 because Trimesh faces are 0-based
                    # trimesh.repair.fix_inversion(_)
                    trimesh.repair.fix_normals(_)
                    # trimesh.repair.fill_holes(_)
                    # trimesh.repair.fix_winding(_)
                    vertices, faces = np.array(_.vertices), np.array(_.faces+1)

                    # --sample mesh--
                    if if_sample_pts_on_mesh:
                        sample_pts, face_index = sample_mesh(vertices, faces, sample_mesh_ratio, sample_mesh_min, sample_mesh_max)
                        self.sample_pts_list.append(sample_pts)
                        # print(sample_pts.shape[0])

                    # --simplify mesh--
                    if if_simplify_mesh and simplify_mesh_ratio != 1.: # not simplying for mesh with very few faces
                        vertices, faces, (N_triangles, target_number_of_triangles) = simplify_mesh(vertices, faces, simplify_mesh_ratio, simplify_mesh_min, simplify_mesh_max, if_remesh=if_remesh, remesh_max_edge=remesh_max_edge)
                        if N_triangles != faces.shape[0]:
                            print('[%s] Mesh simplified to %d->%d triangles (target: %d).'%(_id, N_triangles, faces.shape[0], target_number_of_triangles))
                else:
                    with open(bbox_file_path, "rb") as f:
                        bbox_dict = pickle.load(f)
                    bverts, bfaces = bbox_dict['bverts'], bbox_dict['bfaces']

                if if_load_mesh:
                    vertices_transformed, _ = transform_with_transforms_xml_list(shape_dict['transforms_list'], vertices)
                bverts_transformed, transforms_converted_list = transform_with_transforms_xml_list(shape_dict['transforms_list'], bverts)

                y_max = bverts_transformed[:, 1].max()
                points_2d = bverts_transformed[abs(bverts_transformed[:, 1] - y_max) < 1e-5, :]
                if points_2d.shape[0] != 4:
                    assert if_load_mesh
                    bverts_transformed, bfaces = computeBox(vertices_transformed) # dealing with cases like pillow, where its y axis after transformation does not align with world's (because pillows do not have to stand straight)
                
                # if not(any(ext in shape_dict['filename'] for ext in ['window', 'door', 'lamp'])):

                is_layout = 'layoutMesh' in str(obj_path) or 'uv_mapped.obj' in str(obj_path)
                shape_dict.update({'is_wall': is_layout, 'is_ceiling': is_layout, 'is_layout': is_layout})
                    # 'is_wall': 'wall' in _id.lower(), 
                    # 'is_ceiling': 'ceiling' in _id.lower(), 
                    # 'is_layout': 'wall' in _id.lower() or 'ceiling' in _id.lower(), 
            
                obj_path_list.append(obj_path)
        
                if if_load_mesh:
                    self.vertices_list.append(vertices_transformed)
                    # self.faces_list.append(faces+self.num_vertices)
                    # if '/uv_mapped.obj' in shape_dict['filename']:
                    #     faces = flip_ceiling_normal(faces, vertices)
                    self.faces_list.append(faces)
                    # self.num_vertices += vertices_transformed.shape[0]
                else:
                    self.vertices_list.append(None)
                    self.faces_list.append(None)
                self.bverts_list.append(bverts_transformed)
                self.ids_list.append(_id)
                
                self.shape_list_valid.append(shape_dict)

                if if_emitter:
                    if shape_dict['emitter_prop']['obj_type'] == 'window':
                        # self.window_list.append((shape_dict, vertices_transformed, faces))
                        self.window_list.append(
                            {'emitter_prop': shape_dict['emitter_prop'], 'vertices': vertices_transformed, 'faces': faces}
                            )
                    elif shape_dict['emitter_prop']['obj_type'] == 'obj':
                        # self.lamp_list.append((shape_dict, vertices_transformed, faces))
                        self.lamp_list.append(
                            {'emitter_prop': shape_dict['emitter_prop'], 'vertices': vertices_transformed, 'faces': faces}
                        )

        self.if_loaded_shapes = True

        print(blue_text('[openroomsScene3D] DONE. load_shapes: %d total, %d/%d windows lit, %d/%d area lights lit'%(
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

        if self.if_loaded_layout: return
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
        self.layout_hull_2d, self.layout_hull_pts = minimum_bounding_rectangle(self.v_2d)

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

        self.if_loaded_layout = True

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


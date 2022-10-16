from pathlib import Path, PosixPath
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(suppress=True)

from tqdm import tqdm
import pickle
import trimesh

from lib.utils_io import load_HDR, to_nonHDR
from lib.utils_misc import blue_text, get_list_of_keys, white_blue

from .class_openroomsScene import openroomsScene

from lib.utils_OR.utils_OR_vis_3D import vis_cube_plt, vis_axis, vis_axis_xyz, set_axes_equal, Arrow3D
from lib.utils_OR.utils_OR_mesh import minimum_bounding_rectangle, mesh_to_contour, load_trimesh, remove_top_down_faces, v_pairs_from_v3d_e, v_pairs_from_v2d_e, mesh_to_skeleton, transform_v
from lib.utils_OR.utils_OR_xml import get_XML_root, parse_XML_for_shapes_global
from lib.utils_OR.utils_OR_mesh import loadMesh, computeBox
from lib.utils_OR.utils_OR_transform import transform_with_transforms_xml_list
from lib.utils_OR.utils_OR_emitter import load_emitter_dat_world, render_3SG_envmap, vis_envmap_plt

class openroomsScene3D(openroomsScene):
    '''
    A class used to visualize OpenRooms (public/public-re versions) scene contents (2D/2.5D per-pixel DENSE properties for inverse rendering).
    For high-level semantic properties (e.g. layout, objects, emitters, use class: openroomsScene3D)
    '''
    def __init__(
        self, 
        root_path_dict: dict, 
        scene_params_dict: dict, 
        modality_list: list, 
        modality_list_vis: list=[], 
        im_params_dict: dict={'im_H_load': 480, 'im_W_load': 640, 'im_H_resize': 480, 'im_W_resize': 640}, 
        BRDF_params_dict: dict={}, 
        lighting_params_dict: dict={'env_row': 120, 'env_col': 160, 'SG_num': 12, 'env_height': 16, 'env_width': 32}, # params to load & convert lighting SG & envmap to 
        shape_params_dict: dict={'if_load_mesh': True}, 
        emitter_params_dict: dict={'N_ambient_rep': '3SG-SkyGrd'},
        if_vis_debug_with_plt: bool=False
    ):

        for _ in modality_list:
            assert _ in super().valid_modalities + self.valid_modalities_3D, 'Invalid modality: %s'%_

        self.if_vis_debug_with_plt = if_vis_debug_with_plt
        self.modality_list_vis = modality_list_vis
        for _ in modality_list_vis:
            assert _ in super().valid_modalities + self.valid_modalities_3D_vis, 'Invalid modality_vis: %s'%_

        self.shape_params_dict = shape_params_dict
        self.emitter_params_dict = emitter_params_dict
        self.if_loaded_colors = False

        super().__init__(
            root_path_dict = root_path_dict, 
            scene_params_dict = scene_params_dict, 
            modality_list = modality_list, 
            im_params_dict = im_params_dict, 
            BRDF_params_dict = BRDF_params_dict, 
            lighting_params_dict = lighting_params_dict, 
    )

        '''
        load everything
        '''
        self.load_modalities_3D()

    @property
    def valid_modalities_3D(self):
        return ['layout', 'shapes']

    @property
    def valid_modalities_3D_vis(self):
        return ['layout', 'shapes', 'emitters', 'emitter_envs']

    @property
    def if_has_layout(self):
        return all([_ in self.modality_list for _ in ['layout']])

    @property
    def if_has_shapes(self): # objs + emitters
        return all([_ in self.modality_list for _ in ['shapes']])

    @property
    def if_has_colors(self):
        return self.if_loaded_colors

    def load_modalities_3D(self):
        for _ in self.modality_list:
            if _ == 'layout': self.load_layout()
            if _ == 'shapes': self.load_shapes(self.shape_params_dict) # shapes of 1(i.e. furniture) + emitters

        if self.if_vis_debug_with_plt:
            # plt.draw()
            plt.show()


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
        self.shapes_root, self.envmaps_root = get_list_of_keys(self.root_path_dict, ['shapes_root', 'envmaps_root'], [PosixPath, PosixPath])
        main_xml_file = self.scene_xml_path / ('%s.xml'%self.meta_split.split('_')[0]) # load from one of [main, mainDiffLight, mainDiffMat]
        # print(main_xml_file)
        root = get_XML_root(main_xml_file)

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
        self.bverts_list = []
        
        light_axis_list = []
        # self.num_vertices = 0
        obj_path_list = []
        self.shape_list_valid = []

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

            bbox_file_path = obj_path.replace('.obj', 'pickle')
            
            if_load_mesh = if_load_obj_mesh if not if_emitter else if_load_emitter_mesh

            if if_load_mesh:
                vertices, faces = loadMesh(obj_path) # based on L430 of adjustObjectPoseCorrectChairs.py
                bverts, bfaces = computeBox(vertices)
                if not Path(bbox_file_path).exists():
                    with open(bbox_file_path, "wb") as f:
                        pickle.dump(dict(bverts=bverts, bfaces=bfaces), f)

            else:
                assert Path(bbox_file_path).exists(), 'Need to run once with if_load_mesh=True to dump pickle files for shapes'
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
                self.faces_list.append(faces)
                # self.num_vertices += vertices_transformed.shape[0]
            else:
                self.vertices_list.append(None)
                self.faces_list.append(None)

            self.bverts_list.append(bverts_transformed)
            
            self.shape_list_valid.append(shape)

        print(blue_text('[openroomsScene3D] DONE. load_shapes'),len(self.shape_list_valid))

        if self.if_vis_debug_with_plt:
            ax = None
            if 'shapes' in self.modality_list_vis:
                ax = self.vis_shapes()
            if 'emitters' in self.modality_list_vis:
                self.vis_emitters(ax)
            if 'emitter_envs' in self.modality_list_vis:
                self.vis_emitter_envs()

    def load_layout(self):
        '''
        load and visualize layout in 3D & 2D: 

        images/demo_layout_3D.png
        images/demo_layout_2D.png

        '''

        print(white_blue('[openroomsScene3D] load_layout for scene...'))

        self.layout_root = get_list_of_keys(self.root_path_dict, ['layout_root'], [PosixPath])[0]
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

        if self.if_vis_debug_with_plt and 'layout' in self.modality_list_vis:
            self.vis_layout()

    def vis_layout(self):
        fig = plt.figure(figsize=(15, 4))
        fig.suptitle('layout mesh 3D')

        ax1 = plt.subplot(131, projection='3d')
        ax1.set_title('original layout mesh')
        ax1.set_proj_type('ortho')
        ax1.set_aspect("auto")
        vis_axis(ax1)
        v_pairs = v_pairs_from_v3d_e(self.v, self.e)
        for v_pair in v_pairs:
            ax1.plot3D(v_pair[0], v_pair[1], v_pair[2])

        ax2 = plt.subplot(132, projection='3d')
        ax2.set_title('layout mesh->skeleton')
        ax2.set_proj_type('ortho')
        ax2.set_aspect("auto")
        vis_axis(ax2)
        v_pairs = v_pairs_from_v3d_e(self.v_skeleton, self.e_skeleton)
        for v_pair in v_pairs:
            ax2.plot3D(v_pair[0], v_pair[1], v_pair[2])
        vis_cube_plt(self.layout_box_3d, ax2, 'b', linestyle='--')

        ax3 = plt.subplot(133, projection='3d')
        ax3.set_title('[FINAL COORDS] layout skeleton bbox in transformed coordinates')
        ax3.set_proj_type('ortho')
        ax3.set_aspect("auto")
        vis_axis(ax3)
        v_pairs = v_pairs_from_v3d_e(self.v_skeleton_transformed, self.e_skeleton)
        for v_pair in v_pairs:
            ax3.plot3D(v_pair[0], v_pair[1], v_pair[2])
        for v_idx, v in enumerate(self.layout_box_3d_transformed):
            ax3.text(v[0], v[1], v[2], str(v_idx))
        ax3.view_init(elev=-71, azim=-65)

        plt.show(block=False)

        # visualize floor of original layout, and rectangle hull, in 2D
        fig = plt.figure()
        fig.suptitle('layout 2D')
        # ax = fig.gca()
        ax1 = plt.subplot(111)
        ax1.set_title('layout 2D BEV')
        ax1.set_aspect("equal")
        v_pairs = v_pairs_from_v2d_e(self.v_2d, self.e_2d)
        for v_pair in v_pairs:
            ax1.plot(v_pair[0], v_pair[1])

        hull_pair_idxes = [[0, 1], [1, 2], [2, 3], [3, 0]]
        hull_v_pairs = [([self.layout_hull_2d[idx[0]][0], self.layout_hull_2d[idx[1]][0]], [self.layout_hull_2d[idx[0]][1], self.layout_hull_2d[idx[1]][1]]) for idx in hull_pair_idxes]
        for v_pair in hull_v_pairs:
            ax1.plot(v_pair[0], v_pair[1], 'b--')
        plt.grid()

        plt.show(block=False)
        
    def vis_shapes(self):

        self.load_colors()

        fig = plt.figure(figsize=(10, 10))
        ax = plt.subplot(111, projection='3d')
        ax.set_proj_type('ortho')
        # v_pairs = v_pairs_from_v3d_e(self.v_skeleton, self.e_skeleton)
        # for v_pair in v_pairs:
        #     ax.plot3D(v_pair[0], v_pair[1], v_pair[2])
        ax.view_init(elev=-36, azim=89)
        vis_axis(ax)

        for shape_idx, shape in enumerate(self.shape_list_valid):
            if 'scene' in shape['filename']:
                continue

            bverts_transformed = self.bverts_list[shape_idx]
            if_emitter = shape['if_in_emitter_dict']
            
            if np.amax(bverts_transformed[:, 1]) <= np.amin(bverts_transformed[:, 1]):
                obj_color = 'k' # black for invalid objects
                cat_name = 'INVALID'
                # continue/uv_mapped/
            else:
                # obj_color = 'r'
                obj_path = shape['filename']
                cat_id_str = str(obj_path).split('/')[-3]
                assert cat_id_str in self.OR_mapping_cat_str_to_id_name_dict, 'not valid cat_id_str: %s; %s'%(cat_id_str, obj_path)
                cat_id, cat_name = self.OR_mapping_cat_str_to_id_name_dict[cat_id_str]
                obj_color = self.OR_mapping_id_to_color_dict[cat_id]
                obj_color = [float(x)/255. for x in obj_color]
                linestyle = '-'
                linewidth = 1
                if if_emitter:
                    linewidth = 3
                    linestyle = '--'

            vis_cube_plt(bverts_transformed, ax, color=obj_color, linestyle=linestyle, linewidth=linewidth, label=cat_name)
            print(if_emitter, shape_idx, shape['id'], cat_id, cat_name, Path(obj_path).relative_to(self.shapes_root))

            
        if_layout = 'layout' in self.modality_list

        if if_layout:
            vis_cube_plt(self.layout_box_3d_transformed, ax, 'b', '--')
        
        # ===== cameras
        # vis_axis_xyz(ax, xaxis.flatten(), yaxis.flatten(), zaxis.flatten(), origin.flatten(), suffix='_c') # cameras

        # a = Arrow3D([origin[0][0], lookat[0][0]*2-origin[0][0]], [origin[1][0], lookat[1][0]*2-origin[1][0]], [origin[2][0], lookat[2][0]*2-origin[2][0]], mutation_scale=20,
        #                 lw=1, arrowstyle="->", color="k")
        # ax.add_artist(a)
        # a_up = Arrow3D([origin[0][0], origin[0][0]+up[0][0]], [origin[1][0], origin[1][0]+up[1][0]], [origin[2][0], origin[2][0]+up[2][0]], mutation_scale=20,
        #                 lw=1, arrowstyle="->", color="r")
        # ax.add_artist(a_up)

        ax.set_box_aspect([1,1,1])
        set_axes_equal(ax) # IMPORTANT - this is also required
        ax.view_init(elev=-55, azim=120)

        # plt.show(block=False)
        plt.draw()

        return ax

    def vis_emitters(self, ax=None):

        if ax is None:
            fig = plt.figure(figsize=(10, 10))
            ax = plt.subplot(111, projection='3d')
            ax.set_proj_type('ortho')
            # v_pairs = v_pairs_from_v3d_e(self.v_skeleton, self.e_skeleton)
            # for v_pair in v_pairs:
            #     ax.plot3D(v_pair[0], v_pair[1], v_pair[2])
            ax.view_init(elev=-36, azim=89)
            vis_axis(ax)

        for shape_idx, shape in enumerate(self.shape_list_valid):
            if not shape['if_in_emitter_dict']:
                continue

            # print('===EMITTER', shape['random_id'], shape_idx, shape['emitter_prop']['if_env'], shape['emitter_prop'].keys())
            coords = shape['emitter_prop']['box3D_world']['coords']
            if shape['emitter_prop']['if_lit_up']:
                vis_cube_plt(coords, ax, 'k', '--')
            else:
                vis_cube_plt(coords, ax, 'gray', '--')

            bverts_transformed = self.bverts_list[shape_idx]
            light_center = np.mean(bverts_transformed, 0).flatten()

            # if 'axis_world' in shape['emitter_prop']:
            if shape['emitter_prop']['obj_type'] == 'window':
                label_SG_list = ['', 'Sky', 'Grd']

                for label_SG, color, lw in zip(label_SG_list, ['k', 'b', 'g'], [5, 3, 3]):
                    light_axis_world = np.asarray(shape['emitter_prop']['axis%s_world'%label_SG]).flatten()
                    light_axis_world = light_axis_world / np.linalg.norm(light_axis_world)

                    light_axis_end = np.asarray(light_center).reshape(3,) + light_axis_world * np.log(shape['emitter_prop']['intensity'+label_SG]) * 0.5
                    light_axis_end = light_axis_end.flatten()

                    a_light = Arrow3D([light_center[0], light_axis_end[0]], [light_center[1], light_axis_end[1]], [light_center[2], light_axis_end[2]], mutation_scale=20,
                                    lw=lw, arrowstyle="-|>", color=color)
                    ax.add_artist(a_light)

                # axes for window half envmaps; transformation consistent with [renderOpenRooms] code/utils_OR/func_render_emitter_N_ambient -> axis = np.sin(theta) * np.cos(phi) * envAxis_x \...
                env_x_axis = shape['emitter_prop']['envAxis_x_world'].reshape((3,))
                env_y_axis = shape['emitter_prop']['envAxis_y_world'].reshape((3,))
                env_z_axis = shape['emitter_prop']['envAxis_z_world'].reshape((3,))
                vis_axis_xyz(ax, env_x_axis, env_y_axis, env_z_axis, origin=light_center, suffix='_{env}', colors=['r', 'g', 'b'])

            # axes for emitter bbox
            # light_z_axis = shape['emitter_prop']['box3D_world']['zAxis'].reshape((3,))
            # light_y_axis = shape['emitter_prop']['box3D_world']['yAxis'].reshape((3,))
            # light_x_axis = shape['emitter_prop']['box3D_world']['xAxis'].reshape((3,))
            # vis_axis_xyz(ax, light_x_axis, light_y_axis, light_z_axis, origin=light_center, suffix='_bbox', colors=['r', 'g', 'b'])

        plt.draw()

    def vis_emitter_envs(self):
        '''
        visualize 
            (1) emitter_env 
            (2) envmaps (SGs) converted from all windows ()
        
        images/demo_emitter_envs_3D.png
        compare with:
        - images/demo_emitters_3D_re1.png # note the X_w, Y_w, Z_w, as also appeared in the GLOBAL envmaps
        - images/demo_emitters_3D_re2.png # another viewpoint; note the X_env, Y_env, Z_env, as also appeared in the LOCAL envmaps
        
        note that in (2), when approxing renderer half envmaps with 3SGs, the half envmaps are renderer **with envScale -> 1.** (see [renderOpenRooms] code/utils_OR/func_render_emitter_N_ambient -> scale.set('value', str(1.)))
        '''
        env_map_path = self.emitter_env['emitter_prop']['emitter_filename']
        im_envmap_ori = load_HDR(Path(env_map_path))
        im_envmap_ori_SDR, _ = to_nonHDR(im_envmap_ori)


        self.window_3SG_list_of_dicts = []

        self.global_env_scale = self.emitter_env['emitter_prop']['emitter_scale']

        for shape_idx, shape in enumerate(self.shape_list_valid):
            if not shape['if_in_emitter_dict']:
                continue
            if shape['emitter_prop']['obj_type'] == 'window':
                label_SG_list = ['', 'Sky', 'Grd']
                window_3SG_dict = {}
                for label_SG, color, lw in zip(label_SG_list, ['k', 'b', 'g'], [5, 3, 3]):
                    light_axis_world = np.asarray(shape['emitter_prop']['axis%s_world'%label_SG]).flatten()
                    light_axis_world = light_axis_world / np.linalg.norm(light_axis_world)
                    window_3SG_dict['light_axis%s_world'%label_SG] = light_axis_world

                    window_3SG_dict['weight%s_SG'%label_SG] = np.asarray(shape['emitter_prop']['intensity%s'%label_SG])
                    print(shape['emitter_prop']['intensity%s'%label_SG])
                    window_3SG_dict['lamb%s_SG'%label_SG] = shape['emitter_prop']['lamb%s'%label_SG]

                window_3SG_dict['imHalfEnvName'] = shape['emitter_prop']['imHalfEnvName']
                window_3SG_dict['recHalfEnvName'] = shape['emitter_prop']['recHalfEnvName']

                self.window_3SG_list_of_dicts.append(window_3SG_dict)

        num_windows = len(self.window_3SG_list_of_dicts)
        total_rows = 1 + num_windows * 2

        fig = plt.figure(figsize=(15, total_rows*4))
        ax = plt.subplot(total_rows, 2, 1)
        ax.set_title('GT - GLOBAL envmap (world coords; Y_w+: up)')

        vis_envmap_plt(ax, im_envmap_ori_SDR, ['Z_w-', 'X_w+', 'Z_w+', 'X_w-'])

        for window_idx, self.window_3SG_list_of_dicts in enumerate(self.window_3SG_list_of_dicts):
            ax = plt.subplot(total_rows, 2, window_idx*4+3)
            ax.set_title('[window %d] 3SG - GLOBAL envmap (world coords; Y_w+: up)'%window_idx)
            _3SG_envmap = render_3SG_envmap(window_3SG_dict, intensity_scale=1.) # [IMPORTANT] intensity_scale=1. because half envmaps were rendered with envScale->1.
            _3SG_envmap_SDR, _ = to_nonHDR(_3SG_envmap)
            vis_envmap_plt(ax, _3SG_envmap_SDR, ['Z_w-', 'X_w+', 'Z_w+', 'X_w-'])

            ax = plt.subplot(total_rows, 2, window_idx*4+4)
            ax.set_title('[window %d] GT - HALF envmap (LOCAL ENV coords; Z_env+: inside)'%window_idx)
            im_half_env = load_HDR(Path(window_3SG_dict['imHalfEnvName']))
            im_half_env_SDR, _ = to_nonHDR(im_half_env)
            vis_envmap_plt(ax, im_half_env_SDR, ['X_env-', 'Y_env-', 'X_env+', 'Y_env+'])

            ax = plt.subplot(total_rows, 2, window_idx*4+6)
            ax.set_title('[window %d] 3SG - HALF envmap (LOCAL ENV coords; Z_env+: inside)'%window_idx)
            im_half_env = load_HDR(Path(window_3SG_dict['recHalfEnvName']))
            im_half_env_SDR, _ = to_nonHDR(im_half_env)
            vis_envmap_plt(ax, im_half_env_SDR, ['X_env-', 'Y_env-', 'X_env+', 'Y_env+'])

        plt.show()


            
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
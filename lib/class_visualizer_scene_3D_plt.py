import numpy as np
from tqdm import tqdm
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import open3d.visualization as vis

from pathlib import Path, PosixPath
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(suppress=True)

import trimesh
from sympy import Point3D, Line3D, Plane, sympify, Rational
from copy import deepcopy
import torch
import copy

# Import the library using the alias "mi"
import mitsuba as mi
# Set the variant of the renderer
# from lib.global_vars import mi_variant
# mi.set_variant(mi_variant)

from lib.class_openroomsScene2D import openroomsScene2D
from lib.class_openroomsScene3D import openroomsScene3D
from lib.class_mitsubaScene3D import mitsubaScene3D

from lib.utils_OR.utils_OR_mesh import v_pairs_from_v3d_e, v_pairs_from_v2d_e
from lib.utils_OR.utils_OR_vis_3D import vis_cube_plt, vis_axis, vis_axis_xyz, set_axes_equal, Arrow3D
from lib.utils_io import load_HDR, to_nonHDR
from lib.utils_OR.utils_OR_emitter import load_emitter_dat_world, render_3SG_envmap, vis_envmap_plt

class visualizer_scene_3D_plt(object):
    '''
    matplotlib visualizer (debug only; for better visualizations, use lib.class_visualizer_openroomsScene_o3d.visualizer_openroomsScene_o3d)
    A class used to **visualize** OpenRooms (public/public-re versions) scene contents (2D/2.5D per-pixel DENSE properties for inverse rendering + 3D semantics).
    '''
    def __init__(
        self, 
        openrooms_scene, 
        modality_list_vis: list, 
    ):

        assert type(openrooms_scene) in [openroomsScene2D, openroomsScene3D, mitsubaScene3D], '[visualizer_openroomsScene] has to take an object of openroomsScene, openroomsScene3D, or mitsubaScene3D!'

        self.os = openrooms_scene

        self.modality_list_vis = list(set(modality_list_vis))
        for _ in self.modality_list_vis:
            assert _ in self.valid_modalities_3D_vis, 'Invalid modality_vis: %s'%_

    @property
    def valid_modalities_3D_vis(self):
        return ['layout', 'shapes', 'poses', 'emitters', 'emitter_envs']

    def vis_3d_with_plt(self):
        ax = None
        if 'poses' in self.modality_list_vis:
            assert 'shapes' in self.modality_list_vis or 'layout' in self.modality_list_vis, 'poses can only be visualized in layout view or shapes view!'
        if 'shapes' in self.modality_list_vis:
            self.vis_shapes(if_poses='poses' in self.modality_list_vis)
        if 'layout' in self.modality_list_vis:
            self.vis_layout(if_poses='poses' in self.modality_list_vis)
        if 'emitters' in self.modality_list_vis:
            self.vis_emitters(ax)
        if 'emitter_envs' in self.modality_list_vis:
            self.vis_emitter_envs()
        # if 'mi_depth_normal' in self.modality_list_vis:
        #     self.vis_mi_depth_normal()
        # if 'mi_seg' in self.modality_list_vis:
        #     self.vis_mi_seg()
        plt.show()

    def vis_layout(self, if_poses=False):
        '''
        images/demo_layout_3D.png
        images/demo_layout_2D.png
        images/demo_layout_poses_2D_view1_oldpose.png
        '''

        assert self.os.if_has_layout
        
        fig = plt.figure(figsize=(15, 4))
        fig.suptitle('layout mesh 3D')

        ax1 = plt.subplot(131, projection='3d')
        ax1.set_title('original layout mesh')
        ax1.set_proj_type('ortho')
        ax1.set_aspect("auto")
        vis_axis(ax1)
        v_pairs = (self.os.v, self.os.e)
        for v_pair in v_pairs:
            ax1.plot3D(v_pair[0], v_pair[1], v_pair[2])

        ax2 = plt.subplot(132, projection='3d')
        ax2.set_title('layout mesh->skeleton')
        ax2.set_proj_type('ortho')
        ax2.set_aspect("auto")
        vis_axis(ax2)
        v_pairs = v_pairs_from_v3d_e(self.os.v_skeleton, self.os.e_skeleton)
        for v_pair in v_pairs:
            ax2.plot3D(v_pair[0], v_pair[1], v_pair[2])
        vis_cube_plt(self.os.layout_box_3d, ax2, 'b', linestyle='--')

        ax3 = plt.subplot(133, projection='3d')
        ax3.set_title('[FINAL COORDS] layout skeleton bbox in transformed coordinates')
        ax3.set_proj_type('ortho')
        ax3.set_aspect("auto")
        vis_axis(ax3)
        v_pairs = v_pairs_from_v3d_e(self.os.v_skeleton_transformed, self.os.e_skeleton)
        for v_pair in v_pairs:
            ax3.plot3D(v_pair[0], v_pair[1], v_pair[2])
        for v_idx, v in enumerate(self.os.layout_box_3d_transformed):
            ax3.text(v[0], v[1], v[2], str(v_idx))
        ax3.view_init(elev=-71, azim=-65)

        plt.show(block=False)

        # visualize floor of original layout, and rectangle hull, in 2D
        fig = plt.figure(figsize=(10, 10))
        fig.suptitle('layout 2D')
        # ax = fig.gca()
        ax = plt.subplot(111)
        ax.set_title('layout 2D BEV')
        ax.set_aspect("equal")
        v_pairs = v_pairs_from_v2d_e(self.os.v_2d_transformed, self.os.e_2d)
        for v_pair in v_pairs:
            ax.plot(v_pair[0], v_pair[1])

        hull_pair_idxes = [[0, 1], [1, 2], [2, 3], [3, 0]]
        hull_v_pairs = [([self.os.layout_hull_2d_transformed[idx[0]][0], self.os.layout_hull_2d_transformed[idx[1]][0]], [self.os.layout_hull_2d_transformed[idx[0]][1], self.os.layout_hull_2d_transformed[idx[1]][1]]) for idx in hull_pair_idxes]
        for v_pair in hull_v_pairs:
            ax.plot(v_pair[0], v_pair[1], 'b--')

        if if_poses:
            assert self.os.if_has_poses
            for (origin, lookatvector, up) in self.os.origin_lookatvector_up_list:
                origin_ = origin.flatten()
                lookatvector_ = lookatvector.flatten()
                lookat_ = origin_ + lookatvector.flatten() * 0.5 # arrows are 0.5 in length
                ax.scatter(origin_[0], origin_[2], color='k', marker='*')
                ax.arrow(origin_[0], origin_[2], lookatvector_[0], lookatvector_[2], lw=1, joinstyle='bevel', color='b')
                # a = Arrow3D([origin_[0], lookat_[0]], [origin_[1], lookat_[1]], [origin_[2], lookat_[2]], mutation_scale=5,
                #     lw=1, arrowstyle="->", color='b')
                # ax.add_artist(a)

        plt.grid()

        plt.show(block=False)
        
    def vis_shapes(self, if_poses=False):
        '''
        images/demo_shapes_3D.png
        images/demo_shapes_poses_3D_view1_oldpose.png
        images/demo_shapes_poses_3D_view2_oldpose.png
        '''
        if not self.os.if_has_colors:
            self.os.load_colors()

        fig = plt.figure(figsize=(10, 10))
        ax = plt.subplot(111, projection='3d')
        ax.set_proj_type('ortho')
        # v_pairs = v_pairs_from_v3d_e(self.v_skeleton, self.e_skeleton)
        # for v_pair in v_pairs:
        #     ax.plot3D(v_pair[0], v_pair[1], v_pair[2])
        ax.view_init(elev=-36, azim=89)
        vis_axis(ax)

        for shape_idx, shape in enumerate(self.os.shape_list_valid):
            if 'scene' in shape['filename']:
                continue

            bverts_transformed = self.os.bverts_list[shape_idx]
            if_emitter = shape['if_in_emitter_dict']
            
            if np.amax(bverts_transformed[:, 1]) <= np.amin(bverts_transformed[:, 1]):
                obj_color = 'k' # black for invalid objects
                cat_name = 'INVALID'
                # continue/uv_mapped/
            else:
                # obj_color = 'r'
                obj_path = shape['filename']
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

            vis_cube_plt(bverts_transformed, ax, color=obj_color, linestyle=linestyle, linewidth=linewidth, label=cat_name)
            print(if_emitter, shape_idx, shape['id'], cat_id, cat_name, Path(obj_path).relative_to(self.os.shapes_root))

        if_layout = 'layout' in self.modality_list_vis

        if if_layout:
            vis_cube_plt(self.os.layout_box_3d_transformed, ax, 'b', '--')
        
        if if_poses:
            assert self.os.if_has_poses
            for (origin, lookatvector, up) in self.os.origin_lookatvector_up_list:
                origin_ = origin.flatten()
                lookat_ = origin_ + lookatvector.flatten() * 0.5 # arrows are 0.5 in length
                ax.scatter(origin_[0], origin_[1], origin_[2], color='k', marker='*')
                a = Arrow3D([origin_[0], lookat_[0]], [origin_[1], lookat_[1]], [origin_[2], lookat_[2]], mutation_scale=5,
                    lw=1, arrowstyle="->", color='b')
                ax.add_artist(a)

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
        # ax.view_init(elev=-55, azim=120) # side view
        ax.view_init(elev=0, azim=90) # BEV

        # plt.show(block=False)
        plt.draw()

        return ax

    def vis_emitters(self, ax=None):
        '''
        visualize in 3D:
            - SGs representation of emitters

        - images/demo_emitters_3D_re1.png # note the X_w, Y_w, Z_w, as also appeared in the GLOBAL envmaps
        - images/demo_emitters_3D_re2.png # another viewpoint; note the X_env, Y_env, Z_env, as also appeared in the LOCAL envmaps
        '''

        if ax is None:
            fig = plt.figure(figsize=(10, 10))
            ax = plt.subplot(111, projection='3d')
            ax.set_proj_type('ortho')
            # v_pairs = v_pairs_from_v3d_e(self.v_skeleton, self.e_skeleton)
            # for v_pair in v_pairs:
            #     ax.plot3D(v_pair[0], v_pair[1], v_pair[2])
            ax.view_init(elev=-36, azim=89)
            vis_axis(ax)

        for shape_idx, shape in enumerate(self.os.shape_list_valid):
            if not shape['if_in_emitter_dict']:
                continue

            # print('===EMITTER', shape['random_id'], shape_idx, shape['emitter_prop']['if_env'], shape['emitter_prop'].keys())
            coords = shape['emitter_prop']['box3D_world']['coords']
            if shape['emitter_prop']['if_lit_up']:
                vis_cube_plt(coords, ax, 'k', '--')
            else:
                vis_cube_plt(coords, ax, 'gray', '--')

            bverts_transformed = self.os.bverts_list[shape_idx]
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
        visualize in 2D:
            (1) emitter_env 
            (2) envmaps (SGs) converted from all windows
        
        images/demo_emitter_envs_3D.png

        compare with:
        - images/demo_emitters_3D_re1.png # note the X_w, Y_w, Z_w, as also appeared in the GLOBAL envmaps
        - images/demo_emitters_3D_re2.png # another viewpoint; note the X_env, Y_env, Z_env, as also appeared in the LOCAL envmaps
        
        note that in (2), when approxing renderer half envmaps with 3SGs, the half envmaps are renderer **with envScale -> 1.** (see [renderOpenRooms] code/utils_OR/func_render_emitter_N_ambient -> scale.set('value', str(1.)))
        '''
        env_map_path = self.os.emitter_env['emitter_prop']['emitter_filename']
        im_envmap_ori = load_HDR(Path(env_map_path))
        im_envmap_ori_SDR, _ = to_nonHDR(im_envmap_ori)

        self.os.window_3SG_list_of_dicts = []

        self.os.global_env_scale = self.os.emitter_env['emitter_prop']['emitter_scale']

        for shape_idx, shape in enumerate(self.os.shape_list_valid):
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

                self.os.window_3SG_list_of_dicts.append(window_3SG_dict)

        num_windows = len(self.os.window_3SG_list_of_dicts)
        total_rows = 1 + num_windows * 2

        fig = plt.figure(figsize=(15, total_rows*4))
        ax = plt.subplot(total_rows, 2, 1)
        ax.set_title('GT - GLOBAL envmap (world coords; Y_w+: up)')

        vis_envmap_plt(ax, im_envmap_ori_SDR, ['Z_w-', 'X_w+', 'Z_w+', 'X_w-'])

        for window_idx, self.os.window_3SG_list_of_dicts in enumerate(self.os.window_3SG_list_of_dicts):
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

        plt.show(block=False)
import sys

# host = 'mm1'
host = 'apple'
PATH_HOME = {
    'apple': '/Users/jerrypiglet/Documents/Projects/OpenRooms_RAW_loader', 
    'mm1': '/home/ruizhu/Documents/Projects/OpenRooms_RAW_loader', 
    'qc': '/usr2/rzh/Documents/Projects/directvoxgorui', 
}[host]
OR_RAW_ROOT = {
    'apple': '/Users/jerrypiglet/Documents/Projects/data', 
    'mm1': '/newfoundland2/ruizhu/siggraphasia20dataset', 
    'qc': '', 
}[host]

sys.path.insert(0, PATH_HOME)
from pathlib import Path
import numpy as np
np.set_printoptions(suppress=True)

from lib.class_openroomsScene3D import openroomsScene3D
from lib.class_visualizer_scene_2D import visualizer_scene_2D
from lib.class_visualizer_scene_3D_o3d import visualizer_scene_3D_o3d
from lib.class_visualizer_openroomsScene_3D_plt import visualizer_openroomsScene_3D_plt
from lib.class_renderer_openroomsScene_3D import renderer_openroomsScene_3D

from lib.utils_misc import str2bool
import argparse
parser = argparse.ArgumentParser()
# visualizers
parser.add_argument('--vis_3d_plt', type=str2bool, nargs='?', const=True, default=False, help='whether to visualize 3D with plt for debugging')
parser.add_argument('--vis_3d_o3d', type=str2bool, nargs='?', const=True, default=True, help='whether to visualize in open3D')
parser.add_argument('--vis_2d_plt', type=str2bool, nargs='?', const=True, default=False, help='whether to show projection onto one image with plt (e.g. layout, object bboxes')
parser.add_argument('--if_shader', type=str2bool, nargs='?', const=True, default=False, help='')
# options for visualizers
parser.add_argument('--pcd_color_mode_dense_geo', type=str, default='rgb', help='colormap for all points in fused geo')
parser.add_argument('--if_set_pcd_color_mi', type=str2bool, nargs='?', const=True, default=False, help='if create color map for all points of Mitsuba; required: input_colors_tuple')
parser.add_argument('--if_add_rays_from_renderer', type=str2bool, nargs='?', const=True, default=False, help='if add camera rays and emitter sample rays from renderer')
# differential renderer
parser.add_argument('--render_3d', type=str2bool, nargs='?', const=True, default=False, help='differentiable surface rendering')
parser.add_argument('--renderer_option', type=str, default='PhySG', help='differentiable renderer option')
# debug
parser.add_argument('--if_debug_info', type=str2bool, nargs='?', const=True, default=False, help='if show debug info')
opt = parser.parse_args()

# intrinsics_path = Path(PATH_HOME) / 'data/intrinsic.txt'
semantic_labels_root = Path(PATH_HOME) / 'files_openrooms'
layout_root = Path(OR_RAW_ROOT) / 'layoutMesh'
shapes_root = Path(OR_RAW_ROOT) / 'uv_mapped'
envmaps_root = Path(OR_RAW_ROOT) / 'EnvDataset' # not publicly availale
shape_pickles_root = Path(PATH_HOME) / 'data/openrooms_shape_pickles' # for caching shape bboxes so that we do not need to load meshes very time if only bboxes are wanted
if not shape_pickles_root.exists():
    shape_pickles_root.mkdir()

'''
The classroom scene: one lamp (lit up) + one window (less sun)
data/public_re_3/main_xml1/scene0552_00_more/im_4.png
'''
# meta_split = 'main_xml1'
# scene_name = 'scene0552_00_more'
# frame_ids = [0, 1, 2, 3, 4] + list(range(5, 87, 10))

'''
The classroom scene: one lamp (dark) + one window (directional sun)
data/public_re_3/mainDiffLight_xml1/scene0552_00_more/im_4.png
'''
meta_split = 'mainDiffLight_xml1'
scene_name = 'scene0552_00_more'
frame_ids = [0, 1, 2, 3, 4] + list(range(5, 87, 10))
frame_ids = [2]
# frame_ids = list(range(87))

'''
The lounge with very specular floor and 3 lamps
data/public_re_3/main_xml/scene0008_00_more/im_58.png
'''
meta_split = 'main_xml'
scene_name = 'scene0008_00_more'
# frame_ids = [0, 1, 2, 3, 4] + list(range(5, 102, 10))
# frame_ids = [114]
frame_ids = list(range(102))

'''
The conference room with one lamp
data/public_re_3/main_xml/scene0005_00_more/im_3.png
'''
meta_split = 'main_xml'
scene_name = 'scene0005_00_more'
frame_ids = [0, 1, 2, 3, 4] + list(range(5, 102, 10))
# frame_ids = [3]
# frame_ids = list(range(102))
frame_ids = list(range(3, 102, 10))

'''
- more & better cameras
'''
dataset_version = 'public_re_3_v3pose_2048'
meta_split = 'main_xml'
scene_name = 'scene0008_00_more'
frame_ids = list(range(0, 345, 10))
# frame_ids = [321]

base_root = Path(PATH_HOME) / 'data' / dataset_version
xml_root = Path(PATH_HOME) / 'data' / dataset_version / 'scenes'

openrooms_scene = openroomsScene3D(
    if_debug_info=opt.if_debug_info, 
    host=host, 
    root_path_dict = {'PATH_HOME': Path(PATH_HOME), 'rendering_root': base_root, 'xml_scene_root': xml_root, 'semantic_labels_root': semantic_labels_root, 'shape_pickles_root': shape_pickles_root, 
        'layout_root': layout_root, 'shapes_root': shapes_root, 'envmaps_root': envmaps_root}, 
    scene_params_dict={'meta_split': meta_split, 'scene_name': scene_name, 'frame_id_list': frame_ids}, 
    # modality_list = ['im_sdr', 'im_hdr', 'seg', 'poses', 'albedo', 'roughness', 'depth', 'normal', 'lighting_SG', 'lighting_envmap'], 
    modality_list = [
        'im_sdr', 
        'poses', 
        'seg', 'im_hdr', 
        'albedo', 'roughness', 
        # 'depth', 'normal', 
        # 'lighting_SG', 
        # 'lighting_envmap', 
        'layout', 
        'shapes', # objs + emitters, geometry shapes + emitter properties
        'mi', # mitsuba scene, loading from scene xml file
        ], 
    im_params_dict={
        'im_H_load': 480, 'im_W_load': 640, 
        # 'im_H_resize': 240, 'im_W_resize': 320
        'im_H_resize': 120, 'im_W_resize': 160, # to use for rendering so that im dimensions == lighting dimensions
        'if_direct_lighting': False, # if load direct lighting envmaps and SGs inetad of total lighting
        }, 
    lighting_params_dict={
        'env_row': 120, 'env_col': 160, 'SG_num': 12, 
        # 'env_height': 16, 'env_width': 32, 
        'env_height': 8, 'env_width': 16, 
        'if_convert_lighting_SG_to_global': True, 
        'if_use_mi_geometry': True, 
    }, 
    shape_params_dict={
        'if_load_obj_mesh': True, # set to False to not load meshes for objs (furniture) to save time
        'if_load_emitter_mesh': True,  # default True: to load emitter meshes, because not too many emitters
        },
    emitter_params_dict={
        'N_ambient_rep': '3SG-SkyGrd'
        },
    mi_params_dict={
        'if_also_dump_xml_with_lit_lamps_only': True,  # True: to dump a second file containing lit-up lamps only
        'debug_dump_mesh': True, # [DEBUG] True: to dump all object meshes to mitsuba/meshes_dump; load all .ply files into MeshLab to view the entire scene: images/demo_mitsuba_dump_meshes.png
        'debug_render_test_image': False, # [DEBUG][slow] True: to render an image with first camera, usig Mitsuba: images/demo_mitsuba_render.png
        'if_sample_rays_pts': True, # True: to sample camera rays and intersection pts given input mesh and camera poses
        'if_get_segs': True, # True: to generate segs similar to those in openroomsScene2D.load_seg()
        },
)

'''
Matploblib 2D viewer
'''
if opt.vis_2d_plt:
    visualizer_2D = visualizer_scene_2D(
        openrooms_scene, 
        modality_list_vis=[
            'im', 
            'layout', 
            # 'shapes', 
            # 'depth', 'mi_depth', 
            # 'normal', 'mi_normal', # compare depth & normal maps from mitsuba sampling VS OptixRenderer: **mitsuba does no anti-aliasing**: images/demo_mitsuba_ret_depth_normals_2D.png
            # 'lighting_SG', # convert to lighting_envmap and vis: images/demo_lighting_SG_envmap_2D_plt.png
            # 'lighting_envmap', 
            # 'seg_area', 'seg_env', 'seg_obj', 
            # 'mi_seg_area', 'mi_seg_env', 'mi_seg_obj', # compare segs from mitsuba sampling VS OptixRenderer: **mitsuba does no anti-aliasing**: images/demo_mitsuba_ret_seg_2D.png
            ], 
        # frame_idx_list=[0, 1, 2, 3, 4], 
        frame_idx_list=[0], 
    )
    visualizer_2D.vis_2d_with_plt(
        lighting_params={
            'lighting_scale': 0.01, # rescaling the brightness of the envmap
            }, 
            )

'''
Matploblib 3D viewer
'''
if opt.vis_3d_plt:
    visualizer_3D_plt = visualizer_openroomsScene_3D_plt(
        openrooms_scene, 
        modality_list_vis = [
            'layout', 
            'poses', # camera center + optical axis
            # 'shapes', # boxes and labels (no meshes in plt visualization)
            # 'emitters', # emitter properties
            # 'emitter_envs', # emitter envmaps for (1) global envmap (2) half envmap & SG envmap of each window
            ], 
    )
    visualizer_3D_plt.vis_3d_with_plt()

'''
Differential renderers
'''
if opt.render_3d:
    renderer_3D = renderer_openroomsScene_3D(
        openrooms_scene, 
        renderer_option=opt.renderer_option, 
        host=host, 
        renderer_params={
            'pts_from': 'mi', 
        }
    )
    
    renderer_return_dict = renderer_3D.render(
        frame_idx=0, 
        if_show_rendering_plt=True, 
        render_params={
            'max_plate': 256, 
            'emitter_type_index': ('lamp', 0), 
        })
    
    if opt.renderer_option == 'ZQ_emitter':
        ts = np.median(renderer_return_dict['ts'], axis=1)
        visibility = np.amax(renderer_return_dict['visibility'], axis=1)
        print('visibility', visibility.shape, np.sum(visibility)/float(visibility.shape[0]))
        # from scipy import stats
        # visibility = stats.mode(renderer_return_dict['visibility'], axis=1)[0].flatten()

'''
Open3D 3D viewer
'''
if opt.vis_3d_o3d:
    visualizer_3D_o3d = visualizer_scene_3D_o3d(
        openrooms_scene, 
        modality_list_vis=[
            # 'dense_geo', 
            'cameras', 
            # 'lighting_SG', # images/demo_lighting_SG_o3d.png; arrows in blue
            # 'lighting_envmap', # images/demo_lighting_envmap_o3d.png; arrows in pink
            # 'layout', 
            'shapes', # bbox and (if loaded) meshs of shapes (objs + emitters)
            'emitters', # emitter properties (e.g. SGs, half envmaps)
            'mi', # mitsuba sampled rays, pts
            ], 
        if_debug_info=opt.if_debug_info, 
    )

    if opt.if_set_pcd_color_mi:
        '''
        use results from renderer to colorize Mitsuba/fused points
        '''
        assert opt.render_3d
        assert openrooms_scene.if_has_mitsuba_rays_pts
        visualizer_3D_o3d.set_mi_pcd_color_from_input(
            # input_colors_tuple=(ts, 'dist'), # get from renderer, etc.: images/demo_mitsuba_ret_pts_pcd-color-mode-mi_renderer-t.png
            input_colors_tuple=([visibility], 'mask'), # get from renderer, etc.: images/demo_mitsuba_ret_pts_pcd-color-mode-mi_renderer-visibility-any.png
        )
    
    if opt.if_add_rays_from_renderer:
        assert opt.render_3d
        assert openrooms_scene.if_has_mitsuba_rays_pts
        
        # _pts_idx = list(range(openrooms_scene.W*openrooms_scene.H)); _sample_rate = 1000 # visualize for all scene points;
        _pts_idx = 60 * openrooms_scene.W + 80; _sample_rate = 1 # only visualize for one scene point w.r.t. all lamp points
        visibility = renderer_return_dict['visibility'][_pts_idx].reshape(-1,)[::_sample_rate]
        visualizer_3D_o3d.add_extra_geometry([
            ('rays', {
                'ray_o': renderer_return_dict['ray_o'][_pts_idx].reshape(-1, 3)[::_sample_rate][visibility==1], 
                'ray_e': renderer_return_dict['ray_e'][_pts_idx].reshape(-1, 3)[::_sample_rate][visibility==1], 
                # 'visibility': renderer_return_dict['visibility'].reshape(-1,)[::100], 
                # 't': renderer_return_dict['t'], 
            }), 
            # ('pts', {
            #     'pts': renderer_return_dict['ray_o'][_pts_idx].reshape(-1, 3)[::_sample_rate][visibility==1], 
            # }), 
            ])

    visualizer_3D_o3d.run_o3d(
        if_shader=opt.if_shader, # set to False to disable faycny shaders 
        cam_params={
            'if_cam_axis_only': False, 
            'if_cam_traj': False, 
            }, 
        dense_geo_params={
            'subsample_pcd_rate': 1, # change this according to how sparse the points you would like to be (also according to num of frame_ids)
            'if_ceiling': False, # remove ceiling points to better see the furniture 
            'if_walls': False, # remove wall points to better see the furniture 
            'if_normal': False, # turn off normals to avoid clusters
            'subsample_normal_rate_x': 2, 
            'pcd_color_mode': opt.pcd_color_mode_dense_geo, 
            }, 
        lighting_params={
            'subsample_lighting_pts_rate': 100, # change this according to how sparse the lighting arrows you would like to be (also according to num of frame_ids)
            # 'lighting_scale': 1., 
            # 'lighting_keep_ratio': 0.05, 
            # 'lighting_clip_ratio': 0.1, 
            'lighting_scale': 0.5, 
            # 'lighting_keep_ratio': 0.2, # - good for lighting_SG
            # 'lighting_clip_ratio': 0.3, 
            'lighting_keep_ratio': 0.1, # - good for lighting_envmap
            'lighting_clip_ratio': 0.2, 
            'lighting_autoscale': True, 
            }, 
        shapes_params={
            'simply_ratio': 0.1, # simply num of triangles to #triangles * simply_ratio
            'if_meshes': True, # if show meshes for objs + emitters (False: only show bboxes)
            'if_labels': False, # if show labels (False: only show bboxes)
        },
        emitters_params={
            'if_half_envmap': False, # if show half envmap as a hemisphere for window emitters (False: only show bboxes)
            'scale_SG_length': 2., 
            'if_sampling_emitter': True, # if sample and visualize points on emitter surface; show intensity as vectors along normals: images/demo_envmap_o3d_sampling.png
        },
        mi_params={
            'if_pts': False, # if show pts sampled by mi; should close to backprojected pts from OptixRenderer depth maps
            'if_pts_colorize_rgb': True, 
            'pts_subsample': 1,
            'if_ceiling': False, # remove ceiling points to better see the furniture 
            'if_walls': True, # remove wall points to better see the furniture 

            'if_cam_rays': False, 
            'cam_rays_if_pts': True, # if cam rays end in surface intersections; set to False to visualize rays of unit length
            'cam_rays_subsample': 10, 
            
            'if_normal': False, 
            'normal_subsample': 50, 
            'normal_scale': 0.2, 
        }, 
    )


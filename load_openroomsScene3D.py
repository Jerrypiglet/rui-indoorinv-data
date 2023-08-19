'''
Load and visualize re-rendered OpenRooms scenes with multi-view poses, 3D modalities.

To run on the openrooms-public (i.e. less frames per scene):



'''
import sys

# host = 'mm1'
host = 'apple'
# host = 'r4090'

from lib.global_vars import PATH_HOME_dict, INV_NERF_ROOT_dict, MONOSDF_ROOT_dict, OR_RAW_ROOT_dict, OR_MODALITY_FRAMENAME_DICT, query_host
PATH_HOME = PATH_HOME_dict[host]
sys.path.insert(0, PATH_HOME)
OR_RAW_ROOT = OR_RAW_ROOT_dict[host]
INV_NERF_ROOT = INV_NERF_ROOT_dict[host]
MONOSDF_ROOT = MONOSDF_ROOT_dict[host]

from pathlib import Path
import numpy as np
np.set_printoptions(suppress=True)
from pyhocon import ConfigFactory, ConfigTree
from lib.utils_misc import str2bool, check_exists, yellow

from lib.class_openroomsScene3D import openroomsScene3D

from lib.class_visualizer_scene_2D import visualizer_scene_2D
from lib.class_visualizer_scene_3D_o3d import visualizer_scene_3D_o3d
from lib.class_visualizer_scene_3D_plt import visualizer_scene_3D_plt

from lib.class_diff_renderer_openroomsScene_3D import diff_renderer_openroomsScene_3D

# from lib.class_eval_rad import evaluator_scene_rad
# from lib.class_eval_inv import evaluator_scene_inv
# from lib.class_eval_monosdf import evaluator_scene_monosdf
from lib.class_eval_scene import evaluator_scene_scene

import argparse
parser = argparse.ArgumentParser()
# visualizers
parser.add_argument('--vis_3d_plt', type=str2bool, nargs='?', const=True, default=False, help='whether to visualize 3D with plt for debugging')
parser.add_argument('--vis_3d_o3d', type=str2bool, nargs='?', const=True, default=True, help='whether to visualize in open3D')
parser.add_argument('--vis_2d_plt', type=str2bool, nargs='?', const=True, default=False, help='whether to show (1) pixel-space modalities (2) projection onto one image (e.g. layout, object bboxes), with plt')
parser.add_argument('--if_shader', type=str2bool, nargs='?', const=True, default=False, help='')

# options for visualizers
parser.add_argument('--pcd_color_mode_dense_geo', type=str, default='rgb', help='colormap for all points in fused geo')
parser.add_argument('--if_set_pcd_color_mi', type=str2bool, nargs='?', const=True, default=False, help='if create color map for all points of Mitsuba; required: input_colors_tuple')
parser.add_argument('--if_add_rays_from_renderer', type=str2bool, nargs='?', const=True, default=False, help='if add camera rays and emitter sample rays from renderer')

# differential renderer
parser.add_argument('--render_3d', type=str2bool, nargs='?', const=True, default=False, help='differentiable surface rendering')
parser.add_argument('--renderer_option', type=str, default='PhySG', help='differentiable renderer option')

# evaluator for rad-MLP
parser.add_argument('--eval_rad', type=str2bool, nargs='?', const=True, default=False, help='eval trained rad-MLP')
parser.add_argument('--rad_lighting_sample_type', default='emission', const='all', nargs='?', choices=['emission', 'incident'], help='from supported sample types (default: %(default)s)')
parser.add_argument('--if_add_rays_from_eval', type=str2bool, nargs='?', const=True, default=True, help='if add rays from evaluating MLPs')
parser.add_argument('--if_add_est_from_eval', type=str2bool, nargs='?', const=True, default=True, help='if add estimations from evaluating MLPs')
parser.add_argument('--if_add_color_from_eval', type=str2bool, nargs='?', const=True, default=True, help='if colorize mesh vertices with values from evaluator')
# evaluator for inv-MLP
parser.add_argument('--eval_inv', type=str2bool, nargs='?', const=True, default=False, help='eval trained inv-MLP')
# evaluator for MonoSDF
parser.add_argument('--eval_monosdf', type=str2bool, nargs='?', const=True, default=False, help='eval trained MonoSDF')
# evaluator over scene shapes
parser.add_argument('--eval_scene', type=str2bool, nargs='?', const=True, default=False, help='eval over scene (e.g. shapes for coverage)')

# debug
parser.add_argument('--if_debug_info', type=str2bool, nargs='?', const=True, default=False, help='if show debug info')

# utils
parser.add_argument('--if_sample_poses', type=str2bool, nargs='?', const=True, default=False, help='if sample camera poses instead of loading from pose file')
parser.add_argument('--export', type=str2bool, nargs='?', const=True, default=False, help='if export entire scene to mitsubaScene data structure')
parser.add_argument('--export_format', type=str, default='monosdf', help='')
parser.add_argument('--export_appendix', type=str, default='', help='')
parser.add_argument('--force', type=str2bool, nargs='?', const=True, default=False, help='if force to overwrite existing files')
parser.add_argument('--sample_type', type=str, default='', help='for evaluator')

# SPECIFY scene HERE!
parser.add_argument('--scene', type=str, default='main_xml-scene0288_01', help='load conf file: confs/openrooms/\{opt.scene\}.conf')
# parser.add_argument('--dataset', type=str, default='mainDiffLight_xml1-scene0552_00', help='load conf file: confs/openrooms/\{opt.scene\}.conf')
parser.add_argument('--dataset', default='openrooms', const='all', nargs='?', choices=['openrooms', 'openrooms_public'], help='openrooms: re-rendered version, ~200 frames; openrooms_public: original version, ~10-20 frames')

opt = parser.parse_args()


DATASET = opt.dataset
conf_base_path = Path('confs/%s.conf'%DATASET); check_exists(conf_base_path)
CONF = ConfigFactory.parse_file(str(conf_base_path))
conf_scene_path = Path('confs/%s/%s.conf'%(DATASET, opt.scene))
if not conf_scene_path.exists():
    print(yellow('scene conf file not found: %s; NOT merged into base conf!'%conf_scene_path))
    CONF.scene_params_dict.scene_name = opt.scene
else:
    # check_exists(conf_scene_path)
    conf_scene = ConfigFactory.parse_file(str(conf_scene_path))
    CONF = ConfigTree.merge_configs(CONF, conf_scene)

dataset_root = Path(PATH_HOME) / CONF.data.dataset_root
xml_root = dataset_root / 'scenes'
semantic_labels_root = Path(PATH_HOME) / 'files_openrooms'

layout_root = Path(OR_RAW_ROOT) / 'layoutMesh'
shapes_root = Path(OR_RAW_ROOT) / 'uv_mapped'
envmaps_root = Path(OR_RAW_ROOT) / 'EnvDataset' # not publicly availale
shape_pickles_root = Path(PATH_HOME) / 'data/openrooms_shape_pickles' # for caching shape bboxes so that we do not need to load meshes very time if only bboxes are wanted
if not shape_pickles_root.exists():
    shape_pickles_root.mkdir(parents=True, exist_ok=True)

'''
default
'''
frame_id_list = CONF.scene_params_dict.frame_id_list

# meta_split = 'mainDiffLight_xml1'
# scene_name = 'scene0552_00_more'
# frame_id_list = [0, 11, 10, 64, 81]
# + list(range(5, 87, 10))

frame_id_list = [2, 3, 4, 5, 10]

'''
update confs
'''
CONF.modality_filename_dict = query_host(OR_MODALITY_FRAMENAME_DICT, host)

CONF.scene_params_dict.update({
    # 'split': opt.split, # train, val, train+val
    'frame_id_list': frame_id_list, 
})


CONF.im_params_dict.update({
    'im_H_resize': 240, 'im_W_resize': 320, 
})

# DEBUG
CONF.shape_params_dict.update({
    'force_regenerate_tsdf': True
})

# Mitsuba options
CONF.mi_params_dict.update({
    'if_mi_scene_from_xml': True, # !!!! set to False to load from shapes (single shape or tsdf fused shape (with tsdf in modality_list))
})

# TSDF options
CONF.shape_params_dict.update({
    'if_force_fuse_tsdf': True, # !!!! set to True to force replace existing tsdf shape
})

if opt.export:
    if opt.export_format == 'mitsuba':
        CONF.im_params_dict.update({
            'im_H_resize': 480, 'im_W_resize': 640, # inv-nerf
        })
    elif opt.export_format == 'lieccv22':
        CONF.im_params_dict.update({
            'im_H_resize': 480, 'im_W_resize': 640, 
        })
    elif opt.export_format == 'monosdf':
        CONF.im_params_dict.update({
            'im_H_resize': 480, 'im_W_resize': 640, # monosdf
        })

radiance_scale_vis = 0.001 # GT max radiance ~300. -> ~3.

scene_obj = openroomsScene3D(
    CONF = CONF, 
    if_debug_info = opt.if_debug_info, 
    host = host, 
    root_path_dict = {
        'PATH_HOME': Path(PATH_HOME), 
        'dataset_root': Path(dataset_root), 
        'xml_root': Path(xml_root), 
        'semantic_labels_root': semantic_labels_root, 
        'shape_pickles_root': shape_pickles_root, 
        'layout_root': layout_root, 'shapes_root': shapes_root, 'envmaps_root': envmaps_root, # RAW scene files
        }, 
    modality_list = [
        'im_sdr', 
        # 'im_hdr', 
        'poses', 
        # 'seg', 
        # 'albedo', 'roughness', 
        # 'depth', 'normal',
        
        'semseg', 
        'matseg', 
        'instance_seg', 
        
        # 'lighting_SG', 
        # 'lighting_envmap', 
        
        # 'layout', 
        # 'shapes', # objs + emitters, geometry shapes + emitter properties
        # 'tsdf', 
        'mi', # mitsuba scene, loading from scene xml file
        ], 
)

eval_return_dict = {}

'''
Evaluator for scene
'''
if opt.eval_scene:
    evaluator_scene = evaluator_scene_scene(
        host=host, 
        scene_object=scene_obj, 
    )

    '''
    sample visivility to camera centers on vertices
    '''
    _ = evaluator_scene.sample_shapes(
        sample_type=opt.sample_type, # e.g. ['vis_count', 't', 'rgb_hdr', 'rgb_sdr', 'face_normal', 'mi_normal', 'semseg', 'instance_seg']
        # sample_type='vis_count', # ['']
        # sample_type='t', # ['']
        shape_params={
        }
    )
    for k, v in _.items():
        if k in eval_return_dict:
            eval_return_dict[k].update(_[k])
        else:
            eval_return_dict[k] = _[k]

'''
Differential renderers
'''
if opt.render_3d:
    renderer_3D = diff_renderer_openroomsScene_3D(
        scene_obj, 
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
            'max_plate': 64, 
            'emitter_type_index_list': emitter_type_index_list, 
        })
    
    if opt.renderer_option == 'ZQ_emitter':
        ts = np.median(renderer_return_dict['ts'], axis=1)
        visibility = np.amax(renderer_return_dict['visibility'], axis=1)
        print('visibility', visibility.shape, np.sum(visibility)/float(visibility.shape[0]))
        # from scipy import stats
        # visibility = stats.mode(renderer_return_dict['visibility'], axis=1)[0].flatten()

if opt.export:
    from lib.class_exporter import exporter_scene
    exporter = exporter_scene(
        scene_object=scene_obj,
        format=opt.export_format, 
        modality_list = [
            'poses', 
            'im_hdr', 
            'im_sdr', 
            'im_mask', 
            'shapes', 
            'mi_normal', 
            'mi_depth', 
            ], 
        if_force=opt.force, 
        # convert from y+ (native to indoor synthetic) to z+
        # extra_transform = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.float32),  # y=z, z=x, x=y
        # extra_transform = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=np.float32),  # z=y, y=x, x=z
        
    )
    if opt.export_format == 'monosdf':
        exporter.export_monosdf_fvp_mitsuba(
            split=opt.split, 
            format='monosdf',
            )
    if opt.export_format == 'mitsuba':
        exporter.export_monosdf_fvp_mitsuba(
            # split=opt.split, 
            format='mitsuba',
            modality_list = [
                # 'poses', 
                # 'im_hdr', 
                # 'im_sdr', 
                # 'im_mask', 
                # 'shapes', 
                # 'mi_normal', 
                # 'mi_depth', 
                # 'matseg', 
                'albedo', 
                'roughness', 
                'mi_seg_env', 'mi_seg_area', 'mi_seg_obj',
                ], 
            appendix=opt.export_appendix, 
            )
    if opt.export_format == 'fvp':
        exporter.export_monosdf_fvp_mitsuba(
            split=opt.split, 
            format='fvp',
            modality_list = [
                'poses', 
                # 'im_hdr', 
                # 'im_sdr', 
                # 'im_mask', 
                'shapes', 
                ], 
            appendix=opt.export_appendix, 
        )
    if opt.export_format == 'lieccv22':
        exporter.export_lieccv22(
            modality_list = [
            'im_sdr', 
            'mi_seg', 
            'mi_depth', 
            'lighting', # ONLY available after getting BRDFLight result from testRealBRDFLight.py
            ], 
            split=opt.split, 
            assert_shape=(240, 320),
            window_area_emitter_id_list=CONF.scene_params_dict.window_area_emitter_id_list, # need to manually specify in XML: e.g. <emitter type="area" id="lamp_oven_0">
            merge_lamp_id_list=CONF.scene_params_dict.merge_lamp_id_list,  # need to manually specify in XML
            BRDF_results_folder='BRDFLight_size0.200_int0.001_dir1.000_lam0.001_ren1.000_visWin120000_visLamp119540_invWin200000_invLamp150000', # transfer this back once get BRDF results
            # center_crop_HW=(240, 320), 
            if_no_gt_appendix=True, # do not append '_gt' to the end of the file name
        )


'''
Matploblib 2D viewer
'''
if opt.vis_2d_plt:
    visualizer_2D = visualizer_scene_2D(
        scene_obj, 
        modality_list_vis=[
            'im', 
            # 'layout', 
            # 'shapes', 
            # 'albedo', 
            # 'roughness', 
            
            'semseg', 
            
            'matseg', 
            'matseg-sem', # ![](https://i.imgur.com/jtqwMgF.png)
            # 'instance_seg', 
            # 'instance_seg-sem', # ![](https://i.imgur.com/ThpMRyL.png)
            
            # 'depth', 
            # 'normal', 
            # 'mi_depth', 
            # 'mi_normal', # compare depth & normal maps from mitsuba sampling VS OptixRenderer: **mitsuba does no anti-aliasing**: images/demo_mitsuba_ret_depth_normals_2D.png
            # 'lighting_SG', # convert to lighting_envmap and vis: images/demo_lighting_SG_envmap_2D_plt.png
            # 'lighting_envmap', 
            # 'seg_area', 'seg_env', 'seg_obj', 
            # 'mi_seg_area', 'mi_seg_env', 'mi_seg_obj', # compare segs from mitsuba sampling VS OptixRenderer: **mitsuba does no anti-aliasing**: images/demo_mitsuba_ret_seg_2D.png
            ], 
        # frame_idx_list=[0, 1, 2, 3, 4], 
        # frame_idx_list=[0], 
        frame_idx_list=None, # to use ALL frames in the scene
    )
    if opt.if_add_est_from_eval:
        for modality in ['lighting_envmap']:
            if modality in eval_return_dict:
                scene_obj.add_modality(eval_return_dict[modality], modality, 'EST')

    visualizer_2D.vis_2d_with_plt(
        lighting_params={
            'lighting_scale': 0.1, # rescaling the brightness of the envmap
            }, 
        other_params={
            'mi_normal_vis_coords': 'opengl', 
            'mi_depth_if_sync_scale': True, 
            }, 
    )

'''
Matploblib 3D viewer
'''
if opt.vis_3d_plt:
    visualizer_3D_plt = visualizer_scene_3D_plt(
        scene_obj, 
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
Open3D 3D viewer
'''
if opt.vis_3d_o3d:
    visualizer_3D_o3d = visualizer_scene_3D_o3d(
        scene_obj, 
        modality_list_vis=[
            # 'dense_geo', 
            # 'poses', 
            # 'lighting_SG', # images/demo_lighting_SG_o3d.png; arrows in blue
            # 'lighting_envmap', # images/demo_lighting_envmap_o3d.png; arrows in pink
            'layout', 
            'shapes', # bbox and (if loaded) meshs of shapes (objs + emitters)
            # 'emitters', # emitter properties (e.g. SGs, half envmaps)
            'mi', # mitsuba sampled rays, pts
            'tsdf', 
            ], 
        if_debug_info=opt.if_debug_info, 
    )

    lighting_params_vis={
        'subsample_lighting_pts_rate': 100, # change this according to how sparse the lighting arrows you would like to be (also according to num of frame_id_list)
        # 'lighting_keep_ratio': 0.05, 
        # 'lighting_further_clip_ratio': 0.1, 
        'lighting_scale': 2., 
        # 'lighting_keep_ratio': 0.2, # - good for lighting_SG
        # 'lighting_further_clip_ratio': 0.3, 
        'lighting_keep_ratio': 0.1, # - good for lighting_envmap
        'lighting_further_clip_ratio': 0.3, 
        # 'lighting_keep_ratio': 0., # - debug
        # 'lighting_further_clip_ratio': 0., 
        'lighting_autoscale': False, 
        'lighting_if_show_hemisphere': True, # mainly to show hemisphere and local axes: images/demo_lighting_envmap_hemisphere_axes_o3d.png
        }

    if opt.if_set_pcd_color_mi:
        '''
        use results from renderer to colorize Mitsuba/fused points
        '''
        assert opt.render_3d
        assert scene_obj.if_has_mitsuba_rays_pts
        visualizer_3D_o3d.set_mi_pcd_color_from_input(
            # input_colors_tuple=(ts, 'dist'), # get from renderer, etc.: images/demo_mitsuba_ret_pts_pcd-color-mode-mi_renderer-t.png
            input_colors_tuple=([visibility], 'mask'), # get from renderer, etc.: images/demo_mitsuba_ret_pts_pcd-color-mode-mi_renderer-visibility-any.png
        )
    
    if opt.if_add_rays_from_renderer:
        assert opt.render_3d and scene_obj.if_has_mitsuba_rays_pts
        # _pts_idx = list(range(scene_obj.W*scene_obj.H)); _sample_rate = 1000 # visualize for all scene points;
        _pts_idx = 60 * scene_obj.W + 80; _sample_rate = 1 # only visualize for one scene point w.r.t. all lamp points
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
    
    if opt.if_add_rays_from_eval:
        if 'emitter_rays_list' in eval_return_dict:
            assert opt.eval_rad or opt.eval_inv
            for emitter_ray in eval_return_dict['emitter_rays_list']:
                lpts = emitter_ray['v']
                lpts_end = lpts + emitter_ray['d'] * emitter_ray['l'] * radiance_scale_vis
                # for (lpts, lpts_end) in eval_return_dict['emitter_rays_list']:
                visualizer_3D_o3d.add_extra_geometry([
                    ('rays', {
                        'ray_o': lpts, 'ray_e': lpts_end, 'ray_c': np.array([[0., 1., 0.]]*lpts.shape[0]), # GREEN for EST
                    }),
                ])
                # print('---', lpts.shape, np.amax(lpts), np.amin(lpts),np.amax(lpts_end), np.amin(lpts_end))

        if 'cam_rays' in eval_return_dict:
            lpts = eval_return_dict['cam_rays']['v']
            lpts_end = lpts + eval_return_dict['cam_rays']['d'] * eval_return_dict['cam_rays']['l']
            visualizer_3D_o3d.add_extra_geometry([
                ('rays', {
                    'ray_o': lpts, 'ray_e': lpts_end, 'ray_c': np.array([[0., 1., 0.]]*lpts.shape[0]), # GREEN for EST
                }),
            ])
                
        if 'lighting_fused_list' in eval_return_dict:
            assert opt.eval_rad
            for lighting_fused_dict in eval_return_dict['lighting_fused_list']:
                geometry_list = visualizer_3D_o3d.process_lighting(
                    lighting_fused_dict, 
                    lighting_params=lighting_params_vis, 
                    lighting_source='lighting_envmap', 
                    lighting_color=[0., 0., 1.], 
                    if_X_multiplied=True, 
                    # if_use_pts_end=True,
                    if_use_pts_end=False,
                    )
                visualizer_3D_o3d.add_extra_geometry(geometry_list, if_processed_geometry_list=True)

    if opt.if_add_color_from_eval:
        if 'samples_v_dict' in eval_return_dict:
            assert opt.eval_rad or opt.eval_inv or opt.eval_monosdf or opt.eval_scene
            visualizer_3D_o3d.extra_input_dict['samples_v_dict'] = eval_return_dict['samples_v_dict']

    visualizer_3D_o3d.run_o3d(
        if_shader=opt.if_shader, # set to False to disable faycny shaders 
        cam_params={
            'if_cam_axis_only': False, 
            'if_cam_traj': False, 
            'if_labels': True, 
            }, 
        dense_geo_params={
            'subsample_pcd_rate': 100, # change this according to how sparse the points you would like to be (also according to num of frame_id_list)
            'if_ceiling': False, # remove ceiling points to better see the furniture 
            'if_walls': True, # remove wall points to better see the furniture 
            'if_normal': False, # turn off normals to avoid clusters
            'subsample_normal_rate_x': 2, 
            'pcd_color_mode': opt.pcd_color_mode_dense_geo, 
            }, 
        lighting_params=lighting_params_vis, 
        shapes_params={
            'simply_mesh_ratio_vis': 0.1, # simply num of triangles to #triangles * simply_mesh_ratio_vis
            'if_meshes': True, # if show meshes for objs + emitters (False: only show bboxes)
            'if_labels': True, # if show labels (False: only show bboxes)
            'if_voxel_volume': False, # [OPTIONAL] if show unit size voxel grid from shape occupancy: images/demo_shapes_voxel_o3d.png
            'if_ceiling': False, # remove ceiling **triangles** to better see the furniture 
            # 'if_ceiling': True, 
            'if_walls': False, # remove wall **triangles** to better see the furniture 
            # 'if_walls': True, 
            # 'mesh_color_type': 'eval-emission_mask', # ['obj_color', 'face_normal', 'eval-rad', 'eval-emission_mask']
            'mesh_color_type': 'obj_color' if opt.sample_type == '' else 'eval-'+opt.sample_type, # ['obj_color', 'face_normal', 'eval-rad', 'eval-emission_mask']
        },
        emitter_params={
            'if_half_envmap': True, # if show half envmap as a hemisphere for window emitters (False: only show bboxes)
            'scale_SG_length': 2., 
            'if_sampling_emitter': True, # if sample and visualize points on emitter surface; show intensity as vectors along normals (RED for GT): images/demo_emitter_o3d_sampling.png
            'max_plate': 64, 
            'radiance_scale_vis': radiance_scale_vis, 
        },
        mi_params={
            'if_pts': False, # if show pts sampled by mi; should close to backprojected pts from OptixRenderer depth maps
            'if_pts_colorize_rgb': True, 
            'pts_subsample': 1,
            'if_ceiling': True, # remove ceiling **points** to better see the furniture 
            'if_walls': True, # remove wall **points** to better see the furniture 

            'if_cam_rays': False, 
            'cam_rays_if_pts': True, # if cam rays end in surface intersections; set to False to visualize rays of unit length
            'cam_rays_subsample': 10, 
            
            'if_normal': False, 
            'normal_subsample': 50, 
            'normal_scale': 0.2, 
        }, 
    )


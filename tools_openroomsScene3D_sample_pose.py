import sys

# host = 'mm1'
host = 'apple'

from lib.global_vars import PATH_HOME_dict, INV_NERF_ROOT_dict, MONOSDF_ROOT_dict, OR_RAW_ROOT_dict
PATH_HOME = PATH_HOME_dict[host]
sys.path.insert(0, PATH_HOME)
OR_RAW_ROOT = OR_RAW_ROOT_dict[host]
INV_NERF_ROOT = INV_NERF_ROOT_dict[host]
MONOSDF_ROOT = MONOSDF_ROOT_dict[host]

from pathlib import Path
import numpy as np
np.set_printoptions(suppress=True)
from lib.utils_misc import str2bool

from lib.class_openroomsScene3D import openroomsScene3D

from lib.class_visualizer_scene_2D import visualizer_scene_2D
from lib.class_visualizer_scene_3D_o3d import visualizer_scene_3D_o3d
from lib.class_visualizer_openroomsScene_3D_plt import visualizer_openroomsScene_3D_plt

from lib.class_diff_renderer_openroomsScene_3D import diff_renderer_openroomsScene_3D

from lib.class_eval_rad import evaluator_scene_rad
from lib.class_eval_inv import evaluator_scene_inv
from lib.class_eval_monosdf import evaluator_scene_monosdf
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

opt = parser.parse_args()

semantic_labels_root = Path(PATH_HOME) / 'files_openrooms'
layout_root = Path(OR_RAW_ROOT) / 'layoutMesh'
shapes_root = Path(OR_RAW_ROOT) / 'uv_mapped'
envmaps_root = Path(OR_RAW_ROOT) / 'EnvDataset' # not publicly availale
shape_pickles_root = Path(PATH_HOME) / 'data/openrooms_shape_pickles' # for caching shape bboxes so that we do not need to load meshes very time if only bboxes are wanted
if not shape_pickles_root.exists():
    shape_pickles_root.mkdir(parents=True, exist_ok=True)


dataset_version = 'public_re_3_v3pose_2048'
up_axis = 'y+'
# meta_split = 'main_xml'
# scene_name = 'scene0008_00_more'
# emitter_type_index_list = [('lamp', 0)]
# frame_ids = list(range(0, 345, 10))

'''
The classroom scene: one lamp (dark) + one window (directional sun)
data/public_re_3/mainDiffLight_xml1/scene0552_00/im_4.png
'''
dataset_version = 'public_re_0203'
meta_split = 'main_xml1'
scene_name = 'scene0552_00'
emitter_type_index_list = [('lamp', 0)]
# frame_ids = list(range(0, 345, 10))
frame_ids = list(range(200))

'''
The conference room with one lamp
data/public_re_3/main_xml/scene0005_00_more/im_3.png
images/demo_eval_scene_shapes-vis_count-train-public_re_0203_main_xml_scene0005_00.png
'''
dataset_version = 'public_re_0203'
meta_split = 'main_xml'
scene_name = 'scene0005_00'
frame_ids = list(range(200))
# frame_ids = [0]

'''
The classroom scene: one lamp (lit up) + one window (less sun)
data/public_re_3/main_xml1/scene0552_00/im_4.png
'''
dataset_version = 'public_re_0203'
meta_split = 'main_xml1'
scene_name = 'scene0552_00'
frame_ids = list(range(200))
# frame_ids = [0, 1, 2, 3, 4] + list(range(5, 87, 10))

'''
orange-ish room with direct light
images/demo_eval_scene_shapes-vis_count-train-public_re_0203_main_xml_scene0002_00.png
'''
dataset_version = 'public_re_0203'
meta_split = 'main_xml'
scene_name = 'scene0002_00'
frame_ids = list(range(200))

'''
green-ish room with window, with guitar
images/demo_eval_scene_shapes-vis_count-train-public_re_0203_mainDiffMat_xml1_scene0608_01.png
'''
dataset_version = 'public_re_0203'
meta_split = 'mainDiffMat_xml1'
scene_name = 'scene0608_01'
frame_ids = list(range(200))

'''
toy room with lit lamp and dark window
images/demo_eval_scene_shapes-vis_count-train-public_re_0203_mainDiffMat_xml_scene0603_00.png
'''
dataset_version = 'public_re_0203'
meta_split = 'mainDiffMat_xml'
scene_name = 'scene0603_00'
frame_ids = list(range(200))

'''
default
'''
base_root = Path(PATH_HOME) / 'data' / dataset_version
xml_root = Path(PATH_HOME) / 'data' / dataset_version / 'scenes'
radiance_scale_vis = 0.001 # GT max radiance ~300. -> ~3.

eval_models_dict = {
    'inv-MLP_ckpt_path': '20230109-014709-inv_v3pose_2048_main_xml_scene0008_00_more_specT_re/last.ckpt', 
    'rad-MLP_ckpt_path': '20230104-162138-rad_v3pose_2048_main_xml_scene0008_00_more_specT/last.ckpt', 
    }
monosdf_shape_dict = {}

'''
umcomment to use estimated geometry and radiance from monosdf
'''
# monosdf_shape_dict = {
#     '_shape_normalized': 'normalized', 
#     'shape_file': str(Path(MONOSDF_ROOT) / 'exps/public_re_3_v3pose_2048-main_xml-scene0008_00_more_HDR_EST_mlp_gamma2_L2loss_4xreg_lr1e-4_decay25_scan1/2023_01_20_13_38_20/plots/public_re_3_v3pose_2048-main_xml-scene0008_00_more_HDR_EST_mlp_gamma2_L2loss_4xreg_lr1e-4_decay25_scan1_epoch380.ply'), 
#     'camera_file': str(Path(MONOSDF_ROOT) / 'data/public_re_3_v3pose_2048-main_xml-scene0008_00_morerescaledSDR/scan1/cameras.npz'), 
#     'monosdf_conf_path': 'public_re_3_v3pose_2048-main_xml-scene0008_00_more_HDR_EST_mlp_gamma2_L2loss_4xreg_lr1e-4_decay25_scan1/2023_01_20_13_38_20/runconf.conf', 
#     'monosdf_ckpt_path': 'public_re_3_v3pose_2048-main_xml-scene0008_00_more_HDR_EST_mlp_gamma2_L2loss_4xreg_lr1e-4_decay25_scan1/2023_01_20_13_38_20/checkpoints/ModelParameters/latest.pth', 
#     'inv-MLP_ckpt_path': '20230201-185742-inv-OR/last.ckpt' # OVERRIDING eval_models_dict
#     } # load shape from MonoSDF and un-normalize with scale/offset loaded from camera file: images/demo_shapes_monosdf.png

scene_obj = openroomsScene3D(
    if_debug_info=opt.if_debug_info, 
    host=host, 
    root_path_dict = {'PATH_HOME': Path(PATH_HOME), 'rendering_root': base_root, 'xml_scene_root': xml_root, 'semantic_labels_root': semantic_labels_root, 'shape_pickles_root': shape_pickles_root, 
        'layout_root': layout_root, 'shapes_root': shapes_root, 'envmaps_root': envmaps_root}, 
    scene_params_dict={
        'meta_split': meta_split, 
        'scene_name': scene_name, 
        'frame_id_list': frame_ids, 
        'up_axis': up_axis, 
        'monosdf_shape_dict': monosdf_shape_dict, # comment out if load GT shape from XML; otherwise load shape from MonoSDF to **'shape' and Mitsuba scene**
        }, 
    # modality_list = ['im_sdr', 'im_hdr', 'seg', 'poses', 'albedo', 'roughness', 'depth', 'normal', 'lighting_SG', 'lighting_envmap'], 
    modality_list = [
        # 'im_sdr', 
        'poses', 
        # 'seg', 'im_hdr', 
        # 'albedo', 'roughness', 
        # 'depth', 
        # 'normal', 
        # 'lighting_SG', 
        # 'lighting_envmap', 
        # 'layout', 
        'shapes', # objs + emitters, geometry shapes + emitter properties
        # 'mi', # mitsuba scene, loading from scene xml file
        ], 
    modality_filename_dict = {
        # 'poses', 
        'im_hdr': 'im_%d.hdr', 
        'im_sdr': 'im_%d.png', 
        # 'lighting_envmap', 
        'albedo': 'imbaseColor_%d.png', 
        'roughness': 'imroughness_%d.png', 
        'depth': 'imdepth_%d.dat', 
        'normal': 'imnormal_%d.png', 
        # 'lighting_SG', 
        # 'layout', 
        # 'shapes', # objs + emitters, geometry shapes + emitter properties
    }, 
    im_params_dict={
        'im_H_load': 480, 'im_W_load': 640, 
        # 'im_H_resize': 240, 'im_W_resize': 320, 
        # 'im_H_resize': 120, 'im_W_resize': 160, # to use for rendering so that im dimensions == lighting dimensions
        'im_H_resize': 480, 'im_W_resize': 640, # to use for rendering so that im dimensions == lighting dimensions
        # 'im_H_resize': 6, 'im_W_resize': 8, # to use for rendering so that im dimensions == lighting dimensions
        'if_direct_lighting': False, # if load direct lighting envmaps and SGs inetad of total lighting
        # 'im_hdr_ext': 'hdr', 
        }, 
    cam_params_dict={
        'near': 0.1, 'far': 10., 
        # == params for sample camera poses
        'sampleNum': 3, 
        'heightMin' : 0.8, # camera height min
        'heightMax' : 1.8, # camera height max
        'distMin': 0.5, # to wall distance min
        'distMax': 4.5, # to wall distance max
        'thetaMin': -60, # theta min: pitch angle; up+ 
        'thetaMax' : 60, # theta max: pitch angle; up+
        'phiMin': -60, # yaw angle min
        'phiMax': 60, # yaw angle max
        'distRaysMin': 0.5, # min dist of all camera rays to the scene; [!!!] set to -1 to disable checking
        'distRaysMedianMin': 0.8, # median dist of all camera rays to the scene; [!!!] set to -1 to disable checking
        # ==> if sample poses and render images 
        'if_sample_poses': opt.if_sample_poses, # True to generate camera poses following Zhengqin's method (i.e. walking along walls)
        'sample_pose_num': 200, # Number of poses to sample; set to -1 if not sampling
        'sample_pose_if_vis_plt': False, # images/demo_sample_pose.png
    }, 
    lighting_params_dict={
        'SG_num': 12,
        'env_row': 120, 'env_col': 160, # resolution to load; FIXED
        'env_downsample_rate': 20, # (120, 160) -> (6, 8)

        # 'env_height': 8, 'env_width': 16,
        'env_height': 16, 'env_width': 32, 
        # 'env_height': 128, 'env_width': 256, 
        # 'env_height': 64, 'env_width': 128, 
        # 'env_height': 2, 'env_width': 4, 
        
        'if_convert_lighting_SG_to_global': True, 
        'if_use_mi_geometry': True, 
    }, 
    shape_params_dict={
        'if_load_obj_mesh': True, # set to False to not load meshes for objs (furniture) to save time
        'if_load_emitter_mesh': True,  # default True: to load emitter meshes, because not too many emitters

        'if_sample_pts_on_mesh': False,  # default True: sample points on each shape -> self.sample_pts_list
        'sample_mesh_ratio': 0.1, # target num of VERTICES: len(vertices) * sample_mesh_ratio
        'sample_mesh_min': 10, 
        'sample_mesh_max': 100, 

        'if_simplify_mesh': False,  # default True: simply triangles
        'simplify_mesh_ratio': 0.1, # target num of FACES: len(faces) * simplify_mesh_ratio
        'simplify_mesh_min': 100, 
        'simplify_mesh_max': 1000,
        'if_remesh': True, # False: images/demo_shapes_3D_NO_remesh.png; True: images/demo_shapes_3D_YES_remesh.png
        'remesh_max_edge': 0.15,  
        },
    emitter_params_dict={
        'N_ambient_rep': '3SG-SkyGrd', 
        },
    mi_params_dict={
        # 'if_also_dump_xml_with_lit_area_lights_only': True,  # True: to dump a second file containing lit-up lamps only
        'debug_dump_mesh': True, # [DEBUG] True: to dump all object meshes to mitsuba/meshes_dump; load all .ply files into MeshLab to view the entire scene: images/demo_mitsuba_dump_meshes.png
        'debug_render_test_image': False, # [DEBUG][slow] True: to render an image with first camera, usig Mitsuba: images/demo_mitsuba_render.png
        'if_sample_rays_pts': True, # True: to sample camera rays and intersection pts given input mesh and camera poses
        'if_get_segs': False, # True: to generate segs similar to those in openroomsScene2D.load_seg()
        },
)

eval_return_dict = {}
'''
Evaluator for rad-MLP
'''
if opt.eval_rad:
    evaluator_rad = evaluator_scene_rad(
        scene_object=scene_obj, 
        host=host, 
        INV_NERF_ROOT = INV_NERF_ROOT, 
        # ckpt_path='rad_3_v3pose_2048_main_xml_scene0008_00_more/last.ckpt', # 166, 208
        # ckpt_path='rad_3_v5pose_2048_main_xml_scene0008_00_more/last-v1.ckpt', # 110
        ckpt_path=monosdf_shape_dict.get('rad-MLP_ckpt_path', eval_models_dict['rad-MLP_ckpt_path']), 
        # dataset_key='-'.join(['OR', dataset_version]), 
        dataset_key='OR-public_re_3_v3pose_2048', # has to be one of the keys from inv-nerf/configs/scene_options.py
        rad_scale=1./5., 
        spec=True, 
    )

    '''
    render one image by querying rad-MLP: images/demo_eval_radMLP_render.png
    '''
    # evaluator_rad.render_im(0, if_plt=True) 

    '''
    sample and visualize **radiance** on emitter surface; show intensity as vectors along normals (BLUE for EST): images/demo_emitter_o3d_sampling.png
    '''
    eval_return_dict.update(
        evaluator_rad.sample_emitter(
            emitter_params={
                'max_plate': 64, 
                'radiance_scale': radiance_scale, 
                'emitter_type_index_list': emitter_type_index_list, 
                }))
    
    '''
    sample non-emitter locations along envmap (hemisphere) directions radiance from rad-MLP: images/demo_envmap_o3d_sampling.png
    '''
    # eval_return_dict.update(
    #     evaluator_rad.sample_lighting(
    #         sample_type=opt.rad_lighting_sample_type, # 'emission', 'incident'
    #         subsample_rate_pts=1, 
    #         lighting_scale=0.1, # rescaling the brightness of the envmap
    #         if_vis_envmap_2d_plt=False, 
    #     )
    # )

'''
Evaluator for inv-MLP
'''
if opt.eval_inv:
    evaluator_inv = evaluator_scene_inv(
        host=host, 
        scene_object=scene_obj, 
        INV_NERF_ROOT = INV_NERF_ROOT, 
        ckpt_path=monosdf_shape_dict.get('inv-MLP_ckpt_path', eval_models_dict['inv-MLP_ckpt_path']), 
        # ckpt_path='20230109-014709-inv_v3pose_2048_main_xml_scene0008_00_more_specT_re/last.ckpt', # 110
        # dataset_key='-'.join(['OR', scene_name]), # has to be one of the keys from inv-nerf/configs/scene_options.py
        dataset_key='OR-public_re_3_v3pose_2048', # has to be one of the keys from inv-nerf/configs/scene_options.py
        spec=True, 
        if_monosdf=monosdf_shape_dict=={}, 
        rad_scale=1./5., 
        monosdf_shape_dict=monosdf_shape_dict, 
    )

    '''
    sample emission mask on shape vertices
    '''
    _ = evaluator_inv.sample_shapes(
        sample_type='emission_mask_bin', # ['']
        shape_params={
        }, 
        emitter_params={
            'max_plate': 64, 
            'radiance_scale': 1., 
            },
        if_dump_emitter_mesh=True, 
    )
    for k, v in _.items():
        if k in eval_return_dict:
            eval_return_dict[k].update(_[k])
        else:
            eval_return_dict[k] = _[k]

if opt.eval_monosdf:
    assert scene_obj.hdr_scale_list.count(scene_obj.hdr_scale_list[0]) == len(scene_obj.hdr_scale_list)

    evaluator_monosdf = evaluator_scene_monosdf(
        host=host, 
        scene_object=scene_obj, 
        MONOSDF_ROOT = MONOSDF_ROOT, 
        conf_path=monosdf_shape_dict['monosdf_conf_path'], 
        ckpt_path=monosdf_shape_dict['monosdf_ckpt_path'], 
        rad_scale= 1. / scene_obj.hdr_scale_list[0] * (1./5.), 
    )

    # evaluator_monosdf.export_mesh()

    evaluator_monosdf.render_im_scratch(
        frame_id=0, offset_in_scan=202, 
        # if_integrate=False, 
        if_integrate=True, 
        if_plt=True, 
        )

    # evaluator_monosdf.render_im(
    #     frame_id=0, offset_in_scan=202, 
    #     if_plt=False
    #     )

    # [!!!] set 'mesh_color_type': 'eval-rad'
    # eval_return_dict.update(
    #     evaluator_monosdf.sample_shapes(
    #         sample_type='rad', # ['rad']
    #         shape_params={
    #             'radiance_scale': 1., 
    #         }
    #     )
    # )
    # np.save('test_files/eval_return_dict.npy', eval_return_dict)

# eval_return_dict = np.load('test_files/eval_return_dict.npy', allow_pickle=True).item(); opt.eval_monosdf = True

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
    [!!!] set 'mesh_color_type': 'eval-vis_count'
    '''
    _ = evaluator_scene.sample_shapes(
        sample_type='vis_count', # ['']
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


'''
Matploblib 2D viewer
'''
if opt.vis_2d_plt:
    visualizer_2D = visualizer_scene_2D(
        scene_obj, 
        modality_list_vis=[
            'im', 
            'layout', 
            # 'shapes', 
            # 'albedo', 
            # 'roughness', 
            'depth', 
            'normal', 
            'mi_depth', 
            'mi_normal', # compare depth & normal maps from mitsuba sampling VS OptixRenderer: **mitsuba does no anti-aliasing**: images/demo_mitsuba_ret_depth_normals_2D.png
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
    visualizer_3D_plt = visualizer_openroomsScene_3D_plt(
        scene_obj, 
        modality_list_vis = [
            'layout', 
            # 'poses', # camera center + optical axis
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
            'cameras', 
            # 'lighting_SG', # images/demo_lighting_SG_o3d.png; arrows in blue
            # 'lighting_envmap', # images/demo_lighting_envmap_o3d.png; arrows in pink
            # 'layout', 
            'shapes', # bbox and (if loaded) meshs of shapes (objs + emitters)
            # 'emitters', # emitter properties (e.g. SGs, half envmaps)
            # 'mi', # mitsuba sampled rays, pts
            ], 
        if_debug_info=opt.if_debug_info, 
    )

    lighting_params_vis={
        'subsample_lighting_pts_rate': 100, # change this according to how sparse the lighting arrows you would like to be (also according to num of frame_ids)
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
        # dense_geo_params={
        #     'subsample_pcd_rate': 100, # change this according to how sparse the points you would like to be (also according to num of frame_ids)
        #     'if_ceiling': False, # remove ceiling points to better see the furniture 
        #     'if_walls': True, # remove wall points to better see the furniture 
        #     'if_normal': False, # turn off normals to avoid clusters
        #     'subsample_normal_rate_x': 2, 
        #     'pcd_color_mode': opt.pcd_color_mode_dense_geo, 
        #     }, 
        lighting_params=lighting_params_vis, 
        shapes_params={
            'simply_mesh_ratio_vis': 0.1, # simply num of triangles to #triangles * simply_mesh_ratio_vis
            'if_meshes': True, # if show meshes for objs + emitters (False: only show bboxes)
            'if_labels': False, # if show labels (False: only show bboxes)
            'if_voxel_volume': False, # [OPTIONAL] if show unit size voxel grid from shape occupancy: images/demo_shapes_voxel_o3d.png
            'if_ceiling': True, 
            'if_walls': True, 
            # 'if_ceiling': False, 
            # 'if_walls': False, 
            'mesh_color_type': 'eval-emission_mask', # ['obj_color', 'face_normal', 'eval-rad', 'eval-emission_mask']
        },
        emitter_params={
            'if_half_envmap': False, # if show half envmap as a hemisphere for window emitters (False: only show bboxes)
            'scale_SG_length': 2., 
            'if_sampling_emitter': True, # if sample and visualize points on emitter surface; show intensity as vectors along normals (RED for GT): images/demo_emitter_o3d_sampling.png
            'max_plate': 64, 
            'radiance_scale_vis': radiance_scale_vis, 
        },
        mi_params={
            'if_pts': True, # if show pts sampled by mi; should close to backprojected pts from OptixRenderer depth maps
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

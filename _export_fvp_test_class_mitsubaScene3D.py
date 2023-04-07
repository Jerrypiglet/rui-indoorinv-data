'''
work with Mitsuba/Blender scenes
'''
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
from lib.utils_misc import red, str2bool, white_magenta
import argparse

from lib.class_mitsubaScene3D import mitsubaScene3D

from lib.class_visualizer_scene_2D import visualizer_scene_2D
from lib.class_visualizer_scene_3D_o3d import visualizer_scene_3D_o3d
from lib.class_visualizer_scene_3D_plt import visualizer_scene_3D_plt

from lib.class_eval_rad import evaluator_scene_rad
from lib.class_eval_monosdf import evaluator_scene_monosdf
from lib.class_eval_inv import evaluator_scene_inv
from lib.class_eval_scene import evaluator_scene_scene

from lib.class_renderer_mi_mitsubaScene_3D import renderer_mi_mitsubaScene_3D
from lib.class_renderer_blender_mitsubaScene_3D import renderer_blender_mitsubaScene_3D

parser = argparse.ArgumentParser()
# visualizers
parser.add_argument('--vis_3d_plt', type=str2bool, nargs='?', const=True, default=False, help='whether to visualize 3D with plt for debugging')
parser.add_argument('--vis_3d_o3d', type=str2bool, nargs='?', const=True, default=True, help='whether to visualize in open3D')
parser.add_argument('--vis_2d_plt', type=str2bool, nargs='?', const=True, default=False, help='whether to show (1) pixel-space modalities (2) projection onto one image (e.g. layout, object bboxes), with plt')
parser.add_argument('--if_shader', type=str2bool, nargs='?', const=True, default=False, help='')
# options for visualizers
parser.add_argument('--pcd_color_mode_dense_geo', type=str, default='rgb', help='colormap for all points in fused geo')
parser.add_argument('--if_set_pcd_color_mi', type=str2bool, nargs='?', const=True, default=False, help='if create color map for all points of Mitsuba; required: input_colors_tuple')
# parser.add_argument('--if_add_rays_from_renderer', type=str2bool, nargs='?', const=True, default=False, help='if add camera rays and emitter sample rays from renderer')

parser.add_argument('--split', type=str, default='train', help='')

# differential renderer
# parser.add_argument('--render_diff', type=str2bool, nargs='?', const=True, default=False, help='differentiable surface rendering')
# parser.add_argument('--renderer_option', type=str, default='PhySG', help='differentiable renderer option')

# renderer (mi/blender)
parser.add_argument('--render_2d', type=str2bool, nargs='?', const=True, default=False, help='render 2D modalities')
parser.add_argument('--renderer', type=str, default='blender', help='mi, blender')

# ==== Evaluators
parser.add_argument('--if_add_rays_from_eval', type=str2bool, nargs='?', const=True, default=True, help='if add rays from evaluating MLPs (e.g. emitter radiance rays')
parser.add_argument('--if_add_est_from_eval', type=str2bool, nargs='?', const=True, default=True, help='if add estimations from evaluating MLPs (e.g. ennvmaps)')
parser.add_argument('--if_add_color_from_eval', type=str2bool, nargs='?', const=True, default=True, help='if colorize mesh vertices with values from evaluator')
# evaluator for rad-MLP
parser.add_argument('--eval_rad', type=str2bool, nargs='?', const=True, default=False, help='eval trained rad-MLP')
# evaluator for inv-MLP
parser.add_argument('--eval_inv', type=str2bool, nargs='?', const=True, default=False, help='eval trained inv-MLP')
# evaluator over scene shapes
parser.add_argument('--eval_scene', type=str2bool, nargs='?', const=True, default=False, help='eval over scene (e.g. shapes for coverage)')
# evaluator for MonoSDF
parser.add_argument('--eval_monosdf', type=str2bool, nargs='?', const=True, default=False, help='eval trained MonoSDF')

# debug
parser.add_argument('--if_debug_info', type=str2bool, nargs='?', const=True, default=False, help='if show debug info')

# utils
parser.add_argument('--if_sample_poses', type=str2bool, nargs='?', const=True, default=False, help='if sample camera poses instead of loading from pose file')
parser.add_argument('--export', type=str2bool, nargs='?', const=True, default=False, help='if export entire scene to mitsubaScene data structure')
parser.add_argument('--export_format', type=str, default='monosdf', help='')
parser.add_argument('--export_appendix', type=str, default='', help='')
parser.add_argument('--force', type=str2bool, nargs='?', const=True, default=False, help='if force to overwrite existing files')

opt = parser.parse_args()

dataset_root = Path(PATH_HOME) / 'data/indoor_synthetic'
xml_root = Path(PATH_HOME) / 'data/indoor_synthetic'
# intrinsics_path = Path(PATH_HOME) / 'data/indoor_synthetic/intrinsic_mitsubaScene.txt'

# xml_filename = 'scene_v3.xml'
# xml_filename = 'test.xml'
xml_filename = 'test-relight.xml'
emitter_type_index_list = [('lamp', 0)]; radiance_scale = 0.1; 
shape_file = ''

frame_ids = []
invalid_frame_id_list = []

# scene_name = 'kitchen_new'; 
# shape_file = 'data/indoor_synthetic/kitchen_new/scene_subdiv.obj'
# shape_file = 'data/indoor_synthetic/kitchen_new/scene.obj'
# shape_file = 'data/indoor_synthetic/RESULTS_monosdf/20230226-021300-mm3-EVAL-20230225-135237kitchen_NEW_HDR_grids_trainval.ply'

# scene_name = 'bedroom'
# shape_file = 'data/indoor_synthetic/bedroom/scene_subdiv.obj'
# shape_file = 'data/indoor_synthetic/bedroom/scene_subdiv_large.obj'
# shape_file = 'data/indoor_synthetic/bedroom/scene.obj'
# shape_file = 'data/indoor_synthetic/RESULTS_monosdf/20230225-135215-mm1-EVAL-20230219-211718-bedroom_HDR_grids_trainval.ply'

# scene_name = 'bathroom'
# # shape_file = 'data/indoor_synthetic/bathroom/scene.obj'
# shape_file = 'data/indoor_synthetic/bathroom/scene_subdiv.obj'

scene_name = 'livingroom'
# shape_file = 'data/indoor_synthetic/livingroom/scene.obj'
# shape_file = 'data/indoor_synthetic/livingroom/scene_subdiv.obj'
shape_file = 'data/indoor_synthetic/RESULTS_monosdf/20230225-135959-mm1-EVAL-20230219-211728-livingroom_HDR_grids_trainval.ply'

# scene_name = 'kitchen-resize'
# scene_name = 'kitchen'
# invalid_frame_id_list = [197]
# scene_name = 'kitchen_new_400'

# scene_name = 'livingroom0'
# scene_name = 'livingroom-test'

# ZQ
# frame_ids = [21]
# frame_ids = [64]
# frame_ids = [197]

# frame_ids = list(range(202))
# frame_ids = list(range(10))
# frame_ids = list(range(0, 202, 40))
# frame_ids = list(range(0, 4, 1))
# frame_ids = list(range(197))
# frame_ids = [0]
# frame_ids = list(range(189))

'''
default
'''
eval_models_dict = {
    'inv-MLP_ckpt_path': '20230111-191305-inv_kitchen_190-10_specT/last.ckpt', 
    'rad-MLP_ckpt_path': '20230110-132112-rad_kitchen_190-10_specT/last.ckpt', 
    }
# monosdf_shape_dict = {}

'''
umcomment👇 to use estimated geometry and radiance from monosdf
'''
# monosdf_shape_dict = {
#     '_shape_normalized': 'normalized', 
#     'shape_file': str(Path(MONOSDF_ROOT) / 'exps/20230125-161557-kitchen_HDR_EST_grids_EVALTRAIN2023_01_23_21_23_38_trainval/latest/plots/20230125-161557-kitchen_HDR_EST_grids_EVALTRAIN2023_01_23_21_23_38_trainval_epoch2780.ply'), 
#     'camera_file': str(Path(MONOSDF_ROOT) / 'data/kitchen/trainval/cameras.npz'), 
#     'monosdf_conf_path': '20230125-161557-kitchen_HDR_EST_grids_EVALTRAIN2023_01_23_21_23_38_trainval/latest/runconf.conf', 
#     'monosdf_ckpt_path': 'kitchen_HDR_EST_grids_gamma2_randomPixel_fixedDepthHDR_trainval/2023_01_23_21_23_38/checkpoints/ModelParameters/latest.pth', 
#     'inv-MLP_ckpt_path': '20230127-001044-inv-kitchen/last.ckpt' # OVERRIDING eval_models_dict
#     } # load shape from MonoSDF and un-normalize with scale/offset loaded from camera file: images/demo_shapes_monosdf.png
# monosdf_shape_dict = {}

scene_obj = mitsubaScene3D(
    if_debug_info=opt.if_debug_info, 
    host=host, 
    root_path_dict = {'PATH_HOME': Path(PATH_HOME), 'dataset_root': dataset_root, 'xml_root': xml_root}, 
    scene_params_dict={
        'xml_filename': xml_filename, 
        'scene_name': scene_name, 
        'split': opt.split, # train, val, train+val
        # 'frame_id_list': frame_ids, 
        'mitsuba_version': '3.0.0', 
        'intrinsics_path': Path(PATH_HOME) / 'data/indoor_synthetic' / scene_name / 'intrinsic_mitsubaScene.txt', 
        'axis_up': 'y+', 
        'extra_transform': np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=np.float32), # z=y, y=x, x=z # convert from y+ (native to indoor synthetic) to z+
        'invalid_frame_id_list': invalid_frame_id_list, 
        # 'pose_file': ('Blender', 'train.npy'), # requires scaled Blender scene!
        # 'pose_file': ('OpenRooms', 'cam.txt'), 
        'pose_file': ('json', 'transforms.json'), # requires scaled Blender scene! in comply with Liwen's IndoorDataset (https://github.com/william122742/inv-nerf/blob/bake/utils/dataset/indoor.py)
        'shape_file': shape_file, 
        # 'monosdf_shape_dict': monosdf_shape_dict, # comment out if load GT shape from XML; otherwise load shape from MonoSDF to **'shape' and Mitsuba scene**
        }, 
    mi_params_dict={
        # 'if_also_dump_xml_with_lit_area_lights_only': True,  # True: to dump a second file containing lit-up lamps only
        'debug_render_test_image': True, # [DEBUG][slow] True: to render an image with first camera, usig Mitsuba: images/demo_mitsuba_render.png
        'debug_dump_mesh': True, # [DEBUG] True: to dump all object meshes to mitsuba/meshes_dump; load all .ply files into MeshLab to view the entire scene: images/demo_mitsuba_dump_meshes.png
        'if_sample_rays_pts': True, # True: to sample camera rays and intersection pts given input mesh and camera poses
        'if_get_segs': True, # [depend on if_sample_rays_pts] True: to generate segs similar to those in openroomsScene2D.load_seg()
        },
    # modality_list = ['im_sdr', 'im_hdr', 'seg', 'poses', 'albedo', 'roughness', 'depth', 'normal', 'lighting_SG', 'lighting_envmap'], 
    modality_list = [
        'poses', 
        'im_hdr', 
        'im_sdr', 
        # 'lighting_envmap', 
        # 'albedo', 'roughness', 
        # 'emission', 
        # 'depth', 'normal', 
        # 'lighting_SG', 
        'layout', 
        'shapes', # objs + emitters, geometry shapes + emitter properties
        ], 
    modality_filename_dict = {
        # 'poses', 
        'im_hdr': 'Image/%03d_0001.exr', 
        'im_sdr': 'Image/%03d_0001.png', 
        # 'lighting_envmap', 
        'albedo': 'DiffCol/%03d_0001.exr', 
        'roughness': 'Roughness/%03d_0001.exr', 
        'emission': 'Emit/%03d_0001.exr', 
        'depth': 'Depth/%03d_0001.exr', 
        'normal': 'Normal/%03d_0001.exr', 
        # 'lighting_SG', 
        # 'layout', 
        # 'shapes', # objs + emitters, geometry shapes + emitter properties
    }, 
    im_params_dict={
        'im_H_load': 320, 'im_W_load': 640, 
        'im_H_resize': 320, 'im_W_resize': 640, 
        # 'im_H_load': 240, 'im_W_load': 320, 
        # 'im_H_resize': 160, 'im_W_resize': 320, 
        'spp': 4096, 
        # 'spp': 16, 
        # 'im_H_resize': 120, 'im_W_resize': 160, # to use for rendering so that im dimensions == lighting dimensions
        # 'im_hdr_ext': 'exr', 
        }, 
    cam_params_dict={
        'near': 0.1, 'far': 10., 
        'sampleNum': 3, 
        
        # == params for sample camera poses
        'heightMin' : 0.7, # camera height min
        'heightMax' : 3, # camera height max
        'distMin': 0.2, # to wall distance min
        'distMax': 3, # to wall distance max
        'thetaMin': -60, # theta min: pitch angle; up+ 
        'thetaMax' : 40, # theta max: pitch angle; up+
        'phiMin': -60, # yaw angle min
        'phiMax': 60, # yaw angle max
        'distRaysMin': 0.2, # min dist of all camera rays to the scene; [!!!] set to -1 to disable checking
        'distRaysMedianMin': 0.6, # median dist of all camera rays to the scene; [!!!] set to -1 to disable checking

        # ==> if sample poses and render images 
        'if_sample_poses': opt.if_sample_poses, # True to generate camera poses following Zhengqin's method (i.e. walking along walls)
        'sample_pose_num': 00 if 'train' in opt.split else 20, # Number of poses to sample; set to -1 if not sampling
        'sample_pose_if_vis_plt': True, # images/demo_sample_pose.png, images/demo_sample_pose_bathroom.png
    }, 
    lighting_params_dict={
        'SG_num': 12, 
        'env_row': 8, 'env_col': 16, # resolution to load; FIXED
        'env_downsample_rate': 2, # (8, 16) -> (4, 8)

        # 'env_height': 2, 'env_width': 4, 
        # 'env_height': 8, 'env_width': 16, 
        # 'env_height': 128, 'env_width': 256, 
        'env_height': 256, 'env_width': 512, 
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
        'simplify_mesh_max': 100000, 
        'if_remesh': True, # False: images/demo_shapes_3D_kitchen_NO_remesh.png; True: images/demo_shapes_3D_kitchen_YES_remesh.png
        'remesh_max_edge': 0.05,  
        
        'if_dump_shape': False, # True to dump fixed shape to obj file
        'if_fix_watertight': False, 
        },
    emitter_params_dict={
        },
)

'''
Mitsuba/Blender 2D renderer
'''
if opt.render_2d:
    assert opt.renderer in ['mi', 'blender']
    modality_list = [
        'im', # both hdr and sdr
        # 'poses', 
        # 'seg', 
        # 'albedo', 
        # 'roughness', 
        # 'depth', 'normal', 
        # 'lighting_envmap', 
        ]
    if opt.renderer == 'mi':
        renderer = renderer_mi_mitsubaScene_3D(
            scene_obj, 
            modality_list=modality_list, 
            im_params_dict={}, 
            cam_params_dict={}, 
            mi_params_dict={},
        )
    if opt.renderer == 'blender':
        renderer = renderer_blender_mitsubaScene_3D(
            scene_obj, 
            modality_list=modality_list, 
            host=host, 
            FORMAT='OPEN_EXR', 
            # FORMAT='PNG', 
            im_params_dict={}, 
            cam_params_dict={}, 
            mi_params_dict={},
        )
    host=host, 
    renderer.render()

eval_return_dict = {}
'''
Evaluator for rad-MLP
'''
if opt.eval_rad:
    evaluator_rad = evaluator_scene_rad(
        host=host, 
        scene_object=scene_obj, 
        INV_NERF_ROOT = INV_NERF_ROOT, 
        ckpt_path=monosdf_shape_dict.get('rad-MLP_ckpt_path', eval_models_dict['rad-MLP_ckpt_path']), 
        dataset_key='-'.join(['Indoor', scene_name]), # has to be one of the keys from inv-nerf/configs/scene_options.py
        split=split, 
        rad_scale=1., 
    )

    '''
    render one image by querying rad-MLP: images/demo_eval_radMLP_render.png
    '''
    # evaluator_rad.render_im(7, if_plt=True) 

    '''
    sample and visualize points on emitter surface; show intensity as vectors along normals (BLUE for EST): images/demo_emitter_o3d_sampling.png
    '''
    eval_return_dict.update(
        evaluator_rad.sample_emitter(
            emitter_params={
                'max_plate': 32, 
                'radiance_scale': radiance_scale, 
                'emitter_type_index_list': emitter_type_index_list, 
                }))
    
    '''
    sample non-emitter locations along envmap (hemisphere) directions radiance from rad-MLP: images/demo_envmap_o3d_sampling.png
    '''
    # eval_return_dict.update(
    #     evaluator_rad.sample_lighting(
    #         # sample_type='rad', # 'rad', 'incident-rad'
    #         sample_type='incident', # 'rad', 'incident-rad'
    #         subsample_rate_pts=1, 
    #         if_use_loaded_envmap_position=True, # assuming lighting envmap endpoint position dumped by Blender renderer
    #     )
    # )

    '''
    sample radiance field on shape vertices
    '''
    eval_return_dict.update(
        evaluator_rad.sample_shapes(
            sample_type='rad', # ['rad']
            shape_params={
                'radiance_scale': 1., 
            }
        )
    )

'''
Evaluator for inv-MLP
'''
if opt.eval_inv:
    evaluator_inv = evaluator_scene_inv(
        host=host, 
        scene_object=scene_obj, 
        INV_NERF_ROOT = INV_NERF_ROOT, 
        ckpt_path=monosdf_shape_dict.get('inv-MLP_ckpt_path', eval_models_dict['inv-MLP_ckpt_path']), 
        dataset_key='-'.join(['Indoor', scene_name]), # has to be one of the keys from inv-nerf/configs/scene_options.py
        split=split, 
        spec=True, 
        if_monosdf=monosdf_shape_dict=={}, 
        monosdf_shape_dict=monosdf_shape_dict, 
    )

    '''
    sample emission mask on shape vertices
    '''
    _ = evaluator_inv.sample_shapes(
        sample_type='emission_mask_bin', # ['emission_mask', 'emission_mask_bin', 'albedo', 'metallic', 'roughness']
        shape_params={
        }
    )
    for k, v in _.items():
        if k in eval_return_dict:
            eval_return_dict[k].update(_[k])
        else:
            eval_return_dict[k] = _[k]

if opt.eval_monosdf:
    evaluator_monosdf = evaluator_scene_monosdf(
        host=host, 
        scene_object=scene_obj, 
        MONOSDF_ROOT = MONOSDF_ROOT, 
        conf_path=monosdf_shape_dict['monosdf_conf_path'], 
        ckpt_path=monosdf_shape_dict['monosdf_ckpt_path'], 
        rad_scale=1., 
    )

    # evaluator_monosdf.export_mesh()

    evaluator_monosdf.render_im_scratch(
        frame_id=0, offset_in_scan=202, 
        if_integrate=False, 
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
    if 'scene' not in shape_file:
        print(red('Skipped fkipping for est geometry'))
    else:
        evaluator_scene = evaluator_scene_scene(
            host=host, 
            scene_object=scene_obj, 
        )

        '''
        sample visivility to camera centers on vertices
        [!!!] set 'mesh_color_type': 'eval-vis_count'
        '''
        _ = evaluator_scene.sample_shapes(
            # sample_type='vis_count', # ['']
            # sample_type='t', # ['']
            sample_type='face_normal', # ['']
            shape_params={
            }
        )
        for k, v in _.items():
            if k in eval_return_dict:
                eval_return_dict[k].update(_[k])
            else:
                eval_return_dict[k] = _[k]
                
        if 'face_normals_flipped_mask' in eval_return_dict and opt.export:
            face_normals_flipped_mask = eval_return_dict['face_normals_flipped_mask']
            assert face_normals_flipped_mask.shape[0] == scene_obj.faces_list[0].shape[0]
            if np.sum(face_normals_flipped_mask) > 0:
                validate_idx = np.where(face_normals_flipped_mask)[0][0]
                print(validate_idx, scene_obj.faces_list[0][validate_idx])
                scene_obj.faces_list[0][face_normals_flipped_mask] = scene_obj.faces_list[0][face_normals_flipped_mask][:, [0, 2, 1]]
                print(white_magenta('[FLIPPED] %d/%d inward face normals'%(np.sum(face_normals_flipped_mask), scene_obj.faces_list[0].shape[0])))
                print(validate_idx, '->', scene_obj.faces_list[0][validate_idx])

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
    if opt.export_format == 'fvp':
        exporter.export_monosdf_fvp_mitsuba(
            split=opt.split, 
            format='fvp',
            modality_list = [
                'poses', 
                'im_hdr', 
                'im_sdr', 
                'im_mask', 
                'shapes', 
                'lighting', # new lights
                ], 
            appendix=opt.export_appendix, 
        )
    if opt.export_format == 'lieccv22':
        exporter.export_lieccv22(
            modality_list = [
            'im_sdr', 
            'mi_depth', 
            'mi_seg', 
            ], 
            split=opt.split, 
            assert_shape=(240, 320),
            window_area_emitter_id_list=['window_area_emitter'], # need to manually specify in XML: e.g. <emitter type="area" id="lamp_oven_0">
            merge_lamp_id_list=['lamp_oven_0', 'lamp_oven_1', 'lamp_oven_2'],  # need to manually specify in XML
            # center_crop_HW=(240, 320), 
            if_no_gt_appendix=True, 
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
            # 'emission', 
            # 'depth', 
            # 'normal', 
            # 'mi_depth', 
            'mi_normal', # compare depth & normal maps from mitsuba sampling VS OptixRenderer: **mitsuba does no anti-aliasing**: images/demo_mitsuba_ret_depth_normals_2D.png
            # 'lighting_SG', # convert to lighting_envmap and vis: images/demo_lighting_SG_envmap_2D_plt.png
            # 'lighting_envmap', # renderer with mi/blender: images/demo_lighting_envmap_mitsubaScene_2D_plt.png
            # 'seg_area', 'seg_env', 'seg_obj', 
            # 'mi_seg_area', 'mi_seg_env', 'mi_seg_obj', # compare segs from mitsuba sampling VS OptixRenderer: **mitsuba does no anti-aliasing**: images/demo_mitsuba_ret_seg_2D.png
            ], 
        # frame_idx_list=[0, 1, 2, 3, 4], 
        frame_idx_list=[0], 
        # frame_idx_list=[6, 10, 12], 
    )
    if opt.if_add_est_from_eval:
        for modality in ['lighting_envmap']:
            if modality in eval_return_dict:
                scene_obj.add_modality(eval_return_dict[modality], modality, 'EST')

    visualizer_2D.vis_2d_with_plt(
        lighting_params={
            'lighting_scale': 1., # rescaling the brightness of the envmap
            }, 
        other_params={
            # 'mi_normal_vis_coords': 'world-blender', 
            'mi_normal_vis_coords': 'opencv', 
            'mi_depth_if_sync_scale': False, 
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
            # 'dense_geo', # fused from 2D
            'poses', 
            # 'lighting_SG', # images/demo_lighting_SG_o3d.png; arrows in blue
            # 'lighting_envmap', # images/demo_lighting_envmap_o3d.png; arrows in pink
            'layout', 
            'shapes', # bbox and (if loaded) meshs of shapes (objs + emitters SHAPES); CTRL + 9
            # 'emitters', # emitter PROPERTIES (e.g. SGs, half envmaps)
            'mi', # mitsuba sampled rays, pts
            ], 
        if_debug_info=opt.if_debug_info, 
    )

    lighting_params_vis={
        'if_use_mi_geometry': True, 
        'if_use_loaded_envmap_position': True, # assuming lighting envmap endpoint position dumped by Blender renderer
        'subsample_lighting_pts_rate': 1, # change this according to how sparse the lighting arrows you would like to be (also according to num of frame_ids)
        'subsample_lighting_wi_rate': 500, # subsample on lighting directions: too many directions (e.g. 128x256)
        # 'lighting_keep_ratio': 0.05, 
        # 'lighting_further_clip_ratio': 0.1, 
        'lighting_scale': 2, 
        # 'lighting_keep_ratio': 0.2, # - good for lighting_SG
        # 'lighting_further_clip_ratio': 0.3, 
        'lighting_keep_ratio': 0., # - good for lighting_envmap
        'lighting_further_clip_ratio': 0., 
        # 'lighting_keep_ratio': 0., # - debug
        # 'lighting_further_clip_ratio': 0., 
        'lighting_autoscale': False, 
        'lighting_if_show_hemisphere': True, # mainly to show hemisphere and local axes: images/demo_lighting_envmap_hemisphere_axes_o3d.png
        }

    if opt.if_add_rays_from_eval:
        for key in eval_return_dict.keys():
            if 'rays_list' in key:
            # assert opt.eval_rad
                lpts, lpts_end = eval_return_dict[key]['v'], eval_return_dict[key]['v']+eval_return_dict[key]['d']*eval_return_dict[key]['l'].reshape(-1, 1)
                visualizer_3D_o3d.add_extra_geometry([
                    ('rays', {
                        'ray_o': lpts, 'ray_e': lpts_end, 'ray_c': np.array([[0., 0., 1.]]*lpts.shape[0]), # BLUE for EST
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
                    if_use_pts_end=True,
                    )
                visualizer_3D_o3d.add_extra_geometry(geometry_list, if_processed_geometry_list=True)
        
    if opt.if_add_color_from_eval:
        if 'samples_v_dict' in eval_return_dict:
            assert opt.eval_rad or opt.eval_inv or opt.eval_scene or opt.eval_monosdf
            visualizer_3D_o3d.extra_input_dict['samples_v_dict'] = eval_return_dict['samples_v_dict']
        
    visualizer_3D_o3d.run_o3d(
        if_shader=opt.if_shader, # set to False to disable faycny shaders 
        cam_params={
            'if_cam_axis_only': False, 
            }, 
        lighting_params=lighting_params_vis, 
        shapes_params={
            # 'simply_mesh_ratio_vis': 1., # simply num of triangles to #triangles * simply_mesh_ratio_vis
            'if_meshes': True, # [OPTIONAL] if show meshes for objs + emitters (False: only show bboxes)
            'if_labels': False, # [OPTIONAL] if show labels (False: only show bboxes)
            'if_voxel_volume': False, # [OPTIONAL] if show unit size voxel grid from shape occupancy: images/demo_shapes_voxel_o3d.png; USEFUL WHEN NEED TO CHECK SCENE SCALE (1 voxel = 1 meter)
            # 'if_ceiling': True if opt.eval_scene else False, # [OPTIONAL] remove ceiling meshes to better see the furniture 
            # 'if_walls': True if opt.eval_scene else False, # [OPTIONAL] remove wall meshes to better see the furniture 
            'if_ceiling': True, # [OPTIONAL] remove ceiling meshes to better see the furniture 
            'if_walls': True, # [OPTIONAL] remove wall meshes to better see the furniture 
            'if_sampled_pts': False, # [OPTIONAL] is show samples pts from scene_obj.sample_pts_list if available
            'mesh_color_type': 'eval-', # ['obj_color', 'face_normal', 'eval-' ('rad', 'emission_mask', 'vis_count', 't')]
        },
        emitter_params={
            # 'if_half_envmap': False, # [OPTIONAL] if show half envmap as a hemisphere for window emitters (False: only show bboxes)
            # 'scale_SG_length': 2., 
            'if_sampling_emitter': True, 
            'radiance_scale': radiance_scale, 
            'max_plate': 32, 
        },
        mi_params={
            'if_pts': False, # if show pts sampled by mi; should close to backprojected pts from OptixRenderer depth maps
            'if_pts_colorize_rgb': True, 
            'pts_subsample': 10,
            # 'if_ceiling': True, # [OPTIONAL] remove ceiling points to better see the furniture 
            # 'if_walls': True, # [OPTIONAL] remove wall points to better see the furniture 

            'if_cam_rays': True, 
            'cam_rays_if_pts': True, # if cam rays end in surface intersections; set to False to visualize rays of unit length
            'cam_rays_subsample': 10, 
            
            'if_normal': False, 
            'normal_subsample': 50, 
            'normal_scale': 0.2, 

        }, 
    )


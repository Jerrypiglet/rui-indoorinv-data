'''
Script to load Blender scenes, render with Blender (bpy)/Mitsuba and fuse rendered RGB images onto a TSDF volume-bases mesh, to check multi-view consistency of the renderings.

> git clone --branch add_blender_debug https://github.com/Jerrypiglet/rui-indoorinv-data.git
> cd rui-indoorinv-data

Install the dependencies via: 

> install Torch (>2.0.0) following: https://pytorch.org/get-started/locally/
> pip install -r test_scripts/debug_blender_requirements.txt

Download the scene with: 

> mkdir -p data/debug_scenes
> cd data/debug_scenes
> wget -m -np http://rui4090.ucsd.edu/mclab-data-public/test_files/test_scenes_fipt/
> mv rui4090.ucsd.edu/mclab-data-public/test_files/test_scenes_fipt/debug_scenes/* .

So that the scenes are organized as:

- ./data/debug_scenes/
    - kitchen_diy/
    - cornel_box/ # export cameras from .blend file via: python test_scripts/load_blender_file.py
    
Add your local path and config to lib/global_vars.py, by replacing all the #TODO entries. 
    
Run: 

> cd test_scripts
> python debug_blender_mitsubaScene3D.py --scene kitchen_diy --renderer blender

Choose --scene as one of the scene names; choose --renderer between 'mi' and 'blender'.

Output:

- Rendered images in: debug_scenes/kitchen_diy/train/Image/*.exr (*.exr for Blender renderings; *_mi.exr for Mitsuba renderings)
- Fused TSDF shape at: debug_scenes/kitchen_diy/fused_tsdf.ply (fused_tsdf.ply for fused Blender renderings; fused_tsdf_mi.ply for fused Mitsuba renderings)
- Exported .blend files for each frame: debug_scenes/kitchen_diy/test_frame_{}.blend

Options:

- Choose a subset of frames to render/fuse, via: frame_id_list = [61, 63, 66, 77]
- Set --spp properly for Blender/Mitsuba renderings to reduce noise level (e.g. spp>=128 for Blender; spp>=512 for Mitsuba)

'''
 
import os, sys
from pathlib import Path
path = os.getcwd()
PATH_HOME = Path(os.path.abspath(os.path.join(path, os.pardir)))
sys.path.insert(0, str(PATH_HOME))

# host = 'debug'
host = 'apple'

import numpy as np
np.set_printoptions(suppress=True)
import argparse
from pyhocon import ConfigFactory, ConfigTree
from lib.utils_misc import str2bool, white_magenta, check_exists
from lib.class_mitsubaScene3D import mitsubaScene3D
from lib.class_visualizer_scene_2D import visualizer_scene_2D
from lib.class_visualizer_scene_3D_o3d import visualizer_scene_3D_o3d

from lib.class_eval_scene import evaluator_scene_scene

from lib.class_renderer_mi_mitsubaScene_3D import renderer_mi_mitsubaScene_3D
from lib.class_renderer_blender_mitsubaScene_3D import renderer_blender_mitsubaScene_3D

parser = argparse.ArgumentParser()
# visualizers
parser.add_argument('--split', type=str, default='train', help='')

# renderer (mi/blender)
parser.add_argument('--renderer', type=str, default='blender', help='mi, blender')
parser.add_argument('--spp', type=int, default=128, help='spp for mi/blender')

# ==== Evaluators
parser.add_argument('--if_add_color_from_eval', type=str2bool, nargs='?', const=True, default=True, help='if colorize mesh vertices with values from evaluator')
parser.add_argument('--eval_scene', type=str2bool, nargs='?', const=True, default=False, help='eval over scene (e.g. shapes for coverage)')

# debug
parser.add_argument('--if_debug_info', type=str2bool, nargs='?', const=True, default=False, help='if show debug info')

# utils
parser.add_argument('--if_sample_poses', type=str2bool, nargs='?', const=True, default=False, help='if sample camera poses instead of loading from pose file')
parser.add_argument('--force', type=str2bool, nargs='?', const=True, default=True, help='if force to overwrite existing files')

# === after refactorization
parser.add_argument('--DATASET', type=str, default='debug_scenes', help='load conf file: confs/\{DATASET\}')
parser.add_argument('--scene', type=str, default='kitchen_diy', help='load conf file: confs/\{DATASET\/\{opt.scene\}.conf')

opt = parser.parse_args()

conf_base_path = PATH_HOME / Path('confs/%s.conf'%opt.DATASET); check_exists(conf_base_path)
CONF = ConfigFactory.parse_file(str(conf_base_path))
conf_scene_path = PATH_HOME / Path('confs/%s/%s.conf'%(opt.DATASET, opt.scene)); check_exists(conf_scene_path)
conf_scene = ConfigFactory.parse_file(str(conf_scene_path))
CONF = ConfigTree.merge_configs(CONF, conf_scene)

dataset_root = Path(PATH_HOME) / CONF.data.dataset_root
xml_root = Path(PATH_HOME) / CONF.data.xml_root

frame_id_list = CONF.scene_params_dict.frame_id_list
invalid_frame_id_list = CONF.scene_params_dict.invalid_frame_id_list

# [debug] override
# frame_id_list = [61, 63, 66, 77]

'''
update confs
'''

CONF.scene_params_dict.update({
    'split': opt.split, # train, val, train+val
    'frame_id_list': frame_id_list, 
    'invalid_frame_id_list': invalid_frame_id_list, 
    })
    
CONF.cam_params_dict.update({
    # ==> if sample poses and render images 
    'if_sample_poses': opt.if_sample_poses, # True to generate camera poses following Zhengqin's method (i.e. walking along walls)
    'sample_pose_num': 200 if 'train' in opt.split else 20, # Number of poses to sample; set to -1 if not sampling
    'sample_pose_if_vis_plt': True, # images/demo_sample_pose.png, images/demo_sample_pose_bathroom.png
    })

CONF.mi_params_dict.update({
    'if_sample_rays_pts': True, # True: to sample camera rays and intersection pts given input mesh and camera poses
    'if_get_segs': True, # [depend on if_sample_rays_pts=True] True: to generate segs similar to those in openroomsScene2D.load_seg()
    # 'if_mi_scene_from_xml': False, 
    })

# CONF.im_params_dict.update({
#     'im_H_load': 320, 'im_W_load': 640, 
#     'im_H_resize': 320, 'im_W_resize': 640, 
#     # 'im_H_resize': 640, 'im_W_resize': 1280, 
#     })

CONF.shape_params_dict.update({
    'if_load_obj_mesh': True, # set to False to not load meshes for objs (furniture) to save time
    'if_load_emitter_mesh': True, # default True: to load emitter meshes, because not too many emitters
    })

'''
create scene obj
'''
# if not opt.render_2d:
#     modality_list += ['im_hdr', 'im_sdr', 'depth', 'normal', 'tsdf']

scene_obj = mitsubaScene3D(
    CONF = CONF, 
    if_debug_info = opt.if_debug_info, 
    host = host, 
    root_path_dict = {'PATH_HOME': Path(PATH_HOME), 'dataset_root': dataset_root, 'xml_root': xml_root}, 
    modality_list = ['poses'],
)

'''
Mitsuba/Blender 2D renderer
'''
assert opt.renderer in ['mi', 'blender']
if opt.renderer == 'mi':
    renderer = renderer_mi_mitsubaScene_3D(
        scene_obj, 
        modality_list=[
            'im', # both hdr and sdr
        ], 
        im_params_dict=
        {
            # 'im_H_load': 640, 'im_W_load': 1280, 
            # 'im_H_load': 320, 'im_W_load': 640, 
            'spp': opt.spp, 
            # 'spp': 4096, # default 4096, because of lack of denosing 
        }, # override
        cam_params_dict={}, 
        mi_params_dict={},
        if_skip_check=True,
    )
if opt.renderer == 'blender':
    renderer = renderer_blender_mitsubaScene_3D(
        scene_obj, 
        modality_list=[
            'im', 
            'albedo', 
            'roughness', 
            'depth', 
            'normal', 
            'index', 
            'emission', 
            # 'lighting_envmap', 
            ], 
        host=host, 
        FORMAT='OPEN_EXR', 
        # FORMAT='PNG', 
        im_params_dict=
        {
            # 'im_H_load': 640, 'im_W_load': 1280, 
            # 'im_H_load': 320, 'im_W_load': 640, 
            'spp': opt.spp
            # 'spp': 1024, # default 1024
        }, # override
        cam_params_dict={}, 
        mi_params_dict={},
        # blender_file_name='test_blender_export_reimport.blend', 
        # if_skip_check=False,
    )
    
renderer.render(if_force=opt.force)

# compare HDR images from mi/blender if both are available
# renderer.compare_blender_mi_Image()

'''
fuse TSDF volume
'''

CONF.shape_params_dict.update({
    'if_force_fuse_tsdf': True, 
    })

if opt.renderer == 'mi':
    CONF.modality_filename_dict.update({
        'im_hdr': 'Image/%03d_0001_mi.exr', 
        'im_sdr': 'Image/%03d_0001_mi.png', 
    })
    CONF.shape_params_dict.update({
        'tsdf_file': 'fused_tsdf_mi.ply', 
        })

'''
Dump TSDF file
'''
scene_obj = mitsubaScene3D(
    CONF = CONF, 
    if_debug_info = opt.if_debug_info, 
    host = host, 
    root_path_dict = {'PATH_HOME': Path(PATH_HOME), 'dataset_root': dataset_root, 'xml_root': xml_root}, 
    # modality_list = ['poses', 'im_hdr', 'im_sdr'],
    modality_list = ['poses', 'im_hdr', 'im_sdr', 'tsdf'],
)

'''
Evaluator for scene
'''

# eval_return_dict = {}

# CONF.mi_params_dict.update({
#     'if_mi_scene_from_xml': False, 
#     })

# scene_obj = mitsubaScene3D(
#     CONF = CONF, 
#     if_debug_info = opt.if_debug_info, 
#     host = host, 
#     root_path_dict = {'PATH_HOME': Path(PATH_HOME), 'dataset_root': dataset_root, 'xml_root': xml_root}, 
#     modality_list = ['poses', 'im_hdr', 'im_sdr', 'tsdf'],
# )

# if opt.eval_scene:
#     evaluator_scene = evaluator_scene_scene(
#         host=host, 
#         scene_object=scene_obj, 
#         eval_scene_from='tsdf', 
#     )

#     '''
#     sample visivility to camera centers on vertices
#     [!!!] set 'mesh_color_type': 'eval-vis_count'
#     '''
#     _ = evaluator_scene.sample_shapes(
#         sample_type='vis_count', # ['']
#         # sample_type='t', # ['']
#         # sample_type='face_normal', # ['']
#         # sample_type='rgb_sdr', 
#         shape_params={
#         }
#     )
#     for k, v in _.items():
#         if k in eval_return_dict:
#             eval_return_dict[k].update(_[k])
#         else:
#             eval_return_dict[k] = _[k]
            
#     if 'face_normals_flipped_mask' in eval_return_dict and opt.export:
#         face_normals_flipped_mask = eval_return_dict['face_normals_flipped_mask']
#         assert face_normals_flipped_mask.shape[0] == scene_obj.faces_list[0].shape[0]
#         if np.sum(face_normals_flipped_mask) > 0:
#             validate_idx = np.where(face_normals_flipped_mask)[0][0]
#             print(validate_idx, scene_obj.faces_list[0][validate_idx])
#             scene_obj.faces_list[0][face_normals_flipped_mask] = scene_obj.faces_list[0][face_normals_flipped_mask][:, [0, 2, 1]]
#             print(white_magenta('[FLIPPED] %d/%d inward face normals'%(np.sum(face_normals_flipped_mask), scene_obj.faces_list[0].shape[0])))
#             print(validate_idx, '->', scene_obj.faces_list[0][validate_idx])

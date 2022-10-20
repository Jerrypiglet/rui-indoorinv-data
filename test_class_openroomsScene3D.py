import sys

device = 'local'
PATH_HOME = {
    'local': '/Users/jerrypiglet/Documents/Projects/OpenRooms_RAW_loader', 
}[device]
OR_RAW_ROOT = {
    'local': '/Users/jerrypiglet/Documents/Projects/data', 
}[device]

sys.path.insert(0, PATH_HOME)
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import numpy as np
np.set_printoptions(suppress=True)
from lib.utils_io import load_matrix, load_img, load_binary, load_h5

from lib.class_openroomsScene3D import openroomsScene3D
from lib.class_visualizer_openroomsScene_o3d import visualizer_openroomsScene_o3d
from lib.class_visualizer_openroomsScene_2D import visualizer_openroomsScene_2D

from lib.utils_misc import str2bool
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--vis_3d_plt', type=str2bool, nargs='?', const=True, default=False, help='whether to visualize 3D with plt for debugging')
parser.add_argument('--vis_o3d', type=str2bool, nargs='?', const=True, default=True, help='whether to render in open3D')
parser.add_argument('--vis_2d_proj', type=str2bool, nargs='?', const=True, default=False, help='whether to show projection onto one image with plt (e.g. layout, object bboxes')
parser.add_argument('--if_shader', type=str2bool, nargs='?', const=True, default=True, help='')
opt = parser.parse_args()

base_root = Path(PATH_HOME) / 'data/openrooms_public'
xml_root = Path(PATH_HOME) / 'data/openrooms_public/scenes'
intrinsics_path = Path(PATH_HOME) / 'data/intrinsic.txt'
semantic_labels_root = Path(PATH_HOME) / 'data'
layout_root = Path(OR_RAW_ROOT) / 'layoutMesh'
shapes_root = Path(OR_RAW_ROOT) / 'uv_mapped'
envmaps_root = Path(OR_RAW_ROOT) / 'EnvDataset' # not publicly availale

'''
The classroom scene: one lamp (lit up) + one window (less sun)

data/openrooms_public_re_2/main_xml1/scene0552_00_more/im_1.png
'''

base_root = Path(PATH_HOME) / 'data/openrooms_public_re_2'
xml_root = Path(PATH_HOME) / 'data/openrooms_public_re_2/scenes'
meta_split = 'main_xml1'
scene_name = 'scene0552_00_more'
frame_ids = list(range(0, 87, 10))

openrooms_scene = openroomsScene3D(
    root_path_dict = {'rendering_root': base_root, 'xml_scene_root': xml_root, 'semantic_labels_root': semantic_labels_root, 
        'intrinsics_path': intrinsics_path, 'layout_root': layout_root, 'shapes_root': shapes_root, 'envmaps_root': envmaps_root}, 
    scene_params_dict={'meta_split': meta_split, 'scene_name': scene_name, 'frame_id_list': frame_ids}, 
    # modality_list = ['im_sdr', 'im_hdr', 'seg', 'poses', 'albedo', 'roughness', 'depth', 'normal', 'lighting_SG', 'lighting_envmap'], 
    modality_list = [
        'im_sdr', 'poses', 'seg', 
        'depth', 'normal', 
        # 'lighting_SG', 
        # 'lighting_envmap', 
        'layout', 
        'shapes', # objs + emitters, geometry shapes + emitter properties
        ], 
    im_params_dict={'im_H_load': 480, 'im_W_load': 640, 'im_H_resize': 240, 'im_W_resize': 320}, 
    shape_params_dict={
        'if_load_obj_mesh': False, # set to False to not load meshes for objs (furniture) to save time
        'if_load_emitter_mesh': True,  # default set to True to load emitter meshes, because not too many emitters
        },
    emitter_params_dict={'N_ambient_rep': '3SG-SkyGrd'},
    if_vis_debug_with_plt=opt.vis_3d_plt, 
    modality_list_vis = [
        'layout', 
        'shapes', # boxes and labels (no meshes in plt visualization)
        'emitters', # emitter properties
        'emitter_envs', # emitter envmaps for (1) global envmap (2) half envmap & SG envmap of each window
        ], 
)

# scene_rendering_dir = Path(base_root) / meta_split / scene_name
# scene_xml_dir = Path(xml_root) / (meta_split.split('_')[1]) / scene_name

if opt.vis_2d_proj:
    vis_2D = visualizer_openroomsScene_2D(
        openrooms_scene, 
        modality_list=[
            'layout', 
            # 'shapes', 
            ], 
        frame_idx_list=[0, 1, 2, 3, 4], 
    )
    vis_2D.vis_2d_with_plt()

if opt.vis_o3d:
    vis_o3d = visualizer_openroomsScene_o3d(
        openrooms_scene, 
        modality_list=[
            'dense_geo', 
            'cameras', 
            # 'lighting_SG', 
            'layout', 
            'shapes', # bbox and meshs of shapes (objs + emitters)
            'emitters' # emitter properties (e.g. SGs, half envmaps)
            ], 
    )

    vis_o3d.run_o3d(
        if_shader=opt.if_shader, # set to False to disable shaders 
        cam_params={}, 
        dense_geo_params={
            'subsample_pcd_rate': 1, # change this according to how sparse the points you would like to be (also according to num of frame_ids)
            'if_ceiling': False, # [OPTIONAL] remove ceiling points to better see the furniture 
            'if_walls': False, # [OPTIONAL] remove wall points to better see the furniture 
            'if_normal': False, # [OPTIONAL] turn off normals to avoid clusters
            'subsample_normal_rate_x': 2, 
            }, 
        lighting_SG_params={
            'subsample_lighting_SG_rate': 200, # change this according to how sparse the lighting arrows you would like to be (also according to num of frame_ids)
            # 'SG_scale': 1., 
            # 'SG_keep_ratio': 0.05, 
            # 'SG_clip_ratio': 0.1, 
            'SG_scale': 0.5, 
            'SG_keep_ratio': 0.2, 
            'SG_clip_ratio': 0.3, 
            'SG_autoscale': True, 
            }, 
        shapes_params={
            'simply_ratio': 0.1, # simply num of triangles to #triangles * simply_ratio
            'if_meshes': True, # [OPTIONAL] if show meshes for objs + emitters (False: only show bboxes)
            'if_labels': False, # [OPTIONAL] if show labels (False: only show bboxes)
        },
        emitters_params={
            'if_half_envmap': False, # [OPTIONAL] if show half envmap as a hemisphere for window emitters (False: only show bboxes)
            'scale_SG_length': 2., 
        },

    )

# dump_path = Path(PATH_HOME) / ('logs/pickles/OR_public_re_gt_%s_#MOD_openrooms.pickle'%scene_name[5:])
# OR.fuse_3D_geometry(dump_path=dump_path)

# dump_path = Path(PATH_HOME) / ('logs/pickles/OR_public_re_gt_%s_#MOD_openrooms.pickle'%scene_name[5:])
# OR.fuse_3D_geometry(dump_path=dump_path)

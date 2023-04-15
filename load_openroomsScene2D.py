'''
work with multi-view OpenRooms scene, with 2D modalities
'''

import sys
from pathlib import Path

# host = 'mm1'
host = 'apple'

from lib.global_vars import PATH_HOME_dict# , INV_NERF_ROOT_dict, MONOSDF_ROOT_dict, OR_RAW_ROOT_dict
PATH_HOME = Path(PATH_HOME_dict[host])
sys.path.insert(0, str(PATH_HOME))

import numpy as np
np.set_printoptions(suppress=True)
import argparse
from pyhocon import ConfigFactory, ConfigTree
from lib.utils_misc import str2bool, check_exists

from lib.class_openroomsScene2D import openroomsScene2D
from lib.class_visualizer_scene_2D import visualizer_scene_2D

parser = argparse.ArgumentParser()
parser.add_argument('--vis_2d_plt', type=str2bool, nargs='?', const=True, default=True, help='whether to show (1) pixel-space modalities (2) projection onto one image (e.g. layout, object bboxes), with plt')
parser.add_argument('--if_shader', type=str2bool, nargs='?', const=True, default=True, help='')

# debug
parser.add_argument('--if_debug_info', type=str2bool, nargs='?', const=True, default=False, help='if show debug info')

# === after refactorization
parser.add_argument('--scene', type=str, default='mainDiffLight_xml1-scene0552_00_more', help='load conf file: confs/openrooms/\{opt.scene\}.conf')

opt = parser.parse_args()

DATASET = 'openrooms'
conf_base_path = Path('confs/%s.conf'%DATASET); check_exists(conf_base_path)
CONF = ConfigFactory.parse_file(str(conf_base_path))
conf_scene_path = Path('confs/%s/%s.conf'%(DATASET, opt.scene)); check_exists(conf_scene_path)
conf_scene = ConfigFactory.parse_file(str(conf_scene_path))
CONF = ConfigTree.merge_configs(CONF, conf_scene)

dataset_root = Path(PATH_HOME) / CONF.data.dataset_root
xml_root = dataset_root / 'scenes'
semantic_labels_root = Path(PATH_HOME) / 'data'

# meta_split = 'mainDiffLight_xml1'
# scene_name = 'scene0552_00_more'
frame_id_list = [0, 11, 10, 64, 81]
# + list(range(5, 87, 10))

'''
update confs
'''

CONF.scene_params_dict.update({
    # 'split': opt.split, # train, val, train+val
    'frame_id_list': frame_id_list, 
    })

CONF.im_params_dict.update({
    'im_H_resize': 240, 'im_W_resize': 320, 
    })

openrooms_scene = openroomsScene2D(
    CONF = CONF, 
    if_debug_info = opt.if_debug_info, 
    host = host, 
    root_path_dict = {
        'PATH_HOME': Path(PATH_HOME), 
        'dataset_root': Path(dataset_root), 
        'xml_root': Path(xml_root), 
        'semantic_labels_root': semantic_labels_root, 
        }, 
    modality_list = [
        'im_sdr', 
        'im_hdr', 
        'poses', 
        'seg', 
        'albedo', 'roughness', 
        'depth', 'normal',
        'matseg', 
        # 'semseg', 
        # 'lighting_SG', 'lighting_envmap'
        ], 
)

if opt.vis_2d_plt:
    vis_2D_plt = visualizer_scene_2D(
        openrooms_scene, 
        modality_list_vis=[
            'depth', 'normal', 'albedo', 'roughness', # images/demo_all_2D.png
            'seg_area', 'seg_env', 'seg_obj', 
            'matseg', # images/demo_semseg_matseg_2D.png
            # 'semseg', # images/demo_semseg_matseg_2D.png
            ], 
        frame_idx_list=[0, 1, 2, 3, 4], # 0-based indexing of all selected frames
    )
    vis_2D_plt.vis_2d_with_plt()
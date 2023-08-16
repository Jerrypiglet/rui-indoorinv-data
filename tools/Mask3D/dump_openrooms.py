'''
Rui Zhu

python dump_openrooms.py 

dump all scenes in one for loop; no worries of GPU issues with multiprocessing + mitsuba/torch
'''

from pathlib import Path
'''
set those params according to your environment
'''
DATASET = 'openrooms_public'
# host = 'apple'; PATH_HOME = '/Users/jerrypiglet/Documents/Projects/rui-indoorinv-data'
# host = 'r4090'; PATH_HOME = '/home/ruizhu/Documents/Projects/rui-indoorinv-data'
host = 'mm1'; PATH_HOME = '/home/ruizhu/Documents/Projects/rui-indoorinv-data'
dump_root = Path('/newdata/Mask3D_data/openrooms_public_dump_v2small')

import os, sys
sys.path.insert(0, PATH_HOME)

from lib.global_vars import PATH_HOME_dict, INV_NERF_ROOT_dict, MONOSDF_ROOT_dict, OR_RAW_ROOT_dict
assert PATH_HOME == PATH_HOME_dict[host]

from tqdm import tqdm
import numpy as np
np.set_printoptions(suppress=True)
import copy
import pickle
from pyhocon import ConfigFactory, ConfigTree
from pyhocon.tool import HOCONConverter
from lib.utils_misc import str2bool, check_exists, yellow
from lib.utils_openrooms import get_im_info_list
from lib.class_openroomsScene3D import openroomsScene3D
from lib.class_eval_scene import evaluator_scene_scene
from test_scripts.scannet_utils.utils_visualize import visualize

'''
global vars
'''

OR_RAW_ROOT = OR_RAW_ROOT_dict[host]
INV_NERF_ROOT = INV_NERF_ROOT_dict[host]
MONOSDF_ROOT = MONOSDF_ROOT_dict[host]

conf_base_path = Path('confs/%s.conf'%DATASET); check_exists(conf_base_path)
CONF = ConfigFactory.parse_file(str(conf_base_path))

dataset_root = Path(PATH_HOME) / CONF.data.dataset_root; check_exists(dataset_root)
xml_root = dataset_root / 'scenes'; check_exists(xml_root)
semantic_labels_root = Path(PATH_HOME) / 'files_openrooms'; check_exists(semantic_labels_root)

layout_root = Path(OR_RAW_ROOT) / 'layoutMesh'; check_exists(layout_root)
shapes_root = Path(OR_RAW_ROOT) / 'uv_mapped'; check_exists(shapes_root)
envmaps_root = Path(OR_RAW_ROOT) / 'EnvDataset'; check_exists(envmaps_root)

'''
go over all scenes
'''
black_list = [
    ('main_xml1', 'scene0386_00'), 
    ('main_xml', 'scene0386_00'), 
    ('main_xml', 'scene0608_01'), 
    ('main_xml1', 'scene0608_01'), 
    ('main_xml1', 'scene0211_02'), 
    ('main_xml1', 'scene0126_02'), 
]

frame_list_root = semantic_labels_root / 'public'
assert frame_list_root.exists(), frame_list_root

from dump_openrooms_func import process_one_scene

# for split in ['train', 'val']:
for split in ['val']:
    exclude_scene_list_file = dump_root / ('excluded_scenes_%s.txt'%split)
    # if exclude_scene_list_file.exists():
    #     exclude_scene_list_file.unlink()
    scene_list = get_im_info_list(frame_list_root, split)
    # scene_list = [('mainDiffMat_xml1', 'scene0385_01')]

    import time
    tic = time.time()
    excluded_scenes = []
    for (meta_split, scene_name) in tqdm(scene_list):
        print('++++', meta_split, scene_name, '++++')
        
        # IF_DEBUG = np.random.random() < 0.05
        # IF_DEBUG = np.random.random() <= 1.0
        IF_DEBUG = split == 'val'
        # IF_DEBUG = True
        
        if (meta_split.replace('DiffMat', '').replace('DiffLight', ''), scene_name) in black_list:
            excluded_scenes.append((meta_split, scene_name))
            with open(str(exclude_scene_list_file), 'a') as f:
                f.write('%s %s\n'%(meta_split, scene_name))
                print('[Excluded:]', meta_split, scene_name)
            continue
        
        process_one_scene(CONF, dump_root, exclude_scene_list_file, meta_split, scene_name, IF_DEBUG=False)
                    
    print('=== Done (%d scenes) in %.3f s'%(len(scene_list), time.time() - tic))

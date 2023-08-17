'''
Rui Zhu

Dump all scenes in one for loop; no worries of GPU issues with multiprocessing + mitsuba/torch

Usage: python dump_openrooms.py 

[!!!] Before start, change the file list in dump_openrooms_func.py -> gather_missing_scenes() accordingly!
'''

from pathlib import Path
'''
set those params according to your environment
'''
DATASET = 'openrooms_public'

# host = 'apple'; PATH_HOME = '/Users/jerrypiglet/Documents/Projects/rui-indoorinv-data'
# host = 'r4090'; PATH_HOME = '/home/ruizhu/Documents/Projects/rui-indoorinv-data'
host = 'mm1'; PATH_HOME = '/home/ruizhu/Documents/Projects/rui-indoorinv-data'

# dump_root = Path('/data/Mask3D_data/openrooms_public_dump')
dump_root = Path('/newdata/Mask3D_data/openrooms_public_dump_v3smaller')

import os, sys
sys.path.insert(0, PATH_HOME)

from lib.global_vars import PATH_HOME_dict, OR_RAW_ROOT_dict
assert PATH_HOME == PATH_HOME_dict[host]

from tqdm import tqdm
import numpy as np
np.set_printoptions(suppress=True)
# import copy
# import pickle
from pyhocon import ConfigFactory, ConfigTree
# from pyhocon.tool import HOCONConverter
from lib.utils_misc import str2bool, check_exists, yellow, white_red
from lib.utils_openrooms import get_im_info_list
# from lib.class_openroomsScene3D import openroomsScene3D
# from lib.class_eval_scene import evaluator_scene_scene
# from test_scripts.scannet_utils.utils_visualize import visualize

'''
global vars
'''

OR_RAW_ROOT = OR_RAW_ROOT_dict[host]

conf_base_path = Path('confs/%s.conf'%DATASET); check_exists(conf_base_path)
CONF = ConfigFactory.parse_file(str(conf_base_path))

dataset_root = Path(PATH_HOME) / CONF.data.dataset_root; check_exists(dataset_root)
xml_root = dataset_root / 'scenes'; check_exists(xml_root)
semantic_labels_root = Path(PATH_HOME) / 'files_openrooms'; check_exists(semantic_labels_root)

layout_root = Path(OR_RAW_ROOT) / 'layoutMesh'; check_exists(layout_root)
shapes_root = Path(OR_RAW_ROOT) / 'uv_mapped'; check_exists(shapes_root)
envmaps_root = Path(OR_RAW_ROOT) / 'EnvDataset'; check_exists(envmaps_root)

CONF.im_params_dict.update({'im_H_resize': 240, 'im_W_resize': 320})
CONF.shape_params_dict.update({'force_regenerate_tsdf': True})
# Mitsuba options
CONF.mi_params_dict.update({'if_mi_scene_from_xml': True}) # !!!! set to False to load from shapes (single shape or tsdf fused shape (with tsdf in modality_list))
# TSDF options
CONF.shape_params_dict.update({
    'if_force_fuse_tsdf': True, 
    # 'tsdf_voxel_length': 8.0 / 512.0,
    # 'tsdf_sdf_trunc': 0.05,
    'tsdf_voxel_length': 12.0 / 512.0,
    'tsdf_sdf_trunc': 0.08,
    }) # !!!! set to True to force replace existing tsdf shape

params_dict = dict(
    CONF = CONF, 
    if_debug_info = False, 
    host = host, 
    root_path_dict = {
        'PATH_HOME': Path(PATH_HOME), 
        'dataset_root': Path(dataset_root), 
        'xml_root': Path(xml_root), 
        'semantic_labels_root': semantic_labels_root, 
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
        'tsdf', 
        'mi', # mitsuba scene, loading from scene xml file
        ], 
    )

import pickle
params_dict_path = dump_root / 'params_dict.pkl'
if params_dict_path.exists(): params_dict_path.unlink()
with open(str(params_dict_path), 'wb') as f:
    pickle.dump(params_dict, f)

'''
go over all scenes
'''

frame_list_root = semantic_labels_root / 'public'
assert frame_list_root.exists(), frame_list_root

from openrooms_invalid_scenes import black_list
from dump_openrooms_func import process_one_scene, gather_missing_scenes, read_exclude_scenes_list

# for split in ['train', 'val']:
for split in ['val']:
    scene_list = get_im_info_list(frame_list_root, split)
    
    # scene_list = [('mainDiffMat_xml1', 'scene0385_01')]
    
    _, scene_list = read_exclude_scenes_list(Path(dump_root), split, scene_list)

    exclude_scene_list_file = dump_root / ('excluded_scenes_%s.txt'%split)
    
    '''
    check what is there
    '''
    scene_list = gather_missing_scenes(scene_list, dump_root)
    if len(scene_list) == 0:
        print(white_red('All scenes are processed! Exiting..')); sys.exit()
        
    _ = input(white_red("Processing these scenes?\n"))
    if _ not in ['Y', 'y']: sys.exit()
            
    '''
    render the rest
    '''
    import time
    tic = time.time()
    excluded_scenes = []
    for scene in tqdm(scene_list):
        meta_split, scene_name = scene[0], scene[1]
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
        
        process_one_scene(dump_root, exclude_scene_list_file, meta_split, scene_name, IF_DEBUG=False)
                    
    print('=== Done (%d scenes) in %.3f s'%(len(scene_list), time.time() - tic))

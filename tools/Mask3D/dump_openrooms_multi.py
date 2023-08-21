'''
Rui Zhu

python dump_openrooms_multi.py --gpu_total 8

, to dump all scenes in parallel; does not necessarily bring 8x speedup, but hopefully no worries of GPU issues with multiprocessing + mitsuba/torch

[!!!] Before start, change the file list in dump_openrooms_func.py -> gather_missing_scenes() accordingly!

Simple multi-thread testing: python test_scripts/test_cuda_multi_3.py
'''

import numpy as np
np.set_printoptions(suppress=True)
import os, sys
import torch
# import torch.multiprocessing.spawn as spawn
import torch.multiprocessing as mp
import time

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
'''
Note: [!!!] also CHANGE params in dump_openrooms_func -> process_one_scene
'''

import os, sys
sys.path.insert(0, PATH_HOME)

from lib.global_vars import PATH_HOME_dict, OR_RAW_ROOT_dict, OR_MODALITY_FRAMENAME_DICT, query_host
assert PATH_HOME == PATH_HOME_dict[host]

from pyhocon import ConfigFactory, ConfigTree
from lib.utils_misc import str2bool, check_exists, yellow

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

CONF.modality_filename_dict = query_host(OR_MODALITY_FRAMENAME_DICT, host)

CONF.im_params_dict.update({'im_H_resize': 240, 'im_W_resize': 320})
# CONF.shape_params_dict.update({'force_regenerate_tsdf': True})
CONF.shape_params_dict.update({'force_regenerate_tsdf': False})
# Mitsuba options
CONF.mi_params_dict.update({'if_mi_scene_from_xml': True}) # !!!! set to False to load from shapes (single shape or tsdf fused shape (with tsdf in modality_list))
# TSDF options
CONF.shape_params_dict.update({
    # 'if_force_fuse_tsdf': True, 
    'if_force_fuse_tsdf': False, 
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

from lib.utils_misc import run_cmd

frame_list_root = semantic_labels_root / 'public'
assert frame_list_root.exists(), frame_list_root

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--gpu_total', type=int, default=8, help='total num of gpus available')
opt = parser.parse_args()

def run_one_gpu(i, opt, split, result_queue):
    torch.cuda.is_available()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(i)
    print("pid={} count={}".format(i, torch.cuda.device_count()))
    
    # process_result_list = []
    cmd = 'python tools/Mask3D/dump_openrooms_func.py --gpu_id {} --gpu_total {} --split {} --dump_root {} --frame_list_root {}'.format(i, opt.gpu_total, split, str(dump_root), str(frame_list_root))
    _results = run_cmd(cmd)
    print(_results)
        
    # return process_result_list
    result_queue.put((i, _results))

if __name__ == '__main__':
    
    # if opt.workers_total == -1:
    #     opt.workers_total = opt.gpu_total
    
    # for split in ['train', 'val']:
    for split in ['train']:
    # for split in ['val']:
    
        tic = time.time()
        
        result_queue = mp.Queue()
        for rank in range(opt.gpu_total):
            mp.Process(target=run_one_gpu, args=(rank, opt, split, result_queue)).start()
        
        for _ in range(opt.gpu_total):
            temp_result = result_queue.get()
            print(_, temp_result)
            
        print('==== ...DONE. Took %.2f seconds'%(time.time() - tic))
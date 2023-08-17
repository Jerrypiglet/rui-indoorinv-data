'''
generate clustering-based segmentation (https://github.com/ScanNet/ScanNet/tree/master/Segmentator) for tsdf-fused meshes of openrooms_public scenes
(based on test_scripts/generate_scannet_seg.py)

run 'python dump_openrooms.py' first to generate tsdf files in 'dump_root'
'''

from pathlib import Path
from multiprocessing import Pool
import time

DATASET = 'openrooms_public'
# host = 'apple'; PATH_HOME = '/Users/jerrypiglet/Documents/Projects/rui-indoorinv-data'
# host = 'r4090'; PATH_HOME = '/home/ruizhu/Documents/Projects/rui-indoorinv-data'
host = 'mm1'; PATH_HOME = '/home/ruizhu/Documents/Projects/rui-indoorinv-data'

# dump_root = Path('/data/Mask3D_data/openrooms_public_dump')
dump_root = Path('/newdata/Mask3D_data/openrooms_public_dump_v3smaller')

import os, sys
sys.path.insert(0, PATH_HOME)
from lib.global_vars import PATH_HOME_dict, INV_NERF_ROOT_dict, MONOSDF_ROOT_dict, OR_RAW_ROOT_dict
assert PATH_HOME == PATH_HOME_dict[host]

from tqdm import tqdm
import numpy as np
np.set_printoptions(suppress=True)
import trimesh
import os
from lib.utils_misc import str2bool, check_exists, yellow, green_text, _read_json, run_cmd
from openrooms_invalid_scenes import black_list as openrooms_semantics_black_list
from lib.utils_openrooms import get_im_info_list
from dump_openrooms_func import read_exclude_scenes_list, gather_missing_scenes

semantic_labels_root = Path(PATH_HOME) / 'files_openrooms'; check_exists(semantic_labels_root)

frame_list_root = semantic_labels_root / 'public'
assert frame_list_root.exists(), frame_list_root


'''
gather scene_list from train and val splits
'''
scene_list_dict = {}
# for split in ['train']:
for split in ['train', 'val']:
    scene_list = get_im_info_list(frame_list_root, split)
    print('[%s] %d scenes'%(split, len(scene_list)))
    scene_list = [scene for scene in scene_list if (scene[0].replace('DiffMat', '').replace('DiffLight', ''), scene[1]) not in openrooms_semantics_black_list]
    print('[%s] after removing known invalid scenes from [openrooms_semantics_black_list]-> %d scenes'%(split, len(scene_list)))
    
    _, scene_list = read_exclude_scenes_list(dump_root, split, scene_list)

    # excluded_scenes_file = dump_root / ('excluded_scenes_%s.txt'%split)
    # if excluded_scenes_file.exists():
    #     with open(str(excluded_scenes_file), 'r') as f:
    #         excluded_scenes = f.read().splitlines()
    #     excluded_scenes = [(scene.split(' ')[0].replace('DiffMat', '').replace('DiffLight', ''), scene.split(' ')[1]) for scene in excluded_scenes]
    #     scene_list = [scene for scene in scene_list if (scene[0].replace('DiffMat', '').replace('DiffLight', ''), scene[1]) not in excluded_scenes]
    #     print('[%s] after removing known invalid scenes [from %s] -> %d scenes'%(split, excluded_scenes_file.name, len(scene_list)))
    
    scene_list_dict[split] = scene_list

print('Overlap of train scene_list and val scene_list:', set(scene_list_dict['train']).intersection(set(scene_list_dict['val'])))

'''
generating segmentation
'''
segmentor_path = '/home/ruizhu/Documents/Projects/ScanNet/Segmentator/segmentator' # https://github.com/ScanNet/ScanNet.git

def process_scene(scene):
    scene_idx, meta_split, scene_name, kThresh, segMinVerts, IF_DEBUG = scene[0], scene[1], scene[2], scene[3], scene[4], scene[5]
    scene_dump_root = dump_root / meta_split / scene_name
    _scene_str = '[%d]%s_%s'%(scene_idx, meta_split, scene_name)
    print(yellow('=== Processing %s'%_scene_str))

    tsdf_mesh_path = scene_dump_root / 'fused_tsdf.ply'
    tsdf_mesh = trimesh.load_mesh(str(tsdf_mesh_path), process=False)
    vertices = np.array(tsdf_mesh.vertices)
    faces = np.array(tsdf_mesh.faces)
    print('- Loaded mesh from '+str(tsdf_mesh_path), vertices.shape, faces.shape, np.amin(vertices, axis=0), np.amax(vertices, axis=0), np.amin(faces), vertices.dtype, faces.dtype)
    
    _tmp_trimesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=tsdf_mesh.visual.vertex_colors, process=False)
    _tmp_tsdf_mesh_path = scene_dump_root / '_tmp_fused_tsdf.ply'
    _tmp_trimesh.export(str(_tmp_tsdf_mesh_path))
    
    print('- Running segmentor...')
    cmd = '%s %s %.3f %d'%(str(segmentor_path), str(_tmp_tsdf_mesh_path), kThresh, segMinVerts) # default: kThresh=0.01 segMinVerts=20
    # print(cmd)
    segments_output_file = scene_dump_root / ('_tmp_fused_tsdf.%.6f.segs.json'%kThresh)
    if Path(segments_output_file).exists():
        Path(segments_output_file).unlink()
    # os.system(cmd)
    cmd_output, cmd_err, cmd_p_status = run_cmd(cmd)
    print(cmd_output, cmd_err, cmd_p_status)
    assert cmd_output != ''
    assert Path(segments_output_file).exists(), segments_output_file

    '''
    generate labels txt file
    '''
    segments = _read_json(str(segments_output_file))
    segments = np.array(segments["segIndices"])
    labels_txt_path = Path(str(segments_output_file).replace('.json', '.txt'))
    with open(str(labels_txt_path), "w") as txt_file:
        for line in segments:
            txt_file.write(str(line) + "\n") # works with any number of elements in a line
    print('- labels_txt ([%d] segments) dumped to: '%(np.unique(segments).shape[0]), labels_txt_path)
    
    '''
    debug: generate mesh file colored with segments
    ![](https://i.imgur.com/v0Hx6bL.jpg) /data/Mask3D_data/openrooms_public_dump/main_xml1/scene0552_00/tmp_tsdf_segments.ply
    '''
    if IF_DEBUG:
        out_path = scene_dump_root / 'tmp_tsdf_segments.ply'
        if out_path.exists():
            out_path.unlink()
        cmd = 'python test_scripts/scannet_utils/visualize_labels_on_mesh.py --pred_file %s --mesh_file %s --output_file %s'%(str(labels_txt_path), str(_tmp_tsdf_mesh_path), out_path)
        print(cmd)
        os.system(cmd)
        print('- Output mesh dumped to: ', green_text(out_path))

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--worker_total', type=int, default=32, help='total num of workers; must be dividable by gpu_total, i.e. workers_total/gpu_total jobs per GPU')
opt = parser.parse_args()

kThresh = 0.2
segMinVerts = 2000

        
# for split in ['train', 'val']:
# for split in ['train']:
for split in ['val']:
    '''
    check dumped data
    '''
    scene_list = scene_list_dict[split]
    
    scene_missing_list = gather_missing_scenes(scene_list, Path(dump_root))
    assert len(scene_missing_list) == 0, scene_missing_list

    '''
    generate segs
    '''
    scene_list = gather_missing_scenes(scene_list, Path(dump_root), if_check_seg=True, seg_file_name='_tmp_fused_tsdf.%.6f.segs.json'%kThresh)
    
    IF_DEBUG = split == 'val'
    
    # for scene_idx, scene in enumerate(tqdm(scene_list)):
        
    tic = time.time()
    # print('==== executing %d commands...'%len(cmd_list))
    # p = Pool(processes=opt.workers_total, initializer=init, initargs=(child_env,))
    p = Pool(processes=opt.worker_total)

    # cmd_list = [(_cmd) for _cmd in enumerate(scene_list)]
    scene_list = [(_, scene_list[_][0], scene_list[_][1], kThresh, segMinVerts, IF_DEBUG) for _ in range(len(scene_list))]
    list(tqdm(p.imap_unordered(process_scene, scene_list), total=len(scene_list)))
    p.close()
    p.join()
    print('==== ...DONE. Took %.2f seconds'%(time.time() - tic))


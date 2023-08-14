'''
generate clustering-based segmentation (https://github.com/ScanNet/ScanNet/tree/master/Segmentator) for tsdf-fused meshes of openrooms_public scenes
(based on test_scripts/generate_scannet_seg.py)

run 'python dump_openrooms.py' first to generate tsdf files in 'dump_root'
'''

from pathlib import Path

DATASET = 'openrooms_public'
# host = 'apple'; PATH_HOME = '/Users/jerrypiglet/Documents/Projects/rui-indoorinv-data'
host = 'r4090'; PATH_HOME = '/home/ruizhu/Documents/Projects/rui-indoorinv-data'
dump_root = Path('/data/Mask3D_data/openrooms_public_dump')

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
from lib.utils_openrooms import get_im_info_list, openrooms_semantics_black_list

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

    excluded_scenes_file = dump_root / ('excluded_scenes_%s.txt'%split)
    if excluded_scenes_file.exists():
        with open(str(excluded_scenes_file), 'r') as f:
            excluded_scenes = f.read().splitlines()
        excluded_scenes = [(scene.split(' ')[0].replace('DiffMat', '').replace('DiffLight', ''), scene.split(' ')[1]) for scene in excluded_scenes]
        scene_list = [scene for scene in scene_list if (scene[0].replace('DiffMat', '').replace('DiffLight', ''), scene[1]) not in excluded_scenes]
        print('[%s] after removing known invalid scenes [from %s] -> %d scenes'%(split, excluded_scenes_file.name, len(scene_list)))
    
    scene_list_dict[split] = scene_list

print('Overlap of train scene_list and val scene_list:', set(scene_list_dict['train']).intersection(set(scene_list_dict['val'])))

'''
check dumped data
'''
# for split in ['train', 'val']:
for split in ['train']:
# for split in ['val']:
    scene_list = scene_list_dict[split]
    for scene in tqdm(scene_list):
        meta_split, scene_name = scene[0], scene[1]
        dump_scene_root = dump_root / meta_split / scene_name
        assert dump_scene_root.exists(), dump_scene_root
        for file_name in ['fused_tsdf.ply', 'instance_seg.npy', 'mi_normal.npy', 'semseg.npy']:
            dump_file = dump_scene_root / file_name
            if not dump_file.exists():
                print('Missing', dump_file)
                
'''
generating segmentation
'''
segmentor_path = '/home/ruizhu/Documents/Projects/ScanNet/Segmentator/segmentator' # https://github.com/ScanNet/ScanNet.git

# for split in ['val']:
for split in ['train']:
    scene_list = scene_list_dict[split]
    # scene_list = [('mainDiffLight_xml1', 'scene0032_01')]
    
    IF_DEBUG = split == 'val'
    
    for scene_idx, scene in enumerate(tqdm(scene_list)):
        meta_split, scene_name = scene[0], scene[1]
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
        kThresh=0.2
        segMinVerts=2000
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
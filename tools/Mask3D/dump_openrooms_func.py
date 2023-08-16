'''
funcs to call in dump_openrooms.py (one for loop) or dump_openrooms_multi.py (multiprocessing)
'''

from pathlib import Path
'''
set those params according to your environment
'''
DATASET = 'openrooms_public'
# host = 'apple'; PATH_HOME = '/Users/jerrypiglet/Documents/Projects/rui-indoorinv-data'
# host = 'r4090'; PATH_HOME = '/home/ruizhu/Documents/Projects/rui-indoorinv-data'
host = 'mm1'; PATH_HOME = '/home/ruizhu/Documents/Projects/rui-indoorinv-data'
dump_root = Path('/newdata/Mask3D_data/openrooms_public_dump_v3smaller')

import os, sys
sys.path.insert(0, PATH_HOME)

from lib.global_vars import PATH_HOME_dict, INV_NERF_ROOT_dict, MONOSDF_ROOT_dict, OR_RAW_ROOT_dict
assert PATH_HOME == PATH_HOME_dict[host]

# from tqdm import tqdm
from plyfile import PlyData
import numpy as np
np.set_printoptions(suppress=True)

import copy
import pickle
from pathlib import Path
from tqdm import tqdm
from pyhocon import ConfigFactory, ConfigTree
from pyhocon.tool import HOCONConverter
from lib.utils_misc import str2bool, check_exists, yellow
from lib.utils_openrooms import get_im_info_list
from lib.class_openroomsScene3D import openroomsScene3D
from lib.class_eval_scene import evaluator_scene_scene
from test_scripts.scannet_utils.utils_visualize import visualize

# from multiprocessing import Pool
# import multiprocessing

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

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default='', help='')
parser.add_argument('--gpu_total', type=int, default='', help='')
parser.add_argument('--split', type=str, default='val', help='')
opt = parser.parse_args()

def process_one_scene(CONF, dump_root, exclude_scene_list_file, meta_split, scene_name, IF_DEBUG=False):
    scene_dump_path = dump_root / meta_split / scene_name
    scene_dump_path.mkdir(exist_ok=True, parents=True)
        
    CONF.scene_params_dict.scene_name = '%s-%s'%(meta_split, scene_name)
    CONF.im_params_dict.update({'im_H_resize': 240, 'im_W_resize': 320})
    CONF.shape_params_dict.update({'force_regenerate_tsdf': True})
    # Mitsuba options
    CONF.mi_params_dict.update({'if_mi_scene_from_xml': True}) # !!!! set to False to load from shapes (single shape or tsdf fused shape (with tsdf in modality_list))
    # TSDF options
    CONF.shape_params_dict.update({
        'if_force_fuse_tsdf': True, 
        'tsdf_file': scene_dump_path / 'fused_tsdf.ply',
        # 'tsdf_voxel_length': 8.0 / 512.0,
        # 'tsdf_sdf_trunc': 0.05,
        'tsdf_voxel_length': 12.0 / 512.0,
        'tsdf_sdf_trunc': 0.08,
        }) # !!!! set to True to force replace existing tsdf shape

    # DEBUG
    # CONF.scene_params_dict.update({'frame_id_list': [5]})

    config_json = HOCONConverter.convert(CONF, 'json')
    with open(str(scene_dump_path / 'config.json'), 'w') as f:
        f.write(config_json)

    scene_obj = openroomsScene3D(
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

    if scene_obj.IF_SKIP:
        # excluded_scenes.append((meta_split, scene_name))
        with open(str(exclude_scene_list_file), 'a') as f:
            f.write('%s %s\n'%(meta_split, scene_name))
            print('[Excluded:]', meta_split, scene_name)
        return False

    eval_return_dict = {}

    evaluator_scene = evaluator_scene_scene(
        host=host, 
        scene_object=scene_obj, 
    )

    if IF_DEBUG:
        print('=== Loading mesh file for inspection...' + str(str(scene_obj.tsdf_file_path)))
        with open(str(scene_obj.tsdf_file_path), 'rb') as f:
            plydata = PlyData.read(f)
            num_verts = plydata['vertex'].count
            assert 'red' in plydata['vertex']
        
    '''
    sample different modalities
    '''
    visibility_list_list = []
    for sample_type in ['mi_normal', 'instance_seg']:
    # for sample_type in ['rgb_sdr']:
        return_dict = evaluator_scene.sample_shapes(
            sample_type=sample_type, # e.g. ['vis_count', 't', 'rgb_hdr', 'rgb_sdr', 'face_normal', 'mi_normal', 'semseg', 'instance_seg']
            # sample_type='vis_count', # ['']
            # sample_type='t', # ['']
            shape_params={
            }, 
            visibility_list_list_input = visibility_list_list, 
        )
        
        visibility_list_list = return_dict['visibility_list_list']
        if len(visibility_list_list) > 0:
            print('+++++', sample_type, '++', len(visibility_list_list[0]))
            with open(str(dump_root / meta_split / scene_name / 'visibility_list_list.pkl'), 'wb') as f:
                pickle.dump(visibility_list_list, f)
            print('=== visibility_list_list dumped to:', str(dump_root / meta_split / scene_name / 'visibility_list_list.pkl'))
        else:
            print('+++++', sample_type, '++', 0)
        
        # debug and vis

        # if sample_type == 'rgb_sdr':
        #     rgb_sdr = np.array(return_dict['rgb_sdr'])
        #     assert num_verts == len(rgb_sdr)
        #     rgb_sdr = rgb_sdr.astype(np.float32)
        #     np.save(str(dump_root / meta_split / scene_name / 'rgb_sdr.npy'), rgb_sdr)
            
        if sample_type == 'mi_normal':
            mi_normal = np.array(return_dict['mi_normal'])
            if IF_DEBUG:
                assert num_verts == len(mi_normal)
            mi_normal = mi_normal.astype(np.float32)
            np.save(str(dump_root / meta_split / scene_name / 'mi_normal.npy'), mi_normal)
            
            vertex_view_count = np.array(return_dict['vertex_view_count'])
            if IF_DEBUG:
                assert num_verts == len(vertex_view_count)
            assert np.amax(vertex_view_count) <= 255
            vertex_view_count = vertex_view_count.astype(np.uint8)
            np.save(str(dump_root / meta_split / scene_name / 'vertex_view_count.npy'), vertex_view_count)
            
            if IF_DEBUG:
                '''
                visualize mitsuba normal
                '''
                plydata_copy = copy.deepcopy(plydata)
                colors_normal = np.clip((mi_normal + 1.0) / 2.0, 0., 1.)
                colors_normal = (colors_normal * 255).astype(np.uint8)
                assert num_verts == len(colors_normal)
                for i in range(num_verts):
                    color = colors_normal[i]
                    plydata_copy['vertex']['red'][i] = color[0]
                    plydata_copy['vertex']['green'][i] = color[1]
                    plydata_copy['vertex']['blue'][i] = color[2]
                output_file = dump_root / meta_split / scene_name / 'tmp_tsdf_mi_normal.ply'
                plydata_copy.write(str(output_file))
                
        if sample_type == 'instance_seg':
            
            segments_instance_seg = np.array(return_dict['seg_labels'])
            if IF_DEBUG:
                assert num_verts == len(segments_instance_seg)
            assert np.amax(segments_instance_seg) <= 255
            segments_instance_seg = segments_instance_seg.astype(np.uint8)
            np.save(str(dump_root / meta_split / scene_name / 'instance_seg.npy'), segments_instance_seg)
            
            if IF_DEBUG:
                '''
                visualize instance_seg labels
                '''
                labels_txt_path = dump_root / meta_split / scene_name / 'tmp_instance_seg.txt'
                with open(str(labels_txt_path), "w") as txt_file:
                    for line in segments_instance_seg:
                        txt_file.write(str(line) + "\n") # works with any number of elements in a line
                print('=== labels_txt ([%d] segments_instance_seg) dumped to: '%(np.unique(segments_instance_seg).shape[0]), labels_txt_path)
            
                output_file = dump_root / meta_split / scene_name / 'tmp_tsdf_instance_seg.ply'
                visualize(str(labels_txt_path), str(scene_obj.tsdf_file_path), str(output_file))
                
            '''
            get semseg labels
            '''
            
            segments_semseg = np.empty(segments_instance_seg.shape[0], dtype=segments_instance_seg.dtype)
            segments_semseg.fill(255)
            instance_seg_to_semseg_mapping_dict = {}
            for instance_seg_id in np.unique(segments_instance_seg):
                instance_seg_to_semseg_mapping_dict[instance_seg_id] = []
                for instance_seg_info_frame in scene_obj.instance_seg_info_list:
                    for instance_seg_info in instance_seg_info_frame:
                        if instance_seg_info['id'] == instance_seg_id:
                            instance_seg_to_semseg_mapping_dict[instance_seg_id].append(instance_seg_info['category_id'])
            
                if instance_seg_id != 255:
                    if not len(set(instance_seg_to_semseg_mapping_dict[instance_seg_id])) == 1:
                        print(instance_seg_id, set(instance_seg_to_semseg_mapping_dict[instance_seg_id]))
                        # import ipdb; ipdb.set_trace()
                        # excluded_scenes.append((meta_split, scene_name))
                        with open(str(exclude_scene_list_file), 'a') as f:
                            f.write('%s %s\n'%(meta_split, scene_name))
                            print('[Excluded:]', meta_split, scene_name)
                        return False
                else:
                    instance_seg_to_semseg_mapping_dict[instance_seg_id] = []
                    
                segments_semseg[segments_instance_seg==instance_seg_id] = instance_seg_to_semseg_mapping_dict[instance_seg_id][0] if len(instance_seg_to_semseg_mapping_dict[instance_seg_id]) > 0 else 255
                    
            assert np.amax(segments_semseg) <= 255
            segments_semseg = segments_semseg.astype(np.uint8)
            np.save(str(dump_root / meta_split / scene_name / 'semseg.npy'), segments_semseg)
            
            if IF_DEBUG:
                '''
                visualize semseg labels
                '''
                labels_txt_path = dump_root / meta_split / scene_name / 'tmp_semseg.txt'
                with open(str(labels_txt_path), "w") as txt_file:
                    for line in segments_semseg:
                        txt_file.write(str(line) + "\n") # works with any number of elements in a line
                output_file = dump_root / meta_split / scene_name / 'tmp_tsdf_semseg.ply'
                semseg_color = {k: np.array(v) for k, v in scene_obj.OR_mapping_id45_to_color_dict.items()}
                visualize(str(labels_txt_path), str(scene_obj.tsdf_file_path), str(output_file), colors=semseg_color)
                
    return True

def process_one_gpu(split, gpu_id, gpu_total):
    # IF_DEBUG = np.random.random() < 0.05
    # IF_DEBUG = np.random.random() <= 1.0
    IF_DEBUG = split == 'val'
    # IF_DEBUG = True
    
    excluded_scenes = []
    exclude_scene_list_file = dump_root / ('excluded_scenes_%s_gpu %d.txt'%(split, gpu_id))
    # if exclude_scene_list_file.exists():
    #     exclude_scene_list_file.unlink()
    scene_list = get_im_info_list(frame_list_root, split)
    scene_list = [scene_list[_] for _ in range(len(scene_list)) if _ % gpu_total == gpu_id]
    
    if (meta_split.replace('DiffMat', '').replace('DiffLight', ''), scene_name) in black_list:
        excluded_scenes.append((meta_split, scene_name))
        with open(str(exclude_scene_list_file), 'a') as f:
            f.write('%s %s\n'%(meta_split, scene_name))
            print('[Excluded:]', meta_split, scene_name)
            
    scene_list = [_ for _ in scene_list if _ not in excluded_scenes]
        
    for meta_split, scene_name in tqdm(scene_list):
        print('++++', gpu_id, len(scene_list), meta_split, scene_name, '++++')
        result = process_one_scene(CONF, dump_root, exclude_scene_list_file, meta_split, scene_name, IF_DEBUG=IF_DEBUG)
        
if __name__ == '__main__':
# print("scene={}, torch.cuda.is_available()={}, torch.cuda.device_count={}".format(opt.scene, torch.cuda.is_available(), torch.cuda.device_count()))

    process_one_gpu(opt.split, opt.gpu_id, opt.gpu_total)
    

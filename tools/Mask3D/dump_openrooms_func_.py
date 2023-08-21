'''
funcs to call in dump_openrooms.py (one for loop) or dump_openrooms_multi.py (multiprocessing)
'''

from plyfile import PlyData
import numpy as np
np.set_printoptions(suppress=True)


import sys
from pathlib import Path
# directory reach
directory = Path(__file__).resolve()
# setting path
sys.path.append(str(directory.parent.parent.parent))

import copy
import pickle
from tqdm import tqdm
import pickle
from pyhocon.tool import HOCONConverter
from lib.utils_openrooms import get_im_info_list
from lib.utils_misc import yellow_text, yellow, white_red
from lib.class_openroomsScene3D import openroomsScene3D
from lib.class_eval_scene import evaluator_scene_scene
from test_scripts.scannet_utils.utils_visualize import visualize

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=0, help='', required=False)
parser.add_argument('--gpu_total', type=int, default=1, help='', required=False)
parser.add_argument('--split', type=str, default='val', help='', required=False)
parser.add_argument('--dump_root', type=str, default='', help='', required=False)
parser.add_argument('--frame_list_root', type=str, default='', help='', required=False)
opt = parser.parse_args()

def gather_missing_scenes(scene_list, dump_root, if_check_seg=False, seg_file_name=''):
    file_list = [
        'fused_tsdf.ply', 
        'mi_normal.npy', 
        'semseg.npy', 
        'vertex_view_count.npy', 
        'instance_seg.npy', 
        'visibility_list_list.pkl', 
    ]
    if if_check_seg:
        assert seg_file_name != ''
        file_list = [seg_file_name]

    print(yellow_text('Checking for dumped scenes...'))
    file_missing_list = []
    scene_missing_list = []
    for (meta_split, scene_name) in tqdm(scene_list):
        scene_path = dump_root / meta_split / scene_name
        flag_missing = False
        scene_missing_files = []
        for file in file_list:
            if not (scene_path / file).exists():
                flag_missing = True
                file_missing_list.append((meta_split, scene_name, file))
                scene_missing_files.append(file)
        if flag_missing:
            scene_missing_list.append((meta_split, scene_name, len(scene_missing_files), scene_missing_files))
    print('Checked file_list', file_list)
    print(yellow_text('=== Missing scenes (%d/%d):'%(len(scene_missing_list), len(scene_list))), scene_missing_list[:5])
    
    return scene_missing_list

def read_exclude_scenes_list(dump_root: Path, split: str, scene_list: list=[]):
    exclude_scene_list_files = list(dump_root.glob('excluded_scenes_%s*.txt'%split))
    exclude_scene_list = []
    for exclude_scene_list_file_ in exclude_scene_list_files:
        with open(str(exclude_scene_list_file_), 'r') as f:
            lines = f.readlines()
        exclude_scene_list += [tuple(line.strip().split()) for line in lines]
    print(yellow('Excluded scenes:'), len(exclude_scene_list), exclude_scene_list[:5])
    
    if scene_list != []:
        scene_list = [scene for scene in scene_list if scene not in exclude_scene_list]        
        print(yellow('The rest of the scenes:'), len(scene_list))
        
    return exclude_scene_list, scene_list


def process_one_scene(dump_root, exclude_scene_list_file, meta_split, scene_name, IF_DEBUG=False):
    scene_dump_path = dump_root / meta_split / scene_name
    scene_dump_path.mkdir(exist_ok=True, parents=True)

    # DEBUG
    # CONF.scene_params_dict.update({'frame_id_list': [5]})

    with open(str(dump_root / 'params_dict.pkl'), 'rb') as f:
        params_dict_ = pickle.load(f)
        
    '''
    update scene-specific params
    '''        
    params_dict_['CONF'].scene_params_dict.scene_name = '%s-%s'%(meta_split, scene_name)
    params_dict_['CONF'].shape_params_dict.update({
        'tsdf_file': scene_dump_path / 'fused_tsdf.ply',
        })
    config_json = HOCONConverter.convert(params_dict_['CONF'], 'json')
    with open(str(scene_dump_path / 'config.json'), 'w') as f:
        f.write(config_json)
        
    scene_obj = openroomsScene3D(
        **params_dict_,
    )
    
    if scene_obj.IF_SKIP:
        # excluded_scenes.append((meta_split, scene_name))
        with open(str(exclude_scene_list_file), 'a') as f:
            f.write('%s %s\n'%(meta_split, scene_name))
            print('[Excluded:]', meta_split, scene_name)
        return False

    evaluator_scene = evaluator_scene_scene(
        # host=host, 
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

def process_one_gpu(opt, IF_FORCE=False):
    
    # IF_DEBUG = np.random.random() < 0.05
    # IF_DEBUG = np.random.random() <= 1.0
    IF_DEBUG = opt.split == 'val'
    # IF_DEBUG = True
    
    scene_list = get_im_info_list(Path(opt.frame_list_root), opt.split)
    scene_list = [scene_list[_] for _ in range(len(scene_list)) if _ % opt.gpu_total == opt.gpu_id]
    
    _, scene_list = read_exclude_scenes_list(Path(opt.dump_root), opt.split, scene_list)
    
    exclude_scene_list_file = Path(opt.dump_root) / ('excluded_scenes_%s_gpu %d.txt'%(opt.split, opt.gpu_id))
    
    '''
    check what is there
    '''
    scene_list = gather_missing_scenes(scene_list, Path(opt.dump_root))
    if len(scene_list) == 0:
        print(white_red('All scenes are processed! Exiting..')); sys.exit()
    
    # return
    # scene_list = [('mainDiffMat_xml1', 'scene0608_01')]
            
    for scene in tqdm(scene_list):
        meta_split, scene_name = scene[0], scene[1]
        print('++++', opt.gpu_id, len(scene_list), meta_split, scene_name, '++++')
        result = process_one_scene(Path(opt.dump_root), exclude_scene_list_file, meta_split, scene_name, IF_DEBUG=IF_DEBUG)
        
if __name__ == '__main__':
# print("scene={}, torch.cuda.is_available()={}, torch.cuda.device_count={}".format(opt.scene, torch.cuda.is_available(), torch.cuda.device_count()))

    process_one_gpu(opt)
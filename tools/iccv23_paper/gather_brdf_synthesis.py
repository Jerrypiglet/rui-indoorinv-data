from pathlib import Path
import os
import os
import cv2
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import numpy as np

# export_path = Path('/Users/jerrypiglet/Library/CloudStorage/OneDrive-Personal/[Research]/Projects/FIPT/images/brdf_synthesis')
export_path = Path('/Users/jerrypiglet/Documents/Projects/FIPT/images/brdf_synthesis')
assert Path(export_path).parent.exists()
Path(export_path).mkdir(parents=True, exist_ok=True)
print('Exporting to', export_path)

data_root_path = Path('data/indoor_synthetic')
assert Path(data_root_path).exists()

SPLIT = 'train'
HW = (320, 640)

scene_frame_list = [('kitchen', 6), ('bathroom', 17), ('bedroom', 17)]

def gamma2(x):
    return np.clip(x ** (1./2.2), 0., 1.)

def clamp(x):
    return np.clip(x, 0., 1.)

def resize(x):
    if x.shape[:2] == HW:
        x = cv2.resize(x, (HW[1], HW[0]), interpolation=cv2.INTER_AREA)
    return x

NA_file = Path(str(export_path.parent / 'NA.png'))
where_file = Path(str(export_path.parent / 'where.png'))
wait_file = Path(str(export_path.parent / 'wait.png'))

filename_dict = {
    'GT-im': data_root_path / '#SCENE_NAME' / SPLIT / 'Image' / '%03d_0001.exr', 
    'GT-albedo': data_root_path / '#SCENE_NAME' / SPLIT / 'albedo' / '%03d.exr', 
    'GT-roughness': data_root_path / '#SCENE_NAME' / SPLIT / 'roughness' / '%03d_0001.exr', 
    'GT-emission': data_root_path / '#SCENE_NAME' / SPLIT / 'Emit' / '%03d_0001.exr', 
    'li22-albedo': data_root_path / 'RESULTS/brdf/lieccv22' / '#SCENE_NAME' / '%03d_kd.png', 
    'li22-roughness': data_root_path / 'RESULTS/brdf/lieccv22' / '#SCENE_NAME' / '%03d_roughness.png', 
    'li22-emission': data_root_path / 'RESULTS/brdf/lieccv22' / '#SCENE_NAME' / '%03d_emission.exr', 
    'neilf-albedo': data_root_path / 'RESULTS/brdf/neilf' / '#SCENE_NAME' / '%03d_kd.png', 
    'neilf-roughness': data_root_path / 'RESULTS/brdf/neilf' / '#SCENE_NAME' / '%03d_roughness.png', 
    'neilf-emission': NA_file, 
    'milo-albedo': wait_file, 
    'milo-roughness': wait_file, 
    'milo-emission': wait_file, 
    'ours-albedo': data_root_path / 'RESULTS/brdf/ours' / '#SCENE_NAME' / '%03d_albedo.png', 
    'ours-roughness': data_root_path / 'RESULTS/brdf/ours' / '#SCENE_NAME' / '%03d_roughness.png', 
    'ours-emission': data_root_path / 'RESULTS/brdf/ours' / '#SCENE_NAME' / '%03d_emission.exr', 
}

for scene_name, frame_id in scene_frame_list:
    for method in ['GT', 'ours', 'milo', 'li22', 'neilf']:
        for modality in ['im', 'albedo', 'roughness', 'emission']:
            key = '%s-%s'%(method, modality)
            if key not in filename_dict:
                continue

            if filename_dict[key] in [wait_file, where_file, NA_file]:
                im_ori = filename_dict[key]
            else:
                im_ori = Path(str(filename_dict[key]).replace('#SCENE_NAME', scene_name) % frame_id)
            
            assert im_ori.exists(), 'image file not found: %s'%str(im_ori)
            im_ori = cv2.imread(str(im_ori), cv2.IMREAD_UNCHANGED)
            if im_ori.dtype == np.uint8:
                im_ori = im_ori.astype(np.float32) / 255.
            if modality in ['im', 'emission'] and not filename_dict[key] in [wait_file, where_file, NA_file]:
                im_ori = resize(gamma2(im_ori))
            else:
                im_ori = resize(clamp(im_ori))
            im_ori_target = export_path / ('%s-%s_%s.png'%(method, scene_name, modality))
            im_ori_target.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(im_ori_target), (im_ori * 255).astype(np.uint8))
            print('Saving', str(im_ori_target))
from pathlib import Path
import os
import os
import cv2
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import numpy as np

# export_path = Path('/Users/jerrypiglet/Library/CloudStorage/OneDrive-Personal/[Research]/Projects/FIPT/images/brdf_synthesis')
export_path = Path('/Users/jerrypiglet/Documents/Projects/FIPT/images/render_synthetic')
assert Path(export_path).parent.exists()
Path(export_path).mkdir(parents=True, exist_ok=True)
print('Exporting to', export_path)

data_root_path = Path('data/indoor_synthetic')
assert Path(data_root_path).exists()

SPLIT = 'val'
HW = (320, 640)

scene_frame_list = [('kitchen', (12, 8)), ('bathroom', (11, 2)), ('bedroom', (3, 10))]

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
    'GT-synthesis': data_root_path / '#SCENE_NAME' / SPLIT / 'Image' / '%03d_0001.exr', 
    'GT-relight': data_root_path / ('#SCENE_NAME'+'-relight') / SPLIT / 'Image' / '%03d_0001.exr',
     
    'li22-synthesis': NA_file, 
    'li22-relight': data_root_path / 'RESULTS/relight/lieccv22' / '#SCENE_NAME' / '%03d.exr', 
    
    'fvp-synthesis': data_root_path / 'RESULTS/viewsynthesis/fvp' / '#SCENE_NAME' / '%03d.exr', 
    'fvp-relight': data_root_path / 'RESULTS/relight/fvp' / '#SCENE_NAME' / '%03d.exr', 
    
    'milo-synthesis': wait_file, 
    'milo-relight': wait_file, 
    
    'ours-synthesis': data_root_path / 'RESULTS/viewsynthesis/ours' / '#SCENE_NAME' / '%03d.exr', 
    'ours-relight': data_root_path / 'RESULTS/relight/ours' / '#SCENE_NAME' / '%03d.exr', 
}

for method in ['GT', 'ours', 'milo', 'li22', 'fvp']:
    for modality in ['synthesis', 'relight']:
        for scene_name, frame_ids in scene_frame_list:
            frame_id = frame_ids[0]
            frame_id_inset = frame_ids[1] if modality == 'relight' else None
            key = '%s-%s'%(method, modality)
            if key not in filename_dict:
                continue
            
            im_inset_path = None
            if filename_dict[key] in [wait_file, where_file, NA_file]:
                im_path = filename_dict[key]
            else:
                im_path = Path(str(filename_dict[key]).replace('#SCENE_NAME', scene_name) % frame_id)
                if frame_id_inset is not None:
                    im_inset_path = Path(str(filename_dict[key]).replace('#SCENE_NAME', scene_name) % frame_id_inset)
                print(im_path, im_inset_path)
            if scene_name == 'kitchen':
                im_inset_path = Path('/Volumes/RuiT7/ICCV23/indoor_synthetic/RESULTS/kitchen_lightsource.png')
            if scene_name == 'bathroom':
                im_inset_path = Path('/Volumes/RuiT7/ICCV23/indoor_synthetic/RESULTS/bathroom_lightsource.png')
            
            # print('----', im_path)
            assert im_path.exists(), 'image file not found: %s'%str(im_path)
            im = cv2.imread(str(im_path), cv2.IMREAD_UNCHANGED)[:, :, :3]
            if im.dtype == np.uint8:
                im = im.astype(np.float32) / 255.
            if modality in ['synthesis', 'relight'] and not filename_dict[key] in [wait_file, where_file, NA_file]:
                im = resize(gamma2(im))
            else:
                im = resize(clamp(im))
                
            if im_inset_path is not None and method in ['GT']:
                # print(scene_name, method, modality)
                assert im_inset_path.exists(), 'image file not found: %s'%str(im_inset_path)
                # print('=====', im_inset_path)
                im_inset = cv2.imread(str(im_inset_path), cv2.IMREAD_UNCHANGED)[:, :, :3]
                if im_inset.dtype == np.uint8:
                    im_inset = im_inset.astype(np.float32) / 255.
                if modality in ['synthesis', 'relight'] and not filename_dict[key] in [wait_file, where_file, NA_file] and im_inset_path.suffix == '.exr':
                    im_inset = resize(gamma2(im_inset))
                else:
                    im_inset = resize(clamp(im_inset))
                _H, _W = im.shape[0]//3, im.shape[1]//3
                H, W = im.shape[0], im.shape[1]
                im_inset = cv2.resize(im_inset, (_W, _H), interpolation=cv2.INTER_AREA)
                im[:_H, (W-_W):, :] = im_inset
                im[_H:_H+2, (W-_W):, :] = 1.
                im[0:_H, (W-_W):(W-_W+2), :] = 1.
                im_inset = None
                
            im_target = export_path / ('%s-%s_%s.png'%(method, scene_name, modality))
            im_target.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(im_target), (im * 255).astype(np.uint8))
            # print('Saving', str(im_target))
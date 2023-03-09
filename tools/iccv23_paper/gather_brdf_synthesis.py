from pathlib import Path
import os
import os
import cv2
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import numpy as np
import torch.nn.functional as NF
import torch
import matplotlib.pyplot as plt

# export_path = Path('/Users/jerrypiglet/Library/CloudStorage/OneDrive-Personal/[Research]/Projects/FIPT/images/brdf_synthesis')
export_path = Path('/Users/jerrypiglet/Documents/Projects/FIPT/images/brdf_synthesis')
assert Path(export_path).parent.exists()
Path(export_path).mkdir(parents=True, exist_ok=True)
print('Exporting to', export_path)

data_root_path = Path('data/indoor_synthetic')
assert Path(data_root_path).exists()

SPLIT = 'train'
HW = (320, 640)
# HW_crop = (320, 551)

scene_frame_list = [('kitchen', 192), ('bathroom', 17), ('bedroom', 17)]

def gamma2(x):
    return np.clip(x ** (1./2.2), 0., 1.)

def clamp(x):
    return np.clip(x, 0., 1.)

def resize(x):
    if x.shape[:2] != HW:
        x = cv2.resize(x, (HW[1], HW[0]), interpolation=cv2.INTER_AREA)
    # if x.shape[:2] != HW_crop:
    #     x = x[(HW[1]-HW_crop[1]):, (HW[0]-HW_crop[0]):]
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
    'milo-albedo': data_root_path / 'RESULTS/brdf/milo' / '#SCENE_NAME' / '%03d_albedo.png', 
    'milo-roughness': data_root_path / 'RESULTS/brdf/milo' / '#SCENE_NAME' / '%03d_roughness.png', 
    'milo-emission': data_root_path / 'RESULTS/brdf/milo' / '#SCENE_NAME' / '%03d_emission.exr', 
    'ipt-albedo': data_root_path / 'RESULTS/brdf/ipt' / '#SCENE_NAME' / '%03d_albedo.png', 
    'ipt-roughness': data_root_path / 'RESULTS/brdf/ipt' / '#SCENE_NAME' / '%03d_roughness.png', 
    'ipt-emission': data_root_path / 'RESULTS/brdf/ipt' / '#SCENE_NAME' / '%03d_emission.exr', 
    'ours-albedo': data_root_path / 'RESULTS/brdf/ours' / '#SCENE_NAME' / '%03d_albedo.png', 
    'ours-roughness': data_root_path / 'RESULTS/brdf/ours' / '#SCENE_NAME' / '%03d_roughness.png', 
    'ours-emission': data_root_path / 'RESULTS/brdf/ours' / '#SCENE_NAME' / '%03d_emission.exr', 
}

for scene_name, frame_id in scene_frame_list:
    '''
    get emission errors for all methods, for this frame
    then normalize over all methods
    '''
    emission_GT = None
    emission_error_dict = {}
    for method in ['GT', 'ours', 'ipt', 'milo', 'li22']:
        key = '%s-%s'%(method, 'emission')
        if key not in filename_dict:
            continue

        if filename_dict[key] in [wait_file, where_file, NA_file]:
            im_ori = filename_dict[key]
        else:
            im_ori = Path(str(filename_dict[key]).replace('#SCENE_NAME', scene_name) % frame_id)
        assert im_ori.exists(), 'image file not found: %s'%str(im_ori)
        print('--', scene_name, method, str(im_ori))
        im_ori = cv2.imread(str(im_ori), cv2.IMREAD_UNCHANGED)
        if method == 'GT':
            emission_GT = im_ori.copy()
        if method != 'GT':
            # compute emission error
            grad = NF.conv2d(torch.from_numpy(emission_GT.sum(-1)[None,None]>0).float(), weight=torch.ones(1,1,3,3),padding=1)
            grad = ((grad!=9)&(grad!=0)).float()[0,0]
            grad[0] = 0
            grad[-1] = 0
            grad[:,0] = 0
            grad[:,-1] = 0
            grad_mask = grad.cpu().numpy() != 0
            log_error = np.log(((im_ori-emission_GT)**2).sum(-1)+1)
            emission_error_dict[method] = {'error': log_error, 'mask': grad_mask}

    emission_error_all = np.array([emission_error_dict[method]['error'] for method in ['ours', 'ours', 'ipt', 'milo', 'li22']])
    emission_error_all_max = np.amax(emission_error_all) / 2.

    # for modality in ['emission']:
    for modality in ['im', 'albedo', 'roughness', 'emission']:
        label_GT = None
        for method in ['GT', 'ours', 'milo', 'li22', 'neilf', 'ipt']:
            emission_error_vis = None
            key = '%s-%s'%(method, modality)
            if key not in filename_dict:
                continue

            if filename_dict[key] in [wait_file, where_file, NA_file]:
                im_ori = filename_dict[key]
            else:
                im_ori = Path(str(filename_dict[key]).replace('#SCENE_NAME', scene_name) % frame_id)
            
            assert im_ori.exists(), 'image file not found: %s'%str(im_ori)
            im_ori = cv2.imread(str(im_ori), cv2.IMREAD_UNCHANGED)
            if method == 'GT':
                label_GT = im_ori.copy()
            if method not in ['GT', 'neilf'] and modality in ['emission']:
                assert label_GT is not None
                # compute emission error
                cm = plt.get_cmap('jet') # the larger the hotter
                emission_error = np.clip(emission_error_dict[method]['error'] / emission_error_all_max, 0., 1.)
                emission_error_vis = (cm(emission_error)[:, :, :3] * 255).astype(np.uint8)
                emission_mask = emission_error_dict[method]['mask']
                emission_error_vis[emission_mask] = np.array([255, 255, 255]).reshape((1, 1, 3))

                # plt.subplot(221); plt.imshow(resize(gamma2(label_GT)))
                # plt.subplot(222); plt.imshow(resize(gamma2(im_ori)))
                # plt.subplot(223); plt.imshow(emission_error_vis)
                # plt.subplot(224); plt.imshow(emission_mask)
                # plt.show()
                
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
            
            if emission_error_vis is not None:
                im_ori_target = export_path / ('%s-%s_%s.png'%(method, scene_name, 'emission_vis'))
                im_ori_target.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(im_ori_target), emission_error_vis[:, :, [2, 1, 0]])
                # if scene_name == 'kitchen':
                #     cv2.imwrite(str(im_ori_target), emission_error_vis[:, 89:, [2, 1, 0]])
                # else:
                #     cv2.imwrite(str(im_ori_target), emission_error_vis[:, :551, [2, 1, 0]])
                print('Saving', str(im_ori_target))

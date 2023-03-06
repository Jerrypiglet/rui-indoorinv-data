
from pathlib import Path
from shutil import rmtree
import cv2
import sys
import os
import cv2
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import numpy as np
ROOT_PATH = Path('/Users/jerrypiglet/Documents/Projects/OpenRooms_RAW_loader')
sys.path.insert(0, str(ROOT_PATH))

from lib.utils_io import center_crop

# scene_name = 'indoor_synthetic/bedroom'; appendix = '_est_fullres'

# scene_name = 'indoor_synthetic/bedroom'; appendix = '_gt_fullres'
# split_fvp = 'trainval'; frame_id_list_dict = {'train': list(range(208)), 'val': np.arange(208, 208+14)}

# scene_name = 'indoor_synthetic/kitchen_new'; appendix = '_est_fullres'

# scene_name = 'indoor_synthetic/kitchen_new'; appendix = '_gt_fullres'
# split_fvp = 'trainval'; frame_id_list_dict = {'train': list(range(212)), 'val': np.arange(212, 212+15)}

# scene_name = 'indoor_synthetic/bathroom'; appendix = '_gt_fullres'
# split_fvp = 'trainval'; frame_id_list_dict = {'train': list(range(109)), 'val': np.arange(109, 109+13)}

scene_name = 'indoor_synthetic/livingroom'; appendix = '_est_fullres'
split_fvp = 'trainval'; frame_id_list_dict = {'train': list(range(213)), 'val': np.arange(213, 213+14)}

# ----------

test_root_path = ROOT_PATH / 'data' / scene_name.split('/')[0] / 'RESULTS_fvp' / (scene_name.split('/')[1]+appendix) / split_fvp
assert Path(test_root_path).exists(), str(test_root_path)

TARGET_PATH = ROOT_PATH / 'data' / scene_name.split('/')[0] / 'RESULTS' / '$TASK/fvp' / (scene_name.split('/')[1].replace('_new', '')+appendix.replace('_fullres', '').replace('_gt', ''))
expected_shape = (160, 320)
albedo_filename_str = '16bitData/%d_albedo.png'
relight_filename_str = 'out_EXR/%08d_rdr.exr'
synthesis_filename_str = 'synthesis_out_EXR/%08d_rdr.exr'

IF_ALIGH = True # if align with ours
OFFSET = len(frame_id_list_dict['train'])

# for split in ['train', 'val']:
for split in ['val']:
   for frame_idx, frame_id in enumerate(frame_id_list_dict[split]):
      
      '''
      albedo
      '''
      albedo_vis_path = test_root_path / (albedo_filename_str % frame_id)
      assert albedo_vis_path.exists(), str(albedo_vis_path)
      assert albedo_vis_path.exists(), str(albedo_vis_path)
      albedo = cv2.imread(str(albedo_vis_path), cv2.IMREAD_UNCHANGED)
      
      albedo_target_path = Path(str(TARGET_PATH).replace('$TASK', 'brdf')) / ('%03d_albedo.png'%frame_idx)
      if frame_idx == 0 and albedo_target_path.parent.exists():
         rmtree(str(albedo_target_path.parent), ignore_errors=True)
      albedo_target_path.parent.mkdir(parents=True, exist_ok=True)
      cv2.imwrite(str(albedo_target_path), albedo)
      print('-- albedo results saved to %s'%albedo_target_path)
                  
      
      '''
      relight
      '''
      relight_vis_path = test_root_path / (relight_filename_str % frame_id)
      assert relight_vis_path.exists(), str(relight_vis_path)
      fvp_relight = cv2.imread(str(relight_vis_path), cv2.IMREAD_UNCHANGED)
      
      relight_target_path = Path(str(TARGET_PATH).replace('$TASK', 'relight')) / ('%03d_ori.exr'%frame_idx)
      if frame_idx == 0 and relight_target_path.parent.exists():
         rmtree(str(relight_target_path.parent), ignore_errors=True)
      relight_target_path.parent.mkdir(parents=True, exist_ok=True)
      cv2.imwrite(str(relight_target_path), fvp_relight)
      print('-- relight results saved to %s'%relight_target_path)

      ours_relight_path = ROOT_PATH / 'data' / (scene_name.replace('_new', '')+'-relight') / split / ('Image/%03d_0001.exr'%frame_idx)
      assert ours_relight_path.exists(), str(ours_relight_path)
      ours_relight = cv2.imread(str(ours_relight_path), cv2.IMREAD_UNCHANGED)
      relight_ours_ref_target_path = Path(str(TARGET_PATH).replace('$TASK', 'relight')) / ('%03d_ours_ref.exr'%frame_idx)
      cv2.imwrite(str(relight_ours_ref_target_path), ours_relight)
      print('------- relight reference results saved to %s'%relight_ours_ref_target_path)
      
      if IF_ALIGH:
         gt_render_path = ROOT_PATH / 'data' / scene_name.split('/')[0] / (scene_name.split('/')[1].replace('_new', '')+'-relight') / split / ('Image/%03d_0001.exr'%(frame_id-OFFSET))
         print(frame_id, frame_idx, OFFSET)
         assert gt_render_path.exists(), str(gt_render_path)
         gt_render = cv2.imread(str(gt_render_path), cv2.IMREAD_UNCHANGED)
         gt_render = cv2.resize(gt_render, (expected_shape[1], expected_shape[0]), interpolation=cv2.INTER_AREA)

         gt_render_ = gt_render.flatten()
         sort_index = np.argsort(gt_render_)
         sort_index = sort_index[:int(gt_render_.shape[0]*0.95)]
         gt_render_enery = np.sum(gt_render.flatten()[sort_index])
         fvp_exr_enery = np.sum(fvp_relight.flatten()[sort_index])
         fvp_relight = fvp_relight * gt_render_enery / fvp_exr_enery
         
         cv2.imwrite(str(relight_target_path).replace('_ori.exr', '.exr'), fvp_relight)
         print('-- relight results saved to %s'%relight_target_path)
          
      '''
      synthesis
      '''
      synthesis_vis_path = test_root_path / (synthesis_filename_str % frame_id)
      assert synthesis_vis_path.exists(), str(synthesis_vis_path)
      fvp_synthesis = cv2.imread(str(synthesis_vis_path), cv2.IMREAD_UNCHANGED)
      
      synthesis_target_path = Path(str(TARGET_PATH).replace('$TASK', 'viewsynthesis')) / ('%03d_ori.exr'%frame_idx)
      if frame_idx == 0 and synthesis_target_path.parent.exists():
         rmtree(str(synthesis_target_path.parent), ignore_errors=True)
      synthesis_target_path.parent.mkdir(parents=True, exist_ok=True)
      cv2.imwrite(str(synthesis_target_path), fvp_synthesis)
      print('-- synthesis results saved to %s'%synthesis_target_path)

      ours_synthesis_path = ROOT_PATH / 'data' / scene_name.replace('_new', '') / split / ('Image/%03d_0001.exr'%frame_idx)
      assert ours_synthesis_path.exists(), str(ours_synthesis_path)
      ours_synthesis = cv2.imread(str(ours_synthesis_path), cv2.IMREAD_UNCHANGED)
      synthesis_ours_ref_target_path = Path(str(TARGET_PATH).replace('$TASK', 'viewsynthesis')) / ('%03d_ours_ref.exr'%frame_idx)
      cv2.imwrite(str(synthesis_ours_ref_target_path), ours_synthesis)
      print('------- synthesis reference results saved to %s'%synthesis_ours_ref_target_path)
      
      if IF_ALIGH:
         gt_render_path = ROOT_PATH / 'data' / scene_name.split('/')[0] / scene_name.split('/')[1] / split / ('Image/%03d_0001.exr'%(frame_id-OFFSET))
         assert gt_render_path.exists(), str(gt_render_path)
         gt_render = cv2.imread(str(gt_render_path), cv2.IMREAD_UNCHANGED)
         gt_render = cv2.resize(gt_render, (expected_shape[1], expected_shape[0]), interpolation=cv2.INTER_AREA)

         gt_render_ = gt_render.flatten()
         sort_index = np.argsort(gt_render_)
         sort_index = sort_index[:int(gt_render_.shape[0]*0.95)]
         gt_render_enery = np.sum(gt_render.flatten()[sort_index])
         fvp_exr_enery = np.sum(fvp_synthesis.flatten()[sort_index])
         fvp_synthesis = fvp_synthesis * gt_render_enery / fvp_exr_enery
         
         cv2.imwrite(str(synthesis_target_path).replace('_ori.exr', '.exr'), fvp_synthesis)
         print('-- synthesis results saved to %s'%synthesis_target_path)

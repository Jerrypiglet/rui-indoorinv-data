
from pathlib import Path
import pickle
from shutil import rmtree
import cv2
import sys

import numpy as np
# ROOT_PATH = Path('/Users/jerrypiglet/Documents/Projects/OpenRooms_RAW_loader')
ROOT_PATH = Path('/home/ruizhu/Documents/Projects/OpenRooms_RAW_loader')
assert ROOT_PATH.exists()
sys.path.insert(0, str(ROOT_PATH))

from lib.utils_io import center_crop

test_list_path = ROOT_PATH / 'data/indoor_synthetic_resize/EXPORT_lieccv22/train/testList_kitchen.txt'
scene_name = 'indoor_synthetic/kitchen'

scene_name_write = scene_name.split('/')[1] if '/' in scene_name else scene_name
assert Path(test_list_path).exists(), str(test_list_path)
split = test_list_path.parent.stem
assert split.split('_')[0] in ['train', 'val'], str(split)

TARGET_PATH = ROOT_PATH / 'data/indoor_synthetic/RESULTS/$TASK/lieccv22' / scene_name_write / split

# BRDF_result_folder = 'BRDFLight_size0.200_int0.001_dir1.000_lam0.001_ren1.000_visWin120000_visLamp119540_invWin200000_invLamp150000_optimize'
BRDF_result_folder = 'BRDFLight_size0.200_int0.001_dir1.000_lam0.001_ren1.000_visWin120000_visLamp119540_invWin200000_invLamp150000'

Lighting_result_folder = 'EditedRerendering_size0.200_int0.001_dir1.000_lam0.001_ren1.000_visWin120000_visLamp119540_invWin200000_invLamp150000'

EditedBRDF_result_folder = BRDF_result_folder.replace('BRDFLight', 'EditedBRDFLight')

expected_shape = (160, 320)
if_downsize = True # if downsize to 160x320

IF_ALIGH = True # if align with ours

with open(str(test_list_path), 'r') as f:
   tests = f.read().splitlines()
tests = [_+'/input' for _ in tests]
tests = ['/'.join(_.split('/')[-2:]) for _ in tests]
tests = [Path(test_list_path).parent / _ for _ in tests]
assert all([Path(_).exists() for _ in tests]), str(tests)

for test in tests:
   test_name = str(test).split('/')[-2]
   scene_name_test, frame_id = test_name.split('_')[0], int(test_name.split('_')[1].replace('frame', ''))
   
   if split == 'train':
      BRDF_result_path = test.parent / BRDF_result_folder
      assert BRDF_result_path.exists(), str(BRDF_result_path)
      
      Path(str(TARGET_PATH).replace('$TASK', 'brdf')).mkdir(parents=True, exist_ok=True)
      
      albedo_vis_path = BRDF_result_path / 'albedo.png'
      assert albedo_vis_path.exists(), str(albedo_vis_path)
      albedo = cv2.imread(str(albedo_vis_path), cv2.IMREAD_UNCHANGED)
      albedo = center_crop(albedo, expected_shape)
      albedo = cv2.resize(albedo, (expected_shape[1]*2, expected_shape[0]*2))
      cv2.imwrite(str(Path(str(TARGET_PATH).replace('$TASK', 'brdf')) / ('%03d_kd.png'%frame_id)), albedo)
      
      rough_vis_path = BRDF_result_path / 'rough.png'
      assert rough_vis_path.exists(), str(rough_vis_path)
      rough = cv2.imread(str(rough_vis_path), cv2.IMREAD_UNCHANGED)
      rough = center_crop(rough, expected_shape)
      rough = cv2.resize(rough, (expected_shape[1]*2, expected_shape[0]*2))
      cv2.imwrite(str(Path(str(TARGET_PATH).replace('$TASK', 'brdf')) / ('%03d_roughness.png'%frame_id)), rough)

      INPUT_edited_path = test.parent / 'EditedInput'
      assert INPUT_edited_path.exists(), str(INPUT_edited_path)
      
      emission_path = INPUT_edited_path / 'emission.exr'
      assert emission_path.exists(), str(emission_path)
      emission = cv2.imread(str(emission_path), cv2.IMREAD_UNCHANGED)
      emission = center_crop(emission, expected_shape)
      emission = cv2.resize(emission, (expected_shape[1]*2, expected_shape[0]*2))
      cv2.imwrite(str(Path(str(TARGET_PATH).replace('$TASK', 'brdf')) / ('%03d_emission.exr'%frame_id)), emission)

      print('=== BRDF results saved to %s ==='%str(Path(str(TARGET_PATH).replace('$TASK', 'brdf'))))
    
   '''
   relight
   '''
   if split == 'val':
      Lighting_result_path = test.parent / 'input/envMask.png'
      envMask = cv2.imread(str(Lighting_result_path), cv2.IMREAD_UNCHANGED).astype(np.float32)[:, :, np.newaxis] / 255.
      
      lampMask_files = [_ for _ in (test.parent / 'EditedInput').iterdir() if _.stem.startswith('lampMask')]
      if len(lampMask_files) == 0:
         lampMask = None
         pass
      else:
         assert len(lampMask_files) == 1, str(lampMask_files)
         lampMask_file = lampMask_files[0]
         lampMask = cv2.imread(str(lampMask_file), cv2.IMREAD_UNCHANGED) > 0
         EditedBRDF_result_path = test.parent / EditedBRDF_result_folder
         visLamp_files = [_ for _ in EditedBRDF_result_path.iterdir() if _.stem.startswith('visLampSrc')]
         assert len(visLamp_files) == 1, str(visLamp_files)
         visLamp_file = visLamp_files[0]
         with open(str(visLamp_file), 'rb') as f:
            visLampDict = pickle.load(f)
            _rad = visLampDict['src'].reshape((1, 3))
      
      Lighting_result_path = test.parent / Lighting_result_folder
      assert Lighting_result_path.exists(), str(Lighting_result_path)
      lieccv22_relight_path = Lighting_result_path / 'rendered.hdr'
      assert lieccv22_relight_path.exists(), str(lieccv22_relight_path)
      lieccv22_relight = cv2.imread(str(lieccv22_relight_path), cv2.IMREAD_UNCHANGED)
      lieccv22_relight = lieccv22_relight * envMask
      if lampMask is not None:
         lieccv22_relight[lampMask] = _rad
      lieccv22_relight = center_crop(lieccv22_relight, expected_shape)
      
      relight_target_path = Path(str(TARGET_PATH).replace('$TASK', 'relight')) / ('%03d_ori.exr'%frame_id)
      relight_target_path.parent.mkdir(parents=True, exist_ok=True)
      cv2.imwrite(str(relight_target_path), lieccv22_relight)
      print('-- relight results saved to %s'%relight_target_path)

      # ours_relight_path = ROOT_PATH / 'data' / (scene_name.replace('_new', '')+'-relight') / split / ('Image/%03d_0001.exr'%frame_id)
      ours_relight_path = ROOT_PATH / 'data' / scene_name.split('/')[0] / 'RESULTS/relight/ours' / scene_name.split('/')[1] / ('%03d.exr'%frame_id)
      assert ours_relight_path.exists(), str(ours_relight_path)
      ours_relight = cv2.imread(str(ours_relight_path), cv2.IMREAD_UNCHANGED)
      ours_relight = cv2.resize(ours_relight, (expected_shape[1], expected_shape[0]), interpolation=cv2.INTER_AREA)
      
      relight_ours_ref_target_path = Path(str(TARGET_PATH).replace('$TASK', 'relight')) / ('%03d_ours_ref.exr'%frame_id)
      cv2.imwrite(str(relight_ours_ref_target_path), ours_relight)
      print('------- relight reference results saved to %s'%relight_ours_ref_target_path)
      
      if IF_ALIGH:
         ours_relight_ = ours_relight.flatten()
         sort_index = np.argsort(ours_relight_)
         sort_index = sort_index[:int(ours_relight_.shape[0]*0.95)]
         ours_relight_enery = np.sum(ours_relight.flatten()[sort_index])
         fvp_exr_enery = np.sum(lieccv22_relight.flatten()[sort_index])
         lieccv22_relight = lieccv22_relight * ours_relight_enery / fvp_exr_enery
         
         cv2.imwrite(str(relight_target_path).replace('_ori.exr', '.exr'), lieccv22_relight)
         print('-- relight results saved to %s'%relight_target_path)

from pathlib import Path
import pickle
from shutil import rmtree
import cv2
import sys

import numpy as np
ROOT_PATH = Path('/Users/jerrypiglet/Documents/Projects/rui-indoorinv-data')
assert ROOT_PATH.exists()
sys.path.insert(0, str(ROOT_PATH))

from lib.utils_io import center_crop

# SPLIT = 'train'
SPLIT = 'val'
DATASET = 'indoor_synthetic'
expected_shape = (160, 320)

test_list_path = ROOT_PATH / 'data/indoor_synthetic_resize/EXPORT_lieccv22' / SPLIT / 'testList_kitchen.txt'
scene_name = 'indoor_synthetic/kitchen'

# test_list_path = ROOT_PATH / 'data/indoor_synthetic_resize/EXPORT_lieccv22' / SPLIT / 'testList_bedroom.txt'
# scene_name = 'indoor_synthetic/bedroom' 

# test_list_path = ROOT_PATH / 'data/indoor_synthetic_resize/EXPORT_lieccv22' / SPLIT / 'testList_bathroom.txt'
# scene_name = 'indoor_synthetic/bathroom'

# test_list_path = ROOT_PATH / 'data/indoor_synthetic_resize/EXPORT_lieccv22' / SPLIT / 'testList_livingroom.txt'
# scene_name = 'indoor_synthetic/livingroom'

'''
real
'''
DATASET = 'real'; SPLIT = 'real'; expected_shape = (360, 540)

scene_name = 'real/ConferenceRoomV2_final_supergloo'
test_list_path = ROOT_PATH / 'data/real/EXPORT_lieccv22' / SPLIT / 'testList_ConferenceRoomV2_final_supergloo.txt'
# frame_id_list = [9, 161] # BRDF
frame_id_list = [68, 180] # synthesis/relight 

scene_name = 'real/ClassRoom'
test_list_path = ROOT_PATH / 'data/real/EXPORT_lieccv22' / SPLIT / 'testList_ClassRoom.txt'
# frame_id_list = [235, 120] # BRDF
# frame_id_list = [27, 92] # synthesis
frame_id_list = [278, 275] # relight

scene_name_write = scene_name.split('/')[1] if '/' in scene_name else scene_name
assert Path(test_list_path).exists(), str(test_list_path)
split = test_list_path.parent.stem
# assert split.split('_')[0] in ['train', 'val'], str(split)

TARGET_PATH = ROOT_PATH / 'data' / DATASET / 'RESULTS/$TASK/lieccv22' / scene_name_write

BRDF_result_folder = 'BRDFLight_size0.200_int0.001_dir1.000_lam0.001_ren1.000_visWin120000_visLamp119540_invWin200000_invLamp150000_optimize'
EditedBRDF_result_folder = BRDF_result_folder.replace('BRDFLight', 'EditedBRDFLight')

'''
switch between two re-rendering tasks
'''
# RENDER_TASK = None
# RENDER_TASK = 'viewsynthesis'; Lighting_result_folder = 'Rendering_size0.200_int0.001_dir1.000_lam0.001_ren1.000_visWin120000_visLamp119540_invWin200000_invLamp150000_optimize'
RENDER_TASK = 'relight'; Lighting_result_folder = 'EditedRendering_size0.200_int0.001_dir1.000_lam0.001_ren1.000_visWin120000_visLamp119540_invWin200000_invLamp150000_optimize'

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
   scene_name_test, frame_id = '_'.join(test_name.split('_')[:-1]), int(test_name.split('_')[-1].replace('frame', ''))
   if frame_id not in frame_id_list: continue
   
   if split in ['train', 'real']:
      BRDF_result_path = test.parent / BRDF_result_folder
      assert BRDF_result_path.exists(), str(BRDF_result_path)
      
      Path(str(TARGET_PATH).replace('$TASK', 'brdf')).mkdir(parents=True, exist_ok=True)
      
      albedo_vis_path = BRDF_result_path / 'albedo.png'
      assert albedo_vis_path.exists(), str(albedo_vis_path)
      # albedo = cv2.imread(str(albedo_vis_path), cv2.IMREAD_UNCHANGED).astype(np.float32)/255.
      # albedo = (np.clip(albedo**2.2, 0, 1) * 255).astype(np.uint8)
      albedo = cv2.imread(str(albedo_vis_path), cv2.IMREAD_UNCHANGED)
      # albedo = center_crop(albedo, expected_shape)
      albedo = cv2.resize(albedo, (expected_shape[1]*2, expected_shape[0]*2))
      cv2.imwrite(str(Path(str(TARGET_PATH).replace('$TASK', 'brdf')) / ('%03d_kd.png'%frame_id)), albedo)
      
      rough_vis_path = BRDF_result_path / 'rough.png'
      assert rough_vis_path.exists(), str(rough_vis_path)
      rough = cv2.imread(str(rough_vis_path), cv2.IMREAD_UNCHANGED)
      # rough = center_crop(rough, expected_shape)
      rough = cv2.resize(rough, (expected_shape[1]*2, expected_shape[0]*2))
      cv2.imwrite(str(Path(str(TARGET_PATH).replace('$TASK', 'brdf')) / ('%03d_roughness.png'%frame_id)), rough)

      INPUT_edited_path = test.parent / 'EditedInput'
      assert INPUT_edited_path.exists(), str(INPUT_edited_path)
      
      emission_path = INPUT_edited_path / 'emission.exr'
      assert emission_path.exists(), str(emission_path)
      emission = cv2.imread(str(emission_path), cv2.IMREAD_UNCHANGED)
      # emission = center_crop(emission, expected_shape)
      emission = cv2.resize(emission, (expected_shape[1]*2, expected_shape[0]*2))
      cv2.imwrite(str(Path(str(TARGET_PATH).replace('$TASK', 'brdf')) / ('%03d_emission.exr'%frame_id)), emission)

      print('=== BRDF results saved to %s ==='%str(Path(str(TARGET_PATH).replace('$TASK', 'brdf'))))
   
   # continue
   '''
   relight
   '''
   if RENDER_TASK is None: 
      print('No rendering task specified. DONE.')
      continue
   
   if split in ['val', 'real']:
      Lighting_result_path = test.parent / 'input/envMask.png'
      envMask = cv2.imread(str(Lighting_result_path), cv2.IMREAD_UNCHANGED).astype(np.float32)[:, :, np.newaxis] / 255.
      
      if RENDER_TASK == 'relight':
         emitterMask_files = [_ for _ in (test.parent / 'EditedInput').iterdir() if _.stem.startswith('lampMask')]
      elif RENDER_TASK == 'viewsynthesis':
         emitterMask_files = [_ for _ in (test.parent / 'input').iterdir() if _.stem.startswith('winMask')]
      else:
         raise ValueError(RENDER_TASK)
      
      if len(emitterMask_files) == 0:
         emitterMask_list = []
         print('No emitterMask found for %s'%str(test))
         pass
      else:
         # assert len(emitterMask_files) == 1, str(emitterMask_files)
         # emitterMask_file = emitterMask_files[0]
         emitterMask_list = []
         for emitterMask_file in emitterMask_files:
            emitterMask = cv2.imread(str(emitterMask_file), cv2.IMREAD_UNCHANGED) > 0
            emitter_idx = int(emitterMask_file.stem.split('_')[-1])
            if RENDER_TASK == 'relight':
               _BRDF_result_path = test.parent / EditedBRDF_result_folder
            elif RENDER_TASK == 'viewsynthesis':
               _BRDF_result_path = test.parent / BRDF_result_folder
            else:
               raise ValueError(RENDER_TASK)
               
            if RENDER_TASK == 'relight':
               visEmitter_files = [_ for _ in _BRDF_result_path.iterdir() if _.stem.startswith('visLampSrc_%d'%emitter_idx)]
            elif RENDER_TASK == 'viewsynthesis':
               visEmitter_files = [_ for _ in _BRDF_result_path.iterdir() if _.stem.startswith('visWinSrc_%d'%emitter_idx)]
            else:
               raise ValueError(RENDER_TASK)
            assert len(visEmitter_files) == 1, str(visEmitter_files)
            visEmitter_file = visEmitter_files[0]
            with open(str(visEmitter_file), 'rb') as f:
               visEmitterDict = pickle.load(f)
               _rad = visEmitterDict['src'].flatten()[:3].reshape((1, 3))

            emitterMask_list.append((emitterMask, _rad))
      
      Lighting_result_path = test.parent / Lighting_result_folder
      assert Lighting_result_path.exists(), str(Lighting_result_path)
      lieccv22_relight_path = Lighting_result_path / 'rendered.hdr'
      assert lieccv22_relight_path.exists(), str(lieccv22_relight_path)
      lieccv22_relight = cv2.imread(str(lieccv22_relight_path), cv2.IMREAD_UNCHANGED)
      lieccv22_relight = lieccv22_relight * envMask
      # if emitterMask is not None:
      #    lieccv22_relight[emitterMask] = _rad
      if emitterMask_list != []:
         for emitterMask, _rad in emitterMask_list:
            lieccv22_relight[emitterMask] = _rad
            print('emitterMask.shape', emitterMask.shape, 'assigning _rad:', _rad)
      # lieccv22_relight = center_crop(lieccv22_relight, expected_shape)
      lieccv22_relight = cv2.resize(lieccv22_relight, (expected_shape[1]*2, expected_shape[0]*2))
      
      input_path = test.parent / 'input'
      lampMask = np.zeros((lieccv22_relight.shape[0], lieccv22_relight.shape[1])).astype(bool)
      for _ in input_path.iterdir():
         if _.stem.startswith('lampMask'):
            lampMask_ = cv2.resize(cv2.imread(str(_), cv2.IMREAD_UNCHANGED), (lieccv22_relight.shape[1], lieccv22_relight.shape[0]), cv2.INTER_NEAREST) > 0
            lampMask = np.logical_or(lampMask, lampMask_)
      lieccv22_relight[lampMask] = 0

      relight_target_path = Path(str(TARGET_PATH).replace('$TASK', RENDER_TASK)) / ('%03d_ori.exr'%frame_id)
      relight_target_path.parent.mkdir(parents=True, exist_ok=True)
      cv2.imwrite(str(relight_target_path), lieccv22_relight)
      print('-- %s results saved to %s'%(RENDER_TASK, relight_target_path))

      # ours_render_path = ROOT_PATH / 'data' / (scene_name.replace('_new', '')+'-relight') / split / ('Image/%03d_0001.exr'%frame_id)
      # ours_render_path = ROOT_PATH / 'data' / scene_name.split('/')[0] / 'RESULTS' / RENDER_TASK / 'ours' / scene_name.split('/')[1] / ('%03d.exr'%frame_id)
      # assert ours_render_path.exists(), str(ours_render_path)
      # ours_render = cv2.imread(str(ours_render_path), cv2.IMREAD_UNCHANGED)
      # ours_render = cv2.resize(ours_render, (expected_shape[1], expected_shape[0]), interpolation=cv2.INTER_AREA)
      
      # relight_ours_ref_target_path = Path(str(TARGET_PATH).replace('$TASK', RENDER_TASK)) / ('%03d_ours_ref.exr'%frame_id)
      # cv2.imwrite(str(relight_ours_ref_target_path), ours_render)
      # print('------- %s eference results saved to %s'%(RENDER_TASK, relight_ours_ref_target_path))
      
      if IF_ALIGH:
         if split != 'real':
            gt_render_path = ROOT_PATH / 'data' / scene_name.split('/')[0] / (scene_name.split('/')[1]+'-relight') / split / ('Image/%03d_0001.exr'%frame_id)
         else:
            gt_render_path = ROOT_PATH / 'data' / scene_name.split('/')[0] / (scene_name.split('/')[1]) / ('merged_images/img_%04d.exr'%frame_id)
         assert gt_render_path.exists(), str(gt_render_path)
         gt_render = cv2.imread(str(gt_render_path), cv2.IMREAD_UNCHANGED)
         gt_render = cv2.resize(gt_render, (expected_shape[1], expected_shape[0]), interpolation=cv2.INTER_AREA)
         
         gt_render_ = gt_render.flatten()
         sort_index = np.argsort(gt_render_)
         sort_index = sort_index[:int(gt_render_.shape[0]*0.95)]
         gt_render_enery = np.sum(gt_render.flatten()[sort_index])
         lieccv22_exr_enery = np.sum(lieccv22_relight.flatten()[sort_index])
         lieccv22_relight = lieccv22_relight * gt_render_enery / lieccv22_exr_enery
         
         cv2.imwrite(str(relight_target_path).replace('_ori.exr', '.exr'), lieccv22_relight)
         print('-- %s results saved to %s'%(RENDER_TASK, str(relight_target_path).replace('_ori.exr', '.exr')))
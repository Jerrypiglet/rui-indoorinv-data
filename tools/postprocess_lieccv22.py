
from pathlib import Path
import cv2
import sys
ROOT_PATH = Path('/Users/jerrypiglet/Documents/Projects/OpenRooms_RAW_loader')
sys.path.insert(0, str(ROOT_PATH))

from lib.utils_io import center_crop

# test_list_path = ROOT_PATH / 'data/indoor_synthetic/lieccv22_export/val/testList.txt'
test_list_path = ROOT_PATH / 'data/indoor_synthetic/lieccv22_export/train_gtDepth/testList.txt'
scene_name = 'kitchen'
assert Path(test_list_path).exists(), str(test_list_path)
split = test_list_path.parent.stem
assert split.split('_')[0] in ['train', 'val'], str(split)

TARGET_PATH = ROOT_PATH / 'data/indoor_synthetic/results/brdf/lieccv22' / scene_name / split
TARGET_PATH.mkdir(parents=True, exist_ok=True)
# BRDF_result_folder = 'BRDFLight_size0.200_int0.001_dir1.000_lam0.001_ren1.000_visWin120000_visLamp119540_invWin200000_invLamp150000_optimize'
BRDF_result_folder = 'BRDFLight_size0.200_int0.001_dir1.000_lam0.001_ren1.000_visWin120000_visLamp119540_invWin200000_invLamp150000'
expected_shape = (160, 320)

with open(str(test_list_path), 'r') as f:
   tests = f.read().splitlines()
tests = [_+'/input' for _ in tests]
tests = ['/'.join(_.split('/')[-2:]) for _ in tests]
tests = [Path(test_list_path).parent / _ for _ in tests]
assert all([Path(_).exists() for _ in tests]), str(tests)
   

for test in tests:
    test_name = str(test).split('/')[-2]
    scene_name_test, frame_id = test_name.split('_')[0], int(test_name.split('_')[1].replace('frame', ''))
    
    BRDF_result_path = test.parent / BRDF_result_folder
    assert BRDF_result_path.exists(), str(BRDF_result_path)
    
    albedo_vis_path = BRDF_result_path / 'albedo.png'
    assert albedo_vis_path.exists(), str(albedo_vis_path)
    albedo = cv2.imread(str(albedo_vis_path), cv2.IMREAD_UNCHANGED)
    albedo = center_crop(albedo, expected_shape)
    albedo = cv2.resize(albedo, (expected_shape[1]*2, expected_shape[0]*2))
    cv2.imwrite(str(TARGET_PATH / ('%03d_kd.png'%frame_id)), albedo)

    rough_vis_path = BRDF_result_path / 'rough.png'
    assert rough_vis_path.exists(), str(rough_vis_path)
    rough = cv2.imread(str(rough_vis_path), cv2.IMREAD_UNCHANGED)
    rough = center_crop(rough, expected_shape)
    rough = cv2.resize(rough, (expected_shape[1]*2, expected_shape[0]*2))
    cv2.imwrite(str(TARGET_PATH / ('%03d_roughness.png'%frame_id)), rough)

    
'''
Convert .blender file with cameras to cam.txt and intrinsic_mitsubaScene.txt

> python export_cam_from_blender_file.py

Then render with:

> python debug_blender_mitsubaScene3D.py --scene kitchen_diy --renderer blender --scene cornell_box --spp 128

Blender scene with cameras: ![](https://i.imgur.com/bYRBnra.jpg)
'''

from pathlib import Path
import bpy
import numpy as np
np.set_printoptions(suppress=True)
from scipy.spatial.transform import Rotation
# from lib.utils_OR.utils_OR_cam import R_t_to_origin_lookatvector_up_opencv

def R_t_to_origin_lookatvector_up_opencv(R, t):
    origin = t
    lookatvector = R @ np.array([[0.], [0.], [1.]], dtype=np.float32)
    up = R @ np.array([[0.], [-1.], [0.]], dtype=np.float32)
    return (origin, lookatvector, up)

# scenes_root = Path('/Users/jerrypiglet/Documents/Projects/rui-indoorinv-data/data/debug_scenes')
scenes_root = Path('/home/ruizhu/Documents/Projects/rui-indoorinv-data/data/debug_scenes')

# scene_name = 'cornell_box'
# scene_name = 'cornell_box_tmp'
# scene_name = 'kitchen_manual'
scene_name = 'AI55_004'

blend_file_path = scenes_root / scene_name / 'test.blend'
assert blend_file_path.exists(), 'blend_file_path: %s does not exist!'%blend_file_path
bpy.ops.wm.open_mainfile(filepath=str(blend_file_path))

bpy.context.scene.render.image_settings.file_format = 'OPEN_EXR'
bpy.context.scene.render.image_settings.color_depth = str(16)

im_H, im_W = bpy.context.scene.render.resolution_y, bpy.context.scene.render.resolution_x

(scenes_root / scene_name / 'train').mkdir(exist_ok=True)
pose_file_path = scenes_root / scene_name / 'train' / ('cam.txt')

origin_lookatvector_up_list = []
K_list = []

for cam in bpy.data.objects:
    if not cam.name.replace(scene_name+'_', '').startswith('Camera'): continue # assuming cameras are labelled as Camera{id}, e.g. Camera0, Camera1, etc.
    # https://docs.blender.org/api/current/bpy.types.Camera.html
    if cam.data is None: continue
    print('--- Found camera', cam.name, type(cam))
    assert cam.data.type == 'PERSP', 'cam.data.type: %s is not PERSP!'%cam.data.type
    
    fx = np.float32(cam.data.sensor_width / 2. / cam.data.lens)
    # fy = np.float32(cam.data.sensor_height / 2. / cam.data.lens)
    cx, cy = 0.5, 0.5
    cx_pix, cy_pix = cx * im_W, cy * im_H
    fx_pix = cx_pix / fx
    # fy_pix = cy_pix / fx
    fy_pix = fx_pix
    K = np.array([[fx_pix, 0., cx_pix], [0., fy_pix, cy_pix], [0., 0., 1.]], dtype=np.float32)
    K_list.append(K)
    
    pos = np.array(cam.location).flatten()
    angles = np.array(cam.rotation_euler).flatten()
    
    # inverse of lib/utils_OR/utils_OR_cam.py: convert_OR_poses_to_blender_npy()
    Rs = np.array(Rotation.from_euler('xyz', angles, degrees=False).as_matrix())
    assert np.allclose(np.linalg.det(Rs), 1), 'R is not a rotation matrix!'
    
    coord_conv = [0,2,1]
    Rs[1] = -Rs[1]
    Rs = Rs[coord_conv]
    Rs[:,1] = -Rs[:,1]
    Rs[:,2] = -Rs[:,2]
    pos[1] = -pos[1]
    pos = pos[coord_conv]

    assert np.allclose(np.linalg.det(Rs), 1), 'R is not a rotation matrix!'
    assert Rs.shape == (3,3), 'Rs.shape: %s is not (3,3)'%str(Rs.shape)
    assert pos.shape == (3,), 'pos.shape: %s is not (3,)'%str(pos.shape)
    
    (origin, lookatvector, up) = R_t_to_origin_lookatvector_up_opencv(Rs, pos.reshape((3, 1)))
    print('origin', origin, 'lookatvector', lookatvector, 'up', up)
    
    assert np.abs(np.linalg.norm(up)) - 1. < 1e-5, 'up is not unit vector!'
    assert np.abs(np.linalg.norm(lookatvector)) - 1. < 1e-5, 'lookatvector is not unit vector!'
    
    origin_lookatvector_up_list.append((origin, lookatvector, up))
    
print('%d poses found!'%len(origin_lookatvector_up_list))
assert len(origin_lookatvector_up_list) > 0, 'No camera found!'

with open(str(pose_file_path), 'w') as camOut:
    camOut.write('%d\n'%len(origin_lookatvector_up_list))
    for i, (origin, lookatvector, up) in enumerate(origin_lookatvector_up_list):
        lookat = origin + lookatvector
        for vector in [origin, lookat, up, lookatvector]:
            vector = vector.flatten()
            camOut.write('%.6f %.6f %.6f\n'%(vector[0], vector[1], vector[2]))
print('Pose file written to %s (%d poses).'%(pose_file_path, len(origin_lookatvector_up_list)))

np.savetxt(str(scenes_root / scene_name / 'intrinsic_mitsubaScene.txt'), K_list[0], fmt='%.6f', delimiter=' ')
print('Intrinsic file written to %s.'%(scenes_root / scene_name / 'intrinsic_mitsubaScene.txt'))
print(K_list[0])

K_list_diff = np.array(K_list) - K_list[0]
assert np.amax(K_list_diff) < 1e-5, 'K_list_diff: %s is not all identical! ONLY WROTE FIRST K as intrinsics!'%str(K_list_diff)


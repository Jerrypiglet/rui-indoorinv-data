import numpy as np
import cv2
import torch
import os
from scipy.spatial.transform import Slerp
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R
import json
import trimesh
import glob
import PIL
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

# image_size = 384
# trans_totensor = transforms.Compose([
#     transforms.CenterCrop(image_size*2),
#     transforms.Resize(image_size, interpolation=PIL.Image.BILINEAR),
# ])
# depth_trans_totensor = transforms.Compose([
#     transforms.Resize([968, 1296], interpolation=PIL.Image.NEAREST),
#     transforms.CenterCrop(image_size*2),
#     transforms.Resize(image_size, interpolation=PIL.Image.NEAREST),
# ])

'''
work with Mitsuba/Blender scenes
'''
import sys
# host = 'mm1'
host = 'apple'
PATH_HOME = {
    'apple': '/Users/jerrypiglet/Documents/Projects/OpenRooms_RAW_loader', 
    'mm1': '', 
    'qc': '', 
}[host]
sys.path.insert(0, PATH_HOME)
INV_NERF_ROOT = {
    'apple': '/Users/jerrypiglet/Documents/Projects/inv-nerf', 
    'mm1': '/home/ruizhu/Documents/Projects/inv-nerf', 
    'qc': '', 
}[host]
from pathlib import Path
import numpy as np
np.set_printoptions(suppress=True)

from lib.class_mitsubaScene3D import mitsubaScene3D

base_root = Path(PATH_HOME) / 'data/indoor_synthetic'
xml_root = Path(PATH_HOME) / 'data/indoor_synthetic'

'''
The kitchen scene: data/indoor_synthetic/kitchen/scene_v3.xml
'''
xml_filename = 'scene_v3.xml'
scene_name = 'living-room'; frame_ids_train = list(range(190)); frame_ids_val = list(range(10))
# scene_name = 'kitchen_re'; frame_ids_train = list(range(202)); frame_ids_val = list(range(10))

mitsuba_scene_dict = {}
for split, frame_ids in zip(['train', 'val'], [frame_ids_train, frame_ids_val]):
# for split, frame_ids in zip(['val'], [frame_ids_val]):
    mitsuba_scene_dict[split] = mitsubaScene3D(
        if_debug_info=False, 
        host=host, 
        root_path_dict = {'PATH_HOME': Path(PATH_HOME), 'rendering_root': base_root, 'xml_scene_root': xml_root}, 
        scene_params_dict={
            'xml_filename': xml_filename, 
            'scene_name': scene_name, 
            'split': split, 
            'frame_id_list': frame_ids, 
            'mitsuba_version': '3.0.0', 
            'intrinsics_path': Path(PATH_HOME) / 'data/indoor_synthetic' / scene_name / 'intrinsic_mitsubaScene.txt', 
            'up_axis': 'y+', 
            'pose_file': ('json', 'transforms.json'), # requires scaled Blender scene! in comply with Liwen's IndoorDataset (https://github.com/william122742/inv-nerf/blob/bake/utils/dataset/indoor.py)
            }, 
        mi_params_dict={
            # 'if_also_dump_xml_with_lit_area_lights_only': True,  # True: to dump a second file containing lit-up lamps only
            'debug_render_test_image': False, # [DEBUG][slow] True: to render an image with first camera, usig Mitsuba: images/demo_mitsuba_render.png
            'debug_dump_mesh': True, # [DEBUG] True: to dump all object meshes to mitsuba/meshes_dump; load all .ply files into MeshLab to view the entire scene: images/demo_mitsuba_dump_meshes.png
            'if_sample_rays_pts': True, # True: to sample camera rays and intersection pts given input mesh and camera poses
            'if_get_segs': False, # [depend on if_sample_rays_pts] True: to generate segs similar to those in openroomsScene2D.load_seg()
            'poses_sample_num': 200, # Number of poses to sample; set to -1 if not sampling
            },
        # modality_list = ['im_sdr', 'im_hdr', 'seg', 'poses', 'albedo', 'roughness', 'depth', 'normal', 'lighting_SG', 'lighting_envmap'], 
        modality_list = [
            'poses', 
            'im_hdr', 
            'im_sdr', 
            # 'depth', 'normal', 
            # 'lighting_SG', 
            # 'layout', 
            # 'shapes', # objs + emitters, geometry shapes + emitter properties
            ], 
        modality_filename_dict = {
            # 'poses', 
            'im_hdr': 'Image/%03d_0001.exr', 
            'im_sdr': 'Image/%03d_0001.png', 
            # 'lighting_envmap', 
            'albedo': 'DiffCol/%03d_0001.exr', 
            'roughness': 'Roughness/%03d_0001.exr', 
            'emission': 'Emit/%03d_0001.exr', 
            'depth': 'Depth/%03d_0001.exr', 
            'normal': 'Normal/%03d_0001.exr', 
            # 'lighting_SG', 
            # 'layout', 
            # 'shapes', # objs + emitters, geometry shapes + emitter properties
        }, 
        im_params_dict={
            'im_H_load': 320, 'im_W_load': 640, 
            'im_H_resize': 320, 'im_W_resize': 640, 
            # 'im_H_resize': 160, 'im_W_resize': 320, 
            # 'im_H_resize': 1, 'im_W_resize': 2, 
            # 'im_H_resize': 32, 'im_W_resize': 64, 
            # 'spp': 2048, 
            'spp': 16, 
            # 'im_H_resize': 120, 'im_W_resize': 160, # to use for rendering so that im dimensions == lighting dimensions
            # 'im_hdr_ext': 'exr', 
            }, 
        cam_params_dict={
            'near': 0.1, 'far': 10., 
            'heightMin' : 0.7,  
            'heightMax' : 2.,  
            'distMin': 1., # to wall distance min
            'distMax': 4.5, 
            'thetaMin': -60, 
            'thetaMax' : 40, # theta: pitch angle; up+
            'phiMin': -60, # yaw angle
            'phiMax': 60, 
            'sample_pose_if_vis_plt': False, # images/demo_sample_pose.png
        }, 
        lighting_params_dict={
            'SG_num': 12, 
            'env_row': 8, 'env_col': 16, # resolution to load; FIXED
            'env_downsample_rate': 2, # (8, 16) -> (4, 8)

            # 'env_height': 2, 'env_width': 4, 
            # 'env_height': 8, 'env_width': 16, 
            # 'env_height': 128, 'env_width': 256, 
            'env_height': 256, 'env_width': 512, 
        }, 
        shape_params_dict={
            'if_load_obj_mesh': True, # set to False to not load meshes for objs (furniture) to save time
            'if_load_emitter_mesh': True,  # default True: to load emitter meshes, because not too many emitters

            'if_sample_pts_on_mesh': False,  # default True: sample points on each shape -> self.sample_pts_list
            'sample_mesh_ratio': 0.1, # target num of VERTICES: len(vertices) * sample_mesh_ratio
            'sample_mesh_min': 10, 
            'sample_mesh_max': 100, 

            'if_simplify_mesh': False,  # default True: simply triangles
            'simplify_mesh_ratio': 0.1, # target num of FACES: len(faces) * simplify_mesh_ratio
            'simplify_mesh_min': 100, 
            'simplify_mesh_max': 1000, 
            'if_remesh': True, # False: images/demo_shapes_3D_kitchen_NO_remesh.png; True: images/demo_shapes_3D_kitchen_YES_remesh.png
            'remesh_max_edge': 0.15,  
            },
        emitter_params_dict={
            },
    )


'''
CONVERT
'''
out_path_prefix = str(base_root / scene_name / 'monosdf' / scene_name)
# data_root = '/home/yuzh/Projects/datasets/scannet/'
scenes = [scene_name]
out_names = ['trainval']

for scene, out_name in zip(scenes, out_names):
    out_path = os.path.join(out_path_prefix, out_name)
    os.makedirs(out_path, exist_ok=True)
    print(out_path)

    folders = ["image", "mask", "depth", "normal"]
    for folder in folders:
        out_folder = os.path.join(out_path, folder)
        os.makedirs(out_folder, exist_ok=True)

    # load color 
    
    # color_path = os.path.join(data_root, scene, 'frames', 'color')
    # color_paths = sorted(glob.glob(os.path.join(color_path, '*.jpg')), 
    #     key=lambda x: int(os.path.basename(x)[:-4]))
    # print(color_paths)
    
    # load depth
    
    # depth_path = os.path.join(data_root, scene, 'frames', 'depth')
    # depth_paths = sorted(glob.glob(os.path.join(depth_path, '*.png')), 
    #     key=lambda x: int(os.path.basename(x)[:-4]))
    # print(depth_paths)

    # load intrinsic
    
    # intrinsic_path = os.path.join(data_root, scene, 'frames', 'intrinsic', 'intrinsic_color.txt')
    # camera_intrinsic = np.loadtxt(intrinsic_path)
    # print(camera_intrinsic)
    camera_intrinsic = mitsuba_scene_dict['train'].K # (3, 3)
    camera_intrinsic = np.hstack((camera_intrinsic, np.array([0., 0., 0.], dtype=np.float32).reshape((3, 1))))
    camera_intrinsic = np.vstack((camera_intrinsic, np.array([0., 0., 0., 1.], dtype=np.float32).reshape((1, 4))))
    '''
    array([[585.0939,  -0.0001, 191.8754,   0.    ],
       [  0.    , 585.0942, 191.8754,   0.    ],
       [  0.    ,   0.    ,   1.    ,   0.    ],
       [  0.    ,   0.    ,   0.    ,   1.    ]])
    '''

    # load pose

    # pose_path = os.path.join(data_root, scene, 'frames', 'pose')
    # poses = []
    # pose_paths = sorted(glob.glob(os.path.join(pose_path, '*.txt')),
    #                     key=lambda x: int(os.path.basename(x)[:-4]))
    # for pose_path in pose_paths:
    #     c2w = np.loadtxt(pose_path)
    #     poses.append(c2w)
    poses = mitsuba_scene_dict['train'].pose_list + mitsuba_scene_dict['val'].pose_list # [(3, 4)]
    poses = [np.vstack((pose, np.array([0., 0., 0., 1.], dtype=np.float32).reshape((1, 4)))) for pose in poses]
    poses = np.array(poses)

    # deal with invalid poses
    # valid_poses = np.isfinite(poses).all(axis=2).all(axis=1)
    min_vertices = poses[:, :3, 3].min(axis=0)
    max_vertices = poses[:, :3, 3].max(axis=0)
 
    center = (min_vertices + max_vertices) / 2.
    scale = 2. / (np.max(max_vertices - min_vertices) + 3.)
    print('[pose normalization to unit cube] --center, scale--', center, scale)

    # we should normalized to unit cube
    scale_mat = np.eye(4).astype(np.float32)
    scale_mat[:3, 3] = -center
    scale_mat[:3 ] *= scale 
    scale_mat = np.linalg.inv(scale_mat)

    # copy image
    cameras = {}
    pcds = []
    # H, W = 968, 1296

    # center crop by 2 * image_size
    # offset_x = (W - image_size * 2) * 0.5
    # offset_y = (H - image_size * 2) * 0.5
    # camera_intrinsic[0, 2] -= offset_x
    # camera_intrinsic[1, 2] -= offset_y
    # # resize from 384*2 to 384
    # resize_factor = 0.5
    # camera_intrinsic[:2, :] *= resize_factor
    
    K = camera_intrinsic
    print('--K--', K)
    
    from tqdm import tqdm
    import shutil
    from PIL import Image
    
    im_sdr_file_list = mitsuba_scene_dict['train'].im_sdr_file_list + mitsuba_scene_dict['val'].im_sdr_file_list
    im_hdr_file_list = mitsuba_scene_dict['train'].im_hdr_file_list + mitsuba_scene_dict['val'].im_hdr_file_list
    mi_depth_list = mitsuba_scene_dict['train'].mi_depth_list + mitsuba_scene_dict['val'].mi_depth_list
    mi_normal_global_list = mitsuba_scene_dict['train'].mi_normal_global_list + mitsuba_scene_dict['val'].mi_normal_global_list

    for idx, pose in tqdm(enumerate(poses)):
        # print(idx, valid)
        # if idx % 10 != 0: continue
        # if not valid : continue

        target_image = os.path.join(out_path, "image/%06d.png"%(idx))
        shutil.copy(im_sdr_file_list[idx], target_image)

        target_image_hdr = os.path.join(out_path, "image/%06d.exr"%(idx))
        shutil.copy(im_hdr_file_list[idx], target_image_hdr)

        # print(target_image)
        # img = Image.open(image_path)
        # img_tensor = trans_totensor(img)
        # img_tensor.save(target_image)

        mask = (np.ones((mitsuba_scene_dict['train'].H, mitsuba_scene_dict['train'].W, 3)) * 255.).astype(np.uint8) # [TODO] load masks from mi!
        target_image = os.path.join(out_path, "mask/%03d.png"%(idx))
        cv2.imwrite(target_image, mask)

        # load depth
        target_image = os.path.join(out_path, "depth/%06d.png"%(idx))
        # depth = cv2.imread(depth_path, -1).astype(np.float32) / 1000.
        # depth_PIL = Image.fromarray(depth)
        # new_depth = depth_trans_totensor(depth_PIL)
        # new_depth = np.asarray(new_depth)
        new_depth = mi_depth_list[idx].astype(np.float32) / 1000.
        plt.imsave(target_image, new_depth, cmap='viridis')
        np.save(target_image.replace(".png", ".npy"), new_depth)

        mi_normal_global = mi_normal_global_list[idx].astype(np.float32)
        R = pose[:3, :3]
        mi_normal_cam = (R.T @ mi_normal_global.reshape(-1, 3).T).T.reshape(mitsuba_scene_dict['train'].H, mitsuba_scene_dict['train'].W, 3) # images/demo_normal_gt_scannetScene_opencv.png

        np.save(os.path.join(out_path, "normal/%06d.npy"%(idx)), (mi_normal_cam.transpose(2, 0, 1)+1.)/2.) # (3, H, W), [0., 1.]

        _ = Image.fromarray(((mi_normal_cam+1.)/2.*255.).astype(np.uint8))
        _.save(os.path.join(out_path, "normal/%06d.png"%(idx)))
        
        # save pose
        pcds.append(pose[:3, 3])
        pose = K @ np.linalg.inv(pose)
        
        #cameras["scale_mat_%d"%(idx)] = np.eye(4).astype(np.float32)
        cameras["scale_mat_%d"%(idx)] = scale_mat
        cameras["world_mat_%d"%(idx)] = pose
        
        cameras['split_%d'%idx] = 'train' if idx < len(mitsuba_scene_dict['train'].pose_list) else 'val'
        cameras['frame_id_%d'%idx] = idx if idx < len(mitsuba_scene_dict['train'].pose_list) else idx-len(mitsuba_scene_dict['train'].pose_list)

        '''
        ipdb> scale_mat
        array([[4.4205, 0.    , 0.    , 0.4055],
            [0.    , 4.4205, 0.    , 1.3835],
            [0.    , 0.    , 4.4205, 0.8865],
            [0.    , 0.    , 0.    , 1.    ]], dtype=float32)

        ipdb> pose
        array([[ 611.8527,  -54.7337,  179.557 ,  605.9015],
            [  67.7917, -860.183 ,  -65.4927,  970.0316],
            [   0.7086,   -0.171 ,   -0.6846,    0.4461],
            [   0.    ,    0.    ,    0.    ,    1.    ]], dtype=float32)
        '''
        idx += 1

    cameras['center'] = center
    cameras['scale'] = scale
    #np.savez(os.path.join(out_path, "cameras_sphere.npz"), **cameras)
    np.savez(os.path.join(out_path, "cameras.npz"), **cameras)

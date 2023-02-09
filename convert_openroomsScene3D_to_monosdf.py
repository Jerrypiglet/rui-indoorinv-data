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
OR_RAW_ROOT = {
    'apple': '/Users/jerrypiglet/Documents/Projects/data', 
    'mm1': '/newfoundland2/ruizhu/siggraphasia20dataset', 
    'qc': '', 
}[host]
INV_NERF_ROOT = {
    'apple': '/Users/jerrypiglet/Documents/Projects/inv-nerf', 
    'mm1': '/home/ruizhu/Documents/Projects/inv-nerf', 
    'qc': '', 
}[host]
from pathlib import Path
import numpy as np
np.set_printoptions(suppress=True)

from lib.class_openroomsScene3D import openroomsScene3D

semantic_labels_root = Path(PATH_HOME) / 'files_openrooms'
layout_root = Path(OR_RAW_ROOT) / 'layoutMesh'
shapes_root = Path(OR_RAW_ROOT) / 'uv_mapped'
envmaps_root = Path(OR_RAW_ROOT) / 'EnvDataset' # not publicly availale
shape_pickles_root = Path(PATH_HOME) / 'data/openrooms_shape_pickles' # for caching shape bboxes so that we do not need to load meshes very time if only bboxes are wanted

postfix = ''

dataset_version = 'public_re_3_v3pose_2048'
meta_split = 'main_xml'
scene_name = 'scene0008_00_more'
emitter_type_index_list = [('lamp', 0)]
# frame_ids = list(range(0, 345, 1))
# frame_ids = [0]
# radiance_scale = 0.001

dataset_version = 'public_re_0203'
meta_split = 'mainDiffLight_xml1'
scene_name = 'scene0552_00'
frame_ids = list(range(200))
radiance_rescale = 1./20.

dataset_version = 'public_re_0203'
meta_split = 'main_xml'
scene_name = 'scene0005_00'
frame_ids = list(range(200))
# radiance_rescale = 1./2.
postfix = '_darker'; radiance_rescale = 1./10.

'''
The classroom scene: one lamp (lit up) + one window (less sun)
data/public_re_0203/main_xml1/scene0552_00/im_4.png
'''
dataset_version = 'public_re_0203'
meta_split = 'main_xml1'
scene_name = 'scene0552_00'
frame_ids = list(range(200))
# radiance_rescale = 1./5. # RE
postfix = '_darker'; radiance_rescale = 1./20.

# '''
# orange-ish room with direct light
# data/public_re_0203/main_xml/scene0002_00/im_4.png
# images/demo_eval_scene_shapes-vis_count-train-public_re_0203_main_xml_scene0002_00.png
# '''
# dataset_version = 'public_re_0203'
# meta_split = 'main_xml'
# scene_name = 'scene0002_00'
# frame_ids = list(range(200))
# radiance_rescale = 1.

# '''
# green-ish room with window, with guitar (vert noisy)
# data/public_re_0203/mainDiffMat_xml1/scene0608_01/im_1.png
# images/demo_eval_scene_shapes-vis_count-train-public_re_0203_mainDiffMat_xml1_scene0608_01.png
# '''
# dataset_version = 'public_re_0203'
# meta_split = 'mainDiffMat_xml1'
# scene_name = 'scene0608_01'
# frame_ids = list(range(200))
# radiance_rescale = 5.

# '''
# toy room with lit lamp and dark window
# data/public_re_0203/mainDiffMat_xml/scene0603_00/im_46.png
# images/demo_eval_scene_shapes-vis_count-train-public_re_0203_mainDiffMat_xml_scene0603_00.png
# '''
# dataset_version = 'public_re_0203'
# meta_split = 'mainDiffMat_xml'
# scene_name = 'scene0603_00'
# frame_ids = list(range(200))
# radiance_rescale = 1./2.
# postfix = '_darker'; radiance_rescale = 1./10.

scan_id = 'scan1'

base_root = Path(PATH_HOME) / 'data' / dataset_version
xml_root = Path(PATH_HOME) / 'data' / dataset_version / 'scenes'

openrooms_scene = openroomsScene3D(
    if_debug_info=False, 
    host=host, 
    root_path_dict = {'PATH_HOME': Path(PATH_HOME), 'rendering_root': base_root, 'xml_scene_root': xml_root, 'semantic_labels_root': semantic_labels_root, 'shape_pickles_root': shape_pickles_root, 
        'layout_root': layout_root, 'shapes_root': shapes_root, 'envmaps_root': envmaps_root}, 
    scene_params_dict={
        'meta_split': meta_split, 
        'scene_name': scene_name, 
        'frame_id_list': frame_ids, 
        'up_axis': 'y+', 
        }, 
    # modality_list = ['im_sdr', 'im_hdr', 'seg', 'poses', 'albedo', 'roughness', 'depth', 'normal', 'lighting_SG', 'lighting_envmap'], 
    modality_list = [
        'im_sdr', 
        'poses', 
        'seg', 
        'im_hdr', 
        # 'albedo', 'roughness', 
        # 'depth', 
        # 'normal', 
        # 'lighting_SG', 
        # 'lighting_envmap', 
        # 'layout', 
        # 'shapes', # objs + emitters, geometry shapes + emitter properties
        'mi', # mitsuba scene, loading from scene xml file
        ], 
    modality_filename_dict = {
        # 'poses', 
        'im_hdr': 'im_%d.hdr', 
        'im_sdr': 'im_%d.png', 
        # 'lighting_envmap', 
        'albedo': 'imbaseColor_%d.png', 
        'roughness': 'imroughness_%d.png', 
        'depth': 'imdepth_%d.dat', 
        'normal': 'imnormal_%d.png', 
        # 'lighting_SG', 
        # 'layout', 
        # 'shapes', # objs + emitters, geometry shapes + emitter properties
    }, 
    im_params_dict={
        'im_H_load': 480, 'im_W_load': 640, 
        'im_H_resize': 480, 'im_W_resize': 640, 
        # 'im_H_resize': 240, 'im_W_resize': 320, 
        # 'im_H_resize': 120, 'im_W_resize': 160, # to use for rendering so that im dimensions == lighting dimensions
        # 'im_H_resize': 6, 'im_W_resize': 8, # to use for rendering so that im dimensions == lighting dimensions
        'if_direct_lighting': False, # if load direct lighting envmaps and SGs inetad of total lighting
        # 'im_hdr_ext': 'hdr', 
        }, 
    lighting_params_dict={
        'SG_num': 12,
        'env_row': 120, 'env_col': 160, # resolution to load; FIXED
        'env_downsample_rate': 20, # (120, 160) -> (6, 8)

        # 'env_height': 8, 'env_width': 16,
        'env_height': 16, 'env_width': 32, 
        # 'env_height': 128, 'env_width': 256, 
        # 'env_height': 64, 'env_width': 128, 
        # 'env_height': 2, 'env_width': 4, 
        
        'if_convert_lighting_SG_to_global': True, 
        'if_use_mi_geometry': True, 
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
        'if_remesh': True, # False: images/demo_shapes_3D_NO_remesh.png; True: images/demo_shapes_3D_YES_remesh.png
        'remesh_max_edge': 0.15,  
        },
    emitter_params_dict={
        'N_ambient_rep': '3SG-SkyGrd', 
        },
    mi_params_dict={
        # 'if_also_dump_xml_with_lit_area_lights_only': True,  # True: to dump a second file containing lit-up lamps only
        'debug_dump_mesh': True, # [DEBUG] True: to dump all object meshes to mitsuba/meshes_dump; load all .ply files into MeshLab to view the entire scene: images/demo_mitsuba_dump_meshes.png
        'debug_render_test_image': False, # [DEBUG][slow] True: to render an image with first camera, usig Mitsuba: images/demo_mitsuba_render.png
        'if_sample_rays_pts': True, # True: to sample camera rays and intersection pts given input mesh and camera poses
        'if_get_segs': False, # True: to generate segs similar to those in openroomsScene2D.load_seg()
        },
)


'''
CONVERT
'''
dump_scene_name = '-'.join([dataset_version, meta_split, scene_name]) + '_rescaledSDR' + postfix
out_path_prefix = str(base_root / 'monosdf' / dump_scene_name)
scenes = [dump_scene_name]
out_names = [scan_id]

for scene, out_name in zip(scenes, out_names):
    out_path = os.path.join(out_path_prefix, out_name)
    os.makedirs(out_path, exist_ok=True)
    print('====>', out_path)

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
    camera_intrinsic = openrooms_scene.K # (3, 4)
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
    poses = openrooms_scene.pose_list # [(3, 4)]
    poses = [np.vstack((pose, np.array([0., 0., 0., 1.], dtype=np.float32).reshape((1, 4)))) for pose in poses]
    poses = np.array(poses)

    # deal with invalid poses
    # valid_poses = np.isfinite(poses).all(axis=2).all(axis=1)
    min_vertices = poses[:, :3, 3].min(axis=0)
    max_vertices = poses[:, :3, 3].max(axis=0)
 
    center = (min_vertices + max_vertices) / 2.
    scale = 2. / (np.max(max_vertices - min_vertices) + 3.)
    print(center, scale)

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
    print(K)
    
    from tqdm import tqdm
    import shutil
    from PIL import Image
    import mitsuba as mi
    mi.set_variant('llvm_ad_rgb')

    assert openrooms_scene.hdr_scale_list.count(openrooms_scene.hdr_scale_list[0]) == len(openrooms_scene.hdr_scale_list)

    for idx, pose in tqdm(enumerate(poses)):
        # print(idx, valid)
        # if idx % 10 != 0: continue
        # if not valid : continue
        # ==== HDR
        target_image_hdr = os.path.join(out_path, "image/%06d.exr"%(idx))
        # shutil.copy(openrooms_scene.im_hdr_file_list[idx], target_image_hdr)
        im_hdr_liwenScale = openrooms_scene.im_hdr_list[idx] / openrooms_scene.hdr_scale_list[idx] * radiance_rescale # original max (300.) / 5. -> max ~= 60
        mi.util.write_bitmap(str(target_image_hdr), im_hdr_liwenScale)

        # ==== SDR
        target_image = os.path.join(out_path, "image/%06d.png"%(idx))
        # shutil.copy(openrooms_scene.im_sdr_file_list[idx], target_image)

        # option 2: rescale and convert from HDR
        _ = np.clip((im_hdr_liwenScale) ** (1./2.2), 0., 1.)
        img = Image.fromarray((_*255.).astype(np.uint8))
        img.save(target_image)


        # import matplotlib.pyplot as plt
        # import ipdb; ipdb.set_trace()
        # _ = (im_hdr_liwenScale/5.) ** (1./2.2)

        # x = im_hdr_liwenScale / 2.
        # mask = x <= 0.0031308
        # # mask = x <= 0.05
        # ret = np.zeros_like(x)
        # ret[mask] = 12.92*x[mask]
        # mask = ~mask
        # ret[mask] = 1.055*(x[mask] ** (1/2.4)) - 0.055
        # _ = ret

        # plt.figure()
        # plt.imshow(np.clip(_, 0., 1.))
        # plt.show()
        # import ipdb; ipdb.set_trace()

        # print(target_image)
        # img = Image.open(image_path)
        # img_tensor = trans_totensor(img)
        # img_tensor.save(target_image)

        mask = (np.ones((openrooms_scene.H, openrooms_scene.W, 3)) * 255.).astype(np.uint8) # [TODO] load masks from mi!
        target_image = os.path.join(out_path, "mask/%03d.png"%(idx))
        cv2.imwrite(target_image, mask)

        # load depth
        target_image = os.path.join(out_path, "depth/%06d.png"%(idx))
        # depth = cv2.imread(depth_path, -1).astype(np.float32) / 1000.
        # depth_PIL = Image.fromarray(depth)
        # new_depth = depth_trans_totensor(depth_PIL)
        # new_depth = np.asarray(new_depth)
        new_depth = openrooms_scene.mi_depth_list[idx].astype(np.float32) / 1000.
        new_depth[np.isnan(new_depth)] = 1./1000.
        new_depth[np.isinf(new_depth)] = 1./1000.

        plt.imsave(target_image, new_depth, cmap='viridis')
        np.save(target_image.replace(".png", ".npy"), new_depth)

        mi_normal_global = openrooms_scene.mi_normal_global_list[idx].astype(np.float32)
        R = pose[:3, :3]
        mi_normal_cam = (R.T @ mi_normal_global.reshape(-1, 3).T).T.reshape(openrooms_scene.H, openrooms_scene.W, 3) # images/demo_normal_gt_scannetScene_opencv.png
        mi_normal_cam[np.isnan(mi_normal_cam)] = 0.
        mi_normal_cam[np.isinf(mi_normal_cam)] = 0.

        np.save(os.path.join(out_path, "normal/%06d.npy"%(idx)), (mi_normal_cam.transpose(2, 0, 1)+1.)/2.) # (3, H, W), [0., 1.]

        _ = Image.fromarray(((mi_normal_cam+1.)/2.*255.).astype(np.uint8))
        _.save(os.path.join(out_path, "normal/%06d.png"%(idx)))
        
        # save pose
        pcds.append(pose[:3, 3])
        pose = K @ np.linalg.inv(pose)
        
        #cameras["scale_mat_%d"%(idx)] = np.eye(4).astype(np.float32)
        cameras["scale_mat_%d"%(idx)] = scale_mat
        cameras["world_mat_%d"%(idx)] = pose
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

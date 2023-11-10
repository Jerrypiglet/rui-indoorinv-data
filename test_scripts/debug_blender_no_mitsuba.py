'''
Script to load Blender scenes, render with Blender (bpy) and fuse rendered RGB images onto a TSDF volume-bases mesh, to check multi-view consistency of the renderings.

This version is self-conatined, i.e. no import mitsuba or modules in this project.

> git clone --branch add_blender_debug https://github.com/Jerrypiglet/rui-indoorinv-data.git
> cd rui-indoorinv-data

Orgalize the folder structure as:

- {DATASET_ROOT}
    - debug_scenes/
        - cornel_box/ # export cameras from .blend file via: python test_scripts/dump_cam_from_blender_file.py
            - test.blend # download from https://dsc.cloud/jerrypiglet/test.blend
    
We assume camera intrinsics have been set in .blend file, and multiple cameras are set in the scene, with names 'Camera0', 'Camera1', etc.

Run: 

> python debug_blender_no_mitsuba.py --spp 32 --cycles_device GPU --compute_device_type METAL --scene cornell_box_allblender

Output:

- Rendered images in: debug_scenes/kitchen_diy/Image/*.exr
- Fused TSDF shape at: debug_scenes/kitchen_diy/fused_tsdf.ply
- Exported .blend files for each frame: debug_scenes/kitchen_diy/test_frame_{}.blend

Options:

- Set --spp properly for Blender/Mitsuba renderings to reduce noise level (e.g. spp>=128 for Blender; spp>=512 for Mitsuba)

'''

from pathlib import Path
import numpy as np
np.set_printoptions(suppress=True)
from scipy.spatial.transform import Rotation
import bpy
from tqdm import tqdm

'''
envs and functions
'''

DATASET_ROOT = Path('/Users/jerrypiglet/Documents/Projects/rui-indoorinv-data/data') # change this

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('True', 'yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('False', 'no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expectedl; got: %s'%v)
    
def R_t_to_origin_lookatvector_up_opencv(R, t):
    origin = t
    lookatvector = R @ np.array([[0.], [0.], [1.]], dtype=np.float32)
    up = R @ np.array([[0.], [-1.], [0.]], dtype=np.float32)
    return (origin, lookatvector, up)


import argparse
parser = argparse.ArgumentParser()

# renderer paremeters
parser.add_argument('--spp', type=int, default=128, help='spp for mi/blender')
parser.add_argument('--cycles_device', type=str, default='GPU', help='CPU, GPU')
parser.add_argument('--compute_device_type', type=str, default='CUDA', help='CUDA, OPTIX, METAL')

# === after refactorization
parser.add_argument('--DATASET', type=str, default='debug_scenes', help='load conf file: confs/\{DATASET\}')
parser.add_argument('--scene', type=str, default='kitchen_diy', help='load conf file: confs/\{DATASET\/\{opt.scene\}.conf')

opt = parser.parse_args()

assert DATASET_ROOT.exists(), 'DATASET_ROOT: %s does not exist!'%DATASET_ROOT

scene_path = DATASET_ROOT / opt.DATASET / opt.scene
assert scene_path.exists(), 'scene_path: %s does not exist!'%scene_path
blend_file_path = scene_path / 'test.blend'
assert blend_file_path.exists(), 'blend_file_path: %s does not exist!'%blend_file_path

'''
Load cameras from .blend file
'''

bpy.ops.wm.open_mainfile(filepath=str(blend_file_path))

bpy.context.scene.render.image_settings.file_format = 'OPEN_EXR'
bpy.context.scene.render.image_settings.color_depth = str(16)

im_H, im_W = bpy.context.scene.render.resolution_y, bpy.context.scene.render.resolution_x

pose_list = []
K_list = []
# origin_lookatvector_up_list = []
blender_poses = []

for cam in bpy.data.objects:
    if not cam.name.startswith('Camera'): continue # assuming cameras are labelled as Camera{id}, e.g. Camera0, Camera1, etc.
    # https://docs.blender.org/api/current/bpy.types.Camera.html
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
    blender_poses.append((pos.copy(), angles.copy()))
    
    '''
    convert to OpenCV poses
    '''
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
    
    pose_list.append((Rs.copy(), pos.copy()))
    
    # (origin, lookatvector, up) = R_t_to_origin_lookatvector_up_opencv(Rs, pos.reshape((3, 1)))
    # print('origin', origin, 'lookatvector', lookatvector, 'up', up)
    
    # assert np.abs(np.linalg.norm(up)) - 1. < 1e-5, 'up is not unit vector!'
    # assert np.abs(np.linalg.norm(lookatvector)) - 1. < 1e-5, 'lookatvector is not unit vector!'
    
    # origin_lookatvector_up_list.append((origin, lookatvector, up))
    
print('%d poses found!'%len(pose_list))

K_list_diff = np.array(K_list) - K_list[0]
assert np.amax(K_list_diff) < 1e-5, 'K_list_diff: %s is not all identical!'%str(K_list_diff)

POSE_NUM = len(pose_list)

'''
set up renderer parameters
'''

bpy.context.scene.render.use_motion_blur = False
bpy.context.scene.view_layers[0].cycles.use_denoising = True
# bpy.context.scene.view_layers[0].cycles.use_denoising = False
bpy.context.scene.cycles.samples = opt.spp
bpy.context.scene.view_settings.view_transform = 'Standard'

bpy.context.scene.render.dither_intensity = 0.0
# bpy.context.scene.render.film_transparent = True
bpy.context.scene.render.resolution_percentage = 100
# bpy.context.scene.render.threads = 16
bpy.context.scene.render.views_format = 'MULTIVIEW'

'''
configure render engine and device
'''
print("---------------------------------------------- bpy.app.version_string", bpy.app.version_string)
print('setting up gpu/metal ......')

bpy.context.scene.render.engine = 'CYCLES'
bpy.context.scene.cycles.device = opt.cycles_device
# if opt.cycles_device == 'GPU':
#     bpy.context.scene.cycles.denoiser = 'OPTIX'

bpy.context.preferences.addons["cycles"].preferences.compute_device_type = opt.compute_device_type
bpy.context.preferences.addons["cycles"].preferences.get_devices()
print('==== compute_device_type: ', bpy.context.preferences.addons["cycles"].preferences.compute_device_type)

for d in bpy.context.preferences.addons["cycles"].preferences.devices:
    d.use = True
    if d.type == 'CPU':
        d.use = False
    print("Device '{}' type {} :".format(d.name, d.type), str(d.use) if d.use else str(d.use))
print('setting up gpu/metal done')
print("----------------------------------------------")

'''
modalities: deal with aov modalities
'''

_modality_list = [
    'im', 
    'albedo', 
    'roughness', 
    'depth', 
    'normal', 
    'index', 
    'emission', 
    ]

_modality_folder_maping = {
    'im': 'Image', 
    'depth': 'Depth',
    'normal': 'Normal',
    # 'Alpha',
    # 'IndexOB',
    'albedo': 'DiffCol',
    # 'GlossCol',
    'index': 'IndexMA', 
    'emission': 'Emit', 
    'roughness': 'Roughness', 
    'lighting_envmap': 'LightingEnvmap', 
    # 'Metallic'
    }

AOV_MODALS = []
if 'roughness' in _modality_list:
    AOV_MODALS.append('Roughness') #,'Metallic'
# assigne material id
for i, mat in enumerate(bpy.data.materials):
    mat.pass_index=(i+1)
# link aov output
for aov_modal in AOV_MODALS:
    for mat in bpy.data.materials:
        tree = mat.node_tree
        if tree is None:
            continue
        for node in tree.nodes:
            if 'Bsdf' not in node.bl_idname:
                continue
            if (node.bl_idname =='ShaderNodeBsdfGlossy' or node.bl_idname =='ShaderNodeBsdfGlass') and node.distribution == 'SHARP' and aov_modal=='Roughness':
                modal_value = 0.0
                buffer_node = tree.nodes.new('ShaderNodeValue')
                from_socket = buffer_node.outputs['Value']
                from_socket.default_value = modal_value
            elif aov_modal not in node.inputs.keys():
                modal_value = 0.0
                buffer_node = tree.nodes.new('ShaderNodeValue')
                from_socket = buffer_node.outputs['Value']
                from_socket.default_value = modal_value
            else:
                socket = node.inputs[aov_modal]
                if len(socket.links) == 0:
                    if node.bl_idname == 'ShaderNodeBsdfDiffuse' and aov_modal == 'Roughness':
                        modal_value = 1.0
                    else:
                        modal_value = socket.default_value
                    buffer_node = tree.nodes.new('ShaderNodeValue')
                    from_socket = buffer_node.outputs['Value']
                    from_socket.default_value = modal_value
                    tree.links.new(from_socket,socket)
                else:
                    from_socket=socket.links[0].from_socket

            aov_node = tree.nodes.new('ShaderNodeOutputAOV')
            aov_node.name = aov_modal
            tree.links.new(from_socket,aov_node.inputs['Value'])


'''
camera and scene parameters, AOV
'''
# self.cam = scene.objects['Camera'] # the sensor in XML has to has 'id="Camera"'
# cam = bpy.context.scene.camera #scene.objects['Camera'] # self.cam.data.lens -> 31.17691421508789, in mm (not degrees)

# Add the main camera.
cam = bpy.data.objects.new('Camera', bpy.data.cameras.new('Camera'))
bpy.context.scene.collection.objects.link(cam)
bpy.context.scene.camera = cam
cam.data.clip_start = 0.1
cam.data.clip_end = 50.0
cam.data.type = 'PERSP'

obj_idx = 1
for obj in bpy.context.scene.objects:
    if obj.type in ('MESH'):
        obj.pass_index=obj_idx
        obj_idx += 1

bpy.context.scene.view_layers[0].use_pass_normal = True # "ViewLayer" not found: https://zhuanlan.zhihu.com/p/533843765
bpy.context.scene.view_layers[0].use_pass_object_index = True
bpy.context.scene.view_layers[0].use_pass_z = True
bpy.context.scene.view_layers[0].use_pass_material_index = True
bpy.context.scene.view_layers[0].use_pass_diffuse_color = True
bpy.context.scene.view_layers[0].use_pass_glossy_color = True
bpy.context.scene.view_layers[0].use_pass_emit = True
bpy.context.scene.view_layers[0].use_pass_position = True

bpy.context.scene.use_nodes = True

for aov_modal in AOV_MODALS:
    bpy.ops.scene.view_layer_add_aov()
    # bpy.context.scene.view_layers["ViewLayer"].aovs[-1].name = aov_modal
    # bpy.context.scene.view_layers["ViewLayer"].aovs[-1].type = "VALUE"
    bpy.context.scene.view_layers[0].aovs[-1].name = aov_modal
    bpy.context.scene.view_layers[0].aovs[-1].type = "VALUE"


# bpy.context.scene = bpy.data.scenes["Scene"]
tree = bpy.context.scene.node_tree
for n in tree.nodes:
    tree.nodes.remove(n)
render_layers = tree.nodes.new('CompositorNodeRLayers')

'''
render the scene with bpy
'''

modal_file_outputs = []
for modality in _modality_list:
    modal_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
    folder_name = _modality_folder_maping[modality]
    render_folder_path = scene_path / folder_name
    # if render_folder_path.exists():
        # render_folder_path.unlink()
    render_folder_path.mkdir(exist_ok=True, parents=True)
    modal_file_output.label = folder_name
    tree.links.new(render_layers.outputs[folder_name], modal_file_output.inputs[0])
    modal_file_output.base_path = str(render_folder_path)
    modal_file_outputs.append(modal_file_output)

im_rendering_folder = scene_path / 'Image'

for frame_idx in tqdm(range(POSE_NUM)):
    # t_c2w_b = (t_c2w_b - self.os.trans_m2b) / self.os.scale_m2b # convert to Mitsuba scene scale (to match the dumped Blender scene from Mitsuba)
    im_rendering_path = str(im_rendering_folder / ('%03d_0001'%frame_idx))
    bpy.context.scene.render.filepath = str(im_rendering_path)
    
    cam.location = blender_poses[frame_idx][0]
    cam.rotation_euler[0] = blender_poses[frame_idx][1][0].item()
    cam.rotation_euler[1] = blender_poses[frame_idx][1][1].item()
    cam.rotation_euler[2] = blender_poses[frame_idx][1][2].item()
    
    for modal_file_output in modal_file_outputs:
        modal_file_output.file_slots[0].path = '%03d'%frame_idx + '_'
        
    bpy.ops.render.render(write_still=True)  # render still image
    
    # [Optional] Save the blend file.
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=str(scene_path/ ('test_frame_%d.blend'%frame_idx)))
    
'''
fusing into TSDF volume using rendered RGB images + depths
'''

import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import open3d as o3d

intrinsic = o3d.camera.PinholeCameraIntrinsic(im_W, im_H, fx_pix, fy_pix, cx_pix, cy_pix)
poses = []
T_opengl_opencv = np.array([[-1., 0., 0.], [0., -1., 0.], [0., 0., 1.]], dtype=np.float32) # flip x, y: Liwen's new pose (left-up-forward) -> OpenCV (right-down-forward)

volume = o3d.pipelines.integration.ScalableTSDFVolume(
    voxel_length=8.0 / 512.0,
    sdf_trunc=0.05,
    color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    volume_unit_resolution=16,
    depth_sampling_stride=1, 
)

for p in tqdm(range(5)):
    for frame_idx in range(POSE_NUM):
    # for frame_idx in range(1):
        image_path = scene_path / 'Image' / ('%03d_0001.exr'%frame_idx)
        assert image_path.exists(), 'image_path: %s does not exist!'%image_path
        depth_path = scene_path / 'Depth' / ('%03d_0001.exr'%frame_idx)
        assert depth_path.exists(), 'depth_path: %s does not exist!'%depth_path
        
        im_hdr = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        im_hdr = cv2.cvtColor(im_hdr, cv2.COLOR_BGR2RGB)
        im_hdr = im_hdr.astype(np.float32)[:, :, :3]
        im_sdr = np.clip(im_hdr ** (1./2.2), 0, 1)
        
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)[:, :, 0]
        
        assert im_sdr.shape[:2] == depth.shape, 'im_sdr.shape: %s, depth.shape: %s'%(str(im_sdr.shape), str(depth.shape))

        # print('Loaded frame', frame_idx, im_sdr.shape, np.amax(depth), np.amin(depth), np.amax(im_hdr), np.amin(im_hdr),np.amax(im_sdr), np.amin(im_sdr))
        
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.clip(im_sdr * 255, 0, 255).astype(np.uint8)),
            o3d.geometry.Image(depth.copy()),
            depth_scale=1.0,
            depth_trunc=30.0,
            convert_rgb_to_intensity=False)
        
        R = pose_list[frame_idx][0].reshape((3, 3))
        t = pose_list[frame_idx][1].reshape((3, 1))
        
        extrinsic = np.vstack((np.hstack([R, t]), np.array([0, 0, 0, 1]))) # camera-to-world
        # print(f"extrinsic {extrinsic}")
        volume.integrate(rgbd_image, intrinsic, np.linalg.inv(extrinsic)) # takes the inverse: world-to-camera extrinsics
        
    if p == 0:
        curr_pose = np.concatenate([
            np.dot(extrinsic, np.array([0.0, 0.0, 0.0, 1.0]))[:3],
            np.dot(extrinsic, np.array([0.0, 0.0, 1.0, 1.0]))[:3],
            np.dot(extrinsic, np.array([0.0, 1.0, 0.0, 0.0]))[:3],
        ]).astype(np.float32)
        # print(f"curr_pose {curr_pose}")
        poses.append(curr_pose)

    

# import matplotlib.pyplot as plt
# plt.figure()
# plt.subplot(1, 2, 1)
# plt.imshow(im_sdr)
# plt.subplot(1, 2, 2)
# plt.imshow(depth / np.amax(depth))
# plt.colorbar()
# plt.show()
    
tsdf_mesh_o3d = volume.extract_triangle_mesh()
o3d.io.write_triangle_mesh(str(scene_path / 'fused_tsdf.ply'), tsdf_mesh_o3d, False, True)


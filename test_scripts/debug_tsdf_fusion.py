'''
fuse tsdf given images and poses shared by Sam
'''

from pathlib import Path
import numpy as np
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
from tqdm import tqdm
import open3d as o3d

def get_pixels(w: int, h: int) -> np.ndarray:
    x, y = np.meshgrid(np.arange(w) / (w - 1), np.arange(h) / (h - 1), indexing='xy')
    return np.vstack((x.reshape(-1), y.reshape(-1))).astype(np.float32).T

def pixels_to_rays(pixels: np.ndarray, f: float, cx: float, cy: float, w: int, h: int, normalize: bool) -> np.ndarray:
    uv = (pixels - (cx, cy)) / f
    uv[..., 1] *= h / w
    ray = np.concatenate((uv, np.ones((uv.shape[0], 1), dtype=uv.dtype)), axis=-1)
    return ray if not normalize else ray / np.linalg.norm(ray, axis=-1, keepdims=True)

rendering_root = Path('/Users/jerrypiglet/Documents/TMP')
images = sorted((rendering_root / 'output').glob('color*.png'))
depths = sorted((rendering_root / 'output').glob('depth*.exr'))

calibration_file = rendering_root / 'calibration.npz'
calibration_dict = np.load(str(calibration_file))
rotations = calibration_dict['rotations']
translations = calibration_dict['translations']
f = calibration_dict['f']
cx = calibration_dict['cx']
cy = calibration_dict['cy']

im_sdr_list = []
depth_list = []

points_list = []
colors_list = []
assert len(images) == len(depths) == len(rotations) == len(translations)
print('Loading %d frames...'%len(images))

for image_path, depth_path, rotation, translation in tqdm(zip(images, depths, rotations, translations)):
    # Load the color.
    color = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    w = color.shape[1]
    h = color.shape[0]
    # Load the depth (camera-space z, not actual depth).
    zdepth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    zdepth = zdepth[:, :, 0]
    # Unproject to world space. We don't normalize the rays because:
    # camera_point = ray / |ray| * depth = ray * z
    pixels = get_pixels(w, h)
    rays = pixels_to_rays(pixels, f, cx, cy, w, h, normalize=False)
    camera_points = rays * zdepth.reshape(-1, 1)
    world_points = camera_points @ rotation.T + translation
    idx = np.random.choice(world_points.shape[0], 200000)
    points_list.append(world_points[idx])
    colors_list.append(color.reshape(-1, 3)[idx])
    
    im_sdr_list.append(color.astype(np.float32) / 255.)
    depth = camera_points[..., 2].reshape(h, w)
    depth_list.append(depth)
    
    
'''
Save all points to an obj file.
'''

# all_points = np.concatenate(points_list, axis=0)
# all_colors = np.concatenate(colors_list, axis=0)
# obj_file = rendering_root / 'points.obj'
# with open(str(obj_file), 'w') as f:
#     for point, color in zip(all_points, all_colors):
#         f.write(f'v {point[0]} {point[1]} {point[2]} {color[0]} {color[1]} {color[2]}\n')

'''
fuse tsdf
'''
H, W = 1080, 1920
assert H == h
assert W == w

intrinsic = o3d.camera.PinholeCameraIntrinsic(W, H, W*f, W*f, W*cx, H*cy)
# assert len(depth_path_list) == len(rotations), 'len(depth_path_list): %d != len(rotations): %d'%(len(depth_path_list), len(rotations))
# assert len(depth_path_list) == len(translations), 'len(depth_path_list): %d != len(translations): %d'%(len(depth_path_list), len(translations))

volume = o3d.pipelines.integration.ScalableTSDFVolume(
    voxel_length=8.0 / 512.0,
    sdf_trunc=0.05,
    color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    volume_unit_resolution=16,
    depth_sampling_stride=1, 
)

for p in tqdm(range(5)):
    # for idx, depth_path in enumerate(depth_path_list[0:5]):
    assert len(im_sdr_list) == len(depth_list) == len(rotations) == len(translations)
    for im_sdr, depth, rotation, translation in tqdm(zip(im_sdr_list, depth_list, rotations, translations)):
        
        frame_id = int(depth_path.stem.replace('depth', ''))
        
        assert im_sdr.shape[:2] == depth.shape, 'im_sdr.shape: %s, depth.shape: %s'%(str(im_sdr.shape), str(depth.shape))

        # print('fusing frame_id:', frame_id, im_sdr.shape, np.amax(depth), np.amin(depth))
        
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.clip(im_sdr * 255, 0, 255).astype(np.uint8)),
            o3d.geometry.Image(depth.copy()),
            depth_scale=1.0,
            depth_trunc=10.0,
            convert_rgb_to_intensity=False)
        
        # rotation = rotations[idx].reshape((3, 3))
        # translation = translations[idx].reshape((3, 1))
        # rotation = np.eye(3, dtype=np.float32)
        # translation = np.zeros((3, 1), dtype=np.float32)
        
        # T_opengl_opencv = np.array([[-1., 0., 0.], [0., -1., 0.], [0., 0., 1.]], dtype=np.float32) # flip x, y: Liwen's new pose (left-up-forward) -> OpenCV (right-down-forward)
        # rotation = rotation @ T_opengl_opencv
        
        # rotation = np.linalg.inv(rotation)
        # translation = -np.linalg.inv(rotation) @ translation
        
        extrinsic = np.vstack((np.hstack([rotation, translation.reshape((3,1))]), np.array([0, 0, 0, 1]))) # camera-to-world
        volume.integrate(rgbd_image, intrinsic, np.linalg.inv(extrinsic)) # takes the inverse: world-to-camera extrinsics

# import matplotlib.pyplot as plt
# plt.figure()
# plt.subplot(1, 2, 1)
# plt.imshow(im_sdr)
# plt.subplot(1, 2, 2)
# plt.imshow(depth / np.amax(depth))
# plt.colorbar()
# plt.show()
    
tsdf_mesh_o3d = volume.extract_triangle_mesh()
o3d.io.write_triangle_mesh(str(rendering_root / 'tmp_fused_tsdf.ply'), tsdf_mesh_o3d, False, True)
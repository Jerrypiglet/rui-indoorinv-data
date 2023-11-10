from pathlib import Path
import numpy as np
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import open3d as o3d

rendering_root = Path('/Users/jerrypiglet/Documents/TMP')
image_root = rendering_root / 'output'
depth_path_list = [_ for _ in image_root.glob('*.exr') if _.stem.startswith('depth')]

pose_path = rendering_root / 'calibration.npz'
assert pose_path.exists(), 'pose_path: %s does not exist!'%pose_path
pose_dict = np.load(str(pose_path))
rotations = pose_dict['rotations']
translations = pose_dict['translations']
f = pose_dict['f'].item()
cx = pose_dict['cx'].item()
cy = pose_dict['cy'].item()
H, W = 1080, 1920

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

for idx, depth_path in enumerate(depth_path_list[0:5]):
    frame_id = int(depth_path.stem.replace('depth', ''))
    image_path = image_root / ('color%04d.png'%frame_id)
    assert image_path.exists(), 'image_path: %s does not exist!'%image_path
    
    im_sdr = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    im_sdr = cv2.cvtColor(im_sdr, cv2.COLOR_BGR2RGB)
    im_sdr = im_sdr.astype(np.float32) / 255.
    
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)[:, :, 0]
    
    assert im_sdr.shape[:2] == depth.shape, 'im_sdr.shape: %s, depth.shape: %s'%(str(im_sdr.shape), str(depth.shape))

    print('frame_id', frame_id, im_sdr.shape, np.amax(depth), np.amin(depth))
    
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(np.clip(im_sdr * 255, 0, 255).astype(np.uint8)),
        o3d.geometry.Image(depth.copy()),
        depth_scale=1.0,
        depth_trunc=10.0,
        convert_rgb_to_intensity=False)
    
    rotation = rotations[idx].reshape((3, 3))
    translation = translations[idx].reshape((3, 1))
    # rotation = np.eye(3, dtype=np.float32)
    # translation = np.zeros((3, 1), dtype=np.float32)
    
    # T_opengl_opencv = np.array([[-1., 0., 0.], [0., -1., 0.], [0., 0., 1.]], dtype=np.float32) # flip x, y: Liwen's new pose (left-up-forward) -> OpenCV (right-down-forward)
    # rotation = rotation @ T_opengl_opencv
    
    # rotation = np.linalg.inv(rotation)
    # translation = -np.linalg.inv(rotation) @ translation
    
    extrinsic = np.vstack((np.hstack([rotation, translation]), np.array([0, 0, 0, 1])))
    print(f"extrinsic {extrinsic}")
    volume.integrate(rgbd_image, intrinsic, extrinsic)

# import matplotlib.pyplot as plt
# plt.figure()
# plt.subplot(1, 2, 1)
# plt.imshow(im_sdr)
# plt.subplot(1, 2, 2)
# plt.imshow(depth / np.amax(depth))
# plt.colorbar()
# plt.show()
    
tsdf_mesh_o3d = volume.extract_triangle_mesh()
o3d.io.write_triangle_mesh(str(rendering_root / 'tmp.ply'), tsdf_mesh_o3d, False, True)
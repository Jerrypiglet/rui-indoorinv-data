import numpy as np
import torch

def fuse_depth_pcd_tr(depth_list, pose_list, hwf, subsample_rate=1):
    assert len(depth_list) == len(pose_list)
    H, W = depth_list[0].shape[:2]
    assert H == hwf[0] and W == hwf[1]
    f = hwf[2]
    device = depth_list[0].device

    uu, vv = torch.meshgrid(
        torch.arange(0, W, device=device), 
        torch.arange(0, H, device=device))
    uu = uu.t().float()
    vv = vv.t().float()

    X_global_list = []

    for idx in range(len(depth_list)):
        z_ = depth_list[idx].flatten()[::subsample_rate]
        z_valid_mask = z_ > 0
        z_ = z_[z_valid_mask]

        x_ = (uu - W/2.) * depth_list[idx] / f
        y_ = (vv - H/2.) * depth_list[idx] / f
        x_ = x_.flatten()[::subsample_rate][z_valid_mask]
        y_ = y_.flatten()[::subsample_rate][z_valid_mask]
        X_ = torch.stack([x_, y_, z_], dim=1)
        t = pose_list[idx][:3, -1].reshape((3, 1)).to(device)
        R = pose_list[idx][:3, :3].to(device)

        X_global = (R @ X_.T + t).T

        X_global_list.append(X_global)

    return torch.cat(X_global_list)

from pathlib import Path
import pickle
from lib.utils_io import read_cam_params, normalize_v

def load_OR_public_poses_to_Rt(transforms: np.ndarray, scene_xml_dir: Path, frame_id_list: list, if_inverse_y: bool=False, if_1_based: bool=True):
    '''
    load OpenRooms public pose files (cam.txt and transform.dat[NOT DOING THIS]) and convert to list of per-frame R, t

    # [NOT DOING THIS] Strip R and t of layout-specific transformations (basically transforming/normalizing all objects with a single transformation, for rendering purposes)
    # ↑ layout and objects are transformed independently from their RAW coords as loaded from mesh, to the world coords. 
    # ... Transformations for layout is in transforms[0] while transformations for objects is in their 'transform' as desginated in the XML file (reading the transformations: see parse_XML_for_shapes()->shape_dict['transforms_list')
    # ... as a result, here we DO NOT transform cameras according to the layout transformations.

    if_1_based: openrooms_public is 1-based, while rendering and openrooms_public_re is 0-based
    '''
    assert if_inverse_y == False, 'not handling if_inverse_y=True for now'

    # scale_scene = transforms[0][0][1].reshape((3, 1)) # (3,1)
    # rotMat_scene = transforms[0][1][1] # (3, 3)
    # rotMat_inv_scene = np.linalg.inv(rotMat_scene)
    # trans_scene = transforms[0][2][1].reshape((3, 1)) # (3,1)

    cam_file = scene_xml_dir / 'cam.txt'
    cam_params = read_cam_params(cam_file)

    pose_list = []
    origin_lookatvector_up_list = []

    for frame_id in frame_id_list:
        if if_1_based:
            cam_param = cam_params[frame_id-1]
        else:
            cam_param = cam_params[frame_id]
        origin, lookat, up = np.split(cam_param.T, 3, axis=1)
        
        # origin -= trans_scene
        # lookat -= trans_scene

        # origin = 1./(scale_scene) * (rotMat_inv_scene @ origin)
        # lookat = 1./(scale_scene) * (rotMat_inv_scene @ lookat)
        # up = 1./(scale_scene) * (rotMat_inv_scene @ up)

        origin = origin.flatten()
        lookat = lookat.flatten()
        up = up.flatten()

        at_vector = normalize_v(lookat - origin)
        assert np.amax(np.abs(np.dot(at_vector.flatten(), up.flatten()))) < 2e-3 # two vector should be perpendicular

        t = origin.reshape((3, 1)).astype(np.float32)
        R = np.stack((np.cross(-up, at_vector), -up, at_vector), -1).astype(np.float32)
        # R = R @ np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        
        pose_list.append(np.hstack((R, t)))
        origin_lookatvector_up_list.append((origin.reshape((3, 1)), at_vector.reshape((3, 1)), up.reshape((3, 1))))

    return pose_list, origin_lookatvector_up_list


def convert_SG_angle_to_axis_np(lighting_SG_local_angles):
    '''
    lighting_SG_local_angles: (H, W, SG_num, 2)
    normal: (H, W, 3), in camera coordinates, OpenGL convention
    '''
    theta, phi = np.split(lighting_SG_local_angles, 2, axis=3) # (H, W, SG_num, 1), (H, W, SG_num, 1)
    axisX = np.sin(theta) * np.cos(phi)
    axisY = np.sin(theta) * np.sin(phi)
    axisZ = np.cos(theta)
    axis_local = np.concatenate([axisX, axisY, axisZ], axis=3) # [H, W, 12, 3]; in a local SG (self.ls) coords
    axis_local = axis_local / (np.linalg.norm(axis_local, axis=3, keepdims=True)+1e-6)
    return axis_local

def convert_SG_axis_local_global_np(rL, lighting_params, lighting_SG_local, pose, normal_opengl):
    '''
    split_lighting_SG_local: (H, W, SG_num, 6)
    normal: (H, W, 3), in camera coordinates, OpenGL convention
    '''

    normal_torch = torch.from_numpy(normal_opengl).unsqueeze(0).permute(0, 3, 1, 2) # (1, 3, H, W)
    camx, camy, normalPred = rL.forwardEnv(normal_torch, None, if_normal_only=True) # torch.Size([B, 128, 3, 120, 160]), [B, 3, 120, 160], [B, 3, 120, 160], [B, 3, 120, 160]
    axis_local_SG_flattened = lighting_SG_local[:, :, :, :3].reshape(-1, lighting_params['SG_num'], 3) # (HW, SG_num, 3)
    T_flattened = torch.cat((camx, camy, normalPred), axis=0).permute(2, 3, 0, 1).flatten(0, 1).cpu().numpy() # (HW, 3, 3)
    axis_SG_np = axis_local_SG_flattened @ T_flattened # (HW, SG_num, 3)

    axis_SG_np = np.stack([axis_SG_np[:, :, 0], -axis_SG_np[:, :, 1], -axis_SG_np[:, :, 2]], axis=-1) # transform axis from opengl convention (right-up-backward) to opencv (right-down-forward)
    
    R = pose[:3, :3]
    axis_SG_np_global = axis_SG_np @ (R.T)
    axis_SG_np_global = axis_SG_np_global.reshape(lighting_params['env_row'], lighting_params['env_col'], lighting_params['SG_num'], 3)

    lighting_SG = np.concatenate((axis_SG_np_global, lighting_SG_local[:, :, :, 3:]), axis=3) # (120, 160, 12(SG_num), 7); axis, lamb, weight: 3, 1, 3

    return lighting_SG
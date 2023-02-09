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

def load_OR_public_poses_to_Rt(scene_xml_dir: Path, frame_id_list: list, if_inverse_y: bool=False, if_1_based: bool=True):
    '''
    load OpenRooms public pose files (cam.txt and transform.dat) and transform to R, t

    if_1_based: openrooms_public is 1-based, while rendering and openrooms_scene_dataset is 0-based
    '''
    assert if_inverse_y == False, 'not handling if_inverse_y=True for now'
    transformFile = scene_xml_dir / 'transform.dat'
    with open(transformFile, 'rb') as fIn:
        transforms = pickle.load(fIn)
    scale_scene = transforms[0][0][1].reshape((3, 1)) # (3,1)
    rotMat_scene = transforms[0][1][1] # (3, 3)
    rotMat_inv_scene = np.linalg.inv(rotMat_scene)
    trans_scene = transforms[0][2][1].reshape((3, 1)) # (3,1)

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
        
        origin -= trans_scene
        lookat -= trans_scene

        origin = 1./(scale_scene) * (rotMat_inv_scene @ origin)
        lookat = 1./(scale_scene) * (rotMat_inv_scene @ lookat)
        up = 1./(scale_scene) * (rotMat_inv_scene @ up)

        origin = origin.flatten()
        lookat = lookat.flatten()
        up = up.flatten()

        at_vector = normalize_v(lookat - origin)
        assert np.amax(np.abs(np.dot(at_vector.flatten(), up.flatten()))) < 2e-3 # two vector should be perpendicular

        t = origin.reshape((3, 1))
        R = np.stack((np.cross(-up, at_vector), -up, at_vector), -1)
        # R = R @ np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        
        pose_list.append(np.hstack((R, t)))
        origin_lookatvector_up_list.append((origin, at_vector, up))

    return pose_list, origin_lookatvector_up_list

from lib.utils_rendering_openrooms import renderingLayer
def convert_SG_axis_local_global(lighting_params, split_lighting_SG_list_split, split_pose_list_split, split_normal_list):
    rL = renderingLayer(imWidth=lighting_params['env_col'], imHeight=lighting_params['env_row'], isCuda=False)

    lighting_SG_global_list = []

    for lighting_SG, pose, normal in zip(split_lighting_SG_list_split, split_pose_list_split, split_normal_list):
        lighting_SG_torch = torch.from_numpy(lighting_SG).view(-1, lighting_params['SG_num'], 6)
        theta_torch, phi_torch, _, _ = torch.split(lighting_SG_torch, [1, 1, 1, 3], dim=2)
        theta_torch = theta_torch.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) # (HW, 12(SG_num), 1, 1, 1, 1)
        phi_torch = phi_torch.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        normal_torch = torch.from_numpy(normal).unsqueeze(0).permute(0, 3, 1, 2) # (1, 3, H, W)

        _, camx, camy, normalPred = rL.forwardEnv(normal_torch, None, if_normal_only=True) # torch.Size([B, 128, 3, 120, 160]), [B, 3, 120, 160], [B, 3, 120, 160], [B, 3, 120, 160]
        envNum = lighting_params['env_row'] * lighting_params['env_col']
        camx_reshape = camx.squeeze(0).permute(1, 2, 0).view(envNum, 1, 1, 1, 1, 3)
        camy_reshape = camy.squeeze(0).permute(1, 2, 0).view(envNum, 1, 1, 1, 1, 3)
        camz_reshape = normalPred.squeeze(0).permute(1, 2, 0).view(envNum, 1, 1, 1, 1, 3)

        axisX = torch.sin(theta_torch ) * torch.cos(phi_torch )
        axisY = torch.sin(theta_torch ) * torch.sin(phi_torch )
        axisZ = torch.cos(theta_torch )
        axis_local_SG = torch.cat([axisX, axisY, axisZ], dim=5) # [19200, 12, 1, 1, 1, 3]; in a local SG (self.ls) coords

        axis_SG = axis_local_SG[:, :, :, :, :, 0:1] * camx_reshape \
            + axis_local_SG[:, :, :, :, :, 1:2] * camy_reshape \
            + axis_local_SG[:, :, :, :, :, 2:3] * camz_reshape # transfer from a local camera-dependent coords to the ONE AND ONLY camera coords (LightNet)
        axis_SG = axis_SG.squeeze() # [19200, 12, 3]

        axis_SG_np = axis_SG.cpu().numpy()
        axis_SG_np = np.stack([axis_SG_np[:, :, 0], -axis_SG_np[:, :, 1], -axis_SG_np[:, :, 2]], axis=-1) # transform axis from opengl convention (right-up-backward) to opencv (right-down-forward)
        
        R = pose[:3, :3]
        axis_SG_np_global = axis_SG_np @ (R.T)
        axis_SG_np_global = axis_SG_np_global.reshape(lighting_params['env_row'], lighting_params['env_col'], lighting_params['SG_num'], 3)

        lighting_SG = np.concatenate((axis_SG_np_global, lighting_SG[:, :, :, 2:6]), axis=3) # (120, 160, 12(SG_num), 7); axis, lamb, weight: 3, 1, 3

        lighting_SG_global_list.append(lighting_SG)

    return lighting_SG_global_list

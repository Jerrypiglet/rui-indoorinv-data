import numpy as np
import torch
import time
import torch.nn.functional as F

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


def _convert_local_to_cam_coords(normal):
    '''
    args:
        normal: (H, W, 3), normalized
    return:
        camx, camy, normal

    '''
    # assert normal.shape[:2] == (self.imHeight, self.imWidth)
    up = torch.tensor([0,1,0], device='cpu').float()

    # (3,), (1, 3, 120, 160)

    camyProj = torch.einsum('b,abcd->acd',(up, normal)).unsqueeze(1).expand_as(normal) * normal # project camera up to normal direction https://en.wikipedia.org/wiki/Vector_projection
    camy = F.normalize(up.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand_as(camyProj) - camyProj, dim=1, p=2)
    camx = -F.normalize(torch.cross(camy, normal,dim=1), p=2, dim=1) # torch.Size([1, 3, 120, 160])
    T_cam2local_flattened = torch.cat((camx, camy, normal), axis=0).permute(2, 3, 0, 1).flatten(0, 1).cpu().numpy() # concat as rows: cam2local; (HW, 3, 3)

    return T_cam2local_flattened


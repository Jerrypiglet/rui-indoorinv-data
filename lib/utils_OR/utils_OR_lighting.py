import numpy as np
import torch
from ..utils_openrooms import get_T_local_to_camopengl_np

def downsample_lighting_envmap(lighting_envmap, downsize_ratio: int=8, lighting_scale: float=1.):
    H_grid, W_grid, _, h, w = lighting_envmap.shape
    xx, yy = np.meshgrid(np.arange(0, H_grid, downsize_ratio), np.arange(0, W_grid, downsize_ratio))
    _ = lighting_envmap[xx.T, yy.T, :, :, :] # (30, 40, 3, 8, 16)
    H_grid_after, W_grid_after = _.shape[:2]
    lighting_envmap_downsampled = np.zeros((H_grid_after, W_grid_after, 3, h+1, w+1), dtype=_.dtype) + 1.
    lighting_envmap_downsampled[:, :, :, :h, :w] = _ * lighting_scale
    lighting_envmap_downsampled = lighting_envmap_downsampled.transpose(0, 3, 1, 4, 2).reshape(H_grid*(h+1)//downsize_ratio, W_grid*(w+1)//downsize_ratio, 3)

    return lighting_envmap_downsampled

def get_ls_np(env_height=8, env_width=16):
    '''
    return:
        ls: local hemisphere coordinates (3, env_height, env_width)
    '''

    Az = ( (np.arange(env_width) + 0.5) / env_width - 0.5)* 2 * np.pi
    El = ( (np.arange(env_height) + 0.5) / env_height) * np.pi / 2.0
    Az, El = np.meshgrid(Az, El)
    Az = Az[np.newaxis, :, :]
    El = El[np.newaxis, :, :]
    lx = np.sin(El) * np.cos(Az)
    ly = np.sin(El) * np.sin(Az)
    lz = np.cos(El)
    ls = np.concatenate((lx, ly, lz), axis = 0)
    return ls

class converter_SG_to_envmap():
    '''
        converting SG <-> envmap; always in local coordinates
    '''
    def __init__(self, SG_num=12, env_height=8, env_width=16, device='cpu'):
        self.env_width = env_width
        self.env_height = env_height
        self.device = device

        ls = get_ls_np(env_height=env_height, env_width=env_width)
        self.ls = ls[np.newaxis, np.newaxis, np.newaxis, :, :, :] # (1, 1, 1, 3, 8, 16)
        self.ls_torch = torch.from_numpy(ls.astype(np.float32)).to(self.device)
        self.ls_torch.requires_grad = False

        self.SG_num = SG_num

    def convert_converter_SG_to_envmap_2D(self, axis_local: np.ndarray, lamb: np.ndarray, weight: np.ndarray):
        '''
        axis_local, lamb, weight: (H, W, SG_num, 3/1/3)
        '''
        assert axis_local.shape[:2] == lamb.shape[:2] == weight.shape[:2]
        axis_local = torch.from_numpy(axis_local).unsqueeze(-1).unsqueeze(-1) # -> (120, 160, 12, 3, 1, 1)
        weight = torch.from_numpy(weight).unsqueeze(-1).unsqueeze(-1) # -> (120, 160, 12, 3, 1, 1)
        lamb = torch.from_numpy(lamb).unsqueeze(-1).unsqueeze(-1) # -> (120, 160, 12, 1, 1, 1)
        mi = lamb * (torch.sum(axis_local * self.ls_torch, dim=3).unsqueeze(3) - 1) # -> (120, 160, 12, 1, 8, 16)
        envmaps = weight * torch.exp(mi) # (120, 160, 12, 3, 1, 1), (120, 160, 12, 1, 8, 16) -> (120, 160, 12, 3, 8, 16)
        envmaps = torch.sum(envmaps, dim=2).cpu().numpy() # -> (120, 160, 3, 8, 16)

        return envmaps

    def convert_converter_SG_to_envmap_2D_np(self, axis_local: np.ndarray, lamb: np.ndarray, weight: np.ndarray):
        '''
        axis_local, lamb, weight: (H, W, SG_num, 3/1/3)
        '''
        assert axis_local.shape[:2] == lamb.shape[:2] == weight.shape[:2]
        axis_local = axis_local[..., np.newaxis, np.newaxis] # -> (120, 160, 12, 3, 1, 1)
        weight = weight[..., np.newaxis, np.newaxis] # -> (120, 160, 12, 3, 1, 1)
        lamb = lamb[..., np.newaxis, np.newaxis] # -> (120, 160, 12, 1, 1, 1)
        mi = lamb * (np.sum(axis_local * self.ls, axis=3, keepdims=True) - 1) # -> (120, 160, 12, 1, 8, 16)
        envmaps = weight * np.exp(mi) # (120, 160, 12, 3, 1, 1), (120, 160, 12, 1, 8, 16) -> (120, 160, 12, 3, 8, 16)
        envmaps = np.sum(envmaps, axis=2) # -> (120, 160, 3, 8, 16)

        return envmaps

def convert_lighting_axis_local_to_global_np(lighting_axis_local, pose, normal_opengl):
    '''
    lighting_axis_local: (H, W, wi_num, 3)
    normal: (H, W, 3), in camera coordinates, OpenGL convention
    '''

    # normal_torch = torch.from_numpy(normal_opengl).unsqueeze(0).permute(0, 3, 1, 2) # (1, 3, H, W)
    _, _, wi_num = lighting_axis_local.shape[:3]
    H, W = normal_opengl.shape[:2]
    axis_local_flattened = lighting_axis_local.reshape(-1, wi_num, 3) # (HW, SG_num, 3)
    # camx, camy, normalPred = rL.forwardEnv(normal_torch, None, if_normal_only=True) # [B, 3, 120, 160], [B, 3, 120, 160], [B, 3, 120, 160]
    # T_cam2local_flattened = torch.cat((camx, camy, normalPred), axis=0).permute(2, 3, 0, 1).flatten(0, 1).cpu().numpy() # concat as rows: cam2local; (HW, 3, 3)
    # ts = time.time()
    # T_cam2local_flattened = _convert_local_to_cam_coords(normal_torch)
    # axis_SG_np = axis_local_SG_flattened @ T_cam2local_flattened # (local2cam @ a.T).T -> a @ cam2local, (HW, SG_num, 3)
    # print('---', time.time() - ts)

    # ts = time.time()
    '''
    get_T_local_to_camopengl_np (Numpy only) is about 2x as faster than _convert_local_to_cam_coords (Torch + Numpy)
    '''
    T_local_to_camopengl = get_T_local_to_camopengl_np(normal_opengl)
    axis_opengl_np = axis_local_flattened @ (T_local_to_camopengl.reshape(-1, 3, 3).transpose(0, 2, 1)) # (local2cam @ a.T).T -> a @ cam2local, (HW, num, 3)
    # print('===', time.time() - ts)
    axis_np = np.stack([axis_opengl_np[:, :, 0], -axis_opengl_np[:, :, 1], -axis_opengl_np[:, :, 2]], axis=-1) # transform axis from opengl convention (right-up-backward) to opencv (right-down-forward)
    
    R = pose[:3, :3]
    axis_np_global = axis_np @ (R.T)
    axis_np_global = axis_np_global.reshape(H, W, wi_num, 3)
    axis_np_global = axis_np_global / (np.linalg.norm(axis_np_global, axis=3, keepdims=True)+1e-6)

    return axis_np_global

def convert_SG_angles_to_axis_local_np(lighting_SG_local_angles):
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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)

class rendering_layer_per_point_from_emitter():
    def __init__(self, F0=0.05, device='cpu'):
        self.device = device

        self.F0 = F0
        # self.temp = Variable(torch.FloatTensor(1, 1, 1))
        self.temp = torch.zeros((1, 1, 1)).float() + 2.

        if self.device != 'cpu':
            self.temp = self.temp.to(self.device)

    def forward_rays(
            self, 
            normal, 
            albedo, 
            roughness, 
            v_dirs, 
            l_dirs, 
            pts_intensity_weighted, 
        ):
        '''
        v_dirs: (N, 3)
        l_dirs: (N(HW), M(128), 3)
        '''
        N_pts = normal.shape[0]
        M_lpts = l_dirs.shape[1]
        assert tuple(normal.shape)==(N_pts, 3)
        assert tuple(albedo.shape)==(N_pts, 3)
        assert tuple(roughness.shape)==(N_pts, 1)

        ndl = torch.clamp(torch.sum(normal.unsqueeze(1) * l_dirs, dim=2, keepdim=True), 0, 1) # [!!!] cos in rendering function; normal and l are both in camera coords [n * l]

        albedoBatch = albedo / np.pi
        brdfDiffuse = albedoBatch.unsqueeze(1).expand([N_pts, M_lpts, 3]) * ndl.expand([N_pts, M_lpts, 3])
        colorDiffuse = torch.sum(brdfDiffuse * pts_intensity_weighted, dim=1) # I_d

        # ======== specular
        roughBatch = roughness
        assert torch.amin(roughness) >= 0. and torch.amax(roughness) <= 1.

        k = (roughBatch + 1) * (roughBatch + 1) / 8.0
        alpha = roughBatch * roughBatch
        alpha2 = alpha * alpha

        h_dirs = (v_dirs.unsqueeze(1) + l_dirs) / 2 # (N, 1, 3), (N, M, 3) -> (N(HW), M(128), 3)
        h_dirs = torch.nn.functional.normalize(h_dirs, dim=-1)

        vdh = torch.sum((v_dirs.unsqueeze(1) * h_dirs), dim=2, keepdim=True) # -> (N, M, 1) [w_o * h]
        # self.temp.data[0] = 2.0
        frac0 = self.F0 + (1-self.F0) * torch.pow(self.temp.expand_as(vdh), (-5.55472*vdh-6.98316)*vdh)

        ndh = torch.clamp(torch.sum(normal.unsqueeze(1) * h_dirs, dim=2, keepdim=True), 0, 1) # sum((N, 1, 3) * (N, M, 3)) -> (N, M, 1)
        ndv = torch.clamp(torch.sum(normal * v_dirs.expand_as(normal), dim=1, keepdim=True), 0, 1).unsqueeze(2) # (N, 1, 1) [n * v]
        

        frac = alpha2.unsqueeze(1).expand_as(frac0) * frac0
        nom0 = ndh * ndh * (alpha2.unsqueeze(1).expand_as(ndh) - 1) + 1
        nom1 = ndv * (1 - k.unsqueeze(1).expand_as(ndh)) + k.unsqueeze(1).expand_as(ndh)
        nom2 = ndl * (1 - k.unsqueeze(1).expand_as(ndh)) + k.unsqueeze(1).expand_as(ndh)

        nom = 4*np.pi*nom0*nom0*nom1*nom2
        # nom = torch.clamp(nom, min=1e-6, max=4*np.pi)
        
        specPred = frac / (nom+(1e-6)**4) # f_s
        # print(frac.shape, ndv.shape, ndl.shape) # torch.Size([192, 32768, 1]) torch.Size([192, 1, 1]) torch.Size([192, 32768, 1])

        # envmap = envmap.view([N_pts, 3, self.env_width * self.env_height])
        # envmap = envmap.permute([0, 2, 1]) # (N, self.env_width(8) * self.env_height(16), 3)

        # envmap_mask = torch.max(envmap, dim=2, keepdim=True)[0] > 1e-3

        # brdfDiffuse = albedoBatch.unsqueeze(1).expand([N_pts, self.env_width * self.env_height, 3]) \
        #      * ndl.expand([N_pts, self.env_width * self.env_height, 3])
        # colorDiffuse = torch.sum(brdfDiffuse * pts_intensity_weighted * self.envWeight.expand_as(brdfDiffuse), dim=1) # I_d
        

        # brdfSpec = specPred.expand([N_pts, self.env_width * self.env_height, 3]) \
        #      * ndl.expand([N_pts, self.env_width * self.env_height, 3])
        # colorSpec = torch.sum(brdfSpec * pts_intensity_weighted * self.envWeight.expand_as(brdfSpec), dim=1) # I_s

        brdfSpec = specPred.expand([N_pts, M_lpts, 3]) * ndl.expand([N_pts, M_lpts, 3])
        colorSpec = torch.sum(brdfSpec * pts_intensity_weighted, dim=1) # I_s

        return colorDiffuse, colorSpec
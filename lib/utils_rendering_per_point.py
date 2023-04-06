import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)

class output2env_per_point():
    def __init__(self, SG_num, env_width = 16, env_height = 8, isCuda = True):
        self.env_width = env_width
        self.env_height = env_height

        Az = ( (np.arange(env_width) + 0.5) / env_width - 0.5)* 2 * np.pi
        El = ( (np.arange(env_height) + 0.5) / env_height) * np.pi / 2.0
        Az, El = np.meshgrid(Az, El)
        Az = Az[np.newaxis, :, :]
        El = El[np.newaxis, :, :]
        lx = np.sin(El) * np.cos(Az)
        ly = np.sin(El) * np.sin(Az)
        lz = np.cos(El)
        ls = np.concatenate((lx, ly, lz), axis = 0)
        print(ls.shape)
        ls = ls[np.newaxis, np.newaxis, :, :, :]
        self.ls = Variable(torch.from_numpy(ls.astype(np.float32)))

        self.SG_num = SG_num
        if isCuda:
            self.ls = self.ls.cuda()

        self.ls.requires_grad = False

    def fromSGtoIm(self, axis, lamb, weight):
        bn = axis.size(0)

        # Turn SG parameters to environmental maps
        axis = axis.unsqueeze(-1).unsqueeze(-1)

        weight = weight.view(bn, self.SG_num, 3, 1, 1)
        lamb = lamb.view(bn, self.SG_num, 1, 1, 1)

        mi = lamb.expand([bn, self.SG_num, 1, self.env_height, self.env_width])* \
                (torch.sum(axis.expand([bn, self.SG_num, 3, self.env_height, self.env_width]) * \
                self.ls.expand([bn, self.SG_num, 3, self.env_height, self.env_width]), dim = 2).unsqueeze(2) - 1)
        envmaps = weight.expand([bn, self.SG_num, 3, self.env_height, self.env_width]) * \
            torch.exp(mi).expand([bn, self.SG_num, 3, self.env_height, self.env_width])
        # print(envmaps.shape)

        envmaps = torch.sum(envmaps, dim=1)

        return envmaps

    def postprocess_weight(self, weightOrig):
        weight = 0.999 * weightOrig
        # weight = 0.8 * weightOrig 
        weight = torch.tan(np.pi / 2 * weight)
        return weight

    def postprocess_lamb(self, lambOrig):
        lambOrig = 0.999 * lambOrig
        lamb = torch.tan(np.pi / 2 * lambOrig)
        return lamb

    def output2env(self, axisOrig, lambOrig, weightOrig, if_postprocessing_weight=True, if_postprocessing_lamb=True):
        # bn, _, env_row, env_col = weightOrig.size()

        axis = axisOrig # torch.Size([B, 12(SG_num), 3, 120, 160])
        
        if if_postprocessing_weight:
            weight = self.postprocess_weight(weightOrig)
        else:
            weight = weightOrig

        if if_postprocessing_lamb:
            lamb = self.postprocess_lamb(lambOrig)
        else:
            lamb = lambOrig


        envmaps = self.fromSGtoIm(axis, lamb, weight)

        return envmaps, axis, lamb, weight

class rendering_layer_per_point():
    def __init__(self, imWidth = 640, imHeight = 480, fov=57, F0=0.05, cameraPos = [0, 0, 0], 
            env_width = 16, env_height = 8, SG_num = 12, isCuda = True):
        self.imHeight = imHeight
        self.imWidth = imWidth
        self.env_width = env_width
        self.env_height = env_height
        self.SG_num = SG_num

        self.fov = fov/180.0 * np.pi
        self.F0 = F0
        self.cameraPos = np.array(cameraPos, dtype=np.float32).reshape([3, 1, 1]) # (3, 1, 1)
        self.xRange = 1 * np.tan(self.fov/2)
        self.yRange = float(imHeight) / float(imWidth) * self.xRange
        self.isCuda = isCuda
        x, y = np.meshgrid(np.linspace(-self.xRange, self.xRange, imWidth),
                np.linspace(-self.yRange, self.yRange, imHeight))
        y = np.flip(y, axis=0)
        z = -np.ones( (imHeight, imWidth), dtype=np.float32)

        pCoord = np.stack([x, y, z]).astype(np.float32)
        self.pCoord = pCoord # torch.Size([3, 480, 640])
        v = self.cameraPos - self.pCoord
        v = v / np.sqrt(np.maximum(np.sum(v*v, axis=0, keepdims=True), 1e-12))
        v = v.astype(dtype = np.float32) # torch.Size([3, 480, 640])

        self.v = Variable(torch.from_numpy(v)) # for rendering only: directions from virtual camera plane pixels to camera center (in local cam coords, i.e. gx-y-z with -z forward); 3rd dimension is positive
        self.pCoord = Variable(torch.from_numpy(self.pCoord))

        self.up = torch.Tensor([0, 1, 0])

        # azimuth & elevation angles -> dir vector for each pixel
        Az = ( (np.arange(env_width) + 0.5) / env_width - 0.5)* 2 * np.pi # array([-2.9452, -2.5525, -2.1598, -1.7671, -1.3744, -0.9817, -0.589, -0.1963,  0.1963,  0.589 ,  0.9817,  1.3744,  1.7671,  2.1598,  2.5525,  2.9452])
        El = ( (np.arange(env_height) + 0.5) / env_height) * np.pi / 2.0 # array([0.0982, 0.2945, 0.4909, 0.6872, 0.8836, 1.0799, 1.2763, 1.4726])
        Az, El = np.meshgrid(Az, El)
        Az = Az.reshape(-1, 1)
        El = El.reshape(-1, 1)
        lx = np.sin(El) * np.cos(Az)
        ly = np.sin(El) * np.sin(Az)
        lz = np.cos(El)
        ls = np.concatenate((lx, ly, lz), axis = 1) # dir vector for each pixel (local coords), torch.Size([128, 3])

        envWeight = np.sin(El) * np.pi * np.pi / env_width / env_height

        self.ls = Variable(torch.from_numpy(ls.astype(np.float32)))
        self.envWeight = Variable(torch.from_numpy(envWeight.astype(np.float32)))
        self.envWeight = self.envWeight.unsqueeze(0)

        self.temp = Variable(torch.FloatTensor(1, 1, 1))

        if isCuda:
            self.v = self.v.cuda()
            self.pCoord = self.pCoord.cuda()
            self.up = self.up.cuda()
            self.ls = self.ls.cuda()
            self.envWeight = self.envWeight.cuda()
            self.temp = self.temp.cuda()

    def get_cam2local_transformations(self, normal):
        assert len(normal.shape)==2 and normal.shape[1]==3, 'normal has to be of shape (N, 3)!'
        normal = normal / torch.clamp(torch.linalg.norm(normal, dim=1, keepdim=True), min=1e-6)

        camyProj = torch.einsum('b,ab->a',(self.up, normal)).unsqueeze(1).expand_as(normal) * normal # project camera up to normal direction https://en.wikipedia.org/wiki/Vector_projection
        camy = F.normalize(self.up.unsqueeze(0).expand_as(camyProj) - camyProj, dim=1, p=2)
        camx = -F.normalize(torch.cross(camy, normal, dim=1), p=2, dim=1) # torch.Size([N, 3])

        return torch.stack((camx, camy, normal), axis=1), (camx, camy, normal) # stack along new axis 1 -> (N, 3, 3)

    def forwardEnv(self, rays_uv, normal, envmap, albedo, roughness, camxynormal_tuple=None):
        N_pts = rays_uv.shape[0]
        assert tuple(rays_uv.shape)==(N_pts, 2)
        assert tuple(normal.shape)==(N_pts, 3)
        assert tuple(envmap.shape)==(N_pts, 3, self.env_height, self.env_width)
        assert tuple(albedo.shape)==(N_pts, 3)
        assert tuple(roughness.shape)==(N_pts, 1)

        ldirections = self.ls.unsqueeze(0) # torch.Size([1, 128, 3])

        if camxynormal_tuple is None:        
            normal = normal / torch.clamp(torch.linalg.norm(normal, dim=1, keepdim=True), min=1e-6)
            _, (camx, camy, normal) = self.get_cam2local_transformations(normal)
        else:
            camx, camy, normal = camxynormal_tuple

        # [!!!] multiply the local SG self.ls grid vectors (think of as coefficients) with the LOCAL camera-dependent basis (think of as basis..)
        # ... and then you arrive at a hemisphere in the camera cooords
        l = ldirections[:, :, 0:1] * camx.unsqueeze(1) \
                + ldirections[:, :, 1:2] * camy.unsqueeze(1) \
                + ldirections[:, :, 2:3] * normal.unsqueeze(1)    
            # (1, 128, 1) * (N, 1, 3) -> (N, 128, 3)
        # print(l.shape) # torch.Size([N, 128, 3])

        N_pts = albedo.size(0)

        # print(l.shape, self.v.unsqueeze(1).shape)
        v_rays = self.v[:, rays_uv[:, 1], rays_uv[:, 0]].transpose(0, 1) # (N, 3)
        h = (v_rays.unsqueeze(1) + l) / 2 # (N, 1, 3), (N, 128, 3)
        h = h / torch.sqrt(torch.clamp(torch.sum(h*h, dim=2, keepdim=True), min = 1e-6)) # (N, 128, 3)

        vdh = torch.sum( (v_rays.unsqueeze(1) * h), dim=2, keepdim=True) # (N, 128, 1) [w_o * h]
        self.temp.data[0] = 2.0
        frac0 = self.F0 + (1-self.F0) * torch.pow(self.temp.expand_as(vdh), (-5.55472*vdh-6.98316)*vdh)

        albedoBatch = albedo / np.pi
        # roughBatch = (roughness + 1.0) / 2.0
        roughBatch = roughness
        assert torch.amin(roughness) >= 0. and torch.amax(roughness) <= 1.

        k = (roughBatch + 1) * (roughBatch + 1) / 8.0
        alpha = roughBatch * roughBatch
        alpha2 = alpha * alpha

        ndv = torch.clamp(torch.sum(normal * v_rays.expand_as(normal), dim = 1, keepdim=True), 0, 1).unsqueeze(2) # (N, 1, 1) [n * v]
        ndh = torch.clamp(torch.sum(normal.unsqueeze(1) * h, dim = 2, keepdim=True), 0, 1) # sum((N, 1, 3) * (N, 128, 3)) -> (N, 128, 1)
        ndl = torch.clamp(torch.sum(normal.unsqueeze(1) * l, dim = 2, keepdim=True), 0, 1) # [!!!] cos in rendering function; normal and l are both in camera coords [n * l]

        # print(alpha2.shape, frac0.shape) # torch.Size([N, 1]) torch.Size([N, 128, 1])
        frac = alpha2.unsqueeze(1).expand_as(frac0) * frac0
        nom0 = ndh * ndh * (alpha2.unsqueeze(1).expand_as(ndh) - 1) + 1
        nom1 = ndv * (1 - k.unsqueeze(1).expand_as(ndh)) + k.unsqueeze(1).expand_as(ndh)
        nom2 = ndl * (1 - k.unsqueeze(1).expand_as(ndh)) + k.unsqueeze(1).expand_as(ndh)

        nom = 4*np.pi*nom0*nom0*nom1*nom2
        # nom = torch.clamp(nom, min=1e-6, max=4*np.pi)
        
        # if torch.amin(nom0*nom0*nom1*nom2) < 1e-6/4./np.pi:
        #     print(torch.amax(nom/4./np.pi), torch.amin(nom/4./np.pi))
        #     # print('----', torch.amax(nom0), torch.amax(nom1), torch.amax(nom2)) # nom2 max can be 1 on envmap locations where ndl is ~=1
        #     print('====', torch.amin(nom0), torch.amin(nom1), torch.amin(nom2)) # norm0 can be very small, on very specular roughness (e.g. roughness=0.0441 -> nom=3.7883e-06)
        #     print(alpha2, roughBatch)
        
        specPred = frac / (nom+(1e-6)**4) # f_s
        # print(frac.shape, ndv.shape, ndl.shape) # torch.Size([192, 32768, 1]) torch.Size([192, 1, 1]) torch.Size([192, 32768, 1])

        envmap = envmap.view([N_pts, 3, self.env_width * self.env_height])
        envmap = envmap.permute([0, 2, 1]) # (N, self.env_width(8) * self.env_height(16), 3)

        # envmap_mask = torch.max(envmap, dim=2, keepdim=True)[0] > 1e-3

        brdfDiffuse = albedoBatch.unsqueeze(1).expand([N_pts, self.env_width * self.env_height, 3]) * \
                    ndl.expand([N_pts, self.env_width * self.env_height, 3])
        colorDiffuse = torch.sum(brdfDiffuse * envmap * self.envWeight.expand_as(brdfDiffuse), dim=1) # I_d

        # # print(envmap_mask.shape, self.envWeight.expand_as(brdfDiffuse).shape)

        brdfSpec = specPred.expand([N_pts, self.env_width * self.env_height, 3]) * \
                    ndl.expand([N_pts, self.env_width * self.env_height, 3])
        colorSpec = torch.sum(brdfSpec * envmap * self.envWeight.expand_as(brdfSpec), dim=1) # I_s

        # print(ndl.shape, specPred.shape)

        # envWeight = self.envWeight.expand_as(brdfDiffuse)
        # envWeight = envWeight * envmap_mask / (torch.sum(envWeight * envmap_mask, dim=1, keepdim=True) / torch.sum(envWeight, dim=1, keepdim=True))
        # colorDiffuse = torch.sum(brdfDiffuse * envmap * envWeight, dim=1) # I_d
        # colorSpec = torch.sum(brdfSpec * envmap * envWeight, dim=1) # I_s

        return colorDiffuse, colorSpec
        # return colorDiffuse, colorSpec, ndl, specPred
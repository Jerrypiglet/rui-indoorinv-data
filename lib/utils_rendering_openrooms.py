import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class output2env():
    def __init__(self, SG_num, env_width = 16, env_height = 8, isCuda = True ):
        self.env_width = env_width
        self.env_height = env_height

        Az = ( (np.arange(env_width) + 0.5) / env_width - 0.5 )* 2 * np.pi
        El = ( (np.arange(env_height) + 0.5) / env_height) * np.pi / 2.0
        Az, El = np.meshgrid(Az, El)
        Az = Az[np.newaxis, :, :]
        El = El[np.newaxis, :, :]
        lx = np.sin(El) * np.cos(Az)
        ly = np.sin(El) * np.sin(Az)
        lz = np.cos(El)
        ls = np.concatenate((lx, ly, lz), axis = 0)
        ls = ls[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis, :, :]
        self.ls = Variable(torch.from_numpy(ls.astype(np.float32 ) ) )

        self.SG_num = SG_num
        if isCuda:
            self.ls = self.ls.cuda()

        self.ls.requires_grad = False

    def fromSGtoIm(self, axis, lamb, weight ):
        bn = axis.size(0)
        env_row, env_col = weight.size(2), weight.size(3)

        # Turn SG parameters to environmental maps
        axis = axis.unsqueeze(-1).unsqueeze(-1)

        weight = weight.view(bn, self.SG_num, 3, env_row, env_col, 1, 1)
        lamb = lamb.view(bn, self.SG_num, 1, env_row, env_col, 1, 1)

        mi = lamb.expand([bn, self.SG_num, 1, env_row, env_col, self.env_height, self.env_width] )* \
                (torch.sum(axis.expand([bn, self.SG_num, 3, env_row, env_col, self.env_height, self.env_width]) * \
                self.ls.expand([bn, self.SG_num, 3, env_row, env_col, self.env_height, self.env_width] ), dim = 2).unsqueeze(2) - 1)
        envmaps = weight.expand([bn, self.SG_num, 3, env_row, env_col, self.env_height, self.env_width] ) * \
            torch.exp(mi).expand([bn, self.SG_num, 3, env_row, env_col, self.env_height, self.env_width] )
        # print(envmaps.shape)

        envmaps = torch.sum(envmaps, dim=1)

        return envmaps

    def output2env(self, axisOrig, lambOrig, weightOrig, if_postprocessing=True):
        bn, _, env_row, env_col = weightOrig.size()

        axis = axisOrig # torch.Size([B, 12(SG_num), 3, 120, 160])
        
        if if_postprocessing:
            weight = 0.999 * weightOrig
            # weight = 0.8 * weightOrig 
            weight = torch.tan(np.pi / 2 * weight )
        else:
            weight = weightOrig

        if if_postprocessing:
            lambOrig = 0.999 * lambOrig
            lamb = torch.tan(np.pi / 2 * lambOrig )
        else:
            lamb = lambOrig


        envmaps = self.fromSGtoIm(axis, lamb, weight )

        return envmaps, axis, lamb, weight

class renderingLayer():
    def __init__(self, imWidth = 160, imHeight = 120, fov=57, F0=0.05, cameraPos = [0, 0, 0], 
            env_width = 16, env_height = 8, isCuda = True):
        self.imHeight = imHeight
        self.imWidth = imWidth
        self.env_width = env_width
        self.env_height = env_height

        self.fov = fov/180.0 * np.pi
        self.F0 = F0
        self.cameraPos = np.array(cameraPos, dtype=np.float32).reshape([1, 3, 1, 1])
        self.xRange = 1 * np.tan(self.fov/2)
        self.yRange = float(imHeight) / float(imWidth) * self.xRange
        self.isCuda = isCuda
        x, y = np.meshgrid(np.linspace(-self.xRange, self.xRange, imWidth),
                np.linspace(-self.yRange, self.yRange, imHeight ) )
        y = np.flip(y, axis=0)
        z = -np.ones( (imHeight, imWidth), dtype=np.float32)

        pCoord = np.stack([x, y, z]).astype(np.float32)
        self.pCoord = pCoord[np.newaxis, :, :, :]
        v = self.cameraPos - self.pCoord
        v = v / np.sqrt(np.maximum(np.sum(v*v, axis=1), 1e-12)[:, np.newaxis, :, :] )
        v = v.astype(dtype = np.float32)

        self.v = Variable(torch.from_numpy(v) ) # for rendering only: directions from virtual camera plane pixels to camera center (in local cam coords, i.e. gx-y-z with -z forward); 3rd dimension is positive
        self.pCoord = Variable(torch.from_numpy(self.pCoord) )

        self.up = torch.tensor([0,1,0], device='cpu').float()
        self.temp = Variable(torch.FloatTensor(1, 1, 1, 1, 1) )
        self.temp.data[0] = 2.0

        # azimuth & elevation angles -> dir vector for each pixel
        Az = ( (np.arange(env_width) + 0.5) / env_width - 0.5 )* 2 * np.pi # array([-2.9452, -2.5525, -2.1598, -1.7671, -1.3744, -0.9817, -0.589, -0.1963,  0.1963,  0.589 ,  0.9817,  1.3744,  1.7671,  2.1598,  2.5525,  2.9452])
        El = ( (np.arange(env_height) + 0.5) / env_height) * np.pi / 2.0 # array([0.0982, 0.2945, 0.4909, 0.6872, 0.8836, 1.0799, 1.2763, 1.4726])
        Az, El = np.meshgrid(Az, El)
        Az = Az.reshape(-1, 1)
        El = El.reshape(-1, 1)
        lx = np.sin(El) * np.cos(Az)
        ly = np.sin(El) * np.sin(Az)
        lz = np.cos(El)
        ls = np.concatenate((lx, ly, lz), axis = 1) # dir vector for each pixel (local coords)

        envWeight = np.sin(El ) * np.pi * np.pi / env_width / env_height

        self.ls = Variable(torch.from_numpy(ls.astype(np.float32 ) ) )
        self.envWeight = Variable(torch.from_numpy(envWeight.astype(np.float32 ) ) )
        self.envWeight = self.envWeight.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

        if isCuda:
            self.v = self.v.cuda()
            self.pCoord = self.pCoord.cuda()
            self.up = self.up.cuda()
            self.ls = self.ls.cuda()
            self.temp = self.temp.cuda()
            self.envWeight = self.envWeight.cuda()

    def forwardEnv(self, normal, envmap, albedo=None, roughness=None, if_normal_only=False):
        if envmap is not None:
            envR, envC = envmap.size(2), envmap.size(3)
        else:
            envR, envC = self.imHeight, self.imWidth

        # print(normal.shape, albedo.shape, roughness.shape)
        if albedo is not None and roughness is not None:
            assert normal.shape[-2:] == albedo.shape[-2:] == roughness.shape[-2:]
        
        # print(normal.shape)
        normal = F.adaptive_avg_pool2d(normal, (envR, envC) )
        normal = normal / torch.sqrt( torch.clamp(
            torch.sum(normal * normal, dim=1 ), 1e-6, 1).unsqueeze(1) )

        # assert normal.shape[2:]==(self.imHeight, self.imWidth)

        ldirections = self.ls.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) # torch.Size([1, 128, 3, 1, 1])

        # print(if_normal_only, self.up.shape, normal.shape)
        camyProj = torch.einsum('b,abcd->acd',(self.up, normal)).unsqueeze(1).expand_as(normal) * normal # project camera up to normal direction https://en.wikipedia.org/wiki/Vector_projection
        camy = F.normalize(self.up.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand_as(camyProj) - camyProj, dim=1, p=2)
        camx = -F.normalize(torch.cross(camy, normal,dim=1), p=2, dim=1) # torch.Size([1, 3, 120, 160])

        # print(camx.shape, torch.linalg.norm(camx, dim=1))
        # print(camy.shape, torch.linalg.norm(camy, dim=1))
        # ls (local coords), l (cam coords)
        # \sum [1, 128, 1, 1, 1] * [1, 1, 3, 120, 160] -> [1, 128, 3, 120, 160]
        # single vec: \sum [1, 1, 1, 1, 1] * [1, 1, 3, 1, 1] -> [1, 1, 3, 1, 1]
        
        # print(ldirections[:, :, 0:1, :, :].shape, camx.unsqueeze(1).shape) # torch.Size([1, 128, 1, 1, 1]) torch.Size([2, 1, 3, 120, 160])
        
        # [!!!] multiply the local SG self.ls grid vectors (think of as coefficients) with the LOCAL camera-dependent basis (think of as basis..)
        # ... and then you arrive at a hemisphere in the camera cooords
        l = ldirections[:, :, 0:1, :, :] * camx.unsqueeze(1) \
                + ldirections[:, :, 1:2, :, :] * camy.unsqueeze(1) \
                + ldirections[:, :, 2:3, :, :] * normal.unsqueeze(1)    
        # print(ldirections[:, 20, :, :, :].flatten())
        # print(l.shape) # torch.Size([1, 128, 3, 120, 160])

        if if_normal_only:
            return l, camx, camy, normal

        bn = albedo.size(0)
        albedo = F.adaptive_avg_pool2d(albedo, (envR, envC) )
        roughness = F.adaptive_avg_pool2d(roughness, (envR, envC ) )


        h = (self.v.unsqueeze(1) + l) / 2;
        h = h / torch.sqrt(torch.clamp(torch.sum(h*h, dim=2), min = 1e-6).unsqueeze(2) )
        # print(l.shape, self.v.unsqueeze(1).shape, h.shape) # torch.Size([1, 128, 3, 120, 160]) torch.Size([1, 1, 3, 120, 160]) torch.Size([1, 128, 3, 120, 160])
        # print(self.v.shape, h.shape, (self.v * h).shape)

        vdh = torch.sum( (self.v * h), dim = 2).unsqueeze(2) # torch.Size([1, 128, 1, 120, 160])
        frac0 = self.F0 + (1-self.F0) * torch.pow(self.temp.expand_as(vdh), (-5.55472*vdh-6.98316)*vdh) # torch.Size([1, 128, 1, 120, 160])

        diffuseBatch = (albedo )/ np.pi
        # roughBatch = (roughness + 1.0)/2.0
        roughBatch = roughness

        k = (roughBatch + 1) * (roughBatch + 1) / 8.0
        alpha = roughBatch * roughBatch
        alpha2 = alpha * alpha

        # print(self.v.shape, normal.shape) # torch.Size([1, 3, 120, 160]) torch.Size([1, 3, 120, 160])

        ndv = torch.clamp(torch.sum(normal * self.v.expand_as(normal), dim = 1), 0, 1).unsqueeze(1).unsqueeze(2) # -> torch.Size([1, 1, 1, 120, 160])
        ndh = torch.clamp(torch.sum(normal.unsqueeze(1) * h, dim = 2), 0, 1).unsqueeze(2) # torch.Size([1, 1, 3, 120, 160])，torch.Size([1, 128, 3, 120, 160]) -> ，torch.Size([1, 128, 1, 120, 160]
        ndl = torch.clamp(torch.sum(normal.unsqueeze(1) * l, dim = 2), 0, 1).unsqueeze(2) # [!!!] cos in rendering function; normal and l are both in camera coords

        # print(ndv.shape, ndh.shape, ndl.shape) # torch.Size([1, 1, 1, 120, 160]) torch.Size([1, 128, 1, 120, 160]) torch.Size([1, 128, 1, 120, 160])

        # print(alpha2.shape, frac0.shape) # torch.Size([1, 1, 120, 160]) torch.Size([1, 128, 1, 120, 160])
        frac = alpha2.unsqueeze(1).expand_as(frac0) * frac0
        nom0 = ndh * ndh * (alpha2.unsqueeze(1).expand_as(ndh) - 1) + 1
        nom1 = ndv * (1 - k.unsqueeze(1).expand_as(ndh) ) + k.unsqueeze(1).expand_as(ndh)
        nom2 = ndl * (1 - k.unsqueeze(1).expand_as(ndh) ) + k.unsqueeze(1).expand_as(ndh)
        nom = torch.clamp(4*np.pi*nom0*nom0*nom1*nom2, 1e-6, 4*np.pi)
        specPred = frac / nom # f_s

        envmap = envmap.view([bn, 3, envR, envC, self.env_width * self.env_height ] )
        envmap = envmap.permute([0, 4, 1, 2, 3] ) # [bn, self.env_width * self.env_height, 3, envR(120), envC(160)]

        brdfDiffuse = diffuseBatch.unsqueeze(1).expand([bn, self.env_width * self.env_height, 3, envR, envC] ) * \
                    ndl.expand([bn, self.env_width * self.env_height, 3, envR, envC] )

        # print(brdfDiffuse.shape, envmap.shape, self.envWeight.shape) # torch.Size([1, 128, 3, 120, 160]) torch.Size([1, 128, 3, 120, 160]) torch.Size([1, 128, 1, 1, 1])
        colorDiffuse = torch.sum(brdfDiffuse * envmap * self.envWeight.expand_as(brdfDiffuse), dim=1) # I_d

        brdfSpec = specPred.expand([bn, self.env_width * self.env_height, 3, envR, envC ] ) * \
                    ndl.expand([bn, self.env_width * self.env_height, 3, envR, envC] )
        colorSpec = torch.sum(brdfSpec * envmap * self.envWeight.expand_as(brdfSpec), dim=1) # I_s

        return colorDiffuse, colorSpec
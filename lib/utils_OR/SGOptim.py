import torch
import numpy as np
import cv2
import torch.optim as optim
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

def initializeEnSkyGrd(imOrig ):
    im = np.sum(imOrig, axis=2 )
    height, width = im.shape
    im = im.flatten()
    imId = np.argmax(im )
    rowId = int(imId / width )
    colId = imId - rowId * width


    # The weight
    weight = imOrig[rowId, colId, :]
    # The theta
    theta = (rowId + 0.5) / height * np.pi
    phi = ((colId + 0.5 ) / width - 0.5) * np.pi * 2

    weightSky = np.ones(3, np.float32 )
    weightGrd = np.ones(3, np.float32 )

    thetaSky = np.pi / 2.0 + 0.1
    phiSky = np.pi / 2.0
    thetaGrd = np.pi / 2.0 +  0.1
    phiGrd = -np.pi / 2.0

    return weight, theta, phi, \
        weightSky, thetaSky, phiSky, \
        weightGrd, thetaGrd, phiGrd

class SGEnvOptimSkyGrd():
    def __init__(self,
            weight=np.zeros(3,), theta=0., phi=0.,
            weightSky=np.zeros(3,), thetaSky=0., phiSky=0.,
            weightGrd=np.zeros(3,), thetaGrd=0., phiGrd=0.,
            gpuId = 0, niter = 250, isCuda = True,
            envWidth = 512, envHeight = 256, ch = 3 ):

        self.niter = niter
        self.ch = ch

        Az = ( (np.arange(envWidth) + 0.5) / envWidth - 0.5 ) * 2 * np.pi
        El = ( (np.arange(envHeight) + 0.5) / envHeight ) * np.pi
        Az, El = np.meshgrid(Az, El )
        Az = Az[:, :, np.newaxis ]
        El = El[:, :, np.newaxis ]
        lx = np.sin(El ) * np.cos(Az )
        ly = np.sin(El ) * np.sin(Az )
        lz = np.cos(El )
        self.ls = np.concatenate( (lx, ly, lz), axis = 2)[np.newaxis, :]
        self.ls = torch.from_numpy(self.ls.astype(np.float32) )
        self.envHeight = envHeight
        self.envWidth = envWidth
        self.iterCount = 0

        self.W = torch.from_numpy(np.sin(El.astype(np.float32) ) ).reshape(1, envHeight, envWidth)
        self.W[:, 0:int(envHeight/2), :] = 0
        self.envmap = torch.zeros( (self.ch, self.envHeight, self.envWidth), dtype=torch.float32 )

        self.isCuda = isCuda
        self.gpuId = gpuId

        theta = max(theta, np.pi/2.0 + 0.01 )
        theta = (theta - np.pi/2.0)/ np.pi*4 - 1
        theta = 0.5 * np.log((1 + theta ) / (1 - theta ) )
        phi = phi / np.pi
        phi = 0.5 * np.log((1 + phi ) / (1-phi ) )

        thetaSky = max(thetaSky, np.pi/2.0 + 0.01 )
        thetaSky = (thetaSky - np.pi/2.0)/ np.pi*4 - 1
        thetaSky = 0.5 * np.log((1 + thetaSky ) / (1 - thetaSky ) )
        phiSky = phiSky / np.pi
        phiSky = 0.5 * np.log((1+phiSky) / (1-phiSky) )

        thetaGrd = max(thetaGrd, np.pi/2.0 + 0.01 )
        thetaGrd = (thetaGrd - np.pi/2.0)/ np.pi*4 - 1
        thetaGrd = 0.5 * np.log((1 + thetaGrd ) / (1 - thetaGrd ) )
        phiGrd = phiGrd / np.pi
        phiGrd = 0.5 * np.log((1+phiGrd ) / (1-phiGrd ) )

        weight= np.log(weight + 1e-6).squeeze()
        weight = torch.from_numpy(weight )

        theta = torch.zeros( (1), dtype = torch.float32 ) + theta
        phi = torch.zeros( (1), dtype = torch.float32 ) + phi
        lamb = torch.log(torch.ones(1, dtype = torch.float32 ) * np.pi * 2.0 )

        weightSky = np.log(weightSky + 1e-6).squeeze()
        weightSky = torch.from_numpy(weightSky )

        thetaSky = torch.zeros( (1), dtype = torch.float32 ) + thetaSky
        phiSky = torch.zeros( (1), dtype = torch.float32 ) + phiSky
        lambSky = torch.log(torch.ones(1, dtype = torch.float32 ) * np.pi * 2.0 )

        weightGrd = np.log(weightGrd + 1e-6).squeeze()
        weightGrd = torch.from_numpy(weightGrd )

        thetaGrd = torch.zeros( (1), dtype = torch.float32 ) + thetaGrd
        phiGrd = torch.zeros( (1), dtype = torch.float32 ) + phiGrd
        lambGrd = torch.log(torch.ones(1, dtype = torch.float32 ) * np.pi * 2.0 )

        self.weight = weight.unsqueeze(-1).unsqueeze(-1)
        self.theta = theta.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        self.phi = phi.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        self.lamb = lamb.unsqueeze(-1).unsqueeze(-1)

        self.weightSky = weightSky.unsqueeze(-1).unsqueeze(-1)
        self.thetaSky = thetaSky.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        self.phiSky = phiSky.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        self.lambSky = lambSky.unsqueeze(-1).unsqueeze(-1)

        self.weightGrd = weightGrd.unsqueeze(-1).unsqueeze(-1)
        self.thetaGrd = thetaGrd.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        self.phiGrd = phiGrd.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        self.lambGrd = lambGrd.unsqueeze(-1).unsqueeze(-1)

        if isCuda:
            self.ls = self.ls.cuda(self.gpuId )

            self.weight = self.weight.cuda()
            self.theta = self.theta.cuda()
            self.phi = self.phi.cuda()
            self.lamb = self.lamb.cuda()

            self.weightSky = self.weightSky.cuda()
            self.thetaSky = self.thetaSky.cuda()
            self.phiSky = self.phiSky.cuda()
            self.lambSky = self.lambSky.cuda()

            self.weightGrd = self.weightGrd.cuda()
            self.thetaGrd = self.thetaGrd.cuda()
            self.phiGrd = self.phiGrd.cuda()
            self.lambGrd = self.lambGrd.cuda()

            self.envmap = self.envmap.cuda(self.gpuId )
            self.W = self.W.cuda(self.gpuId )

        self.weight.requires_grad = True
        self.lamb.requires_grad = True
        self.weightSky.requires_grad = True
        self.lambSky.requires_grad = True
        self.weightGrd.requires_grad = True
        self.lambGrd.requires_grad = True

        self.mseLoss = nn.MSELoss(size_average = False )
        self.optEnvAdam = optim.Adam([self.weight, self.lamb,
                                      self.weightSky, self.lambSky, self.weightGrd, self.lambGrd], lr=1e-2 )

    def renderSG(self, theta, phi, lamb, weight,
                 thetaSky, phiSky, lambSky, weightSky,
                 thetaGrd, phiGrd, lambGrd, weightGrd ):

        thetaAll = torch.cat([theta, thetaSky, thetaGrd], dim=0 )
        phiAll = torch.cat([phi, phiSky, phiGrd], dim=0 )
        lambAll = torch.cat([lamb, lambSky, lambGrd], dim=0 )
        weightAll = torch.cat([weight, weightSky, weightGrd], dim=1 )

        ######
        axisX = torch.sin(thetaAll ) * torch.cos(phiAll )
        axisY = torch.sin(thetaAll ) * torch.sin(phiAll )
        axisZ = torch.cos(thetaAll )

        axis = torch.cat([axisX, axisY, axisZ], dim=-1)

        mi = lambAll * torch.clamp(torch.sum(axis * self.ls, dim = -1) -1, max=0 )
        envmaps = weightAll.unsqueeze(-1) * torch.exp(mi.unsqueeze(0) )
        envmap = torch.sum(envmaps, dim=1 )

        return envmap

    def transformPara(self ):
        theta = 0.25*np.pi * (torch.tanh(self.theta )+1) + np.pi/2.0 + 0.01
        phi = np.pi * torch.tanh(self.phi )
        weight = torch.exp(self.weight )
        lamb = torch.clamp(torch.exp(self.lamb ), max=10000 )

        thetaSky = 0.25 * np.pi * (torch.tanh(self.thetaSky )+1) + np.pi/2.0 + 0.01
        phiSky = np.pi * torch.tanh(self.phiSky )
        weightSky = torch.exp(self.weightSky )
        lambSky = torch.clamp(torch.exp(self.lambSky ), max=10000 )

        thetaGrd = 0.25 * np.pi * (torch.tanh(self.thetaGrd )+1) + np.pi/2.0 + 0.01
        phiGrd = np.pi * torch.tanh(self.phiGrd )
        weightGrd = torch.exp(self.weightGrd )
        lambGrd = torch.clamp(torch.exp(self.lambGrd ), max=10000 )

        return theta, phi, lamb, weight, \
            thetaSky, phiSky, lambSky, weightSky, \
            thetaGrd, phiGrd, lambGrd, weightGrd


    def optimizeAdam(self, envmap ):
        assert(envmap.shape[0] == self.ch
            and envmap.shape[1] == self.envHeight
            and envmap.shape[2] == self.envWidth )
        self.envmap.data.copy_(torch.from_numpy(envmap  ) )

        minLoss = 2e20
        recImageBest = None

        weightBest = None
        thetaBest = None
        phiBest = None
        lambBest = None

        weightSkyBest = None
        thetaSkyBest = None
        phiSkyBest = None
        lambSkyBest = None

        weightGrdBest = None
        thetaGrdBest = None
        phiGrdBest = None
        lambGrdBest = None

        self.loss = None

        for i in range(0, self.niter ):
            print('Iteration %d' % i )

            for j in range(0, 100):
                theta, phi, lamb, weight, \
                    thetaSky, phiSky, lambSky, weightSky, \
                    thetaGrd, phiGrd, lambGrd, weightGrd = \
                    self.transformPara()

                recImage = self.renderSG(theta, phi, lamb, weight,
                                         thetaSky, phiSky, lambSky, weightSky,
                                         thetaGrd, phiGrd, lambGrd, weightGrd )
                loss = self.mseLoss(
                        recImage * self.W,
                        self.envmap * self.W )

                self.loss = loss

                self.optEnvAdam.zero_grad()
                loss.backward()
                self.iterCount += 1

                self.optEnvAdam.step()

            print('Step %d Loss: %f' % (self.iterCount, (loss.item() / self.envWidth / self.envHeight / self.ch ) ) )


            if self.loss.cpu().data.item() < minLoss:
                if torch.isnan(torch.sum(self.theta ) ) or \
                        torch.isnan(torch.sum(self.phi ) ) or \
                        torch.isnan(torch.sum(self.weight ) ) or \
                        torch.isnan(torch.sum(self.lamb ) ) or \
                        torch.isinf(torch.sum(self.theta ) ) or \
                        torch.isinf(torch.sum(self.phi ) ) or \
                        torch.isinf(torch.sum(self.weight ) ) or \
                        torch.isinf(torch.sum(self.lamb ) ):
                    break
                else:

                    recImage = self.renderSG(theta, phi, lamb, weight,
                                             thetaSky, phiSky, lambSky, weightSky,
                                             thetaGrd, phiGrd, lambGrd, weightGrd )

                    recImageBest = recImage.cpu().data.numpy()

                    thetaBest = theta.data.cpu().numpy().squeeze()
                    phiBest = phi.data.cpu().numpy().squeeze()
                    lambBest = lamb.data.cpu().numpy().squeeze()
                    weightBest = weight.data.cpu().numpy().squeeze()

                    thetaSkyBest = thetaSky.data.cpu().numpy().squeeze()
                    phiSkyBest = phiSky.data.cpu().numpy().squeeze()
                    lambSkyBest = lambSky.data.cpu().numpy().squeeze()
                    weightSkyBest = weightSky.data.cpu().numpy().squeeze()

                    thetaGrdBest = thetaGrd.data.cpu().numpy().squeeze()
                    phiGrdBest = phiGrd.data.cpu().numpy().squeeze()
                    lambGrdBest = lambGrd.data.cpu().numpy().squeeze()
                    weightGrdBest = weightGrd.data.cpu().numpy().squeeze()

                    minLoss = self.loss.cpu()

                    del theta, phi, weight, lamb, recImage, \
                        thetaSky, phiSky, weightSky, lambSky, \
                        thetaGrd, phiGrd, weightGrd, lambGrd
            else:
                break

        return recImageBest, \
            thetaBest, phiBest, lambBest, weightBest, \
            thetaSkyBest, phiSkyBest, lambSkyBest, weightSkyBest, \
            thetaGrdBest, phiGrdBest, lambGrdBest, weightGrdBest


# {SPLIT}/cam.txt to {SPLIT}.npy

from pathlib import Path
import numpy as np

import torch
import torch.nn.functional as NF
import mitsuba
mitsuba.set_variant('cuda_ad_rgb')

import json

import cv2
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import math
from .convert_helper import *

import matplotlib.pyplot as plt
import scipy
from scipy.spatial.transform import Rotation

if __name__ == "__main__":
    SCENE_PATH = '../../data_gen/kitchen'
    SPLIT = 'train'
    cam_path = Path(SCENE_PATH)/Path(SPLIT)/Path('cam.txt')
    cam_params = read_cam_params(cam_path)

    # read original [R|t]
    pose_list = []
    for i,cam_param in enumerate(cam_params):
        origin, lookat, up = np.split(cam_param.T, 3, axis=1)
        origin = origin.flatten()
        lookat = lookat.flatten()
        up = up.flatten()
        at_vector = normalize_v(lookat - origin)
        assert np.amax(np.abs(np.dot(at_vector.flatten(), up.flatten()))) < 2e-3 # two vector should be perpendicular

        t = origin.reshape((3, 1)).astype(np.float32)
        R = np.stack((np.cross(-up, at_vector), -up, at_vector), -1).astype(np.float32)
        
        pose_list.append(np.hstack((R, t)))

    pose_list = np.stack(pose_list,0)
    pose_list = torch.from_numpy(pose_list)

    pose_list = torch.cat([pose_list,torch.zeros_like(pose_list[:,:1,:])],1)
    pose_list[:,3,3] = 1.0

    # convert to blender format [translation, euler angle]
    coord_conv = [0,2,1]
    blender_pose = np.zeros((len(pose_list),2,3))
    for i,pose in enumerate(pose_list):
        pos = pose[:,3].clone()
        coord_conv = [0,2,1]
        pos = pos[coord_conv]
        pos[1] = -pos[1]
        #pos = pos*scale_to_blend + trans_to_blend
        Rs = pose[:,:3].clone()
        Rs[:,1] = -Rs[:,1]
        Rs[:,2] = -Rs[:,2]
        Rs = Rs[coord_conv]
        Rs[1] = -Rs[1]
        angle = Rotation.from_matrix(Rs).as_euler('xyz',degrees=False)
        blender_pose[i,0] = pos.numpy()
        blender_pose[i,1] = angle

    # save file
    np.save(os.path.join(SCENE_PATH,'{}.npy'.format(SPLIT)),blender_pose)
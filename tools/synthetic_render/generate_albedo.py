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
import sys
sys.path.append('..')
from .convert_helper import *
import drjit as dr

import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.spatial.transform import Rotation


from scipy.spatial.transform import Rotation
def get_albedo(scene,spp,seed=0):
    # mitsuba integrator that calculates material reflectance metric
    integrator = mitsuba.ad.common.ADIntegrator()
    sensor = scene.sensors()[0]
    bsdf_ctx = mitsuba.BSDFContext()
    
    sampler, spp = integrator.prepare(sensor, seed, spp, [])
    
    ray, _, pos, _ = integrator.sample_rays(scene, sensor,sampler)
    
    si = scene.ray_intersect(ray)
    valid = si.is_valid()
    bsdf = si.bsdf(ray)
    bsdf_sample,bsdf_weight = bsdf.sample(bsdf_ctx,si,
                        sampler.next_1d(),sampler.next_2d(),valid)
    
    block = sensor.film().create_block()
    block.set_coalesce(block.coalesce() and spp >= 4)
    block.put(
        pos=pos,
        wavelengths=ray.wavelengths,
        value=bsdf_weight,
        alpha=dr.select(valid, mitsuba.Float(1), mitsuba.Float(0))
    )
    sensor.film().put_block(block)
    image = sensor.film().develop()
    return image

if __name__ == "__main__":
    device = torch.device(0)
    SCENE = 'livingroom0'
    SCENE_PATH = '../../data_gen/living-room'
    SPLIT = 'train'

    # read xml file
    img_hw = [320,640]
    scene = mitsuba.load_file(os.path.join(SCENE_PATH,'test.xml'))
    params = mitsuba.traverse(scene)

    # convert euler angle, translation to [R|t]
    blender_pose = np.load(os.path.join(SCENE_PATH,'{}.npy'.format(SPLIT)))
    coord_conv = [0,2,1]
    mitsuba_pose = []
    for i,pose in enumerate(blender_pose):
        Rs = Rotation.from_euler('xyz',pose[1]).as_matrix()
        # coordinate convention
        Rs[1] = -Rs[1]
        Rs = Rs[coord_conv]
        Rs[:,2] = - Rs[:,2]
        Rs[:,0] *= -1
        
        ts = pose[0]
        #ts = (ts-trans_to_blend)/scale_to_blend
        ts[1] = -ts[1]
        ts = ts[coord_conv]
        mitsuba_pose.append(np.concatenate([Rs,ts.reshape(3,1)],-1))

    OUTPUT_PATH = os.path.join(
                SCENE_PATH,SPLIT,'albedo')
    os.makedirs(OUTPUT_PATH,exist_ok=True)

    # estimate material reflectance
    spp = 256
    img_id = 0
    for pose in tqdm(mitsuba_pose):
        params['PerspectiveCamera.to_world'] = mitsuba.Transform4f.look_at(origin=pose[:3,3], target=pose[:3,3]+pose[:3,2], up=pose[:3,1])
        params.update()
        img = get_albedo(scene,spp)
        cv2.imwrite(os.path.join(OUTPUT_PATH,'{:03d}.exr'.format(img_id)),img.numpy()[...,[2,1,0]])
        img_id += 1
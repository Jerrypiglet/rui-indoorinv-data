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
.convert_helper import *

from argparse import Namespace, ArgumentParser


from scipy.spatial.transform import Rotation
from tqdm import tqdm

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--scene', type=str, required=True)
    parser.add_argument('--split', type=str, required=True)
    
    device = torch.device(0)
    
    args = parser.parse_args()
    
    img_hw = [320,640]#[500,500]

    SCENE_PATH = args.scene#'../../../data_gen/living-room/'
    SPLIT = args.split#'val'
    scene = mitsuba.load_file(os.path.join(SCENE_PATH,'test.xml')) \
          if args.split!='relight' else mitsuba.load_file(os.path.join(SCENE_PATH,'test-relight.xml')) 

    if args.split != 'relight':
        blender_pose = np.load(os.path.join(SCENE_PATH,'{}.npy'.format(SPLIT)))
    else:
        blender_pose = np.load(os.path.join(SCENE_PATH,'val.npy'))
    coord_conv = [0,2,1]
    mitsuba_pose = []
    for i,pose in enumerate(blender_pose):
        Rs = Rotation.from_euler('xyz',pose[1]).as_matrix()
        Rs[1] = -Rs[1]
        Rs = Rs[coord_conv]
        Rs[:,2] = - Rs[:,2]
        Rs[:,0] *= -1

        ts = pose[0]
        #ts = (ts-trans_to_blend)/scale_to_blend
        ts[1] = -ts[1]
        ts = ts[coord_conv]
        mitsuba_pose.append(np.concatenate([Rs,ts.reshape(3,1)],-1))

    # save to json
    JSON_PATH = os.path.join(SCENE_PATH,SPLIT,'transforms.json')
    with open(JSON_PATH,'r')as f:
        meta = json.load(f)

    for pose,frame in zip(mitsuba_pose,meta['frames']):
        frame['transform_matrix'] = listify_matrix(np.concatenate([
            pose,np.array([0,0,0,1])[None]],0))

    with open(JSON_PATH, 'w') as out_file:
        json.dump(meta, out_file, indent=4)

    params = mitsuba.traverse(scene)


    OUT_PATH = os.path.join(SCENE_PATH,SPLIT,'Image')
    os.makedirs(OUT_PATH,exist_ok=True)
    for i in tqdm(range(len(mitsuba_pose))):
        pose = mitsuba_pose[i]
        params['PerspectiveCamera.to_world'] = mitsuba.Transform4f.look_at(origin=pose[:3,3], target=pose[:3,3]+pose[:3,2], up=pose[:3,1])
        params.update()
        img = mitsuba.render(scene,spp=4096).torch().cpu().numpy()
        cv2.imwrite(os.path.join(OUT_PATH,'{:03d}_0001.exr'.format(i)),img[:,:,[2,1,0]])
        cv2.imwrite(os.path.join(OUT_PATH,'{:03d}_0001.png'.format(i)),((img[:,:,[2,1,0]]**(1/2.2)).clip(0,1)*255).astype(np.uint8))
import numpy as np
import os
import cv2
import xml.etree.ElementTree as et
import struct
import scipy.ndimage as ndimage
from pathlib import Path
from utils_misc import yellow, yellow_text
import shutil
import matplotlib.pyplot as plt
from tqdm import tqdm
import mitsuba as mi

from lib.utils_io import load_matrix, resize_intrinsics
from .utils_mitsubaScene_scene import findSupport, adjustHeight, checkPointInPolygon, moveBoxInWall
from utils_OR.utils_OR_cam import origin_lookat_up_to_R_t
# from .utils_objs import loadMesh, writeMesh, computeBox
# from .utils_cam_transform import computeTransform
# from .utils_xml import transformToXml

def func_mitsubaScene_sample_poses(
        mitsubaScene, 
        lverts, boxes,
        cam_params_dict, 
        sample_pose_if_vis_plt: bool=False):

    samplePoint = cam_params_dict['samplePoint']
    sampleNum = cam_params_dict['sampleNum']
    heightMin = cam_params_dict['heightMin']
    heightMax = cam_params_dict['heightMax']
    distMin = cam_params_dict['distMin']
    distMax = cam_params_dict['distMax']
    thetaMin = cam_params_dict['thetaMin']
    thetaMax = cam_params_dict['thetaMax']
    phiMin = cam_params_dict['phiMin']
    phiMax = cam_params_dict['phiMax']
    distRaysMin = cam_params_dict['distRaysMin']
    distRaysMedian = cam_params_dict['distRaysMedianMin']

    cam_loc_bbox = cam_params_dict.get('cam_loc_bbox', [])
    exclude_obj_id_list = cam_params_dict.get('exclude_obj_id_list', [])

    wallVertices = []
    floorHeight = lverts[:, 1].min()
    for n in range(0, lverts.shape[0]):
        vert = lverts[n, :]
        if np.abs(vert[1] - floorHeight) < 0.1:
            wallVertices.append(vert)

    X = [pt[0] for pt in wallVertices]
    Z = [pt[2] for pt in wallVertices]

    wallVertices_center = np.array(wallVertices).mean(axis=0)
    wallVertices_smaller = [(_-wallVertices_center)*0.7+wallVertices_center for _ in wallVertices]

    meanPoint = [sum(X) / len(X), 0, sum(Z) / len(Z)]
    meanPoint = np.array(meanPoint, dtype = np.float32)

    thetaMin = thetaMin / 180.0 * np.pi
    thetaMax = thetaMax / 180.0 * np.pi
    phiMin = phiMin / 180.0 * np.pi
    phiMax = phiMax / 180.0 * np.pi

    # yMin = np.sin(thetaMin)
    # yMax = np.sin(thetaMax)
    # xMin = np.sin(phiMin)
    # xMax = np.sin(phiMax)

    # Compute the segment length
    totalLen = 0
    j = len(wallVertices) - 1
    for i in range(len(wallVertices)):
        l = (X[i] - X[j]) * (X[i] - X[j]) \
                + (Z[i] - Z[j]) * (Z[i] - Z[j])
        l = np.sqrt(l)
        totalLen += l
        j = i
    assert samplePoint != 0
    segLen = totalLen / samplePoint

    # Sample the camera poses
    j = len(wallVertices) - 1

    camPoses = []
    validPointCount = 0

    if sample_pose_if_vis_plt:
        plt.figure(figsize=(15, 12))
        plt.title('[each wall & cams sampled along each wall & normal of each wall] thick solid line (R-G-B-grey)')
        plt.scatter(0, 0, marker='*', color='m')
        # for i in range(4):
        #     plt.plot(wallVertices[i][[0, 2]], wallVertices[(i-1)%4][[0, 2]])

    normal_list = []
    midPoint_list = []
    origin_list = []
    direc_list = []
    totalLen_list = []
    cam_colors = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [0.5, 0.5, 0.5]]

    print('Sampling poses...')

    # visualize additional camera location constraining bbox
    if cam_loc_bbox != [] and sample_pose_if_vis_plt:
        for ii, cam_loc_v in enumerate(cam_loc_bbox):
            jj = (ii + 1) % len(cam_loc_bbox)
            v1 = cam_loc_bbox[ii]; v2 = cam_loc_bbox[jj]
            # print(i, j, v1, v2)
            plt.plot(np.array([v1[0], v2[0]]), np.array([v1[1], v2[1]]), color='k', ls='-', linewidth=6) # additional wall contour as solid line: BLACK


    for i in tqdm(range(len(wallVertices))):
        # compute the segment direction
        direc = np.array( [X[i] - X[j], 0, Z[i] - Z[j]], dtype = np.float32)
        if sample_pose_if_vis_plt:
            # print('=====', i, j, X[i], X[j])
            plt.plot(np.array([X[i], X[j]]), np.array([Z[i], Z[j]]), color=cam_colors[i], ls='-', linewidth=6) # wall contour as solid line: R-G-B-grey
        totalLen = np.sqrt(np.sum(direc * direc))
        if totalLen == 0:
            continue

        direc = direc / totalLen

        # Determine the normal direction
        normal = np.array([direc[2], 0, -direc[0]], dtype = np.float32)
        normal = normal / np.sqrt(np.sum(normal * normal))

        midPoint = np.array([0.5*(X[i] + X[j]), 0, 0.5*(Z[i] + Z[j])], dtype = np.float32)
        sp1 = midPoint + normal * 0.1
        sp2 = midPoint - normal * 0.1

        isIn1 = checkPointInPolygon(wallVertices, sp1)
        isIn2 = checkPointInPolygon(wallVertices, sp2)
        assert(isIn1 != isIn2)

        if isIn1 == False and isIn2 == True:
            normal = -normal

        origin = np.array([X[j], 0, Z[j]], dtype = np.float32)

        origin_list.append(origin)
        normal_list.append(normal)
        midPoint_list.append(midPoint)
        direc_list.append(direc)
        totalLen_list.append(totalLen)

        j = i
        # print(i, j, direc, normal, origin)
        if sample_pose_if_vis_plt:
            print('sampleNum', sampleNum)
            print('normal', normal)
            plt.scatter(midPoint[0], midPoint[2], c='r')
            plt.plot([midPoint[0], midPoint[0]+normal[0]], [midPoint[2], midPoint[2]+normal[2]], color=cam_colors[i], ls='-') # wall normal as dashed line; same color as wall

    if sample_pose_if_vis_plt:
        for box in boxes:
            bverts = box[0][0:4, :]
            plt.plot(box[0][[0,1,2,3,0], 0], box[0][[0,1,2,3,0], 2], color='y', ls='--') # furnitures as green dashed line

    '''
    wallVertices: [np.array(3,)] * 4, clock-wise dir
    '''
    for i, origin, normal, direc, midPoint, totalLen in tqdm(zip(range(len(wallVertices)), origin_list, normal_list, direc_list, midPoint_list, totalLen_list)):
        accumLen = 0.2 * segLen
        cam_color = cam_colors[i]

        while accumLen < totalLen:
            # compute point location
            # cnt = 0
            for cnt in range(0, sampleNum):
            # while cnt < sampleNum:
                pointLoc = origin + accumLen * direc
                # if sample_pose_if_vis_plt:
                #     plt.scatter(pointLoc[0], pointLoc[2], color='y', marker='*')
                pointLoc += (np.random.random() * (distMax - distMin) \
                        + distMin) * normal
                pointLoc[1] = np.random.random() * (heightMax - heightMin) \
                        + heightMin + floorHeight

                if cam_loc_bbox != []:
                    is_in_cam_loc_bbox = checkPointInPolygon([[_v[0], 0., _v[1]] for _v in cam_loc_bbox], pointLoc)
                    if not is_in_cam_loc_bbox:
                        print(cnt, yellow_text('DISCARDED pose: point is outside **cam_loc_bbox**'))
                        continue

                isIn = checkPointInPolygon(wallVertices, pointLoc)
                # isIn = checkPointInPolygon(wallVertices_smaller, pointLoc)
                dist_to_wall_list = [np.sum((pointLoc[[0, 2]]-midPoint_[[0, 2]])*normal_[[0, 2]]) for normal_, midPoint_ in zip(normal_list, midPoint_list)]
                dist_to_wall = min(dist_to_wall_list)

                if not isIn:
                    print(cnt, yellow_text('DISCARDED pose: point is outside the room'))
                    continue
                elif dist_to_wall < distMin:
                    print(cnt, yellow_text('DISCARDED pose: point closer to walls than distMin (%.2f<%.2f)'%(dist_to_wall, distMin)))
                    continue
                else:
                    # check if the point will in any bounding boxes
                    isOverlap = False
                    overlap_shape_id = 'N/A'
                    for box in boxes:
                        bverts = box[0][0:4, :]
                        bminY = box[0][0, 1]
                        bmaxY = box[0][4, 1]

                        if pointLoc[1] > bminY and pointLoc[1] < bmaxY:
                            isOverlap = checkPointInPolygon(bverts, pointLoc)
                            if isOverlap:
                                overlap_shape_id = box[2]
                                break

                    if isOverlap and overlap_shape_id not in exclude_obj_id_list:
                        print(cnt, yellow_text('DISCARDED pose: point overlaps with %s'%overlap_shape_id))
                        continue

                    validPointCount += 1
                    camPose = np.zeros((3, 3), dtype=np.float32)
                    camPose[0, :] = pointLoc

                    xAxis = normal
                    xAxis = xAxis / np.maximum(np.sqrt(
                        np.sum(xAxis * xAxis)), 1e-6)
                    zAxis = np.array([0, 1, 0], dtype=np.float32) # ASSUME y+ is up
                    yAxis = np.cross(zAxis, xAxis)

                    Az_phi = (phiMax - phiMin) * np.random.random() + phiMin # on ground plane
                    # Az_phi = 0. # debug: camera points towards wall normal (zero yaw)
                    El_theta = (thetaMax - thetaMin) * np.random.random() + thetaMin # along up
                    ly = np.sin(Az_phi) * np.cos(El_theta)
                    lx = np.cos(Az_phi) * np.cos(El_theta)
                    lz = np.sin(El_theta)
                    targetDirec = xAxis * lx + yAxis * ly + zAxis * lz
                    target = pointLoc + targetDirec
                    up = zAxis - np.sum(zAxis * targetDirec) * targetDirec
                    up = up / np.sqrt(np.sum(up *  up))

                    # print('pitch y: %.2f'%targetDirec[1], 'targetDirec:', targetDirec, 'up:', up)

                    camPose[1, :] = target
                    camPose[2, :] = up

                    '''
                    render depth with current pose
                    '''
                    min_depth = -1.
                    if distRaysMin > 0:
                        _origin, _lookat, _up = np.split(camPose.T, 3, axis=1)
                        (_R, _t), _lookatvector = origin_lookat_up_to_R_t(_origin, _lookat, _up)
                        H, W = mitsubaScene.im_H_load//4, mitsubaScene.im_W_load//4
                        scale_factor = [t / s for t, s in zip((H, W), (mitsubaScene.im_H_load, mitsubaScene.im_W_load))]
                        K = resize_intrinsics(mitsubaScene.K, scale_factor)
                        _pose = np.hstack((_R, _t))
                        tmp_cam_rays = mitsubaScene.get_cam_rays_list(H, W, [K], [_pose])[0]
                        rays_o, rays_d, ray_d_center = tmp_cam_rays
                        rays_o_flatten, rays_d_flatten = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)

                        xs_mi = mi.Point3f(mitsubaScene.to_d(rays_o_flatten))
                        ds_mi = mi.Vector3f(mitsubaScene.to_d(rays_d_flatten))
                        rays_mi = mi.Ray3f(xs_mi, ds_mi)
                        ret = mitsubaScene.mi_scene.ray_intersect(rays_mi) # https://mitsuba.readthedocs.io/en/stable/src/api_reference.html?highlight=write_ply#mitsuba.Scene.ray_intersect
                        rays_v_flatten = ret.t.numpy()[:, np.newaxis] * rays_d_flatten

                        mi_depth = np.sum(rays_v_flatten.reshape(H, W, 3) * ray_d_center.reshape(1, 1, 3), axis=-1)
                        invalid_depth_mask = np.logical_or(np.isnan(mi_depth), np.isinf(mi_depth))
                        mi_depth_ = mi_depth[~invalid_depth_mask] # (N,)
                        if mi_depth_.size == 0:
                            continue
                        min_depth = np.min(mi_depth_)
                        median_depth = np.median(mi_depth_)
                        if min_depth < distRaysMin or median_depth < distRaysMedian:
                            print(yellow_text('DISCARDED pose: camera too close to the scene: min=%.2f(thres %.2f), median=%.2f(thres %.2f); discarded.'%(min_depth, distRaysMin, median_depth, distRaysMedian)))
                            '''
                            uncomment to show invalid image
                            '''
                            # plt.figure()
                            # plt.imshow(mi_depth)
                            # plt.colorbar()
                            # plt.show()
                            continue
                        # print(yellow('!!!!'), min_depth>distRaysMin, min_depth, distRaysMin)
                        

                    camPoses.append(camPose)
                    print('Appended valid pose; min_depth: %.2f'%min_depth)

                    if sample_pose_if_vis_plt:
                        plt.scatter(pointLoc[0], pointLoc[2], color=cam_color)
                        plt.text(pointLoc[0]+0.1, pointLoc[2]+0.1, '%.2f'%dist_to_wall, color=cam_color)
                        targetVis = pointLoc + 0.5 * targetDirec
                        plt.plot([pointLoc[0], targetVis[0]], [pointLoc[2], targetVis[2]], color=cam_color, ls='-')

                    cnt += 1
            accumLen += segLen

    if sample_pose_if_vis_plt:
        plt.grid()
        plt.axis('equal')
        # plt.title('a')
        plt.show()

    return camPoses

    
def mitsubaScene_sample_poses_one_scene(mitsubaScene, scene_dict: dict, cam_params_dict: dict, path_dict: dict):
    '''
    generate camera files for one scene: cam.txt, camInitial.txt -> dest_scene_path

    Adapted from code/utils_OR/DatasetCreation/sampleCameraPose.py
    '''

    threshold = cam_params_dict.get('threshold', 0.3) # 'the threshold to decide low quality mesh.'
    samplePoint = cam_params_dict.get('samplePoint', 100)

    sample_pose_if_vis_plt = cam_params_dict.get('sample_pose_if_vis_plt', False)
    print(yellow('Num of sample points %d'%(samplePoint)))

    lverts = scene_dict['lverts']
    boxes = scene_dict['boxes']
    cads = scene_dict['cads']

    # Build the relationship and adjust heights
    floorList, boxList = findSupport(lverts, boxes, cats=['']*len(boxes))
    adjustHeight(lverts, boxes, cads, floorList, boxList)

    # Sample initial camera pose
    isMove, isBeyondRange = moveBoxInWall(lverts, boxes, cads, threshold)
    cnt = 0
    while isMove == True and isBeyondRange == False:
        isMove, isBeyondRange = moveBoxInWall(lverts, boxes, cads, threshold)
        print('IterNum %d'%cnt)
        cnt += 1
        if cnt == 5 or isMove == False or isBeyondRange == True:
            break

    camPoses = func_mitsubaScene_sample_poses(
            mitsubaScene=mitsubaScene, 
            lverts=lverts, boxes=boxes, \
            cam_params_dict=cam_params_dict, 
            sample_pose_if_vis_plt=sample_pose_if_vis_plt)
    
    camNum = len(camPoses)
    assert camNum != 0

    return camPoses
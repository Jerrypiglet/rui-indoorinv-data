import numpy as np
import os
import cv2
import xml.etree.ElementTree as et
import struct
import scipy.ndimage as ndimage
from pathlib import Path
from utils_misc import get_list_of_keys, yellow
import shutil
import matplotlib.pyplot as plt
from tqdm import tqdm

from .utils_mitsubaScene_scene import findSupport, adjustHeight, checkPointInPolygon, moveBoxInWall
# from .utils_objs import loadMesh, writeMesh, computeBox
# from .utils_cam_transform import computeTransform
# from .utils_xml import transformToXml

def func_mitsubaScene_sample_poses(lverts, boxes,
        samplePoint, sampleNum,
        heightMin, heightMax,
        distMin, distMax,
        thetaMin, thetaMax,
        phiMin, phiMax,
        if_vis_plt: bool=False):

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

    yMin = np.sin(thetaMin)
    yMax = np.sin(thetaMax)
    xMin = np.sin(phiMin)
    xMax = np.sin(phiMax)

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

    if if_vis_plt:
        plt.figure(figsize=(15, 12))
        plt.scatter(0, 0, color='g')
        # for i in range(4):
            # plt.plot(wallVertices[i][[0, 2]], wallVertices[(i-1)%4][[0, 2]])

    normal_list = []
    midPoint_list = []
    origin_list = []
    direc_list = []
    totalLen_list = []

    print('Sampling poses...')

    for i in tqdm(range(len(wallVertices))):
        # compute the segment direction
        direc = np.array( [X[i] - X[j], 0, Z[i] - Z[j]], dtype = np.float32)
        if if_vis_plt:
            print('=====', i, j, X[i], X[j])
            plt.plot(np.array([X[i], X[j]]), np.array([Z[i], Z[j]]), 'k-')
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
        if if_vis_plt:
            print('sampleNum', sampleNum)
            print('normal', normal)
            plt.scatter(midPoint[0], midPoint[2], c='r')
            plt.plot([midPoint[0], midPoint[0]+normal[0]], [midPoint[2], midPoint[2]+normal[2]], 'k--')

    for i, origin, normal, direc, midPoint, totalLen in tqdm(zip(range(len(wallVertices)), origin_list, normal_list, direc_list, midPoint_list, totalLen_list)):
        accumLen = 0.2 * segLen

        while accumLen < totalLen:
            # compute point location
            for cnt in range(0, sampleNum):
                pointLoc = origin + accumLen * direc
                if if_vis_plt:
                    plt.scatter(pointLoc[0], pointLoc[2], color='y', marker='*')
                pointLoc += (np.random.random() * (distMax - distMin) \
                        + distMin) * normal
                pointLoc[1] = np.random.random() * (heightMax - heightMin) \
                        + heightMin + floorHeight

                isIn = checkPointInPolygon(wallVertices, pointLoc)
                # isIn = checkPointInPolygon(wallVertices_smaller, pointLoc)
                dist_to_wall_list = [np.sum((pointLoc[[0, 2]]-midPoint_[[0, 2]])*normal_[[0, 2]]) for normal_, midPoint_ in zip(normal_list, midPoint_list)]
                dist_to_wall = min(dist_to_wall_list)

                if not isIn:
                    print('Warning: %d point is outside the room'%validPointCount)
                    continue
                elif dist_to_wall < distMin:
                    print('Warning: %d point closer to walls than distMin (%.2f<%.2f)'%(validPointCount, dist_to_wall, distMin))
                    continue
                else:
                    # check if the point will in any bounding boxes
                    isOverlap = False
                    for box in boxes:
                        bverts = box[0][0:4, :]
                        bminY = box[0][0, 1]
                        bmaxY = box[0][4, 1]

                        if pointLoc[1] > bminY and pointLoc[1] < bmaxY:
                            isOverlap = checkPointInPolygon(bverts, pointLoc)
                            if isOverlap:
                                break

                    if isOverlap:
                        print('Warning: %d point overlaps with furniture'%validPointCount)
                        continue

                    validPointCount += 1
                    camPose = np.zeros((3, 3), dtype=np.float32)
                    camPose[0, :] = pointLoc

                    # zAxis = normal
                    # zAxis = zAxis / np.maximum(np.sqrt(
                    #     np.sum(zAxis * zAxis)), 1e-6)
                    # yAxis = np.array([0, 1, 0], dtype=np.float32)
                    # xAxis = np.cross(yAxis, zAxis)

                    # yaw = (xMax - xMin) * np.random.random() + xMin
                    # pitch = (yMax - yMin) * np.random.random() + yMin

                    # targetDirec_ =  zAxis + pitch * yAxis + yaw * xAxis
                    # targetDirec = targetDirec_ / np.sqrt(np.sum(targetDirec_ * targetDirec_))

                    xAxis = normal
                    xAxis = xAxis / np.maximum(np.sqrt(
                        np.sum(xAxis * xAxis)), 1e-6)
                    zAxis = np.array([0, 1, 0], dtype=np.float32)
                    yAxis = np.cross(zAxis, xAxis)

                    Az_phi = (phiMax - phiMin) * np.random.random() + phiMin # on ground plane
                    El_theta = (thetaMax - thetaMin) * np.random.random() + thetaMin # along up
                    lx = np.sin(Az_phi) * np.cos(El_theta)
                    ly = np.cos(Az_phi) * np.cos(El_theta)
                    lz = np.sin(El_theta)
                    targetDirec = xAxis * lx + yAxis * ly + zAxis * lz
                    target = pointLoc + targetDirec
                    up = zAxis - np.sum(zAxis * targetDirec) * targetDirec
                    up = up / np.sqrt(np.sum(up *  up))

                    print('pitch y: %.2f'%targetDirec[1], 'targetDirec:', targetDirec, 'up:', up)

                    camPose[1, :] = target
                    camPose[2, :] = up
                    camPoses.append(camPose)

                    # if (camPose[1, :] - camPose[0, :])[1] > np.sin(20/180*np.pi):
                    #     import ipdb; ipdb.set_trace()
                    # if (camPose[1, :] - camPose[0, :])[1] <np.sin(-60/180*np.pi):
                    #     import ipdb; ipdb.set_trace()

                    if if_vis_plt:
                        plt.scatter(pointLoc[0], pointLoc[2], c='b')
                        plt.text(pointLoc[0]+0.1, pointLoc[2]+0.1, '%.2f'%dist_to_wall)
                        plt.plot([pointLoc[0], target[0]], [pointLoc[2], target[2]], 'b-')

            accumLen += segLen

    if if_vis_plt:
        plt.grid()
        plt.axis('equal')
        plt.title('a')
        plt.show()
        assert False

    return camPoses

    
def mitsubaScene_sample_poses_one_scene(scene_dict: dict, program_dict: dict, param_dict: dict, path_dict: dict):
    '''
    generate camera files for one scene: cam.txt, camInitial.txt -> dest_scene_path

    Adapted from code/utils_OR/DatasetCreation/sampleCameraPose.py
    '''

    threshold = param_dict.get('threshold', 0.3) # 'the threshold to decide low quality mesh.'
    samplePoint = param_dict.get('samplePoint', 100)
    sampleNum = param_dict.get('sampleNum', 3)
    heightMin = param_dict.get('heightMin', 1.4)
    heightMax = param_dict.get('heightMax', 1.8)
    distMin = param_dict.get('distMin', 0.3)
    distMax = param_dict.get('distMax', 1.5)
    thetaMin = param_dict.get('thetaMin', -60)
    thetaMax = param_dict.get('thetaMax', 20)
    phiMin = param_dict.get('phiMin', -45)
    phiMax = param_dict.get('phiMax', 45)

    if_vis_plt = param_dict.get('if_vis_plt', False)
    assert samplePoint > 0
    print(yellow('Num of sample points %d'%(samplePoint)))

    lverts = scene_dict['lverts']
    boxes = scene_dict['boxes']
    cads = scene_dict['cads']

    '''
    dump scene: buggy
    '''
    # import copy
    # num_vertices = 0
    # f_list = []
    # v_list = []
    # # for _ in cads:
    # #     vertices, faces = _[0], _[1]
    # #     f_list.append(copy.deepcopy(faces + num_vertices))
    # #     v_list.append(copy.deepcopy(vertices + num_vertices))
    # #     num_vertices += vertices.shape[0]
    # f = np.array(lfaces)
    # f_list.append(f+num_vertices)

    # v = np.array(lverts)
    # v_list.append(v+num_vertices)
    
    # writeMesh('./tmp_mesh.obj', np.vstack(v_list), np.vstack(f_list))

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

    camPoses = func_mitsubaScene_sample_poses(lverts, boxes, \
            samplePoint, sampleNum,
            heightMin, heightMax, \
            distMin, distMax, \
            thetaMin, thetaMax, \
            phiMin, phiMax, 
            if_vis_plt=if_vis_plt)
    
    camNum = len(camPoses)
    assert camNum != 0

    return camPoses

    with open(str(dest_scene_path / 'camInitial.txt'), 'w') as camOut:
        camOut.write('%d\n'%camNum)
        print('Final sampled camera poses: %d'%len(camPoses))
        for camPose in camPoses:
            for n in range(0, 3):
                camOut.write('%.3f %.3f %.3f\n'%\
                        (camPose[n, 0], camPose[n, 1], camPose[n, 2]))

    # Downsize the size of the image
    oldXML = dest_scene_path / 'main.xml'
    newXML = dest_scene_path / 'mainTemp.xml'
    camFile = dest_scene_path / 'camInitial.txt'
    if not oldXML.exists() or not camFile.exists():
        assert False

    tree = et.parse(oldXML)
    root  = tree.getroot()

    sensors = root.findall('sensor')
    for sensor in sensors:
        film = sensor.findall('film')[0]
        integers = film.findall('integer')
        for integer in integers:
            if integer.get('name') == 'width':
                integer.set('value', '160')
            if integer.get('name') == 'height':
                integer.set('value', '120')

    for elem in root.iter():
        if elem is not None and elem.get('value') is not None:
            if '../../../../../' in elem.get('value'):
                # print('=====', elem.get('value'))
                for tag in ['layoutMesh', 'BRDFOriginDataset', 'uv_mapped', 'EnvDataset']:
                    elem.set('value', elem.get('value').replace('../../../../../%s'%tag, str(Path(path_dict['OR_RAW'])/tag)))

    xmlString = transformToXml(root)
    with open(str(newXML), 'w') as xmlOut:
        xmlOut.write(xmlString)

    # Render depth and normal
    cmd = '%s -f %s -c %s -o %s -m %d'%(program_dict['program'], str(newXML), 'camInitial.txt', 'im.rgbe', 2)
    cmd += ' --forceOutput'
    os.system(cmd)

    cmd = '%s -f %s -c %s -o %s -m %d'%(program_dict['program'], str(newXML), 'camInitial.txt', 'im.rgbe', 4)
    cmd += ' --forceOutput'
    os.system(cmd)

    cmd = '%s -f %s -c %s -o %s -m %d'%(program_dict['program'], str(newXML), 'camInitial.txt', 'im.rgbe', 5)
    cmd += ' --forceOutput'
    os.system(cmd)

    # Load the normal and depth
    normalCosts = []
    depthCosts = []
    for n in range(0, camNum):
        # Load the depth and normal
        normalName = dest_scene_path / ('imnormal_%d.png'%n)
        maskName = dest_scene_path / ('immask_%d.png'%n)
        depthName = dest_scene_path / ('imdepth_%d.dat'%n)

        normal = cv2.imread(str(normalName))
        mask = cv2.imread(str(maskName))
        with open(str(depthName), 'rb') as fIn:
            hBuffer = fIn.read(4)
            height = struct.unpack('i', hBuffer)[0]
            wBuffer = fIn.read(4)
            width = struct.unpack('i', wBuffer)[0]
            dBuffer = fIn.read(4 * width * height)
            depth = np.asarray(struct.unpack('f' * height * width, dBuffer), dtype=np.float32)
            depth = depth.reshape([height, width])

        # Compute the ranking
        mask = mask[:, :, 0] > 0.4
        mask = ndimage.binary_erosion(mask, border_value=1, structure=np.ones((3, 3)))
        mask = mask.astype(np.float32)
        pixelNum = np.sum(mask)

        if pixelNum == 0:
            normalCosts.append(0)
            depthCosts.append(0)
            continue

        normal = normal.astype(np.float32)
        normal_gradx = np.abs(normal[:, 1:] - normal[:, 0:-1])
        normal_grady = np.abs(normal[1:, :] - normal[0:-1, :])
        ncost = (np.sum(normal_gradx) + np.sum(normal_grady)) / pixelNum

        dcost = np.sum(np.log(depth + 1)) / pixelNum

        normalCosts.append(ncost)
        depthCosts.append(dcost)

    normalCosts = np.array(normalCosts, dtype=np.float32)
    depthCosts = np.array(depthCosts, dtype=np.float32)

    normalCosts = (normalCosts - normalCosts.min()) \
            / (normalCosts.max() - normalCosts.min())
    depthCosts = (depthCosts - depthCosts.min()) \
            / (depthCosts.max() - depthCosts.min())

    totalCosts = normalCosts + 0.3 * depthCosts

    camIndex = np.argsort(totalCosts)
    camIndex = camIndex[::-1]

    camPoses_s = []
    selectedDir = dest_scene_path / 'selected'
    if selectedDir.exists():
        # os.system('rm -r %s'%selectedDir)
        shutil.rmtree(selectedDir)
    # os.system('mkdir %s'%selectedDir)
    selectedDir.mkdir(parents=False, exist_ok=False)

    for n in range(0, min(samplePoint, camNum)):
        camPoses_s.append(camPoses[camIndex[n]])

        normalName = dest_scene_path / ('imnormal_%d.png'%(camIndex[n]))
        os.system('cp %s %s'%(str(normalName), str(selectedDir)))

    with open(str(dest_scene_path / 'cam.txt'), 'w') as camOut:
        camOut.write('%d\n'%len(camPoses_s))
        print('Final sampled camera poses: %d'%len(camPoses_s))
        for camPose in camPoses_s:
            # assert (camPose[1, :] - camPose[0, :]) <= np.sin(-20/180*np.pi)
            for n in range(0, 3):
                camOut.write('%.3f %.3f %.3f\n'%\
                        (camPose[n, 0], camPose[n, 1], camPose[n, 2]))

    os.system('rm %s'%str(dest_scene_path / 'mainTemp.xml'))
    os.system('rm %s'%str(dest_scene_path / 'immask_*.png'))
    os.system('rm %s'%str(dest_scene_path / 'imdepth_*.dat'))
    os.system('rm %s'%str(dest_scene_path / 'imnormal_*.png'))

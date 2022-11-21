import numpy as np
from pathlib import Path, PosixPath
from utils_OR.utils_OR_cam import read_cam_params
from utils_misc import get_list_of_keys

def get_cam_params(scene_info_dict):
    dest_scene_path = get_list_of_keys(scene_info_dict, ['dest_scene_path'], [PosixPath])[0]
    cam_file = dest_scene_path / 'cam.txt'
    assert cam_file.exists(), 'cam_file does not exist! %s'%(str(cam_file))
    cam_params = read_cam_params(cam_file)
    return cam_params, len(cam_params)

def writeScene(name, boxes):
    with open(name, 'w') as meshOut:
        vNum = 0
        for group in boxes:
            vertices = group[0]
            faces = group[1]
            for n in range(0, vertices.shape[0]):
                meshOut.write('v %.3f %.3f %.3f\n' %
                        (vertices[n, 0], vertices[n, 1], vertices[n, 2]))
            for n in range(0, faces.shape[0]):
                meshOut.write('f %d %d %d\n' %
                        (faces[n, 0] + vNum, faces[n, 1] + vNum, faces[n, 2] + vNum))
            vNum += vertices.shape[0]

def checkOverlapApproximate(bverts1, bverts2):
    axis_1 = (bverts1[1, :] - bverts1[0, :]).reshape(1, 3)
    xLen = np.sqrt(np.sum(axis_1 * axis_1))
    axis_2 = (bverts1[3, :] - bverts1[0, :]).reshape(1, 3)
    zLen = np.sqrt(np.sum(axis_2 * axis_2))

    origin = bverts1[0, :]
    xCoord = np.sum( (bverts2[0:4, :] - origin) * axis_1 / xLen, axis=1)
    zCoord = np.sum( (bverts2[0:4, :] - origin) * axis_2 / zLen, axis=1)
    minX, maxX = xCoord.min(), xCoord.max()
    minZ, maxZ = zCoord.min(), zCoord.max()

    xOverlap = (min(maxX, xLen) - max(minX, 0))
    zOverlap = (min(maxZ, zLen) - max(minZ, 0))
    if xOverlap < 0 or zOverlap < 0:
        return False

    areaTotal = (maxX - minX) * (maxZ - minZ)
    areaOverlap = xOverlap * zOverlap
    if areaOverlap / areaTotal > 0.7:
        return True
    else:
        return False

def findSupport(lverts, boxes, cats):
    # Find support for every object
    boxList = []
    for n in range(0, len(boxes)):
        bList = []
        top = boxes[n][0][:, 1].max()

        for m in range(0, len(boxes)):
            if m != n:
                bverts = boxes[m][0]
                minY, maxY = bverts[:, 1].min(), bverts[:, 1].max()

                bottom = minY
                if np.abs(top - bottom) < 0.75 * (maxY - minY) and np.abs(top - bottom) < 1:
                    isOverlap = checkOverlapApproximate(boxes[n][0], boxes[m][0])
                    if isOverlap:
                        if m < n:
                            if not n in boxList[m]:
                                bList.append(m)
                        else:
                            bList.append(m)
        boxList.append(bList)


    # Find objects on floor
    floorList = []
    floorHeight = lverts[:, 1].min()
    for n in range(0, len(boxes)):
        isSupported = False
        for bList in boxList:
            if n in bList:
                isSupported = True
                break

        if not isSupported:
            if cats[n] == '03046257' or cats[n] == '03636649' or cats[n] == '02808440':
                bverts = boxes[n][0]
                minY, maxY = bverts[:, 1].min(), bverts[:, 1].max()
                if np.abs(minY - floorHeight) < 1.5 * (maxY - minY) and np.abs(minY - floorHeight) < 1 :
                    floorList.append(n)
            else:
                floorList.append(n)

    return floorList, boxList

def adjustHeightBoxes(boxId, boxes, cads, boxList):
    top = boxes[boxId ][0][:, 1].max()
    for n in boxList[boxId ]:
        bverts = boxes[n][0]
        bottom = bverts[:, 1].min()
        delta = np.array([0, top-bottom, 0]).reshape(1, 3)

        boxes[n][0] = boxes[n][0] + delta
        cads[n][0] = cads[n][0] + delta

        boxes[n].append( ('t', delta.squeeze()))
        cads[n].append( ('t', delta.squeeze()))
        if len(boxList[n]) != 0:
            adjustHeightBoxes(n, boxes, cads, boxList)
            adjustHeightBoxes(n, boxes, cads, boxList)
    return

def adjustHeight(lverts, boxes, cads, floorList, boxList):
    # Adjust the height
    floorHeight = lverts[:, 1].min()
    for n in floorList:
        bverts = boxes[n][0]
        bottom = bverts[:, 1].min()
        delta = np.array([0, floorHeight-bottom, 0]).reshape(1, 3)

        boxes[n][0] = boxes[n][0] + delta
        boxes[n].append( ('t', delta.squeeze()))
        cads[n][0] = cads[n][0] + delta
        cads[n].append( ('t', delta.squeeze()))

        if len(boxList[n]) != 0:
            adjustHeightBoxes(n, boxes, cads, boxList)

    return

def checkPointInPolygon(wallVertices, v):
    ###Given the wall vertices, determine if the pt is inside the polygon
    X = [pt[0] for pt in wallVertices ]
    Z = [pt[2] for pt in wallVertices ]
    j = len(wallVertices) - 1

    oddNodes = False
    x, z = v[0], v[2]
    for i in range(len(wallVertices)):
        if (Z[i] < z and Z[j] >= z) or (Z[j] < z and Z[i] >= z):
            if (X[i] + ((z - Z[i]) / (Z[j] - Z[i]) * (X[j] - X[i]))) <= x:
                oddNodes = not oddNodes
        j=i
    return oddNodes

def calLineParam(pt1, pt2):
    ###Calculate line parameters
    x1, z1 = pt1
    x2, z2 = pt2

    a = z1 - z2
    b = x2 - x1
    c = z2 * x1 - x2 * z1
    return a, b, c


def findNearestPt(w1, w2, pts):
    ###Find the nearest point on the line to a point
    a, b, c = calLineParam(w1, w2)
    x, z = pts
    a2b2 = a ** 2 + b ** 2
    new_x = (b * (b * x - a * z) - a * c) / a2b2
    new_z = (a * (-b * x + a * z) - b * c) / a2b2
    return np.array([new_x, new_z])


def findNearestWall(pt, wallVertices):
    ###Find nearest wall of a point
    minD, result = 100, None
    pt = np.array([pt[0], pt[2]], dtype=np.float32)
    j = len(wallVertices) - 1
    for i in range(len(wallVertices)):
        w1 = np.array([wallVertices[i][0], wallVertices[i][2] ], dtype = np.float32)
        w2 = np.array([wallVertices[j][0], wallVertices[j][2] ], dtype = np.float32)
        if np.linalg.norm(w1 - pt) < np.linalg.norm(w2 - pt):
            d = np.linalg.norm(np.cross(w2 - w1, w1 - pt)) / np.linalg.norm(w2 - w1)
        else:
            d = np.linalg.norm(np.cross(w2 - w1, w2 - pt)) / np.linalg.norm(w2 - w1)
        if d < minD:
            nearestPt = findNearestPt(w1, w2, pt)
            denom, nom  = w1 - w2, w1 - nearestPt
            if(np.sum(denom == 0)):
                denom[denom == 0] = denom[denom != 0]
            check = nom / denom
            if np.mean(check) < 1 and np.mean(check) > 0:
                minD = d
                result = nearestPt
        j = i

    for i in range(len(wallVertices)):
        w1 = np.array([wallVertices[i][0], wallVertices[i][2] ], dtype = np.float32)
        d = np.linalg.norm(w1 - pt)
        if d < minD:
            minD = d
            result = w1
    return minD, result


def moveBox(record):
    pt, nearestPt = record
    vector = ((nearestPt[0] - pt[0]), (nearestPt[1] - pt[2]))
    return vector

def moveBoxInWall(cverts, boxes, cads, threshold = 0.3):
    # find wall_vertices
    wallVertices = []
    floorHeight = cverts[:, 1].min()
    for n in range(0, cverts.shape[0]):
        vert = cverts[n, :]
        if np.abs(vert[1] - floorHeight) < 0.1:
            wallVertices.append(vert)

    isMove = False
    isBeyondRange = False
    for n in range(0, len(boxes)):
        box = boxes[n]
        maxD, record = 0, None
        bverts = box[0]
        for m in range(0, bverts.shape[0]):
            v = bverts[m, :]
            if not checkPointInPolygon(wallVertices, v):
                d, nearestPt = findNearestWall(v, wallVertices)
                if maxD < d:
                    record = (v, nearestPt)
                    maxD = d

        if record != None:
            t_x, t_z = moveBox(record)
            trans = np.array([t_x, 0, t_z], dtype=np.float32)
            if np.linalg.norm(trans) > threshold:
                isBeyondRange = True
            if np.linalg.norm(trans) >= 1e-7:
                isMove = True
                direc = trans / np.linalg.norm(trans)
                trans = trans + direc * 0.04

                boxes[n][0] = boxes[n][0] + trans.reshape(1, 3)
                boxes[n].append( ('t', trans.squeeze()))

                cads[n][0] = cads[n][0] + trans.reshape(1, 3)
                cads[n].append( ('t', trans.squeeze()))

    return isMove, isBeyondRange

import open3d as o3d
import numpy as np
from PIL import Image
import cv2
import os.path as osp
import torch
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt

def x_cam_zq_2_x_cam_rui(K, _vertices_cam: np.ndarray, height = 120, width = 160):
    '''
    Convert the vertices from the camera coordinate system of ZQ's code, to the camera coordinate system of the Rui's camera,
    
    due to the different image size, and different camera intrinsics.
    
    ![](https://i.imgur.com/9CcHHSa.jpg)
    '''
    
    assert len(_vertices_cam.shape) == 2
    assert _vertices_cam.shape[1] == 3
    # assert _vertices_cam.shape[0] == height * width
    
    xx_cam = _vertices_cam[:, 0] / _vertices_cam[:, 2]
    yy_cam = _vertices_cam[:, 1] / _vertices_cam[:, 2]
    
    fov_x = 57.95
    xRange_zq_start = -np.tan(fov_x / 180* np.pi / 2.0 ) # == W/2 / fx
    yRange_zq_start = float(height) / float(width) * xRange_zq_start
    xRange_rui_start = (0.5-K[0][2]) / K[0][0]
    yRange_rui_start = (0.5-K[1][2]) / K[1][1]
    
    xx_rui = (xx_cam - xRange_zq_start) / (2. * (-xRange_zq_start)) * (width * 2-1) + xRange_rui_start # 0.5, ..., W-0.5, total 160x2 points
    yy_rui = (yy_cam - yRange_zq_start) / (2. * (-yRange_zq_start)) * (height * 2-1) + yRange_rui_start # 0.5, ..., H-0.5, total 120x2 points
    
    xx_cam_rui = (xx_rui - K[0][2]) / K[0][0] * _vertices_cam[:, 2]
    yy_cam_rui = (yy_rui - K[1][2]) / K[1][1] * _vertices_cam[:, 2]
    
    X_cam_rui = np.stack([xx_cam_rui, yy_cam_rui, _vertices_cam[:, 2]], axis=1)
    
    return X_cam_rui, (xx_rui, yy_rui)

def x_cam_rui_2_x_cam_zq(K, _vertices_cam_rui: np.ndarray, height = 120, width = 160):
    '''
    doing the inverse of the function x_cam_zq_2_x_cam_rui
    '''
    
    assert len(_vertices_cam_rui.shape) == 2
    assert _vertices_cam_rui.shape[1] == 3
    # assert _vertices_cam_rui.shape[0] == height * width
    
    xx_cam = _vertices_cam_rui[:, 0] / _vertices_cam_rui[:, 2]
    yy_cam = _vertices_cam_rui[:, 1] / _vertices_cam_rui[:, 2]
    
    fov_x = 57.95
    xRange_zq_start = -np.tan(fov_x / 180* np.pi / 2.0 ) # == W/2 / fx
    yRange_zq_start = float(height) / float(width) * xRange_zq_start
    xRange_rui_start = (0.5-K[0][2]) / K[0][0]
    yRange_rui_start = (0.5-K[1][2]) / K[1][1]
    xRange_rui_end = (width*2-0.5-K[0][2]) / K[0][0]
    yRange_rui_end = (height*2-0.5-K[1][2]) / K[1][1]
    
    xx_zq = (xx_cam - xRange_rui_start) / (xRange_rui_end-xRange_rui_start) * (2. * - xRange_zq_start) + xRange_zq_start
    yy_zq = (yy_cam - yRange_rui_start) / (yRange_rui_end-yRange_rui_start) * (2. * - yRange_zq_start) + yRange_zq_start
    
    xx_cam_zq = xx_zq * _vertices_cam_rui[:, 2]
    yy_cam_zq = yy_zq * _vertices_cam_rui[:, 2]
    
    X_cam_rui = np.stack([xx_cam_zq, -yy_cam_zq, -_vertices_cam_rui[:, 2]], axis=1)
    
    return X_cam_rui, (xx_zq, yy_zq)

    
def srgb2rgb(srgb ):
    ret = np.zeros_like(srgb )
    idx0 = srgb <= 0.04045
    idx1 = srgb > 0.04045
    ret[idx0] = srgb[idx0] / 12.92
    ret[idx1] = np.power( (srgb[idx1] + 0.055) / 1.055, 2.4 )
    return ret

def writeErrToScreen(errorName, errorArr, epoch, j):
    print( ('[%d/%d] {0}:' % (epoch, j) ).format(errorName), end=' ')
    for n in range(0, len(errorArr) ):
        if not torch.is_tensor(errorArr[n] ):
            print('%.6f' % errorArr[n], end=' ')
        else:
            print('%.6f' % errorArr[n].data.item(), end = ' ')
    print('.')

def writeNpErrToScreen(errorName, errorArr, epoch, j):
    print( ('[%d/%d] {0}:' % (epoch, j) ).format(errorName), end=' ')
    for n in range(0, len(errorArr) ):
        print('%.6f' % errorArr[n], end = ' ')
    print('.')

def writeErrToFile(errorName, errorArr, fileOut, epoch, j):
    fileOut.write( ('[%d/%d] {0}:'% (epoch, j) ).format(errorName) )
    for n in range(0, len(errorArr) ):
        if not torch.is_tensor(errorArr[n] ):
            fileOut.write('%.6f ' % errorArr[n] )
        else:
            fileOut.write('%.6f ' % errorArr[n].data.item() )
    fileOut.write('.\n')

def writeNpErrToFile(errorName, errorArr, fileOut, epoch, j):
    fileOut.write( ('[%d/%d] {0}:' % (epoch, j) ).format(errorName) )
    for n in range(0, len(errorArr) ):
        fileOut.write('%.6f ' % errorArr[n] )
    fileOut.write('.\n')

def turnErrorIntoNumpy(errorArr):
    errorNp = []
    for n in range(0, len(errorArr) ):
        if not torch.is_tensor(errorArr[n] ):
            errorNp.append(errorArr[n] )
        else:
            errorNp.append(errorArr[n].data.item() )
    return np.array(errorNp)[np.newaxis, :]


def writeImageToFile(imgBatch, nameBatch, isGama = False):
    batchSize = imgBatch.size(0)
    if imBatch.is_cuda:
        imBatch = imBatch.cpu()
    for n in range(0, batchSize):
        img = imgBatch[n, :, :, :].data.numpy()
        img = np.clip(img, 0, 1)
        if isGama:
            img = np.power(img, 1.0/2.2)
        img = (255 *img.transpose([1, 2, 0] ) ).astype(np.uint8)
        if img.shape[2] == 1:
            img = np.concatenate([img, img, img], axis=2)
        img = Image.fromarray(img )
        img.save(nameBatch[n] )


def writeEnvToFile(envmaps, rowNum, envName, nrows=15, ncols=20, envHeight=8,
        envWidth=16, gap=1 ):
    if envmaps.is_cuda:
        envmaps = envmaps.cpu()
    envmaps = envmaps.data.numpy()
    colNum = int(float(envmaps.shape[0] ) / float(rowNum ) + 0.5 )

    envRow, envCol = envmaps.shape[2], envmaps.shape[3]
    interY = int(envRow / nrows )
    interX = int(envCol / ncols )

    lnrows = len(np.arange(0, envRow, interY) )
    lncols = len(np.arange(0, envCol, interX) )

    lenvHeight = lnrows * (envHeight + gap) + gap
    lenvWidth = lncols * (envWidth + gap) + gap

    envmapsLarge = np.zeros( [lenvHeight * colNum,
        lenvWidth * rowNum, 3], dtype = np.float32 )

    for envId in range(0, envmaps.shape[0] ):
        envmap = envmaps[envId, :, :, :, :, :]
        envmap = np.transpose(envmap, [1, 2, 3, 4, 0] )

        envmapLarge = np.zeros([lenvHeight, lenvWidth, 3], dtype=np.float32) + 1.0
        for r in range(0, envRow, interY ):
            for c in range(0, envCol, interX ):
                rId = int(r / interY )
                cId = int(c / interX )

                rs = rId * (envHeight + gap )
                cs = cId * (envWidth + gap )
                envmapLarge[rs : rs + envHeight, cs : cs + envWidth, :] = envmap[r, c, :, :, :]

        rowId = int(envId / rowNum )
        colId = envId - rowId * rowNum
        rs = rowId * lenvHeight
        re = rs + lenvHeight
        cs = colId * lenvWidth
        ce = cs + lenvWidth
        envmapsLarge[rs:re, cs:ce, : ] = envmapLarge

    cv2.imwrite(envName, envmapsLarge[:, :, ::-1] )

def envToShading(env, envWidth = 16, envHeight = 8 ):
    Az = ( (np.arange(envWidth) + 0.5) / envWidth - 0.5 )* 2 * np.pi
    El = ( (np.arange(envHeight) + 0.5) / envHeight ) * np.pi / 2.0
    Az, El = np.meshgrid(Az, El )
    Az = Az[np.newaxis, :, :]
    El = El[np.newaxis, :, :]
    envWeight = np.cos(El) * np.sin(El )
    envWeight = envWeight.reshape(1, 1, 1, 1, envHeight, envWidth ).astype(np.float32 )
    envWeight = torch.from_numpy(envWeight ).cuda()

    shading = torch.sum(torch.sum(envWeight * env, dim = 4 ), dim =4 ) \
            * np.pi / envHeight / envWidth

    return shading

def writeDepthAsPointClouds(depth, normal, mask, fileName, fov = 57.95, isNormalize = True ):
    # From depth to points
    batchSize = depth.size(0 )
    height, width = depth.size(2), depth.size(3)
    if normal.is_cuda == True:
        normal = normal.cpu()
    if depth.is_cuda == True:
        depth = depth.cpu()
    if mask.is_cuda == True:
        mask = mask.cpu()
    normal = normal.detach().numpy()
    depth = depth.detach().numpy()

    xRange = 1 * np.tan(fov / 180* np.pi / 2.0 )
    yRange = float(height) / float(width) * xRange

    x, y = np.meshgrid(np.linspace(-xRange, xRange, width ),
            np.linspace(-yRange, yRange, height ) )

    y = np.flip(y, axis=0 )
    z = -np.ones( (height, width), dtype=np.float32 )

    pCoord = np.stack([x, y, z], axis = 0 )[np.newaxis, :]
    pCoord = pCoord.astype(np.float32 )
    point = pCoord * depth

    point = point.reshape(batchSize, 3, -1 )
    normal = normal.reshape(batchSize, 3, -1 )
    mask = mask.reshape(batchSize, 1, -1 )

    # Output points
    for n in range(0, batchSize ):
        pointArr = []
        normalArr =  []
        for m  in range(0, height * width ):
            if mask[n, 0, m] > 0.9:
                pointArr.append(point[n, :, m].reshape(1, 3) )
                normalArr.append(normal[n, :, m].reshape(1, 3) )
        pointArr = np.concatenate(pointArr, axis=0 )
        normalArr = np.concatenate(normalArr, axis=0 )
        if isNormalize:
            colorArr = (normalArr + 1) * 0.5
        else:
            colorArr = normalArr
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pointArr )
        pcd.normals = o3d.utility.Vector3dVector(normalArr )
        pcd.colors = o3d.utility.Vector3dVector(colorArr )
        o3d.io.write_point_cloud(
                fileName.replace('.ply', '_%d.ply' % n), pcd )
    return

def generateBox(axes, center, offset ):
    faces = []
    faces.append(np.array([1, 2, 3] ) + offset - 1)
    faces.append(np.array([1, 3, 4] ) + offset - 1)
    faces.append(np.array([5, 7, 6] ) + offset - 1)
    faces.append(np.array([5, 8, 7] ) + offset - 1)
    faces.append(np.array([1, 6, 2] ) + offset - 1)
    faces.append(np.array([1, 5, 6] ) + offset - 1)
    faces.append(np.array([2, 7, 3] ) + offset - 1)
    faces.append(np.array([2, 6, 7] ) + offset - 1)
    faces.append(np.array([3, 8, 4] ) + offset - 1)
    faces.append(np.array([3, 7, 8] ) + offset - 1)
    faces.append(np.array([4, 5, 1] ) + offset - 1)
    faces.append(np.array([4, 8, 5] ) + offset - 1)

    xAxis = axes[0, :] * 0.5
    yAxis = axes[1, :] * 0.5
    zAxis = axes[2, :] * 0.5
    corners = []
    corners.append(center - xAxis - yAxis - zAxis )
    corners.append(center + xAxis - yAxis - zAxis )
    corners.append(center + xAxis - yAxis + zAxis )
    corners.append(center - xAxis - yAxis + zAxis )

    corners.append(center - xAxis + yAxis - zAxis )
    corners.append(center + xAxis + yAxis - zAxis )
    corners.append(center + xAxis + yAxis + zAxis )
    corners.append(center - xAxis + yAxis + zAxis )

    corners = np.stack(corners, axis=0 )
    faces = np.stack(faces, axis=0 )

    return corners, faces

def writeLampBatch(axes, center, ons, srcNum, fileName ):
    batchSize = axes.size(0 )
    if axes.is_cuda:
        axes = axes.cpu()
    if center.is_cuda:
        center = center.cpu()
    axes = axes.detach().numpy()
    center = center.detach().numpy()

    for n in range(0, batchSize ):
        for m in range(0, srcNum ):
            if ons[n, m] > 0:
                vertices, faces = generateBox(
                        axes[n, m, :],
                        center[n, m, :],
                        8 * m )

                name = fileName.replace('.ply', '_%d_%d.obj' % (n, m) )
                mesh = o3d.geometry.TriangleMesh()
                mesh.vertices = o3d.utility.Vector3dVector(vertices )
                mesh.triangles = o3d.utility.Vector3iVector(faces )
                o3d.io.write_triangle_mesh(name, mesh )

    return

def writeWindowBatch(centers, ys, xs, ons, srcNum, fileName ):
    batchSize = centers.size(0 )
    if centers.is_cuda:
        centers = centers.cpu()
    if ys.is_cuda:
        ys = ys.cpu()
    if xs.is_cuda:
        xs = xs.cpu()
    centers = centers.detach().numpy()
    xs = xs.detach().numpy()
    ys = ys.detach().numpy()

    for n in range(0, batchSize ):
        center = centers[n, :]
        x = xs[n, :]
        y = ys[n, :]
        for m in range(0, srcNum ):
            if ons[n, m ] > 0:
                vertexArr = []
                faceArr = []
                for r in range(-1, 2, 2 ):
                    for c in range(-1, 2, 2 ):
                        pt = center[m , :] + r * x[m, :] * 0.5 + c * y[m, :] * 0.5
                        vertexArr.append(pt )

                f1 = np.array([0, 1, 2]) + 1
                f2 = np.array([2, 1, 3]) + 1
                faceArr.append(f1 )
                faceArr.append(f2 )

                with open(fileName.replace('.obj', '_%d_%d.obj' % (n, m) ), 'w') as fOut:
                    for v in vertexArr:
                        fOut.write('v %.4f %.4f %.4f\n' % (v[0], v[1], v[2] ) )
                    for f in faceArr:
                        fOut.write('f %d %d %d\n' % (f[0], f[1], f[2] ) )
                        
    return

def writeLampList(centers, depths, normals, masks, srcNum, fileName,
        fov = 57.95 ):
    batchSize = int(len(centers ) / srcNum )
    if depths.is_cuda:
        depths = depths.cpu()
    if normals.is_cuda:
        normals = normals.cpu()
    if masks.is_cuda:
        masks = masks.cpu()

    depths = depths.detach().numpy()
    normals = normals.detach().numpy()
    masks = masks.detach().numpy()
    height, width = depths.shape[2:]

    xRange = 1 * np.tan(fov / 360.0 * np.pi )
    yRange = float(height) / float(width) * xRange
    x, y = np.meshgrid(np.linspace(-xRange, xRange, width ),
                       np.linspace(-yRange, yRange, height ) )

    y = np.flip(y, axis=0 )
    z = -np.ones( (height, width), dtype=np.float32 )

    pCoords = np.stack([x, y, z], axis = 0 )[np.newaxis, :]
    pCoords = pCoords.astype(np.float32 )
    pCoords = pCoords * depths

    pixel_len = xRange / width * 2
    pixel_size = pixel_len * pixel_len
    pNum = width * height

    for n in range(0, batchSize ):
        pCoord = pCoords[n, :, :, :]
        normal = normals[n, :, :, :]
        for m in range(0, srcNum ):
            center = centers[n * srcNum + m ]
            if center is None:
                continue
            if center.is_cuda:
                center = center.cpu()
            center = center.detach().numpy()

            mask = (masks[n, m].squeeze() == 1)
            mask_s = ndimage.binary_erosion(mask, structure = np.ones((3, 3) ) )
            edge = mask.astype(np.float32 ) - mask_s.astype(np.float32 )

            mask = mask.reshape(-1 ).astype(np.float32 )
            edge = edge.reshape(-1 ).astype(np.float32 )

            pCoord = pCoord.reshape(3, pNum )
            normal = normal.reshape(3, pNum )

            fg_center = pCoord[:, mask > 0.9]
            edge_sp = pCoord[:, edge > 0.9]

            plateNum = fg_center.shape[1]
            edgeNum = edge_sp.shape[1]

            fg_normal = normal[:, mask > 0.9]

            fg_area = fg_center[2:3, :] * fg_center[2:3, :] * pixel_size \
                    / np.maximum(np.abs(fg_normal[2:3, :] ), 1e-1 )
            edge_width = np.abs(edge_sp[2:3, :] ) * pixel_len

            # compute background plates
            center = center.reshape(3, 1 )
            direc = center / np.sqrt(np.maximum(np.sum(center * center ), 1e-6 ) )

            bg_center = np.abs(np.sum( (fg_center - center) * direc,
                    axis=0, keepdims=True ) ) * 2 * direc + fg_center
            bg_normal = np.abs(np.sum(fg_normal * direc, axis=0, keepdims=True ) ) \
                    * direc * 2 + fg_normal

            edge_center = np.abs(np.sum( (edge_sp - center) * direc,
                    axis=0, keepdims=True ) ) * direc + edge_sp
            edge_normal = edge_center - center - \
                    np.sum( (edge_center - center ) * direc, axis=0, keepdims=True) * direc
            edge_normal = edge_normal / np.sqrt(np.maximum(
                np.sum(edge_normal * edge_normal, axis=0, keepdims = True ), 1e-6 ) )
            edge_len = 2 * np.sqrt(
                    np.maximum(
                        np.sum((edge_center -  edge_sp) * (edge_center - edge_sp), axis=0, keepdims=True ),
                        1e-6 ) )

            facet_center = np.concatenate([fg_center, bg_center ], axis=1 )
            facet_normal = np.concatenate([fg_normal, bg_normal ], axis=1 )
            facet_area = np.concatenate([fg_area, fg_area ], axis=1 )


            vertices, faces = [], []
            for l in range(0, facet_center.shape[1] ):
                center = facet_center[:, l].reshape(3 )
                zAxis = facet_normal[:, l].reshape(3 )
                area = facet_area[:, l].squeeze()
                plateLen = np.sqrt(area ) / 2.0

                yAxis = np.array([0, 1, 0], dtype = np.float32 )
                yAxis = yAxis - np.sum(yAxis * zAxis ) * zAxis
                yAxis = yAxis / np.sqrt(np.sum(yAxis * yAxis ) )

                xAxis = np.cross(yAxis, zAxis )

                vNum = len(vertices )

                p1 = center + xAxis * plateLen + yAxis * plateLen
                p2 = center + xAxis * plateLen - yAxis * plateLen
                p3 = center - xAxis * plateLen - yAxis * plateLen
                p4 = center - xAxis * plateLen + yAxis * plateLen

                vertices.append(p1 )
                vertices.append(p2 )
                vertices.append(p3 )
                vertices.append(p4 )

                faces.append(np.array([0, 2, 1], dtype=np.int32 ) + vNum )
                faces.append(np.array([0, 3, 2], dtype=np.int32 ) + vNum )

            for l in range(0, edge_center.shape[1] ):
                center = edge_center[:, l].reshape(3 )
                zAxis = edge_normal[:, l].reshape(3 )
                eWidth = edge_width[:, l].squeeze()
                eLength = edge_len[:, l].squeeze()

                yAxis = direc.squeeze()
                yAxis = yAxis - np.sum(yAxis * zAxis ) * zAxis
                yAxis = yAxis / np.sqrt(np.sum(yAxis * yAxis ) )

                xAxis = np.cross(yAxis, zAxis )

                vNum = len(vertices )

                p1 = center + yAxis * eLength / 2.0 + xAxis * eWidth / 2.0
                p2 = center + yAxis * eLength / 2.0 - xAxis * eWidth / 2.0
                p3 = center - yAxis * eLength / 2.0 - xAxis * eWidth / 2.0
                p4 = center - yAxis * eLength / 2.0 + xAxis * eWidth / 2.0

                vertices.append(p1 )
                vertices.append(p2 )
                vertices.append(p3 )
                vertices.append(p4 )

                faces.append(np.array([0, 1, 2], dtype=np.int32 ) + vNum )
                faces.append(np.array([0, 2, 3], dtype=np.int32 ) + vNum )

            if vertices == []:
                continue
            vertices = np.stack(vertices, axis=0 )
            faces = np.stack(faces, axis=0 )

            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(vertices )
            mesh.triangles = o3d.utility.Vector3iVector(faces )
            meshName = fileName.replace('.ply', '_%d_%d.obj' % (n, m ) )
            o3d.io.write_triangle_mesh(meshName, mesh )
    return

def writeWindowList(centers, ys, xs, srcNum, fileName ):
    batchSize = int(len(centers ) / srcNum )

    for n in range(0, batchSize ):
        for m in range(0, srcNum ):
            center = centers[n * srcNum + m ]
            x = xs[n * srcNum + m ]
            y = ys[n * srcNum + m ]
            if center is None:
                continue
            if center.is_cuda:
                center = center.cpu()
            if y.is_cuda:
                y = y.cpu()
            if x.is_cuda:
                x = x.cpu()
            center = center.detach().numpy()
            x = x.detach().numpy()
            y = y.detach().numpy()

            vertexArr = []
            faceArr = []
            for r in range(-1, 2, 2 ):
                for c in range(-1, 2, 2 ):
                    pt = center[0, :] + r * x[0, :] * 0.5 + c * y[0, :] * 0.5
                    vertexArr.append(pt )

            f1 = np.array([0, 1, 2]) + 1
            f2 = np.array([2, 1, 3]) + 1
            faceArr.append(f1 )
            faceArr.append(f2 )

            with open(fileName.replace('.obj', '_%d_%d.obj' % (n, m) ), 'w') as fOut:
                for v in vertexArr:
                    fOut.write('v %.4f %.4f %.4f\n' % (v[0], v[1], v[2]) )
                for f in faceArr:
                    fOut.write('f %d %d %d\n' % (f[0], f[1], f[2]) )
    return

def vis_disp_colormap(disp_array_, file=None, normalize=True, min_and_scale=None, valid_mask=None, cmap_name='jet'):
    disp_array = disp_array_.copy()
    cm = plt.get_cmap(cmap_name) # the larger the hotter
    if valid_mask is not None:
        assert valid_mask.shape==disp_array.shape
        assert valid_mask.dtype==bool
    else:
        valid_mask = np.ones_like(disp_array).astype(bool)
    
    if normalize:
        if min_and_scale is None:
            depth_min = np.amin(disp_array[valid_mask])
            disp_array -= depth_min
            depth_scale = 1./(1e-6+np.amax(disp_array[valid_mask]))
            disp_array = disp_array * depth_scale
            min_and_scale = [depth_min, depth_scale]
        else:
            disp_array -= min_and_scale[0]
            disp_array = disp_array * min_and_scale[1]

    disp_array = np.clip(disp_array, 0., 1.)
    disp_array = (cm(disp_array)[:, :, :3] * 255).astype(np.uint8)
    
    if file is not None:
        from PIL import Image, ImageFont, ImageDraw
        disp_Image = Image.fromarray(disp_array)
        disp_Image.save(file)
    else:
        return disp_array, min_and_scale

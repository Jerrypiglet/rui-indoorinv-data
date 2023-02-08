from PIL import Image, ImageFont, ImageDraw
from pyquaternion import Quaternion
import numpy as np
import open3d as o3d
from pathlib import Path
import matplotlib
import torch
from copy import deepcopy

def text_3d(text, pos, direction=None, degree=0.0, density=10, font='/usr/share/fonts/truetype/freefont/FreeMonoOblique.ttf', font_size=10, text_color=(0, 0, 0)):
    """
    Taken from https://github.com/isl-org/Open3D/issues/2

    Generate a 3D text point cloud used for visualization.
    :param text: content of the text
    :param pos: 3D xyz position of the text upper left corner
    :param direction: 3D normalized direction of where the text faces
    :param degree: in plane rotation of text
    :param font: Name of the font - change it according to your system
    :param font_size: size of the font
    :return: o3d.geoemtry.PointCloud object
    """
    if direction is None:
        direction = (0., 0., 1.)

    assert Path(font).exists()

    font_obj = ImageFont.truetype(font, font_size * density)
    # font_obj = ImageFont.load_default()
    font_dim = font_obj.getsize(text)

    img = Image.new('RGB', font_dim, color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), text, font=font_obj, fill=text_color)
    img = np.asarray(img)
    img_mask = img[:, :, 0] < 128
    indices = np.indices([*img.shape[0:2], 1])[:, img_mask, 0].reshape(3, -1).T

    pcd = o3d.geometry.PointCloud()
    pcd.colors = o3d.utility.Vector3dVector(img[img_mask, :].astype(float) / 255.0)
    points = indices / 1000 / density
    points -= np.mean(points, axis=0, keepdims=True)
    pcd.points = o3d.utility.Vector3dVector(points)
    try:
        downpcd = pcd.voxel_down_sample(voxel_size=(np.amax(points)-np.amin(points))/50.)
        # print(np.asarray(downpcd.points).shape)
        pcd = downpcd
    except ValueError as e:
        pass

    # raxis = np.cross([0.0, 0.0, 1.0], direction)
    raxis = direction
    if np.linalg.norm(raxis) < 1e-6:
        raxis = (0.0, 0.0, 1.0)
    # trans = (Quaternion(axis=raxis, radians=np.arccos(direction[2])) *
    #          Quaternion(axis=direction, degrees=degree)).transformation_matrix
    trans = Quaternion(axis=direction, degrees=degree).transformation_matrix
    # print(trans)
    # trans[0:3, 3] = np.asarray(pos)
    trans[0:3, 3] = -np.mean(points, axis=0).reshape((3,)) + np.array(pos).reshape((3,))
    pcd.transform(trans)
    return pcd

def get_cross_prod_mat(pVec_Arr):
    # pVec_Arr shape (3)
    qCross_prod_mat = np.array([
        [0, -pVec_Arr[2], pVec_Arr[1]], 
        [pVec_Arr[2], 0, -pVec_Arr[0]],
        [-pVec_Arr[1], pVec_Arr[0], 0],
    ])
    return qCross_prod_mat


def caculate_align_mat(pVec_Arr):
    scale = np.linalg.norm(pVec_Arr)
    pVec_Arr = pVec_Arr/ scale
    # must ensure pVec_Arr is also a unit vec. 
    z_unit_Arr = np.array([0,0,1])
    z_mat = get_cross_prod_mat(z_unit_Arr)

    z_c_vec = np.matmul(z_mat, pVec_Arr)
    z_c_vec_mat = get_cross_prod_mat(z_c_vec)

    if np.dot(z_unit_Arr, pVec_Arr) == -1:
        qTrans_Mat = -np.eye(3, 3)
    elif np.dot(z_unit_Arr, pVec_Arr) == 1:   
        qTrans_Mat = np.eye(3, 3)
    else:
        qTrans_Mat = np.eye(3, 3) + z_c_vec_mat + np.matmul(z_c_vec_mat,
                                                    z_c_vec_mat)/(1 + np.dot(z_unit_Arr, pVec_Arr))

    qTrans_Mat *= scale
    return qTrans_Mat

def create_arrow(scale=10):
    """
    Create an arrow in for Open3D
    """
    cone_height = scale*0.2
    cylinder_height = scale*0.8
    cone_radius = scale/10
    cylinder_radius = scale/20
    mesh_frame = o3d.geometry.TriangleMesh.create_arrow(
        cone_radius=cone_radius, 
        cone_height=cone_height,
        cylinder_radius=cylinder_radius,
        cylinder_height=cylinder_height)
    return(mesh_frame)

def get_arrow_o3d(origin: np.array=np.zeros(3, dtype=np.float32), end=None, vec=None, scale=1, color=[0, 1, 1]):
    """
    Taken from https://stackoverflow.com/a/59026582 and [how to compute transformation:] https://stackoverflow.com/a/59829173

    Creates an arrow from an origin point to an end point,
    or create an arrow from a vector vec starting from origin.
    Args:
        - end (): End point. np.array([x, y, z])
        - vec (): Vector. np.array([i, j, k])
    """
    
    if end is not None:
        vec = np.array(end) - origin
    elif vec is not None:
        vec = np.array(vec)

    vec = vec / (np.linalg.norm(vec)+1e-6)
        
    mesh = create_arrow(scale)
    R = caculate_align_mat(vec)

    mesh.rotate(R, center=np.array([0, 0, 0]))
    
    mesh.translate(origin)

    if isinstance(color, str):
        color = matplotlib.colors.to_rgb(color) # e.g. 'k' -> (0.0, 0.0, 0.0)

    mesh.paint_uniform_color(color)
    mesh.compute_vertex_normals()

    return(mesh)

def get_sphere(scale=5., resolution=200, hemisphere_normal=None, envmap=None):
    sphere = o3d.geometry.TriangleMesh.create_sphere(scale, resolution=resolution)
    sphere.compute_vertex_normals()
    sphere.compute_triangle_normals()
    sphere.normalize_normals()

    if hemisphere_normal is not None:
        assert hemisphere_normal.shape==(3,)
        triangles = np.asarray(sphere.triangles).copy()
        triangle_normals = np.asarray(sphere.triangle_normals).copy()
        vertices = np.asarray(sphere.vertices).copy()
        mask = np.sum(triangle_normals * hemisphere_normal.reshape(1, 3), axis=1) > 0
        triangles_new = triangles[mask]

        sphere = deepcopy(sphere)
        sphere.triangles = o3d.utility.Vector3iVector(triangles_new)
        sphere.vertices = o3d.utility.Vector3dVector(vertices)
        sphere.compute_vertex_normals()
        sphere.compute_triangle_normals()

        if envmap is None:
            sphere.vertex_colors = o3d.utility.Vector3dVector(np.random.rand(vertices.shape[0], 3))
        else:
            '''
            sample global envmap with sphere normals
            '''
            vertex_normals = np.asarray(sphere.vertex_normals).copy()
            # /home/ruizhu/Documents/Projects/semanticInverse/train/models_def/models_layout_emitter_lightAccu.py -> sample_envmap()
            vertex_normals_SG = np.stack((vertex_normals[:, 2], -vertex_normals[:, 0], vertex_normals[:, 1]), axis=-1)
            cos_theta = vertex_normals_SG[:, 2]
            theta_SG = np.arccos(cos_theta) # [0, pi]
            cos_phi = vertex_normals_SG[:, 0] / np.sin(theta_SG)
            sin_phi = vertex_normals_SG[:, 1] / np.sin(theta_SG)
            phi_SG = np.arctan2(sin_phi, cos_phi)
            uu_normalized = phi_SG / np.pi # pixel center when align_corners = False; -pi~pi -> -1~1; (N,)
            vv_normalized = theta_SG * 2. / np.pi -1. # pixel center when align_corners = False; 0~pi -> -1~1; (N,)
            uv_normalized = np.stack([uu_normalized, vv_normalized], axis=-1) # (N, 2)
            uv_normalized_torch = torch.from_numpy(uv_normalized).unsqueeze(0).unsqueeze(0).float() # (1, 1, N, 2)
            envmap_torch = torch.from_numpy(envmap).permute(2, 0, 1).unsqueeze(0).float() # (1, 3, H, W)
            sampled_envmap_torch = torch.nn.functional.grid_sample(envmap_torch, uv_normalized_torch, mode='bilinear', align_corners=True) # (1, 3, 1, N)
            sampled_envmap = sampled_envmap_torch.squeeze().numpy().transpose() # (N, 3)
            sphere.vertex_colors = o3d.utility.Vector3dVector(sampled_envmap) # [TODO] not sure how to set triangle colors... the Open3D documentation is pretty confusing and actually does not work... http://www.open3d.org/docs/release/python_api/open3d.t.geometry.TriangleMesh.html

    # self.vis.add_geometry(sphere)
    return sphere

def remove_ceiling(xyz_pcd: np.ndarray, pcd_color: np.ndarray, if_debug_info: bool=False):
    # remove ceiling points; assuming y axis is up
    ceiling_y = np.amax(xyz_pcd[:, 1]) # y axis is up
    pcd_mask = xyz_pcd[:, 1] < (ceiling_y*0.95)
    xyz_pcd = xyz_pcd[pcd_mask]
    pcd_color = pcd_color[pcd_mask]
    if if_debug_info:
        print('Removed points close to ceiling... percentage: %.2f'%(np.sum(pcd_mask)*100./xyz_pcd.shape[0]))

    return xyz_pcd, pcd_color

def remove_walls(layout_bbox_3d: np.ndarray, xyz_pcd: np.ndarray, pcd_color: np.ndarray, if_debug_info: bool=False):
    dists_all = np.zeros((xyz_pcd.shape[0]), dtype=np.float32) + np.inf

    for wall_v_idxes in [(4, 0, 5), (6, 5, 2), (7, 6, 3), (7, 3, 4)]:
        plane_normal = np.cross(layout_bbox_3d[wall_v_idxes[1]]-layout_bbox_3d[wall_v_idxes[0]], layout_bbox_3d[wall_v_idxes[2]]-layout_bbox_3d[wall_v_idxes[0]])
        plane_normal = plane_normal / np.linalg.norm(plane_normal)
        
        l1 = xyz_pcd - layout_bbox_3d[wall_v_idxes[0]].reshape(1, 3)
        dist_ = np.sum(l1 * plane_normal.reshape(1, 3), axis=1)
        dists_all = np.minimum(dist_, dists_all)

    layout_sides = np.vstack((layout_bbox_3d[1]-layout_bbox_3d[0], layout_bbox_3d[3]-layout_bbox_3d[0], layout_bbox_3d[4]-layout_bbox_3d[0]))
    layout_dimensions = np.linalg.norm(layout_sides, axis=1)
    if if_debug_info:
        print(layout_dimensions)

    pcd_mask = dists_all > np.amin(layout_dimensions)*0.05 # threshold is 5% of the shortest room dimension
    xyz_pcd = xyz_pcd[pcd_mask]
    pcd_color = pcd_color[pcd_mask]
    if if_debug_info:
        print('Removed points close to walls... percentage: %.2f'%(np.sum(pcd_mask)*100./xyz_pcd.shape[0]))

    return xyz_pcd, pcd_color
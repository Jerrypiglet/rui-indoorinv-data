from PIL import Image, ImageFont, ImageDraw
from pyquaternion import Quaternion
import numpy as np
import open3d as o3d
import matplotlib
import matplotlib.pyplot as plt

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


    font_obj = ImageFont.truetype(font, font_size * density)
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
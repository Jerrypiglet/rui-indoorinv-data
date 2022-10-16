import numpy as np
import quaternion
from scipy.spatial.transform import Rotation as R

def q_to_R(q):
    assert  type(q) is list and len(q) == 4
    q = np.quaternion(q[0], q[1], q[2], q[3])
    rotMat = quaternion.as_rotation_matrix(q )
    # rotMat = R.from_quat([q[0], q[1], q[2], q[3]]).as_matrix()
    # print(q, rotMat)
    
    # if np.abs(rotMat[1, 1] ) > 0.5:
    #     d = rotMat[1, 1]
    #     rotMat[:, 1] = 0
    #     rotMat[1, :] = 0
    #     if d < 0:
    #         rotMat[1, 1] = -1
    #     else:
    #         rotMat[1, 1] = 1
    return rotMat

def angle_axis_to_rotMat(angle_axis):
    assert  type(angle_axis) is list and len(angle_axis) == 4
    rotMat = R.from_rotvec(angle_axis[0] * np.array([angle_axis[1], angle_axis[2], angle_axis[3]])).as_matrix()
    return rotMat


def apply_R(rotMat, vertices):
    assert len(vertices.shape)==2 and vertices.shape[1]==3 and vertices.shape[0]>0
    vertices = np.matmul(rotMat, vertices.transpose() )
    vertices = vertices.transpose()
    return vertices

def apply_s(s, vertices):
    assert len(vertices.shape)==2 and vertices.shape[1]==3 and vertices.shape[0]>0
    scale = np.array(s, dtype=np.float32 ).reshape(1, 3)
    vertices = vertices * scale
    return vertices    

def apply_t(t, vertices):
    assert len(vertices.shape)==2 and vertices.shape[1]==3 and vertices.shape[0]>0
    trans = np.array(t, dtype=np.float32 ).reshape(1, 3)
    vertices = vertices + trans
    return vertices

def transform_with_transforms_xml_list(transforms_list, vertices, if_only_rotate=False):
    # suitanle for transforms_list read from XML: [{'scale': {'x': 1.519629, 'y': 2.335258, 'z': 1.910475}}, {'rotate': {'angle': 118.880976, 'x': 0.0, 'y': -1.0, 'z': 0.0}}, ...]
    assert len(vertices.shape)==2 and vertices.shape[1]==3 and vertices.shape[0]>0
    transforms_converted_list = []
    for transform in transforms_list:
        transform_name = list(transform.keys())[0]
        assert len(list(transform.keys())) == 1
        assert transform_name in ['scale', 'rotate', 'translate']
        if transform_name == 'scale' and not if_only_rotate:
            scale = transform[transform_name]
            if 'x' in scale:
                s = [scale['x'], scale['y'], scale['z']]
            else:
                s = [scale['value'], scale['value'], scale['value']]
            assert min(s) >= 0
            vertices = apply_s(s, vertices)
            transforms_converted_list.append(('s', np.asarray(s).reshape(3)))
        if transform_name == 'rotate':
            rotate = transform[transform_name]
            angle_axis = [rotate['angle']/180.*np.pi, rotate['x'], rotate['y'], rotate['z']]
            R = angle_axis_to_rotMat(angle_axis)
            vertices = apply_R(R, vertices)
            transforms_converted_list.append(('rot', R))
        if transform_name == 'translate' and not if_only_rotate:
            translate = transform[transform_name]
            t = [translate['x'], translate['y'], translate['z']]
            vertices = apply_t(t, vertices)
            transforms_converted_list.append(('t', np.asarray(t).reshape(3)))
    return vertices, transforms_converted_list

def transform_with_transforms_dat_list(transforms_list, vertices):
    # suitanle for transforms_list read from transform.dat: [('s', array([1., 1., 1.], dtype=float32)), ('rot', array([[ 1., -0.,  0.], [ 0.,  0.,  1.], [-0., -1.,  0.]])), ...]
    assert len(vertices.shape)==2 and vertices.shape[1]==3 and vertices.shape[0]>0
    for transform in transforms_list:
        transform_name = transform[0]
        transform_param = transform[1]
        assert transform_name in ['s', 'rot', 't']
        if transform_name == 's':
            vertices = apply_s(transform_param, vertices)
        if transform_name == 'rot':
            vertices = apply_R(transform_param, vertices)
        if transform_name == 't':
            vertices = apply_t(transform_param, vertices)
    return vertices
        
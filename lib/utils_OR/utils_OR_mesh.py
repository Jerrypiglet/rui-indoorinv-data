from inspect import isdatadescriptor
import trimesh
import numpy as np
import quaternion
import copy
from pathlib import Path

# original obj operations by Zhengqin
def loadMesh(name):
    vertices = []
    faces = []
    with open(str(name), 'r') as meshIn:
        lines = meshIn.readlines()
    lines = [x.strip() for x in lines if len(x.strip()) > 2 ]
    for l in lines:
        if l[0:2] == 'v ':
            vstr = l.split(' ')[1:4]
            varr = [float(x) for x in vstr ]
            varr = np.array(varr).reshape([1, 3])
            vertices.append(varr)
        elif l[0:2] == 'f ':
            fstr = l.split(' ')[1:4]
            farr = [int(x.split('/')[0]) for x in fstr ]
            farr = np.array(farr).reshape([1, 3])
            faces.append(farr)

    vertices = np.concatenate(vertices, axis=0).astype(np.float32)
    faces = np.concatenate(faces, axis=0).astype(np.int32)
    return vertices, faces


def writeMesh(name, vertices, faces):
    with open(name, 'w') as meshOut:
        for n in range(0, vertices.shape[0]):
            meshOut.write('v %.3f %.3f %.3f\n' %
                    (vertices[n, 0], vertices[n, 1], vertices[n, 2]))
        for n in range(0,faces.shape[0]):
            meshOut.write('f %d %d %d\n' %
                    (faces[n, 0], faces[n, 1], faces[n, 2]))

def write_one_mesh_from_v_f_lists(mesh_path: str, vertices_list: list, faces_list: list, ids_list: list=[]):
    assert len(vertices_list) == len(faces_list)
    if ids_list == []:
        ids_list == ['']*len(vertices_list)

    num_vertices = 0
    f_list = []
    for vertices, faces, id in zip(vertices_list, faces_list, ids_list):
        f_list.append(copy.deepcopy(faces + num_vertices))
        num_vertices += vertices.shape[0]

    v_list = copy.deepcopy(vertices_list)
    
    writeMesh(mesh_path, np.vstack(v_list), np.vstack(f_list))

def write_mesh_list_from_v_f_lists(mesh_dir: Path, vertices_list: list, faces_list: list, ids_list: list=[]):
    assert len(vertices_list) == len(faces_list)
    if ids_list == []:
        ids_list == ['']*len(vertices_list)

    for idx, (vertices, faces, id) in enumerate(zip(vertices_list, faces_list, ids_list)):
        writeMesh(str(mesh_dir / ('%d_%s.obj'%(idx, id))), vertices, faces)

def computeBox(vertices):
    minX, maxX = vertices[:, 0].min(), vertices[:, 0].max()
    minY, maxY = vertices[:, 1].min(), vertices[:, 1].max()
    minZ, maxZ = vertices[:, 2].min(), vertices[:, 2].max()

    corners = []
    corners.append(np.array([minX, minY, minZ]).reshape(1, 3))
    corners.append(np.array([maxX, minY, minZ]).reshape(1, 3))
    corners.append(np.array([maxX, minY, maxZ]).reshape(1, 3))
    corners.append(np.array([minX, minY, maxZ]).reshape(1, 3))

    corners.append(np.array([minX, maxY, minZ]).reshape(1, 3))
    corners.append(np.array([maxX, maxY, minZ]).reshape(1, 3))
    corners.append(np.array([maxX, maxY, maxZ]).reshape(1, 3))
    corners.append(np.array([minX, maxY, maxZ]).reshape(1, 3))

    corners = np.concatenate(corners).astype(np.float32)

    faces = []
    faces.append(np.array([1, 2, 3]).reshape(1, 3))
    faces.append(np.array([1, 3, 4]).reshape(1, 3))

    faces.append(np.array([5, 7, 6]).reshape(1, 3))
    faces.append(np.array([5, 8, 7]).reshape(1, 3))

    faces.append(np.array([1, 6, 2]).reshape(1, 3))
    faces.append(np.array([1, 5, 6]).reshape(1, 3))

    faces.append(np.array([2, 7, 3]).reshape(1, 3))
    faces.append(np.array([2, 6, 7]).reshape(1, 3))

    faces.append(np.array([3, 8, 4]).reshape(1, 3))
    faces.append(np.array([3, 7, 8]).reshape(1, 3))

    faces.append(np.array([4, 5, 1]).reshape(1, 3))
    faces.append(np.array([4, 8, 5]).reshape(1, 3))

    faces = np.concatenate(faces).astype(np.int32)

    return corners, faces


def computeTransform(vertices, t, q, s):
    if s != None:
        scale = np.array(s, dtype=np.float32).reshape(1, 3)
        vertices = vertices * scale

    if q != None:
        q = np.quaternion(q[0], q[1], q[2], q[3])
        rotMat = quaternion.as_rotation_matrix(q)
        if np.abs(rotMat[1, 1]) > 0.5:
            d = rotMat[1, 1]
            rotMat[:, 1] = 0
            rotMat[1, :] = 0
            if d < 0:
                rotMat[1, 1] = -1
            else:
                rotMat[1, 1] = 1
        vertices = np.matmul(rotMat, vertices.transpose())
        vertices = vertices.transpose()

    if t != None:
        trans = np.array(t, dtype=np.float32).reshape(1, 3)
        vertices = vertices + trans

    return vertices, trans.squeeze(), rotMat, scale.squeeze()

# mesh operations by Rui
def load_trimesh(layout_obj_file):
    mesh = trimesh.load_mesh(str(layout_obj_file))
    mesh = as_trimesh(mesh)
    return mesh

def as_trimesh(scene_or_mesh):
    """
    Convert a possible scene to a mesh.

    If conversion occurs, the returned mesh has only vertex and face data.
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = trimesh.util.concatenate(
                tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                    for g in scene_or_mesh.geometry.values()))
    else:
        assert(isinstance(scene_or_mesh, trimesh.Trimesh))
        mesh = scene_or_mesh
    return mesh

def remove_top_down_faces(mesh):
    v = np.array(mesh.vertices)
    f = list(np.array(mesh.faces))
    f_after = []
    for f0 in f:
        if not(v[f0[0]][2]==v[f0[1]][2]==v[f0[2]][2]):
            f_after.append(f0)
    new_mesh = trimesh.Trimesh(vertices=v, faces=np.asarray(f_after))
    return new_mesh

def flip_ceiling_normal(faces, vertices):
    '''
    works on uv_mapped.obj files
    '''
    ceiling_z = np.amax(vertices[:, 2])
    ceiling_faces_N = 0
    # print(faces)
    for face_idx, face in enumerate(faces):
        vertices_3 = vertices[face-1]
        if np.all(vertices_3[:, 2] == ceiling_z):
            ceiling_faces_N += 1
            # print(face_idx, face.shape)
            faces[face_idx] = face[[1, 0, 2]] # flip the normal
    assert ceiling_faces_N == 2, 'should be two triangles for the ceiling'
    # print('->', faces)

    return faces

def mesh_to_contour(mesh, if_input_is_v_e=False, vertical_dim=-1):
    if if_input_is_v_e:
        v, e = mesh
    else:
        mesh = remove_top_down_faces(mesh)
        v = np.array(mesh.vertices)
        e = np.array(mesh.edges)

    v_new_id_list = []
    v_new_id = 0
    floor_z = np.amin(v[:, vertical_dim])
    for v0 in v:
        if v0[vertical_dim]==floor_z:
            v_new_id_list.append(v_new_id)
            v_new_id += 1
        else:
            v_new_id_list.append(-1)

    v_new = np.array([np.delete(v[x], vertical_dim) for x in range(len(v)) if v_new_id_list[x]!=-1])
    e_new = np.array([[v_new_id_list[e[x][0]], v_new_id_list[e[x][1]]] for x in range(len(e)) if (v_new_id_list[e[x][0]]!=-1 and v_new_id_list[e[x][1]]!=-1)])

    return v_new, e_new

def mesh_to_skeleton(mesh):
    # mesh = remove_top_down_faces(mesh)
    v = np.array(mesh.vertices)
    e = mesh.edges

    floor_z = np.amin(v[:, -1])
    ceil_z = np.amax(v[:, -1])
    e_new = []
    for e0 in e:
        z0, z1 = v[e0[0]][2], v[e0[1]][2]
        if z0 == z1:
            e_new.append(e0)
        elif np.array_equal(v[e0[0]][:2], v[e0[1]][:2]):
            e_new.append(e0)
    e_new = np.array(e_new)

    return v, e_new, abs(floor_z - ceil_z)

def v_pairs_from_v3d_e(v, e):
    '''
    generate vertice pairs (for each edge), given all v and e
    '''
    v_pairs = [(np.array([v[e0[0]][0], v[e0[1]][0]]), np.array([v[e0[0]][1], v[e0[1]][1]]), np.array([v[e0[0]][2], v[e0[1]][2]])) for e0 in e]
    return v_pairs

def v_pairs_from_v2d_e(v, e):
    v_pairs = [(np.array([v[e0[0]][0], v[e0[1]][0]]), np.array([v[e0[0]][1], v[e0[1]][1]])) for e0 in e]
    return v_pairs

def v_xytuple_from_v2d_e(v, e):
    v_pairs = [(v[e0[0]], v[e0[1]]) for e0 in e]
    return v_pairs

def transform_v(vertices, transforms):
    '''
    transform vertices to world coordinates
    '''
    assert transforms[0][0]=='s' and transforms[1][0]=='rot' and transforms[2][0]=='t'
    # following computeTransform()
    assert len(vertices.shape)==2
    assert vertices.shape[1]==3

    s = transforms[0][1]
    scale = np.array(s, dtype=np.float32).reshape(1, 3)
    vertices = vertices * scale
    
    rotMat = transforms[1][1]
    vertices = np.matmul(rotMat, vertices.transpose())
    vertices = vertices.transpose()
    
    t = transforms[2][1]
    trans = np.array(t, dtype=np.float32).reshape(1, 3)
    vertices = vertices + trans
    
    return vertices

from scipy.spatial import ConvexHull

def minimum_bounding_rectangle(points):
    # https://gis.stackexchange.com/questions/22895/finding-minimum-area-rectangle-for-given-points
    """
    Find the smallest bounding rectangle for a set of points.
    Returns a set of points representing the corners of the bounding box.

    :param points: an nx2 matrix of coordinates
    :rval: an nx2 matrix of coordinates
    """
    from scipy.ndimage.interpolation import rotate
    pi2 = np.pi/2.

    # get the convex hull for the points
    hull_points = points[ConvexHull(points).vertices]

    # calculate edge angles
    edges = np.zeros((len(hull_points)-1, 2))
    edges = hull_points[1:] - hull_points[:-1]

    angles = np.zeros((len(edges)))
    angles = np.arctan2(edges[:, 1], edges[:, 0])

    angles = np.abs(np.mod(angles, pi2))
    angles = np.unique(angles)

    # find rotation matrices
    # XXX both work
    rotations = np.vstack([
        np.cos(angles),
        np.cos(angles-pi2),
        np.cos(angles+pi2),
        np.cos(angles)]).T
#     rotations = np.vstack([
#         np.cos(angles),
#         -np.sin(angles),
#         np.sin(angles),
#         np.cos(angles)]).T
    rotations = rotations.reshape((-1, 2, 2))

    # apply rotations to the hull
    rot_points = np.dot(rotations, hull_points.T)

    # find the bounding points
    min_x = np.nanmin(rot_points[:, 0], axis=1)
    max_x = np.nanmax(rot_points[:, 0], axis=1)
    min_y = np.nanmin(rot_points[:, 1], axis=1)
    max_y = np.nanmax(rot_points[:, 1], axis=1)

    # find the box with the best area
    areas = (max_x - min_x) * (max_y - min_y)
    best_idx = np.argmin(areas)

    # return the best box
    x1 = max_x[best_idx]
    x2 = min_x[best_idx]
    y1 = max_y[best_idx]
    y2 = min_y[best_idx]
    r = rotations[best_idx]

    rval = np.zeros((4, 2))
    rval[0] = np.dot([x1, y2], r)
    rval[1] = np.dot([x2, y2], r)
    rval[2] = np.dot([x2, y1], r)
    rval[3] = np.dot([x1, y1], r)

    return rval

def writeMesh_rect(name, vertices):
    if vertices.shape==(4, 3):
        faces = np.array([[1, 2, 3], [1, 3, 4]])
    elif vertices.shape==(8, 3):
        # faces = np.array([[1, 3, 2], [1, 3, 4], [5, 7, 6], [5, 7, 8], \
        #                  [4, 7, 3], [4, 7, 8], [1, 6, 2], [1, 6, 5], \
        #                   [3, 6, 2], [3, 6, 7], [4, 5, 1], [4, 5, 8]])
        faces = np.array([[0, 3, 4], [7, 4, 3], \
            [6, 2, 1], [1, 5, 6], \
                [0, 4, 1], [5, 1, 4], \
                    [3, 2, 7], [6, 7, 2], \
                        [7, 6, 4], [5, 4, 6], \
                            [1, 2, 3], [0, 1, 3]
                            ])
        faces += 1
        # faces = np.array([[0, 3, 4], [7, 4, 3]])
#
    else:
        raise ValueError('writeMesh_rect: vertices of invalid shape!')

    with open(name, 'w') as meshOut:
        for n in range(0, vertices.shape[0]):
            meshOut.write('v %.3f %.3f %.3f\n' %
                    (vertices[n, 0], vertices[n, 1], vertices[n, 2]))
        for n in range(0,faces.shape[0]):
            meshOut.write('f %d %d %d\n' %
                    (faces[n, 0], faces[n, 1], faces[n, 2]))
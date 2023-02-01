from pathlib import Path
import numpy as np
import trimesh
from utils_OR.utils_OR_mesh import sample_mesh, simplify_mesh, computeBox

def load_monosdf_shape(shape_file: Path, shape_params_dict={}, scale_offset: tuple=()):
    if_sample_mesh = shape_params_dict.get('if_sample_mesh', False)
    sample_mesh_ratio = shape_params_dict.get('sample_mesh_ratio', 1.)
    sample_mesh_min = shape_params_dict.get('sample_mesh_min', 100)
    sample_mesh_max = shape_params_dict.get('sample_mesh_max', 1000)

    if_simplify_mesh = shape_params_dict.get('if_simplify_mesh', False)
    simplify_mesh_ratio = shape_params_dict.get('simplify_mesh_ratio', 1.)
    simplify_mesh_min = shape_params_dict.get('simplify_mesh_min', 100)
    simplify_mesh_max = shape_params_dict.get('simplify_mesh_max', 1000)
    if_remesh = shape_params_dict.get('if_remesh', True) # False: images/demo_shapes_3D_NO_remesh.png; True: images/demo_shapes_3D_YES_remesh.png
    remesh_max_edge = shape_params_dict.get('remesh_max_edge', 0.1)

    if if_sample_mesh:
        sample_pts_list = []

    shape_tri_mesh = trimesh.load_mesh(str(shape_file))
    vertices, faces = shape_tri_mesh.vertices, shape_tri_mesh.faces+1 # faces is 1-based; [TODO] change to 0-based in all methods
    if scale_offset != ():
        (scale, offset) = scale_offset
        vertices = vertices / scale
        vertices = vertices - offset

    _id = Path(shape_file).stem

    # --sample mesh--
    if if_sample_mesh:
        sample_pts, face_index = sample_mesh(vertices, faces, sample_mesh_ratio, sample_mesh_min, sample_mesh_max)
        sample_pts_list.append(sample_pts)
        # print(sample_pts.shape[0])

    # --simplify mesh--
    if if_simplify_mesh and simplify_mesh_ratio != 1.: # not simplying for mesh with very few faces
        vertices, faces, (N_triangles, target_number_of_triangles) = simplify_mesh(vertices, faces, simplify_mesh_ratio, simplify_mesh_min, simplify_mesh_max, if_remesh=if_remesh, remesh_max_edge=remesh_max_edge, _id=_id)
        if N_triangles != faces.shape[0]:
            print('[%s] Mesh simplified to %d->%d triangles (target: %d).'%(_id, N_triangles, faces.shape[0], target_number_of_triangles))

    bverts, bfaces = computeBox(vertices)
    shape_dict = {
        'filename': str(shape_file), 
        'id': _id, 
        'random_id': 'XXXXXX', 
        'if_in_emitter_dict': False, 
        # [IMPORTANT] currently relying on definition of walls and ceiling in XML file to identify those, becuase sometimes they can be complex meshes instead of thin rectangles
        'is_wall': False, 
        'is_ceiling': False, 
        'is_layout': False, 
    }

    return {
        'vertices': vertices, 
        'faces': faces, 
        'bverts': bverts, 
        'bfaces': bfaces, 
        'shape_dict': shape_dict, 
        '_id': _id, 
    }

def load_monosdf_scale_offset(monosdf_pose_file: Path):
    '''
    v_normalized = scale * (v_ori + offset)
    '''
    camera_dict = np.load(str(monosdf_pose_file))
    
    # scale_mat = camera_dict['scale_mat_%d' % 0].astype(np.float32)
    # scale_mat = np.linalg.inv(scale_mat)
    # scale = scale_mat[0][0]
    # scale_mat[:3] = scale_mat[:3] / scale
    # offset = scale_mat[:3, 3:4].reshape(1, 3) # (1, 3)
    scale = camera_dict['scale'].item()
    offset = -camera_dict['center'].reshape(1, 3) # (1, 3)

    scale_mat = np.eye(4).astype(np.float32)
    scale_mat[:3, 3] = -camera_dict['center']
    scale_mat[:3 ] *= scale 
    scale_mat = np.linalg.inv(scale_mat)

    return (scale, offset), scale_mat
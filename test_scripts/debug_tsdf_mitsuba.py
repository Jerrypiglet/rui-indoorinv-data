import mitsuba as mi
mi.set_variant('llvm_ad_rgb')
from pathlib import Path
import trimesh
import numpy as np
import open3d as o3d
import open3d.visualization as vis

tsdf_file_path = Path('fused_tsdf.obj') # downloadable from https://drive.google.com/open?id=1UtlFsVw5lsNoO4yBgHNT36QRMyJcIxCP
tsdf_file_dump_path = Path('fused_tsdf_dump.obj') # downloadable from https://drive.google.com/open?id=1UtlFsVw5lsNoO4yBgHNT36QRMyJcIxCP

tsdf_shape = trimesh.load_mesh(str(tsdf_file_path), process=True, maintain_order=True)
vertices = np.array(tsdf_shape.vertices)
faces = np.array(tsdf_shape.faces)

# vertices, faces = trimesh.remesh.subdivide_to_size(vertices, faces, max_edge=0.5)
_ = trimesh.Trimesh(vertices=vertices, faces=faces)
# _ = _.simplify_quadratic_decimation(int(vertices.shape[0]*0.95))
# trimesh.repair.fix_inversion(_)
# trimesh.repair.fix_normals(_)
# trimesh.repair.fill_holes(_)
# trimesh.repair.fix_winding(_)

_.export(str(tsdf_file_dump_path))

tsdf_shape = trimesh.load_mesh(str(tsdf_file_dump_path), process=False, maintain_order=True)
vertices = np.array(tsdf_shape.vertices)
faces = np.array(tsdf_shape.faces)

shape_id_dict = {
    'type': tsdf_file_dump_path.suffix[1:],
    'filename': str(tsdf_file_dump_path), 
    }
        
mi_scene = mi.load_dict({
    'type': 'scene',
    'shape_id': shape_id_dict, 
})

origin = np.array([[-2.503], [0.526], [0.327]], dtype=np.float32)

origin = np.tile(np.array(origin).reshape((1, 3)), (vertices.shape[0], 1))
ds = vertices - origin
ds_norm = (np.linalg.norm(ds, axis=1, keepdims=1)+1e-6)
ds = ds / ds_norm

xs = origin
xs_mi = mi.Point3f(xs)
ds_mi = mi.Vector3f(ds)
# ray origin, direction, t_max
rays_mi = mi.Ray3f(xs_mi, ds_mi)
ret = mi_scene.ray_intersect(rays_mi) # https://mitsuba.readthedocs.io/en/stable/src/api_reference.html?highlight=write_ply#mitsuba.Scene.ray_intersect
# returned structure contains intersection location, nomral, ray step, ...
t = ret.t.numpy()
# import ipdb; ipdb.set_trace()

'''
visualize the mesh; colorized by t of rays (camera center --> vertices)
'''
o3d_geometry_list = []
tsdf_mesh_o3d = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(tsdf_shape.vertices), o3d.utility.Vector3iVector(tsdf_shape.faces))
tsdf_mesh_o3d.compute_vertex_normals()
tsdf_mesh_o3d.compute_triangle_normals()
# tsdf_mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(self.os.tsdf_fused_dict['colors'])
samples_v_t = t; max_t = np.amax(t)
assert samples_v_t.shape[0] == vertices.shape[0]
samples_v_ = (samples_v_t / float(max_t)).reshape(-1, 1)
samples_v_ = np.array([[1., 0., 0.]]) * samples_v_ + np.array([[0., 0., 1.]]) * (1. - samples_v_)
tsdf_mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(samples_v_) # [TODO] not sure how to set triangle colors... the Open3D documentation is pretty confusing and actually does not work... http://www.open3d.org/docs/release/python_api/open3d.t.geometry.TriangleMesh.html
o3d_geometry_list.append(tsdf_mesh_o3d)

'''
visualize (subset of) rays with t=inf
'''
t_inf_mask = np.isinf(t)
# t[t_inf_mask] = 1
t_inf_mask[20000:] = False
t_inf_mask[:19900] = False

rays_o3d = o3d.geometry.LineSet()
rays_o3d.points = o3d.utility.Vector3dVector(np.vstack((origin[t_inf_mask], vertices[t_inf_mask])))
rays_o3d.colors = o3d.utility.Vector3dVector([[0., 1, 0.] for _ in range(np.sum(t_inf_mask))])
rays_o3d.lines = o3d.utility.Vector2iVector([[_, _+np.sum(t_inf_mask)] for _ in range(np.sum(t_inf_mask))])
o3d_geometry_list.append(rays_o3d)

'''
run o3d instance
'''
vis = o3d.visualization.Visualizer()
W = vis.create_window()
opt = vis.get_render_option()
opt.background_color = np.asarray([1., 1., 1.])

o3d_geometry_list += [o3d.geometry.TriangleMesh.create_coordinate_frame()]
for _ in o3d_geometry_list:
    if isinstance(_, list):
        geo, geo_name = _
    else:
        geo = _
    vis.add_geometry(geo)

vis.run()


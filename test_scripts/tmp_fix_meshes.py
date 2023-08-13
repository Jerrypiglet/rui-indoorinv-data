import trimesh
from pathlib import Path
import numpy as np

# shape_path  = Path('/data/Openrooms_RAW/uv_mapped/04379243/e0229fb0e8d85e1fbfd790572919a77f/alignedNew.obj')
# shape_tri_mesh = trimesh.load_mesh(str(shape_path), process=True)
# # xx = np.array(shape_tri_mesh.vertex_normals)
# # import ipdb; ipdb.set_trace()
# # trimesh.repair.fill_holes(shape_tri_mesh)
# # trimesh.repair.fix_winding(shape_tri_mesh)
# # trimesh.repair.fix_inversion(shape_tri_mesh)
# # trimesh.repair.fix_normals(shape_tri_mesh)

# shape_tri_mesh.export(str(shape_path))



from pathlib import Path
import mitsuba as mi
mi.set_variant('cuda_ad_rgb')
from tqdm import tqdm

# mi.load_file('/home/ruizhu/Documents/Projects/rui-indoorinv-data/mitsuba/tmp_scene_20230812-025631-3BJ0D.xml')

# mi_scene = mi.load_dict({
#         'type': 'scene',
#         'shape_id': {
#             'type': shape_path.suffix[1:],
#             'filename': str(shape_path), 
#             }, 
#     })

# import ipdb; ipdb.set_trace()


all_shape_paths = Path('/data/Openrooms_RAW/uv_mapped/').rglob('*.obj')
print('Found %d shape files.'%len(list(all_shape_paths)))
all_shape_paths = Path('/data/Openrooms_RAW/uv_mapped/').rglob('*.obj')

def as_mesh(scene_or_mesh):
    if isinstance(scene_or_mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate([
            trimesh.Trimesh(vertices=m.vertices, faces=m.faces)
            for m in scene_or_mesh.geometry.values()])
    else:
        mesh = scene_or_mesh
    return mesh

for shape_path in tqdm(list(all_shape_paths)):
    # print('=== Loading mesh file for inspection...' + str(shape_path))
    # if str(shape_path) == '/data/Openrooms_RAW/uv_mapped/04379243/e0229fb0e8d85e1fbfd790572919a77f/alignedNew.obj':
    #     import ipdb; ipdb.set_trace()
    # shape_path  = Path('/data/Openrooms_RAW/uv_mapped/04379243/e0229fb0e8d85e1fbfd790572919a77f/alignedNew.obj')
    
    shape_tri_mesh = trimesh.load_mesh(str(shape_path), process=True)
    shape_tri_mesh = as_mesh(shape_tri_mesh)
    shape_tri_mesh = trimesh.Trimesh(faces=shape_tri_mesh.faces, vertices=shape_tri_mesh.vertices, process=False, maintain_order=True)
    shape_tri_mesh.export(str(shape_path))

    shape_id_dict = {
        'type': shape_path.suffix[1:],
        'filename': str(shape_path), 
        }
    mi_scene = mi.load_dict({
        'type': 'scene',
        'shape_id': shape_id_dict, 
    })


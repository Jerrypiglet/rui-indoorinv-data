import trimesh

# mesh_path = '/data/Openrooms_RAW/uv_mapped/03211117/eb75ac8723230375d465c64f5322e86/alignedNew.obj'
# mesh_path = '/data/Openrooms_RAW/uv_mapped/03211117/d0959256c79f60872a9c9a7e9520eea/alignedNew.obj'
# mesh_path = '/data/Openrooms_RAW/uv_mapped/03211117/6272280e5ee3637d4f8f787d72a46973/alignedNew.obj'
# shape_tri_mesh = trimesh.load_mesh(mesh_path, process=True)
# trimesh.repair.fill_holes(shape_tri_mesh)
# trimesh.repair.fix_winding(shape_tri_mesh)
# trimesh.repair.fix_inversion(shape_tri_mesh)
# trimesh.repair.fix_normals(shape_tri_mesh)

# shape_tri_mesh.export(mesh_path)



from pathlib import Path
import mitsuba as mi
mi.set_variant('cuda_ad_rgb')
from tqdm import tqdm

all_shape_paths = Path('/data/Openrooms_RAW/uv_mapped/').rglob('*.obj')
print('Found %d shape files.'%len(list(all_shape_paths)))
all_shape_paths = Path('/data/Openrooms_RAW/uv_mapped/').rglob('*.obj')

for shape_path in tqdm(list(all_shape_paths)):
    # print('=== Loading mesh file for inspection...' + str(shape_path))
    shape_id_dict = {
        'type': shape_path.suffix[1:],
        'filename': str(shape_path), 
        }
    mi_scene = mi.load_dict({
        'type': 'scene',
        'shape_id': shape_id_dict, 
    })


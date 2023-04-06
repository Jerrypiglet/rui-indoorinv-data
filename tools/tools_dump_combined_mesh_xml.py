import numpy as np
from pathlib import Path
from lib.utils_OR.utils_OR_mesh import get_rectangle_mesh, loadMesh
from lib.utils_OR.utils_OR_xml import get_XML_root
from tqdm import tqdm
import trimesh

xml_file_path = Path('data/indoor_synthetic/kitchen/scene_v3.xml')
assert xml_file_path.exists()
dump_path = Path('test_files/xml_mesh_dump')
dump_path.mkdir(exist_ok=True, parents=True)

root = get_XML_root(str(xml_file_path))
shapes = root.findall('shape')

vertices_list = []
faces_list = []
trimesh_list = []

for shape in tqdm(shapes):
    if shape.get('type') != 'obj':
        assert shape.get('type') == 'rectangle'
        transform_m = np.array(shape.findall('transform')[0].findall('matrix')[0].get('value').split(' ')).reshape(4, 4).astype(np.float32) # [[R,t], [0,0,0,1]]
        (vertices, faces) = get_rectangle_mesh(transform_m[:3, :3], transform_m[:3, 3:4])
    else:
        if not len(shape.findall('string')) > 0: continue
        filename = shape.findall('string')[0]; assert filename.get('name') == 'filename'
        obj_path = xml_file_path.parent / filename.get('value') # [TODO] deal with transform
        vertices, faces = loadMesh(obj_path) # based on L430 of adjustObjectPoseCorrectChairs.py; faces is 1-based!
    trimesh_list.append(trimesh.Trimesh(vertices, faces-1))

trimesh_combined = trimesh.util.concatenate(trimesh_list)
trimesh_combined.export(str(dump_path / 'combined_mesh.obj'))

print('Done. Mesh saved to %s'%str(dump_path / 'combined_mesh.obj'))


        

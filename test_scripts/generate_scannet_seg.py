'''
usage: python test_scripts/generate_scannet_seg.py

For Mask3D: 
given scene ply file, generate preprocessed scene ply file with segment ids, 
using https://github.com/ScanNet/ScanNet/tree/master/Segmentator
'''

segmentor_path = '/Users/jerrypiglet/Documents/Projects/ScanNet/Segmentator/segmentator' # https://github.com/ScanNet/ScanNet.git

# (648472, 3) (1230643, 3) [-3.85378083 -1.26112087 -4.27540642] [2.9140625  1.7583029  4.24371038] 0 float64 int64
mesh_path = '/Users/jerrypiglet/Documents/Projects/rui-indoorinv-data/data/openrooms_public/mainDiffMat_xml/scene0403_01/fused_tsdf.ply'

# (95715, 3) (185307, 3) [-0.03978582 -0.00638135  0.00121044] [3.74416876 3.12823582 2.23039341] 0 float64 int64; 
# 471 segs
# mesh_path = '/Users/jerrypiglet/Documents/Projects/rui-indoorinv-data/data/ScanNet/scene0055_00/scene0055_00_vh_clean_2.ply'

import trimesh
from pathlib import Path
import numpy as np
assert Path(mesh_path).exists()
tsdf_mesh = trimesh.load_mesh(str(mesh_path), process=False)
vertices = np.array(tsdf_mesh.vertices)
faces = np.array(tsdf_mesh.faces)
print('=== Loaded mesh from '+str(mesh_path), vertices.shape, faces.shape, np.amin(vertices, axis=0), np.amax(vertices, axis=0), np.amin(faces), vertices.dtype, faces.dtype)
_tmp_trimesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=tsdf_mesh.visual.vertex_colors)
# _tmp_trimesh = _tmp_trimesh.simplify_quadratic_decimation(int(len(faces) * 0.1))
_tmp_trimesh.export('tmp.ply')

'''
run: generate segs
'''
mesh_path = 'tmp.ply'
kThresh=0.2
segMinVerts=2000
cmd = '%s %s %.3f %d'%(str(segmentor_path), str(mesh_path), kThresh, segMinVerts) # default: kThresh=0.01 segMinVerts=20
import os
print('=== Running segmentor...')
print(cmd)

'''
generate labels txt file
'''
output_file = Path(mesh_path).stem + ('.%.6f.segs.json'%kThresh)
if Path(output_file).exists():
    Path(output_file).unlink()
os.system(cmd)
assert Path(output_file).exists(), output_file

import json
def _read_json(path):
    with open(path) as f:
        file = json.load(f)
    return file

segments = _read_json(str(output_file))
segments = np.array(segments["segIndices"])
labels_txt_path = output_file.replace('.json', '.txt')
with open(str(labels_txt_path), "w") as txt_file:
    for line in segments:
        txt_file.write(str(line) + "\n") # works with any number of elements in a line
print('=== labels_txt ([%d] segments) dumped to: '%(np.unique(segments).shape[0]), labels_txt_path)
        
'''
inspect mesh file for visualization
'''
print('=== Loading mesh file for inspection...' + str(mesh_path))
from plyfile import PlyData
with open(mesh_path, 'rb') as f:
    plydata = PlyData.read(f)
    num_verts = plydata['vertex'].count
    assert num_verts == len(segments)
    assert 'red' in plydata['vertex']

out_path = 'tmp_seg.ply'
cmd = 'python test_scripts/scannet_utils/visualize_labels_on_mesh.py --pred_file %s --mesh_file %s --output_file %s'%(str(labels_txt_path), str(mesh_path), out_path)
print(cmd)
os.system(cmd)
print('=== Output mesh dumped to: ', out_path)

'''
Load the output mesh, you will get something like: 

- ScanNet with default segmentator params:

![](https://i.imgur.com/VfpnJRb.jpg)

- OpenRooms with our params:

![20230811143737](https://i.imgur.com/GjfCYCA.jpg)

'''
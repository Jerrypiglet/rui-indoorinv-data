'''
usage: python test_scripts/visualize_scannet_seg.py

dump labels as npy files (e.g. segment_ids.npy: (N,), int) from Mask3D/notebooks/debug_scannet_processing.ipynb
'''

from pathlib import Path
import numpy as np

root = Path('/Users/jerrypiglet/Documents/Projects/rui-indoorinv-data')
scannet_dir = root / 'data/ScanNet/scene0055_00/'
assert scannet_dir.exists()

labels_path = scannet_dir / 'segment_ids.npy'
assert labels_path.exists()
xx = np.load(str(labels_path))
labels_txt_path = scannet_dir / 'segment_ids.txt'
with open(str(labels_txt_path), "w") as txt_file:
    for line in xx:
        txt_file.write(str(line) + "\n") # works with any number of elements in a line

mesh_path = scannet_dir / 'scene0055_00_vh_clean_2.ply'
assert mesh_path.exists()

out_path = 'tmp_scene0055_00_vh_clean_2_seg.ply'
cmd = 'python test_scripts/scannet_utils/visualize_labels_on_mesh.py --pred_file %s --mesh_file %s --output_file %s'%(str(labels_txt_path), str(mesh_path), out_path)
print(cmd)

'''
Load the output mesh, you will get something like: 

![](https://i.imgur.com/VfpnJRb.jpg)

'''
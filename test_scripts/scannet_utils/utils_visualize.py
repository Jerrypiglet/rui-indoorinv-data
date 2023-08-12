# Example script to visualize labels in the evaluation format on the corresponding mesh.
# Inputs:
#   - predicted labels as a .txt file with one line per vertex
#   - the corresponding *_vh_clean_2.ply mesh
# Outputs a .ply with vertex colors, a different color per value in the predicted .txt file
#
# example usage: visualize_labels_on_mesh.py --pred_file [path to predicted labels file] --mesh_file [path to the *_vh_clean_2.ply mesh] --output_file [output file]

# python imports
import math
import os, sys, argparse
import inspect
import json

try:
    import numpy as np
except:
    print("Failed to import numpy package.")
    sys.exit(-1)
try:
    from plyfile import PlyData, PlyElement
except:
    print("Please install the module 'plyfile' for PLY i/o, e.g.")
    print("pip install plyfile")
    sys.exit(-1)

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
sys.path.insert(0,currentdir)
import util
import util_3d
import colorsys

def _get_colors(num_colors):
    colors=[]
    for i in np.arange(0., 360., 360. / num_colors):
        hue = i/360.
        lightness = (50 + np.random.rand() * 10)/100.
        saturation = (90 + np.random.rand() * 10)/100.
        colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))
    return colors

def visualize(pred_file, mesh_file, output_file, if_random_colors=True, colors=None):
    if not output_file.endswith('.ply'):
        util.print_error('output file must be a .ply file')
    ids = util_3d.load_ids(pred_file)
    
    if colors is None:
        if if_random_colors and colors is None:
            _id_list = np.unique(ids).tolist()
            colors = _get_colors(len(_id_list))
            colors = (np.array(colors) * 255).astype(np.uint8)
            colors = {_id: _color for _id, _color in zip(_id_list, colors)}
            colors[255] = np.array([255, 255, 255], dtype=np.uint8)
            # colors = _get_colors(int(ids.max())+1)
        else:
            colors = util.create_color_palette()
    else:
        assert isinstance(colors, dict) or isinstance(colors, list), 'colors must be a dict or a list'
    
    num_colors = len(colors)
    with open(mesh_file, 'rb') as f:
        plydata = PlyData.read(f)
        num_verts = plydata['vertex'].count
        if num_verts != len(ids):
            util.print_error('#predicted labels = ' + str(len(ids)) + 'vs #mesh vertices = ' + str(num_verts))
        # *_vh_clean_2.ply has colors already
        for i in range(num_verts):
            if ids[i] >= num_colors and not if_random_colors:
                util.print_error('found predicted label ' + str(ids[i]) + ' not in nyu40 label set')
            color = colors[ids[i]]
            plydata['vertex']['red'][i] = color[0]
            plydata['vertex']['green'][i] = color[1]
            plydata['vertex']['blue'][i] = color[2]
    plydata.write(output_file)
    print('saved to ' + output_file)
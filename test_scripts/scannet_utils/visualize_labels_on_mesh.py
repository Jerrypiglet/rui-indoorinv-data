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

parser = argparse.ArgumentParser()
parser.add_argument('--pred_file', required=True, help='path to predicted labels file as .txt evaluation format')
parser.add_argument('--mesh_file', required=True, help='path to the *_vh_clean_2.ply mesh')
parser.add_argument('--output_file', required=True, help='output .ply file')
parser.add_argument('--if_random_colors', required=False, default=True, type=bool, help='False to use nyu40 colors, True to use random colors`')
opt = parser.parse_args()

from utils_visualize import visualize

def main():
    visualize(opt.pred_file, opt.mesh_file, opt.output_file, if_random_colors=opt.if_random_colors)


if __name__ == '__main__':
    main()

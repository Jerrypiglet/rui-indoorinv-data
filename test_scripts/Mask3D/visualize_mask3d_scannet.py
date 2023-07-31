'''
adapted from Mask3D repo: datasets/semseg.py

To visualize the processed scannet data
'''

from pathlib import Path
import numpy as np
import torch
import yaml

import logging
from yaml import CLoader as Loader
from itertools import product
from pathlib import Path
from random import random, sample, uniform
from typing import List, Optional, Tuple, Union
from random import choice
from copy import deepcopy
from random import randrange
import albumentations as A


BASE_PATH = Path('/Users/jerrypiglet/Documents/Projects/rui-indoorinv-data/data/Mask3D')

dataset_name = "scannet"
data_dir = [BASE_PATH / "data/processed/scannet"]
label_db_filepath = BASE_PATH / "data/processed/scannet/label_database.yaml"
color_mean_std = BASE_PATH / "data/processed/scannet/color_mean_std.yaml"

# add from conf/data/datasets/scannet.yaml
mode = "validation"
num_labels = 20
filter_out_classes=[0, 1]
label_offset=2
add_normals = False
add_raw_coordinates = True

add_colors = True
add_instance = False
data_percent = 1.0
ignore_label = 255
volume_augmentations_path = None
image_augmentations_path = None
instance_oversampling=0
place_around_existing=False
max_cut_region=0
point_per_cut=100
flip_in_center=False
noise_rate=0.0
resample_points=0.0
cache_data=False
add_unlabeled_pc=False
cropping=False
cropping_args=None
is_tta=False
crop_min_size=20000
crop_length=6.0
cropping_v1=True
reps_per_epoch=1
area=-1
on_crops=False
eval_inner_core=-1
add_clip=False
is_elastic_distortion=True
color_drop=0.0

SCANNET_COLOR_MAP_20 = {
    0: (0.0, 0.0, 0.0),
    1: (174.0, 199.0, 232.0),
    2: (152.0, 223.0, 138.0),
    3: (31.0, 119.0, 180.0),
    4: (255.0, 187.0, 120.0),
    5: (188.0, 189.0, 34.0),
    6: (140.0, 86.0, 75.0),
    7: (255.0, 152.0, 150.0),
    8: (214.0, 39.0, 40.0),
    9: (197.0, 176.0, 213.0),
    10: (148.0, 103.0, 189.0),
    11: (196.0, 156.0, 148.0),
    12: (23.0, 190.0, 207.0),
    14: (247.0, 182.0, 210.0),
    15: (66.0, 188.0, 102.0),
    16: (219.0, 219.0, 141.0),
    17: (140.0, 57.0, 197.0),
    18: (202.0, 185.0, 52.0),
    19: (51.0, 176.0, 203.0),
    20: (200.0, 54.0, 131.0),
    21: (92.0, 193.0, 61.0),
    22: (78.0, 71.0, 183.0),
    23: (172.0, 114.0, 82.0),
    24: (255.0, 127.0, 14.0),
    25: (91.0, 163.0, 138.0),
    26: (153.0, 98.0, 156.0),
    27: (140.0, 153.0, 101.0),
    28: (158.0, 218.0, 229.0),
    29: (100.0, 125.0, 154.0),
    30: (178.0, 127.0, 135.0),
    32: (146.0, 111.0, 194.0),
    33: (44.0, 160.0, 44.0),
    34: (112.0, 128.0, 144.0),
    35: (96.0, 207.0, 209.0),
    36: (227.0, 119.0, 194.0),
    37: (213.0, 92.0, 176.0),
    38: (94.0, 106.0, 211.0),
    39: (82.0, 84.0, 163.0),
    40: (100.0, 85.0, 144.0),
}

task = "instance_segmentation"
assert task in [
        "instance_segmentation",
        "semantic_segmentation",
    ], "unknown task"


color_map = SCANNET_COLOR_MAP_20
color_map[255] = (255, 255, 255)

'''
helper functions
'''

def _load_yaml(filepath):
    with open(filepath) as f:
        file = yaml.load(f, Loader=Loader)
        # file = yaml.load(f)
    return file

def _select_correct_labels(labels, num_labels):
    number_of_validation_labels = 0
    number_of_all_labels = 0
    for (
        k,
        v,
    ) in labels.items():
        number_of_all_labels += 1
        if v["validation"]:
            number_of_validation_labels += 1

    if num_labels == number_of_all_labels:
        return labels
    elif num_labels == number_of_validation_labels:
        valid_labels = dict()
        for (
            k,
            v,
        ) in labels.items():
            if v["validation"]:
                valid_labels.update({k: v})
        return valid_labels
    else:
        msg = f"""not available number labels, select from:
        {number_of_validation_labels}, {number_of_all_labels}"""
        raise ValueError(msg)

def _remap_from_zero(labels):
    # set invalid label ids to ignore_label (e.g. 255)
    labels[
        ~np.isin(labels, list(_labels.keys()))
    ] = ignore_label
    # remap to the range from 0: because original labels may not start from 0 (e.g. _labels.keys()=dict_keys([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]))
    for i, k in enumerate(_labels.keys()):
        labels[labels == k] = i
    return labels
    
def splitPointCloud(cloud, size=50.0, stride=50, inner_core=-1):
    if inner_core == -1:
        limitMax = np.amax(cloud[:, 0:3], axis=0)
        width = int(np.ceil((limitMax[0] - size) / stride)) + 1
        depth = int(np.ceil((limitMax[1] - size) / stride)) + 1
        cells = [
            (x * stride, y * stride)
            for x in range(width)
            for y in range(depth)
        ]
        blocks = []
        for (x, y) in cells:
            xcond = (cloud[:, 0] <= x + size) & (cloud[:, 0] >= x)
            ycond = (cloud[:, 1] <= y + size) & (cloud[:, 1] >= y)
            cond = xcond & ycond
            block = cloud[cond, :]
            blocks.append(block)
        return blocks
    else:
        limitMax = np.amax(cloud[:, 0:3], axis=0)
        width = int(np.ceil((limitMax[0] - inner_core) / stride)) + 1
        depth = int(np.ceil((limitMax[1] - inner_core) / stride)) + 1
        cells = [
            (x * stride, y * stride)
            for x in range(width)
            for y in range(depth)
        ]
        blocks_outer = []
        conds_inner = []
        for (x, y) in cells:
            xcond_outer = (
                cloud[:, 0] <= x + inner_core / 2.0 + size / 2
            ) & (cloud[:, 0] >= x + inner_core / 2.0 - size / 2)
            ycond_outer = (
                cloud[:, 1] <= y + inner_core / 2.0 + size / 2
            ) & (cloud[:, 1] >= y + inner_core / 2.0 - size / 2)

            cond_outer = xcond_outer & ycond_outer
            block_outer = cloud[cond_outer, :]

            xcond_inner = (block_outer[:, 0] <= x + inner_core) & (
                block_outer[:, 0] >= x
            )
            ycond_inner = (block_outer[:, 1] <= y + inner_core) & (
                block_outer[:, 1] >= y
            )

            cond_inner = xcond_inner & ycond_inner

            conds_inner.append(cond_inner)
            blocks_outer.append(block_outer)
        return conds_inner, blocks_outer

def map2color(self, labels):
    output_colors = list()

    for label in labels:
        output_colors.append(color_map[label])

    return torch.tensor(output_colors)


'''
=== data loading script ===
'''
# loading database files
_data = []
for database_path in data_dir:
    database_path = Path(database_path)
    if not (database_path / f"{mode}_database.yaml").exists():
        print(
            f"generate {database_path}/{mode}_database.yaml first"
        )
        exit()
    _data.extend(
        _load_yaml(database_path / f"{mode}_database.yaml")
    )
if data_percent < 1.0:
    _data = sample(
        _data, int(len(_data) * data_percent)
    )
labels = _load_yaml(Path(label_db_filepath))

LEN_DATA = reps_per_epoch * len(_data)

# if working only on classes for validation - discard others
_labels = _select_correct_labels(labels, num_labels)

if instance_oversampling > 0:
    instance_data = _load_yaml(
        Path(label_db_filepath).parent / "instance_database.yaml"
    )

# normalize color channels
if dataset_name == "s3dis":
    color_mean_std = color_mean_std.replace(
        "color_mean_std.yaml", f"Area_{area}_color_mean_std.yaml"
    )

if Path(str(color_mean_std)).exists():
    print('Loaded color mean and std from .yaml file %s' % color_mean_std)
    color_mean_std = _load_yaml(color_mean_std)
    color_mean, color_std = (
        tuple(color_mean_std["mean"]),
        tuple(color_mean_std["std"]),
    )
elif len(color_mean_std[0]) == 3 and len(color_mean_std[1]) == 3:
    color_mean, color_std = color_mean_std[0], color_mean_std[1]
else:
    assert False, "pass mean and std as tuple of tuples, or as an .yaml file"
    
# augmentations
pass
if add_colors:
    normalize_color = A.Normalize(mean=color_mean, std=color_std)
    
'''
=== load idx-th data frame ===
'''

idx = 0
points = np.load(str(BASE_PATH / _data[idx]["filepath"].replace("../../", ""))) # (N, 12); /processed/scannet/validation/0011_00.npy
coordinates, color, normals, segments, labels = (
    points[:, :3],
    points[:, 3:6],
    points[:, 6:9],
    points[:, 9],
    points[:, 10:12],
)

raw_coordinates = coordinates.copy()
raw_color = color
raw_normals = normals

# normalize color information
pseudo_image = color.astype(np.uint8)[np.newaxis, :, :]
color = np.squeeze(normalize_color(image=pseudo_image)["image"])

# prepare labels and map from 0 to 20(40)
labels = labels.astype(np.int32)
'''
ipdb> np.amax(labels, axis=0), np.amin(labels, axis=0)
    array([255,  32], dtype=int32), array([ 0, -1], dtype=int32)
ipdb> np.amax(segments, axis=0), np.amin(segments, axis=0)
    1610.0, 0.0
'''
if labels.size > 0:
    labels[:, 0] = _remap_from_zero(labels[:, 0])

labels = np.hstack((labels, segments[..., None].astype(np.int32)))

features = color
if add_raw_coordinates:
    if len(features.shape) == 1:
        features = np.hstack((features[None, ...], coordinates))
    else:
        features = np.hstack((features, coordinates))

data_tuple = (
    coordinates, # (N, 3)
    features, # (N, 6)
    labels, # (N, 3), int32
    _data[idx]["raw_filepath"].split("/")[-2], # e.g. 'scene0011_00'
    raw_color, # (N, 3), floar32, [0., 255.]
    raw_normals, # (N, 3), float32, [-1., 1.], normalized
    raw_coordinates, # (N, 3), float32, same as 'coordinates'
    idx,
)

'''
visualization

![](https://i.imgur.com/lVcf3fZ.jpg)
![](https://i.imgur.com/ZPbn1gD.jpg)
'''

import open3d as o3d

o3d_geometry_list = []
points = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(coordinates))
points.colors = o3d.utility.Vector3dVector(raw_color / 255.)

o3d_geometry_list.append(points)
o3d_geometry_list += [o3d.geometry.TriangleMesh.create_coordinate_frame()]

dirs = o3d.geometry.LineSet()
dirs.points = o3d.utility.Vector3dVector(np.vstack((raw_coordinates, raw_coordinates+raw_normals*0.2)))
# dirs.colors = o3d.utility.Vector3dVector([[1., 0., 0.] if vis == 1 else [0.8, 0.8, 0.8] for vis in visibility]) # red: visible; blue: not visible
ray_c = [[0.5, 0.5, 0.5]] * raw_coordinates.shape[0]
dirs.colors = o3d.utility.Vector3dVector(ray_c)
dirs.lines = o3d.utility.Vector2iVector([[_, _+raw_coordinates.shape[0]] for _ in range(raw_coordinates.shape[0])])
o3d_geometry_list.append(dirs)

o3d.visualization.draw_geometries(o3d_geometry_list)

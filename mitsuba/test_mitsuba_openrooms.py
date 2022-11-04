import sys
sys.path.append('../lib')
from pathlib import Path
import numpy as np
import copy

# Import the library using the alias "mi"
import mitsuba as mi
# Set the variant of the renderer
# from lib.global_vars import mi_variant
# mi.set_variant(mi_variant)

import xml.etree.ElementTree as et
from utils_misc import gen_random_str, transformToXml
from utils_io import read_cam_params

'''
    loading an OpenRooms scene and transform to Mitsuba 0.6.0 (not 3.0.0)
    https://mitsuba.readthedocs.io/en/latest/src/key_topics/scene_format.html#scene-xml-file-format
'''

device = 'mbp'
PATH_HOME = {
    'mbp': '/Users/jerrypiglet/Documents/Projects/OpenRooms_RAW_loader', 
    'mm1': '/home/ruizhu/Documents/Projects/OpenRooms_RAW_loader', 
    'qc': '/usr2/rzh/Documents/Projects/directvoxgorui'
}[device]
OR_RAW_ROOT = {
    'mbp': '/Users/jerrypiglet/Documents/Projects/data', 
    'mm1': '/newfoundland2/ruizhu/siggraphasia20dataset', 
    'qc': ''
}[device]

layout_root = Path(OR_RAW_ROOT) / 'layoutMesh'
shapes_root = Path(OR_RAW_ROOT) / 'uv_mapped'
envmaps_root = Path(OR_RAW_ROOT) / 'EnvDataset' # not publicly availale

scene_xml_dir = Path(PATH_HOME) / 'data/openrooms_public_re_2/scenes/xml1/scene0552_00_more'

xml_file = scene_xml_dir / 'mainDiffLight.xml'
tree = et.parse(xml_file)
root = copy.deepcopy(tree.getroot())
root.attrib.pop('verion', None)
# scene = mi.load_file("../scenes/cbox.xml")
root.set('version', '0.6.0')

# remove microfacet bsdfs FOR NOW (incompatible with mitsuba) # [TODO] fix bsdf
# https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_bsdfs.html
for bsdf in root.findall('bsdf'):
    root.remove(bsdf)
for shape in root.findall('shape'):
    bsdf_refs = shape.findall('ref')
    for bsdf_ref in bsdf_refs:
        if bsdf_ref.get('name') == 'bsdf':
            shape.remove(bsdf_ref)     

# fix duplicate shape ids
shape_ids = []
for shape in root.findall('shape'):
    shape_id = shape.get('id')
    if shape_id not in shape_ids:
        shape_ids.append(shape_id)
    else:
        shape_id += gen_random_str(5)
        shape.set('id', shape_id)
    shape_ids.append(shape_id)

    filename = shape.findall('string')[0]
    if 'uv_mapped/' in filename.get('value'):
        obj_path = shapes_root / filename.get('value').split('uv_mapped/')[1]
        assert obj_path.exists(), str(obj_path)
        filename.set('value', str(obj_path))
    if 'layoutMesh' in filename.get('value'):
        obj_path = layout_root / filename.get('value').split('layoutMesh')[1][1:]
        assert obj_path.exists(), str(obj_path)
        filename.set('value', str(obj_path))

for emitter in root.findall('emitter'):
    filename = emitter.findall('string')[0]
    if 'EnvDataset' in filename.get('value'):
        obj_path = envmaps_root / filename.get('value').split('EnvDataset')[1][1:]
        assert obj_path.exists(), str(obj_path)
        filename.set('value', str(obj_path))

# sampler: adaptive -> independent
sensor = root.findall('sensor')[0]
sampler = sensor.findall('sampler')[0]
sampler.set('type', 'independent')

# sensor: set transform as first frame
cam_file = scene_xml_dir / 'cam.txt'
cam_params = read_cam_params(cam_file)
cam_param = cam_params[0]
origin, lookat, up = np.split(cam_param.T, 3, axis=1)
sensor_transform = et.SubElement(
    sensor,
    "transform",
    name="toWorld"
)
et.SubElement(
    sensor_transform,
    "lookat",
    origin=', '.join(['%.4f'%_ for _ in origin.flatten().tolist()]),
    target=', '.join(['%.4f'%_ for _ in lookat.flatten().tolist()]),
    up=', '.join(['%.4f'%_ for _ in up.flatten().tolist()])
)

xmlString = transformToXml(root)
with open('scene.xml', 'w') as xmlOut:
    xmlOut.write(xmlString )

'''
Render.
Should be consistent with OpenRooms renderings in layout:
- mitsuba/my_first_render.png
- /home/ruizhu/Documents/Projects/renderOpenRooms/public_re_3/main_xml1/scene0552_00_more/im_0.png
'''
scene = mi.load_file("scene.xml")
image = mi.render(scene, spp=256)
mi.util.write_bitmap("my_first_render.png", image)
mi.util.write_bitmap("my_first_render.exr", image)
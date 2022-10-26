from dataclasses import replace
import xml.etree.ElementTree as et
import copy
import time
import numpy as np
import random
from pathlib import Path

from lib.utils_OR.utils_OR_xml import transformToXml
from lib.utils_misc import get_datetime, gen_random_str

def replace_str_xml(filename, lookup_dict: dict={}):
    for k, path_root in lookup_dict.items():
        if k in filename.get('value'):
            obj_path = path_root / filename.get('value').split(k)[1]
            assert obj_path.exists(), str(obj_path)
            filename.set('value', str(obj_path))
    return filename

def dump_OR_xml_for_mi(
    xml_file: str, 
    shapes_root: Path, layout_root: Path, envmaps_root: Path, 
    xml_dump_dir: Path, 
    origin_lookatvector_up_tuple: tuple=(), 
    if_no_emitter_shape: bool=False, # remove all window frames and lamps (lit-up & dark)
    if_also_dump_lit_lamps: bool=True) -> Path:

    t = 1000 * time.time() # current time in milliseconds
    np.random.seed(int(t) % 2**32)
    random.seed(int(t) % 2**32)

    xml_dump_path = xml_dump_dir / ('tmp_scene_%s-%s.xml'%(get_datetime(), gen_random_str()))
    for _ in xml_dump_dir.iterdir():
        if _.name.startswith('tmp_scene') and _.name.endswith('.xml'):
            _.unlink()

    tree = et.parse(xml_file)
    root = copy.deepcopy(tree.getroot())
    root.attrib.pop('verion', None)
    # scene = mi.load_file("../scenes/cbox.xml")
    root.set('version', '0.6.0')

    lookup_dict = {
        'uv_mapped/': shapes_root, 
        'layoutMesh/': layout_root, 
        'EnvDataset/': envmaps_root
    }

    # remove microfacet bsdfs FOR NOW (incompatible with mitsuba) # [TODO] fix bsdf
    # https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_bsdfs.html
    for bsdf in root.findall('bsdf'):
        root.remove(bsdf)
    for shape in root.findall('shape'):
        bsdf_refs = shape.findall('ref')
        for bsdf_ref in bsdf_refs:
            if bsdf_ref.get('name') == 'bsdf':
                shape.remove(bsdf_ref)     

    emitters_env = root.findall('emitter')[0]
    root.remove(emitters_env)

    shape_ids = []
    lit_up_lamp_list = []
    for shape in root.findall('shape'):
        # optionally remove lamps and windows
        filename_str = shape.findall('string')[0].get('value')
        if 'window' in filename_str:
            if if_no_emitter_shape:
                root.remove(shape)
                continue
        if ((filename_str.find('ceiling_lamp') != -1 or filename_str.find('03636649') != -1) \
        and ('aligned_light.obj' in filename_str or 'alignedNew.obj' in filename_str)):
            if len(shape.findall('emitter')) != 0:
                assert len(shape.findall('emitter')) == 1
                emitter = shape.findall('emitter')[0]
                intensity = [float(x) for x in emitter.findall('rgb')[0].get('value').split(' ')]
                if max(intensity) > 1e-3:
                    lamp = copy.deepcopy(shape)
                    for emitter in lamp.findall('emitter'):
                        lamp.remove(emitter)
                    replace_str_xml(lamp.findall('string')[0], lookup_dict)
                    if lamp.get('id') in [_.get('id') for _ in lit_up_lamp_list]:
                        lamp.set('id', lamp.get('id')+'_'+gen_random_str())
                    lit_up_lamp_list.append(lamp)
            if if_no_emitter_shape:
                root.remove(shape)
                continue

        # fix duplicate shape ids
        shape_id = shape.get('id')
        if shape_id not in shape_ids:
            shape_ids.append(shape_id)
        else:
            shape_id += gen_random_str(5)
            shape.set('id', shape_id)
        shape_ids.append(shape_id)

        for emitter in shape.findall('emitter'):
            shape.remove(emitter) # removing emitter associated with shapes for now

            # for rgb in emitter.findall('rgb'):
            #     rgb.set('name', 'a random name')

        replace_str_xml(shape.findall('string')[0], lookup_dict)

        # if 'uv_mapped/' in filename.get('value'):
        #     obj_path = shapes_root / filename.get('value').split('uv_mapped/')[1]
        #     assert obj_path.exists(), str(obj_path)
        #     filename.set('value', str(obj_path))
        # if 'layoutMesh' in filename.get('value'):
        #     obj_path = layout_root / filename.get('value').split('layoutMesh')[1][1:]
        #     assert obj_path.exists(), str(obj_path)
        #     filename.set('value', str(obj_path))


    for emitter in root.findall('emitter'):
        replace_str_xml(emitter.findall('string')[0], lookup_dict)
        # if 'EnvDataset' in filename.get('value'):
        #     obj_path = envmaps_root / filename.get('value').split('EnvDataset')[1][1:]
        #     assert obj_path.exists(), str(obj_path)
        #     filename.set('value', str(obj_path))

    # sampler: adaptive -> independent
    sensor = root.findall('sensor')[0]
    sampler = sensor.findall('sampler')[0]
    sampler.set('type', 'independent')

    # sensor: set transform as first frame
    # cam_file = scene_xml_dir / 'cam.txt'
    # cam_params = read_cam_params(cam_file)
    # cam_param = cam_params[0]
    # if cam_param is not None:
    if origin_lookatvector_up_tuple != ():
        # origin, lookat, up = np.split(cam_param.T, 3, axis=1)
        origin, lookatvector, up = origin_lookatvector_up_tuple
        sensor_transform = et.SubElement(
            sensor,
            "transform",
            name="toWorld"
        )
        target = origin + lookatvector
        et.SubElement(
            sensor_transform,
            "lookat",
            origin=', '.join(['%.4f'%_ for _ in origin.flatten().tolist()]),
            target=', '.join(['%.4f'%_ for _ in target.flatten().tolist()]),
            up=', '.join(['%.4f'%_ for _ in up.flatten().tolist()])
        )

    xmlString = transformToXml(root)
    with open(str(xml_dump_path), 'w') as xmlOut:
        xmlOut.write(xmlString )

    if if_also_dump_lit_lamps:
        for shape in root.findall('shape'):
            root.remove(shape)
        for lamp in lit_up_lamp_list:
            root.append(lamp)
        xmlString = transformToXml(root)
        with open(str(xml_dump_path).replace('.xml', '_lit_up_lamps_only.xml'), 'w') as xmlOut:
            xmlOut.write(xmlString )

    return xml_dump_path
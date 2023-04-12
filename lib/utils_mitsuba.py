from dataclasses import replace
import xml.etree.ElementTree as et
import copy
import time
import numpy as np
import random
from pathlib import Path
import mitsuba as mi

from lib.utils_OR.utils_OR_xml import transformToXml, loadMesh, transform_with_transforms_xml_list
from lib.utils_OR.utils_OR_mesh import write_one_mesh_from_v_f_lists, write_mesh_list_from_v_f_lists, flip_ceiling_normal
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
    xml_dump_dir: Path=Path('.'), 
    origin_lookatvector_up_tuple: tuple=(), 
    if_no_emitter_shape: bool=False, # remove all window frames and lamps (lit-up & dark)
    if_dump_mesh: bool=False, 
    dump_mesh_path: str='', 
    dump_mesh_dir: Path=Path('.'), 
    if_also_dump_xml_with_lit_area_lights_only: bool=False) -> Path:

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
    vertices_list, faces_list, ids_list = [], [], []

    for shape in root.findall('shape'):
        filename_str = shape.findall('string')[0].get('value')

        if 'container' in filename_str:
            root.remove(shape)
            continue

        # optionally remove lamps and windows
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
                import ipdb; ipdb.set_trace()
                continue

        replace_str_xml(shape.findall('string')[0], lookup_dict)

        if if_dump_mesh:
            obj_path = shape.findall('string')[0].get('value')
            transforms = shape.findall('transform')
            transforms_list = []
            if len(transforms) != 0:
                transforms = transforms[0]
                assert transforms.get('name') == 'toWorld', transforms.get('name')
                for transform in transforms:
                    transform_name = transform.tag
                    assert transform_name in ['scale', 'rotate', 'translate']
                    transform_dict = {transform_name: {key: float(
                        transform.get(key)) for key in transform.keys()}}
                    transforms_list.append(transform_dict)
            vertices, faces = loadMesh(obj_path) # based on L430 of adjustObjectPoseCorrectChairs.py
            if '/uv_mapped.obj' in obj_path:
                faces = flip_ceiling_normal(faces, vertices)
            vertices_transformed, _ = transform_with_transforms_xml_list(transforms_list, vertices)
            vertices_list.append(vertices_transformed)
            faces_list.append(faces)
            ids_list.append(shape.get('id'))

        # fix duplicate shape ids
        shape_id = shape.get('id')
        if shape_id not in shape_ids:
            shape_ids.append(shape_id)
        else:
            shape_id += gen_random_str(5)
            shape.set('id', shape_id)
        shape_ids.append(shape_id)

        # for emitter in shape.findall('emitter'):
        #     shape.remove(emitter) # removing emitter associated with shapes for now

        # for compatibility with Mitsuba
        for emitter in shape.findall('emitter'):
            rgbs = emitter.findall('rgb')
            for rgb in rgbs:
                rgb.set('name', 'radiance')

    if if_dump_mesh:
        write_one_mesh_from_v_f_lists(dump_mesh_path, vertices_list, faces_list, ids_list)
        write_mesh_list_from_v_f_lists(dump_mesh_dir, vertices_list, faces_list, ids_list)
 
    for emitter in root.findall('emitter'):
        replace_str_xml(emitter.findall('string')[0], lookup_dict)

    # sampler: adaptive -> independent
    sensor = root.findall('sensor')[0]
    sampler = sensor.findall('sampler')[0]
    sampler.set('type', 'independent')

    # sensor: set transform as first frame
    # cam_file = scene_xml_dir / 'cam.txt'
    # cam_params = read_cam_params_OR(cam_file)
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

    # if if_also_dump_xml_with_lit_area_lights_only:
    #     for shape in root.findall('shape'):
    #         root.remove(shape)
    #     for lamp in lit_up_lamp_list:
    #         root.append(lamp)
    #     xmlString = transformToXml(root)
    #     with open(str(xml_dump_path).replace('.xml', '_lit_up_area_lights_only.xml'), 'w') as xmlOut:
    #         xmlOut.write(xmlString )

    return xml_dump_path

def dump_Indoor_area_lights_only_xml_for_mi(
    xml_file: str, 
):
    '''
        Dump xml_with_lit_area_lights_only to *_lit_up_area_lights_only.xml
        work with the Indoor dataset (e.g. kitchen scene)
    '''
    xml_dump_path = str(xml_file).replace('.xml', '_lit_up_area_lights_only.xml')
    tree = et.parse(xml_file)
    root = copy.deepcopy(tree.getroot())
    shapes = root.findall('shape')
    for shape in shapes:
        if_has_lit_up_area_light = False
        emitters = shape.findall('emitter')
        if len(emitters) > 0:
            assert len(emitters) == 1
            emitter = emitters[0]
            assert emitter.get('type') == 'area'
            rgb = emitter.findall('rgb')[0]
            assert rgb.get('name') == 'radiance'
            radiance = np.array(rgb.get('value').split(', ')).astype(np.float32).reshape(3,)
            if np.amax(radiance) > 1e-3:
                if_has_lit_up_area_light = True

        if not if_has_lit_up_area_light:
            root.remove(shape)

    xmlString = transformToXml(root)
    with open(xml_dump_path, 'w') as xmlOut:
        xmlOut.write(xmlString )
    
    return xml_dump_path

def get_rad_meter_sensor(origin, direction, spp):
    '''
    https://mitsuba.readthedocs.io/en/stable/src/generated/plugins_sensors.html#radiance-meter-radiancemeter
    '''
    return mi.load_dict({
        'type': 'radiancemeter',
        'origin': origin.flatten(),
        'direction': direction.flatten(),
        'sampler': {
            'type': 'independent',
            'sample_count': spp
        }, 
        'film': {
            'type': 'hdrfilm',
            'width': 1,
            'height': 1,
            'pixel_format': 'rgb',
            'rfilter': {
                'type': 'box'
            }
        },
    })

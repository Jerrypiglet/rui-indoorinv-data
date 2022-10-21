import xml.etree.ElementTree as et
import copy
from pathlib import Path
from lib.utils_OR.utils_OR_xml import gen_random_str, transformToXml

def dump_OR_xml_for_mi(xml_file: str, shapes_root: Path, layout_root: Path, envmaps_root: Path, xml_out_path: str, origin_lookat_up_tuple: tuple=(), if_no_emitter_shape: bool=True):

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

    shape_ids = []
    for shape in root.findall('shape'):
        # remove lamps and windows
        if if_no_emitter_shape:
            filename_str = shape.findall('string')[0].get('value')
            if 'window' in filename_str \
                or ((filename_str.find('ceiling_lamp') != -1 or filename_str.find('03636649') != -1) \
                and ('aligned_light.obj' in filename_str or 'alignedNew.obj' in filename_str)):
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
    # cam_file = scene_xml_dir / 'cam.txt'
    # cam_params = read_cam_params(cam_file)
    # cam_param = cam_params[0]
    # if cam_param is not None:
    if origin_lookat_up_tuple != ():
        # origin, lookat, up = np.split(cam_param.T, 3, axis=1)
        origin, lookat, up = origin_lookat_up_tuple
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
    with open(xml_out_path, 'w') as xmlOut:
        xmlOut.write(xmlString )

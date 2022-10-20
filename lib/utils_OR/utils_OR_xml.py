import sys
from tqdm import tqdm
import xml.etree.ElementTree as et
import numpy as np
from pathlib import Path
import copy
import random
import string
import xml.etree.ElementTree as et
from xml.dom import minidom

import xml.etree.ElementTree as et
from xml.dom import minidom

from icecream import ic

from .utils_OR_mesh import loadMesh, computeBox, computeTransform, writeMesh
from .utils_OR_transform import transform_with_transforms_xml_list


def get_XML_root(main_xml_file):
    # L202 of sampleCameraPoseFromScanNet.py
    tree = et.parse(str(main_xml_file))
    root = tree.getroot()
    return root

def transformToXml(root):
    rstring = et.tostring(root, 'utf-8')
    pstring = minidom.parseString(rstring)
    xmlString = pstring.toprettyxml(indent="    ")
    xmlString= xmlString.split('\n')
    xmlString = [x for x in xmlString if len(x.strip()) != 0 ]
    xmlString = '\n'.join(xmlString)
    return xmlString

def parse_XML_for_intrinsics(root):
    sensors = root.findall('sensor')
    assert len(sensors) == 1
    sensor = sensors[0]

    film = sensor.findall('film')[0]
    integers = film.findall('integer')
    for integer in integers:
        if integer.get('name') == 'width':
            width = int(integer.get('value'))
        if integer.get('name') == 'height':
            height = int(integer.get('value'))
    fov_entry = sensor.findall('float')[0]
    assert fov_entry.get('name') == 'fov'
    fov = float(fov_entry.get('value'))
    f_px = width / 2. / np.tan(fov / 180. * np.pi / 2.)
    cam_K = np.array(
        [[-f_px, 0., width/2.], [0., -f_px, height/2.], [0., 0., 1.]])
    return cam_K, {'fov': fov, 'f_px': f_px, 'width': width, 'height': height}


def box3D_center_axis_len_to_coords(box3D):
    center, xAxis, yAxis, zAxis, xLen, yLen, zLen = box3D['center'], box3D[
        'xAxis'], box3D['yAxis'], box3D['zAxis'], box3D['xLen'], box3D['yLen'], box3D['zLen']
    return np.vstack([
        center - 0.5 * xAxis * xLen - 0.5 * yAxis * yLen - 0.5 * zAxis * zLen,
        center + 0.5 * xAxis * xLen - 0.5 * yAxis * yLen - 0.5 * zAxis * zLen,
        center + 0.5 * xAxis * xLen - 0.5 * yAxis * yLen + 0.5 * zAxis * zLen,
        center - 0.5 * xAxis * xLen - 0.5 * yAxis * yLen + 0.5 * zAxis * zLen,
        center - 0.5 * xAxis * xLen + 0.5 * yAxis * yLen - 0.5 * zAxis * zLen,
        center + 0.5 * xAxis * xLen + 0.5 * yAxis * yLen - 0.5 * zAxis * zLen,
        center + 0.5 * xAxis * xLen + 0.5 * yAxis * yLen + 0.5 * zAxis * zLen,
        center - 0.5 * xAxis * xLen + 0.5 * yAxis * yLen + 0.5 * zAxis * zLen,
    ])


def _parse_XML_for_shapes(root, root_uv_mapped, root_layoutMesh, if_return_emitters=False, light_dat_lists=None, random_seed=None, scene_name_frame_id=None, if_not_write_representation_specific_info=False, main_xml_file=''):
    assert False, 'obsolete; only useful for loading per-frame properties (e.g. cam coords)'
    # For windows only returns when outside is not dark
    # ['emitter_prop']['if_env'] == True when is the env map outside, otherwise can be windows or lamps
    # ['emitter_prop']['obj_type'] in ['window', 'obj'], where 'obj' is basically lamps
    # shape_dict['if_in_emitter_dict'] == True if in emitter_dict else in shape_dict
    if scene_name_frame_id is None:
        scene_name_frame_id_str = ''
    else:
        scene_name_frame_id_str = scene_name_frame_id[0] + '_' + scene_name_frame_id[1] + '_' + str(scene_name_frame_id[2])
    shapes = root.findall('shape')
    shapes_list = []
    emitters_list = []
    max_envmap_scale = -np.inf
    if if_return_emitters:
        assert light_dat_lists is not None
        # get envmap emitter(s), as the 1st emitter
        emitters_env = root.findall('emitter')
        assert len(emitters_env) == 1
        emitter_env = emitters_env[0]
        assert emitter_env.get('type') == 'envmap'
        emitter_dict = {'if_emitter': True}
        emitter_dict['emitter_prop'] = {'emitter_type': 'envmap'}
        emitter_dict['emitter_prop']['if_env'] = True
        assert len(emitter_env.findall('string')) == 1
        assert emitter_env.findall('string')[0].get('name') == 'filename'
        emitter_dict['emitter_prop']['emitter_filename'] = emitter_env.findall('string')[
            0].get('value')
        emitter_env.findall('float')[0].get('name') == 'scale'
        emitter_dict['emitter_prop']['emitter_scale'] = float(
            emitter_env.findall('float')[0].get('value'))
        max_envmap_scale = max(
            max_envmap_scale, emitter_dict['emitter_prop']['emitter_scale'])
        emitters_list.append(emitter_dict)
        if_bright_outside = max_envmap_scale > 1e-3

    if random_seed is not None:
        random.seed(random_seed)

    for shape in shapes:
        shape_dict = {'id': shape.get('id'), 'random_id': ''.join(
            random.choices(string.ascii_uppercase + string.digits, k=5))}

        shape_dict[shape.findall('string')[0].get('name')] = shape.findall('string')[0].get('value')

        # assert len(shape.findall('transform')) == 1, 'Shape of id %s has not transform!'%shape_dict['id']
        transforms = shape.findall('transform')
        if len(transforms) == 0:
            shape_dict['transforms_list'] = None
        else:
            transforms = transforms[0]
            assert transforms.get('name') == 'toWorld', transforms.get('name')
            transforms_list = []
            for transform in transforms:
                transform_name = transform.tag
                assert transform_name in ['scale', 'rotate', 'translate']
                transform_dict = {transform_name: {key: float(
                    transform.get(key)) for key in transform.keys()}}
                transforms_list.append(transform_dict)
            shape_dict['transforms_list'] = transforms_list
        shape_dict['if_correct_path'] = False

        # find emitter property: get objs emitter(s)
        emitters = shape.findall('emitter')
        shape_dict['emitter_prop'] = {
            'max_envmap_scale': max_envmap_scale, 'if_lit_up': False}
        shape_dict['if_emitter'] = False
        shape_dict['if_in_emitter_dict'] = False
        light_dat = None

        # print('--', shape_dict['filename'])
        if 'uv_mapped/' in shape_dict['filename']:
            shape_dict['filename'] = str(Path(root_uv_mapped) / shape_dict['filename'].split('uv_mapped/')[1])
        elif 'layoutMesh/' in shape_dict['filename']:
            shape_dict['filename'] = str(Path(root_layoutMesh) / shape_dict['filename'].split('layoutMesh/')[1])
        else:
            assert False, shape_dict['filename']
        # print('---->', shape_dict['filename'], len(emitters))

        if 'window' in shape_dict['filename']:
            if if_bright_outside:  # window that has light shining through
                shape_dict['if_emitter'] = True
                shape_dict['emitter_prop']['if_env'] = False
                shape_dict['emitter_prop']['obj_type'] = 'window'
                shape_dict['emitter_prop']['combined_filename'] = str(
                    root_uv_mapped / shape_dict['filename'].replace('../../../../../uv_mapped/', ''))
                if light_dat_lists['windows']:
                    _ = light_dat_lists['windows'].pop(0)
                    if len(_) == 3:
                        light_path, light_dat, light_file_obj_name = _
                        assert light_file_obj_name == shape.get('id')
                    elif len(_) == 2:
                        light_path, light_dat = _
                    else:
                        light_dat = _
                        light_path = ''

                    # shape_dict['emitter_prop']['if_lit_up'] = max([float(x) for x in light_dat['intensity']]) > 1e-3
                    shape_dict['emitter_prop']['if_lit_up'] = light_dat['envScale'] > 1e-3

                    if not if_not_write_representation_specific_info:
                        save_prefix = light_dat['N_ambient_rep']

                        shape_dict['emitter_prop']['light_path'] = light_path
                        shape_dict['emitter_prop'].update(light_dat)

                        label_SG_list = ['']
                        assert save_prefix in [
                            '3SG-SkyGrd'], '20211014: Does not work for other representation any more; to make it work make sure to write envScale-independent intensity for windows to light dat files, and bring back the true scale here!'
                        if save_prefix in ['3SG-SkyGrd']:
                            label_SG_list = ['', 'Sky', 'Grd']
                        for label_SG in label_SG_list:
                            shape_dict['emitter_prop']['intensity'+label_SG] = [
                                float(x) for x in light_dat['intensity'+label_SG]]
                            # the window is light up (bright outside)
                            if shape_dict['emitter_prop']['if_lit_up']:
                                # print('axis_RAW_cam' in light_dat, light_path, light_dat.keys(), light_dat['isWindow'], max([float(x) for x in light_dat['intensity']]) > 1e-3)
                                # print('--------', light_path, light_dat.keys(), light_dat['axis_world'])
                                # shape_dict['emitter_prop']['axis_RAW_cam'+label_SG] = [
                                #     float(x) for x in light_dat['axis_RAW_cam'+label_SG]]
                                shape_dict['emitter_prop']['axis%s_world'%label_SG] = [
                                    float(x) for x in light_dat['axis%s_world'%label_SG]]

                        # assert np.sum(shape_dict['emitter_prop']['box3D_world']['zAxis']
                                    #   * shape_dict['emitter_prop']['box3D_world']['center']) <= 0
                        shape_dict['emitter_prop']['box3D_world'] = light_dat['box3D_world']
                        shape_dict['emitter_prop']['box3D_world']['coords'] = box3D_center_axis_len_to_coords(
                            light_dat['box3D_world'])

                    if 'box2D' in light_dat:
                        shape_dict['emitter_prop']['box2D'] = light_dat['box2D']
            emitter_dict = copy.deepcopy(shape_dict)
            emitter_dict['if_in_emitter_dict'] = True
            emitters_list.append(emitter_dict)

        elif (shape_dict['filename'].find('ceiling_lamp') != -1 or shape_dict['filename'].find('03636649') != -1) \
                and ('aligned_light.obj' in shape_dict['filename'] or 'alignedNew.obj' in shape_dict['filename']):
            # lamps, ceilimg_lamps
            # they are written as two objs in XML: aligned_shape.obj, aligned_light.obj; only record the latter, and then get bbox that encapsulates both aligned_shape.obj, aligned_light.obj in the postprocessing below
            shape_dict['if_emitter'] = True
            shape_dict['emitter_prop']['if_env'] = False
            shape_dict['emitter_prop']['obj_type'] = 'obj'
            # assert len(emitters) == 1, scene_name_frame_id_str
            if len(emitters) == 1:
                emitter = emitters[0]
                shape_dict['emitter_prop']['emitter_type'] = emitter.get('type')
                assert len(emitter.findall('rgb')) == 1, scene_name_frame_id_str
                shape_dict['emitter_prop']['intensity'] = [
                    float(x) for x in emitter.findall('rgb')[0].get('value').split(' ')]
            else:
                shape_dict['emitter_prop']['intensity'] = [0., 0., 0.]
            shape_dict['emitter_prop']['if_lit_up'] = max(shape_dict['emitter_prop']['intensity']) > 1e-3

            if light_dat_lists is not None and light_dat_lists['objs']:
                # print('----->', light_dat_lists['objs'].pop(0))
                _ = light_dat_lists['objs'].pop(0)
                if len(_) == 3:
                    light_path, light_dat, light_file_obj_name = _
                    assert light_file_obj_name == shape.get('id')
                elif len(_) == 2:
                    light_path, light_dat = _
                else:
                    light_dat = _
                    light_path = ''

                error = [abs(x-y) for x, y in zip(shape_dict['emitter_prop']['intensity'], light_dat['intensity'])]
                if max(error) >= 1e-3:
                    print('------', shape_dict['emitter_prop']['intensity'], light_dat['intensity'], shape_dict['filename'], shape.get('id'))
                    print('----', shape_dict)
                    print('--', light_dat, light_path)
                    print('--', main_xml_file)
                assert max(error) < 1e-3, scene_name_frame_id_str
                if not if_not_write_representation_specific_info:
                    shape_dict['emitter_prop']['light_path'] = light_path
                    shape_dict['emitter_prop'].update(light_dat)
                    # shape_dict['emitter_prop']['axis'] = [float(x) for x in light_dat['axis']]
                if 'box3D_world' in light_dat:
                    shape_dict['emitter_prop']['box3D_world'] = light_dat['box3D_world']
                    # shape_dict['emitter_prop']['box3D_world'] = {}
                    shape_dict['emitter_prop']['box3D_world']['coords'] = box3D_center_axis_len_to_coords(
                        light_dat['box3D_world'])
                else:
                    shape_dict['emitter_prop']['box3D_world'] = {}
                    shape_dict['emitter_prop']['box3D_world']['coords'] = None
                if 'box2D' in light_dat:
                    shape_dict['emitter_prop']['box2D'] = light_dat['box2D']

            emitter_dict = copy.deepcopy(shape_dict)
            emitter_dict['if_in_emitter_dict'] = True
            emitters_list.append(emitter_dict)

        else:
            shape_dict['if_emitter'] = False

        shapes_list.append(shape_dict)

    if light_dat_lists is not None:
        assert len(light_dat_lists['windows']) == 0, scene_name_frame_id_str
        assert len(light_dat_lists['objs']) == 0, scene_name_frame_id_str

    # ===== Post-process to merge lamp light and lamp base
    for idx, emitter_dict in enumerate(emitters_list[1:]):
        # print(shape_dict['id'], shape_dict['filename'])
        # if emitter_dict['emitter_prop']['if_env']:
        #     continue
        # emitter_filename_abs = root_uv_mapped / \
        #     emitter_dict['filename'].replace('../../../../../uv_mapped/', '')
        emitter_filename_abs = str(Path(root_uv_mapped) / emitter_dict['filename'].split('uv_mapped/')[1])

        # already using merged/single CAD model
        if 'aligned_light.obj' not in emitter_dict['filename']:
            emitter_dict['emitter_prop']['combined_filename'] = str(
                emitter_filename_abs)
            emitter_dict['emitter_prop']['emitter_filename'] = str(
                emitter_filename_abs)
            emitter_dict['emitter_prop']['emitter_part_random_id'] = emitter_dict['random_id']
            continue

        if_found_emitter, if_found_base = False, False
        for shape_dict in shapes_list:  # look for the light ``base`` obj
            if emitter_dict['filename'].replace('aligned_light.obj', 'aligned_shape.obj') == shape_dict['filename']:
                if_found_base = True
                combined_filename_abs = Path(str(emitter_filename_abs).replace(
                    'aligned_light.obj', 'alignedNew.obj'))
                if not combined_filename_abs.exists():
                    other_part_filename_abs = Path(str(emitter_filename_abs).replace(
                        'aligned_light.obj', 'aligned_shape.obj'))
                    assert other_part_filename_abs.exists()
                    vertices_0, faces_0 = loadMesh(str(emitter_filename_abs))
                    vertices_1, faces_1 = loadMesh(
                        str(other_part_filename_abs))
                    vertices_combine = np.vstack([vertices_0, vertices_1])
                    faces_combine = np.vstack(
                        [faces_0, faces_1+vertices_0.shape[0]])
                    writeMesh(str(combined_filename_abs),
                              vertices_combine, faces_combine)
                    print('NEW mesh written to %s' %
                          str(combined_filename_abs))
                emitter_dict['emitter_prop']['combined_filename'] = str(
                    combined_filename_abs)
            if emitter_dict['filename'] == shape_dict['filename']:
                if_found_emitter = True
                emitter_dict['emitter_prop']['emitter_filename'] = str(
                    emitter_filename_abs)
                emitter_dict['emitter_prop']['emitter_part_random_id'] = shape_dict['random_id']
                emitter_dict['emitter_prop']['emitter_part_obj_dict'] = shape_dict
        assert if_found_emitter and if_found_base, 'Not both base and emitter are found!'

    if if_return_emitters:
        return shapes_list, emitters_list
    else:
        return shapes_list

def parse_XML_for_shapes_global(root, root_uv_mapped, root_layoutMesh, scene_xml_path, root_EnvDataset, if_return_emitters=False, light_dat_lists=None, random_seed=0, scene_name_frame_id=None, if_not_write_representation_specific_info=False, main_xml_file=''):
    '''
    load global info ONLY
    '''
    # For windows only returns when outside is not dark
    # ['emitter_prop']['if_env'] == True when is the env map outside, otherwise can be windows or lamps
    # ['emitter_prop']['obj_type'] in ['window', 'obj'], where 'obj' is basically lamps
    # shape_dict['if_in_emitter_dict'] == True if in emitter_dict else in shape_dict

    if scene_name_frame_id is None: # including scene/frame info; for degging purposes
        scene_name_frame_id_str = ''
    else:
        scene_name_frame_id_str = scene_name_frame_id[0] + '_' + scene_name_frame_id[1] + '_' + str(scene_name_frame_id[2])

    shapes = root.findall('shape')
    shapes_list = []
    emitters_list = []
    max_envmap_scale = -np.inf
    if if_return_emitters:
        assert light_dat_lists is not None
        # get envmap emitter(s), as the 1st emitter
        emitters_env = root.findall('emitter')
        assert len(emitters_env) == 1
        emitter_env = emitters_env[0]
        assert emitter_env.get('type') == 'envmap'
        emitter_dict = {'if_emitter': True}
        emitter_dict['emitter_prop'] = {'emitter_type': 'envmap'}
        emitter_dict['emitter_prop']['if_env'] = True
        assert len(emitter_env.findall('string')) == 1
        assert emitter_env.findall('string')[0].get('name') == 'filename'
        emitter_dict['emitter_prop']['emitter_filename'] = emitter_env.findall('string')[
            0].get('value')
        emitter_dict['emitter_prop']['emitter_filename'] = str(Path(root_EnvDataset) / Path(emitter_dict['emitter_prop']['emitter_filename']).name)

        emitter_env.findall('float')[0].get('name') == 'scale'
        emitter_dict['emitter_prop']['emitter_scale'] = float(
            emitter_env.findall('float')[0].get('value'))
        max_envmap_scale = max(
            max_envmap_scale, emitter_dict['emitter_prop']['emitter_scale'])

        emitters_list.append(emitter_dict)
        if_bright_outside = max_envmap_scale > 1e-3

    if random_seed is not None:
        random.seed(random_seed)

    for shape in shapes:
        shape_dict = {'id': shape.get('id'), 'random_id': ''.join(
            random.choices(string.ascii_uppercase + string.digits, k=5))}

        shape_dict[shape.findall('string')[0].get('name')] = shape.findall('string')[0].get('value')

        # assert len(shape.findall('transform')) == 1, 'Shape of id %s has not transform!'%shape_dict['id']
        transforms = shape.findall('transform')
        if len(transforms) == 0:
            shape_dict['transforms_list'] = None
        else:
            transforms = transforms[0]
            assert transforms.get('name') == 'toWorld', transforms.get('name')
            transforms_list = []
            for transform in transforms:
                transform_name = transform.tag
                assert transform_name in ['scale', 'rotate', 'translate']
                transform_dict = {transform_name: {key: float(
                    transform.get(key)) for key in transform.keys()}}
                transforms_list.append(transform_dict)
            shape_dict['transforms_list'] = transforms_list
        shape_dict['if_correct_path'] = False

        # find emitter property: get objs emitter(s)
        emitters = shape.findall('emitter')
        shape_dict['emitter_prop'] = {
            'max_envmap_scale': max_envmap_scale, 'if_lit_up': False}
        shape_dict['if_emitter'] = False
        shape_dict['if_in_emitter_dict'] = False

        if 'uv_mapped/' in shape_dict['filename']:
            shape_dict['filename'] = str(Path(root_uv_mapped) / shape_dict['filename'].split('uv_mapped/')[1])
        elif 'layoutMesh/' in shape_dict['filename']:
            shape_dict['filename'] = str(Path(root_layoutMesh) / shape_dict['filename'].split('layoutMesh/')[1])
        else:
            assert False, shape_dict['filename']

        if 'window' in shape_dict['filename']:
            if if_bright_outside:  # window that has light shining through
                shape_dict['if_emitter'] = True
                shape_dict['emitter_prop']['if_env'] = False
                shape_dict['emitter_prop']['obj_type'] = 'window'
                shape_dict['emitter_prop']['combined_filename'] = str(Path(root_uv_mapped) / shape_dict['filename'].split('uv_mapped/')[1])

                if light_dat_lists['windows']:
                    light_path, light_dat = light_dat_lists['windows'].pop(0) #  ['shapeId', 'intensity', 'envScale', 'lamb', 'axis_world', 'intensitySky', 'lambSky', 'axisSky_world', 'intensityGrd', 'lambGrd', 'axisGrd_world', 'theta_local', 'phi_local', 'envAxis_x_world', 'envAxis_y_world', 'envAxis_z_world', 'lightXmlFile', 'envMapPath', 'recHalfEnvName', 'imHalfEnvName', 'isWindow', 'box3D_world', 'N_ambient_rep']

                    # shape_dict['emitter_prop']['if_lit_up'] = max([float(x) for x in light_dat['intensity']]) > 1e-3
                    shape_dict['emitter_prop']['if_lit_up'] = light_dat['envScale'] > 1e-3

                    if not if_not_write_representation_specific_info:
                        save_prefix = light_dat['N_ambient_rep']

                        shape_dict['emitter_prop']['light_path'] = light_path
                        shape_dict['emitter_prop'].update(light_dat)
                        for envmap_key in ['envMapPath']:
                            shape_dict['emitter_prop'][envmap_key] = str(Path(root_EnvDataset) / Path(shape_dict['emitter_prop'][envmap_key]).name)
                        for envmap_key in ['recHalfEnvName', 'imHalfEnvName', 'lightXmlFile']:
                            shape_dict['emitter_prop'][envmap_key] = str(Path(scene_xml_path) / Path(shape_dict['emitter_prop'][envmap_key]).name)

                        label_SG_list = ['']
                        assert save_prefix in [
                            '3SG-SkyGrd'], '20211014: Does not work for other representation any more; to make it work make sure to write envScale-independent intensity for windows to light dat files, and bring back the true scale here!'
                        if save_prefix in ['3SG-SkyGrd']:
                            label_SG_list = ['', 'Sky', 'Grd']
                        for label_SG in label_SG_list:
                            shape_dict['emitter_prop']['intensity'+label_SG] = [
                                float(x) for x in light_dat['intensity'+label_SG]]
                            # the window is light up (bright outside)
                            if shape_dict['emitter_prop']['if_lit_up']:
                                shape_dict['emitter_prop']['axis%s_world'%label_SG] = [
                                    float(x) for x in light_dat['axis%s_world'%label_SG]]

                        shape_dict['emitter_prop']['box3D_world'] = light_dat['box3D_world']
                        shape_dict['emitter_prop']['box3D_world']['coords'] = box3D_center_axis_len_to_coords(
                            light_dat['box3D_world'])

            emitter_dict = copy.deepcopy(shape_dict)
            emitter_dict['if_in_emitter_dict'] = True
            emitters_list.append(emitter_dict)

        elif (shape_dict['filename'].find('ceiling_lamp') != -1 or shape_dict['filename'].find('03636649') != -1) \
                and ('aligned_light.obj' in shape_dict['filename'] or 'alignedNew.obj' in shape_dict['filename']):
            # lamps, ceilimg_lamps
            # they are written as two objs in XML: aligned_shape.obj, aligned_light.obj; only record the latter, and then get bbox that encapsulates both aligned_shape.obj, aligned_light.obj in the postprocessing below
            shape_dict['if_emitter'] = True
            shape_dict['emitter_prop']['if_env'] = False
            shape_dict['emitter_prop']['obj_type'] = 'obj'
            # assert len(emitters) == 1, scene_name_frame_id_str
            if len(emitters) == 1:
                emitter = emitters[0]
                shape_dict['emitter_prop']['emitter_type'] = emitter.get('type')
                assert len(emitter.findall('rgb')) == 1, scene_name_frame_id_str
                shape_dict['emitter_prop']['intensity'] = [
                    float(x) for x in emitter.findall('rgb')[0].get('value').split(' ')]
            else:
                shape_dict['emitter_prop']['intensity'] = [0., 0., 0.]
            shape_dict['emitter_prop']['if_lit_up'] = max(shape_dict['emitter_prop']['intensity']) > 1e-3

            if light_dat_lists is not None and light_dat_lists['objs']:
                # print('----->', light_dat_lists['objs'].pop(0))
                light_path, light_dat = light_dat_lists['objs'].pop(0)

                error = [abs(x-y) for x, y in zip(shape_dict['emitter_prop']['intensity'], light_dat['intensity'])]
                if max(error) >= 1e-3:
                    print('------', shape_dict['emitter_prop']['intensity'], light_dat['intensity'], shape_dict['filename'], shape.get('id'))
                    print('----', shape_dict)
                    print('--', light_dat, light_path)
                    print('--', main_xml_file)
                assert max(error) < 1e-3, scene_name_frame_id_str
                if not if_not_write_representation_specific_info:
                    shape_dict['emitter_prop']['light_path'] = light_path
                    shape_dict['emitter_prop'].update(light_dat)
                if 'box3D_world' in light_dat:
                    shape_dict['emitter_prop']['box3D_world'] = light_dat['box3D_world']
                    shape_dict['emitter_prop']['box3D_world']['coords'] = box3D_center_axis_len_to_coords(
                        light_dat['box3D_world'])
                else:
                    shape_dict['emitter_prop']['box3D_world'] = {}
                    shape_dict['emitter_prop']['box3D_world']['coords'] = None

            emitter_dict = copy.deepcopy(shape_dict)
            emitter_dict['if_in_emitter_dict'] = True
            emitters_list.append(emitter_dict)

        else:
            shape_dict['if_emitter'] = False

        shapes_list.append(shape_dict)

    if light_dat_lists is not None:
        assert len(light_dat_lists['windows']) == 0, scene_name_frame_id_str
        assert len(light_dat_lists['objs']) == 0, scene_name_frame_id_str

    # ===== Post-process to merge lamp light and lamp base into a new mesh --write to-> emitter_dict['emitter_prop']['combined_filename']
    for idx, emitter_dict in enumerate(emitters_list[1:]):
        emitter_filename_abs = str(Path(root_uv_mapped) / emitter_dict['filename'].split('uv_mapped/')[1])

        # already using merged/single CAD model
        if 'aligned_light.obj' not in emitter_dict['filename']:
            emitter_dict['emitter_prop']['combined_filename'] = str(emitter_filename_abs)
            emitter_dict['emitter_prop']['emitter_filename'] = str(emitter_filename_abs)
            emitter_dict['emitter_prop']['emitter_part_random_id'] = emitter_dict['random_id']
            continue

        if_found_emitter, if_found_base = False, False
        for shape_dict in shapes_list:  # look for the light ``base`` obj
            if emitter_dict['filename'].replace('aligned_light.obj', 'aligned_shape.obj') == shape_dict['filename']:
                if_found_base = True
                combined_filename_abs = Path(str(emitter_filename_abs).replace(
                    'aligned_light.obj', 'alignedNew.obj'))
                if not combined_filename_abs.exists():
                    other_part_filename_abs = Path(str(emitter_filename_abs).replace(
                        'aligned_light.obj', 'aligned_shape.obj'))
                    assert other_part_filename_abs.exists()
                    vertices_0, faces_0 = loadMesh(str(emitter_filename_abs))
                    vertices_1, faces_1 = loadMesh(
                        str(other_part_filename_abs))
                    vertices_combine = np.vstack([vertices_0, vertices_1])
                    faces_combine = np.vstack(
                        [faces_0, faces_1+vertices_0.shape[0]])
                    writeMesh(str(combined_filename_abs),
                              vertices_combine, faces_combine)
                    print('NEW mesh written to %s' %
                          str(combined_filename_abs))
                emitter_dict['emitter_prop']['combined_filename'] = str(combined_filename_abs)
            if emitter_dict['filename'] == shape_dict['filename']:
                if_found_emitter = True
                emitter_dict['emitter_prop']['emitter_filename'] = str(emitter_filename_abs)
                emitter_dict['emitter_prop']['emitter_part_random_id'] = shape_dict['random_id']
                emitter_dict['emitter_prop']['emitter_part_obj_dict'] = shape_dict
        assert if_found_emitter and if_found_base, 'Not both base and emitter are found!'

    if if_return_emitters:
        return shapes_list, emitters_list
    else:
        return shapes_list

def gen_random_str(length=5):
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))

def transformToXml(root):
    rstring = et.tostring(root, 'utf-8')
    pstring = minidom.parseString(rstring)
    xmlString = pstring.toprettyxml(indent="    ")
    xmlString = xmlString.split('\n')
    xmlString = [x for x in xmlString if len(x.strip()) != 0]
    xmlString = '\n'.join(xmlString)
    return xmlString


# sys.path.insert(0, '/home/ruizhu/Documents/Projects/Total3DUnderstanding')
# root_uv_mapped = Path('/newfoundland2/ruizhu/siggraphasia20dataset/uv_mapped')
# root_layoutMesh = Path(
#     '/newfoundland2/ruizhu/siggraphasia20dataset/layoutMesh')


def xml_scene_to_mesh(xml_file, output_path='', skip_uv_mapped=False, skipped_walls=False, base_path=None):
    root = get_XML_root(xml_file)
    shape_list, emitters_list = parse_XML_for_shapes(
        root, root_uv_mapped, if_return_emitters=True)
    vertices_list = []
    faces_list = []
    num_vertices = 0

    for shape_idx, shape in tqdm(enumerate(shape_list)):
        if 'container' in shape['filename']:
            continue

        if skip_uv_mapped and ('layoutMesh' in shape['filename']):
            print('Skipped '+shape['filename'])
            continue

        if skipped_walls and ('mesh_wall' in shape['filename'] or 'mesh_patch' in shape['filename']):
            print('Skipped '+shape['filename'])
            continue

        if_emitter = shape['if_in_emitter_dict']
        if if_emitter:
            obj_path = shape['emitter_prop']['emitter_filename']
        else:
            filename = shape['filename']
            obj_path = filename
            if 'uv_mapped' in shape['filename']:
                obj_path = root_uv_mapped / \
                    filename.replace('../../../../../uv_mapped/', '')
            if 'layoutMesh' in shape['filename']:
                obj_path = root_layoutMesh / \
                    filename.replace('../../../../../layoutMesh/', '')
            if base_path is not None:
                obj_path = str(Path(base_path) / obj_path)
                # print(base_path, obj_path, type(obj_path))
                print(obj_path)
                obj_path = str(
                    Path(base_path) / (str(obj_path).replace('bunny.ply', 'bunny.obj')))
                # obj_path = str(Path(base_path) / obj_path.replace('bunny.ply', 'sphere.obj'))

        # print(shape_idx, if_emitter, shape)

        # print(obj_path)
        # based on L430 of adjustObjectPoseCorrectChairs.py
        vertices, faces = loadMesh(obj_path)

        if shape['transforms_list'] is not None:
            vertices_transformed, _ = transform_with_transforms_xml_list(
                shape['transforms_list'], vertices)
        else:
            vertices_transformed = vertices

        vertices_list.append(vertices_transformed)

        faces_list.append(faces+num_vertices)
        num_vertices += vertices.shape[0]

    # write to obj
    vertices_combine = np.vstack(vertices_list)
    faces_combine = np.vstack(faces_list)

    if output_path != '':
        writeMesh(output_path, vertices_combine, faces_combine)
        print('Mesh saved to %s; loaded from %s.' % (output_path, xml_file))

    return vertices_combine, faces_combine

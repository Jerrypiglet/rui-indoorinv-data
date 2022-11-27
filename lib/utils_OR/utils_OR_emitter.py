import glob
from pathlib import Path
import pickle
import numpy as np
import torch
from .SGOptim import SGEnvOptimSkyGrd

window_keys = ['shapeId', 'N_ambient_rep', 'intensity', 'isWindow', 'box3D_world']
lamp_keys = ['shapeId', 'intensity', 'envScale', 'lamb', 'axis_world', 'intensitySky', 'lambSky', 'axisSky_world', 'intensityGrd', 'lambGrd', 'axisGrd_world', 'envAxis_x_world', 'envAxis_y_world', 'envAxis_z_world', 'lightXmlFile', 'isWindow', 'box3D_world']

def merge_light_dat_list_of_lists_to_global(light_dat_list_of_lists):
    '''
    merge list of light_dat_lists (one for each frame) to one list, containing only global info of emitters
    '''

    light_dat_lists_merged = {'windows': [], 'objs': []}
    for emitter_type in ['windows', 'objs']:
        shape_id_list = []
        for light_dat_lists in light_dat_list_of_lists: # over frames
            for (light_file, light_dat_emitter) in light_dat_lists[emitter_type]: # over emitters
                shape_id = light_dat_emitter['shapeId']
                if shape_id in shape_id_list:
                    pass
                else:
                    shape_id_list.append(shape_id)
                    light_dat_lists_merged[emitter_type].append({
                        key: light_dat_emitter[key] for key in window_keys+lamp_keys if key in light_dat_emitter
                    })

    return light_dat_lists_merged

def load_emitter_dat_world(light_dir: Path, N_ambient_rep: str, if_save_storage: bool=False):
    '''
    load .dat files (global/world); not loading per-frame .dat files (light{}/*.dat)

    return:

    emitter_dict_of_lists_world: {'objs': [...], 'windows': [...]}
    '''
    assert N_ambient_rep in ['1ambient', '2ambient', '3SG-SkyGrd'], 'Support for other approx to be added.'

    light_files = sorted(glob.glob(str(light_dir / ('light_%s_*.dat'%N_ambient_rep))))
    if if_save_storage:
        box_files = sorted(glob.glob((str(light_dir / 'box*.dat').replace('DiffLight', ''))))
    else:
        box_files = sorted(glob.glob((str(light_dir / 'box*.dat'))))

    assert len(light_files) != 0, 'No light .dat files found at: '+str(light_dir)
    assert len(light_files) == len(box_files)

    emitter_dict_of_lists_world = {'objs': [], 'windows': []}
    
    for light_id, (light_file, box_file) in enumerate(zip(light_files, box_files)):
        with open(light_file, 'rb') as fIn:
            light_dat = pickle.load(fIn)
        with open(box_file, 'rb') as fIn:
            box_dat = pickle.load(fIn)
        light_dat.update(box_dat)
        # print(light_file, light_dat, light_dat['isWindow'])
        if light_dat['isWindow']:
            light_dat['N_ambient_rep'] = N_ambient_rep
            emitter_dict_of_lists_world['windows'].append((light_file, light_dat))
        else:
            emitter_dict_of_lists_world['objs'].append((light_file, light_dat))

        # print('-----', frame_id, light_id)
        # print(light_dat.keys())
    
    return emitter_dict_of_lists_world


# world -> SG local
def render_3SG_envmap(_3SG_dict: dict, intensity_scale=1.):
    light_axis_world_SG = np.array([_3SG_dict['light_axis_world'][2], -_3SG_dict['light_axis_world'][0], _3SG_dict['light_axis_world'][1]])
    cos_theta = light_axis_world_SG[2]
    theta_SG = np.arccos(cos_theta) # [0, pi]
    cos_phi = light_axis_world_SG[0] / np.sin(theta_SG)
    sin_phi = light_axis_world_SG[1] / np.sin(theta_SG)
    phi_SG = np.arctan2(sin_phi, cos_phi)
    assert phi_SG >= -np.pi and phi_SG <= np.pi

    light_axisSky_world_SG = np.array([_3SG_dict['light_axisSky_world'][2], -_3SG_dict['light_axisSky_world'][0], _3SG_dict['light_axisSky_world'][1]])
    cos_theta = light_axisSky_world_SG[2]
    thetaSky_SG = np.arccos(cos_theta) # [0, pi]
    cos_phi = light_axisSky_world_SG[0] / np.sin(theta_SG)
    sin_phi = light_axisSky_world_SG[1] / np.sin(theta_SG)
    phiSky_SG = np.arctan2(sin_phi, cos_phi)
    assert phiSky_SG >= -np.pi and phiSky_SG <= np.pi

    light_axisGrd_world_SG = np.array([_3SG_dict['light_axisGrd_world'][2], -_3SG_dict['light_axisGrd_world'][0], _3SG_dict['light_axisGrd_world'][1]])
    cos_theta = light_axisGrd_world_SG[2]
    thetaGrd_SG = np.arccos(cos_theta) # [0, pi]
    cos_phi = light_axisGrd_world_SG[0] / np.sin(theta_SG)
    sin_phi = light_axisGrd_world_SG[1] / np.sin(theta_SG)
    phiGrd_SG = np.arctan2(sin_phi, cos_phi)
    assert phiGrd_SG >= -np.pi and phiGrd_SG <= np.pi

    envOptim = SGEnvOptimSkyGrd(isCuda=False)

    recImg = envOptim.renderSG(
        torch.tensor(theta_SG).reshape((1, 1, 1, 1, 1, 1)), torch.tensor(phi_SG).reshape((1, 1, 1, 1, 1, 1)), \
        torch.tensor(_3SG_dict['lamb_SG']).reshape((1, 1, 1, 1, 1)), torch.tensor(_3SG_dict['weight_SG']).reshape((1, 1, 3, 1, 1))*intensity_scale, \
        torch.tensor(thetaSky_SG).reshape((1, 1, 1, 1, 1, 1)), torch.tensor(phiSky_SG).reshape((1, 1, 1, 1, 1, 1)), \
        torch.tensor(_3SG_dict['lambSky_SG']).reshape((1, 1, 1, 1, 1)), torch.tensor(_3SG_dict['weightSky_SG']).reshape((1, 1, 3, 1, 1))*intensity_scale, \
        torch.tensor(thetaGrd_SG).reshape((1, 1, 1, 1, 1, 1)), torch.tensor(phiGrd_SG).reshape((1, 1, 1, 1, 1, 1)), \
        torch.tensor(_3SG_dict['lambGrd_SG']).reshape((1, 1, 1, 1, 1)), torch.tensor(_3SG_dict['weightGrd_SG']).reshape((1, 1, 3, 1, 1))*intensity_scale, \
            )
    recImg_gen_HDR = recImg.squeeze().cpu().numpy().transpose(1, 2, 0)
    return recImg_gen_HDR
    # recImg_gen_uint8, _ = to_nonhdr(recImg_gen)

def vis_envmap_plt(ax, im_envmap: np.ndarray, quad_labels: list=[]):
    
    ax.imshow(im_envmap)

    if quad_labels != []:
        assert len(quad_labels) == 4
        H, W = im_envmap.shape[:2]
        ax.plot([W/4., W/4.], [0, H], 'w')
        ax.plot([W/4.*3., W/4.*3.], [0, H], 'w')
        ax.plot([W/2., W/2.], [0, H], 'w')
        ax.plot([0., W], [H/2., H/2.], 'w--')
        ax.text(0., H, quad_labels[0], color='k', fontsize=10)
        ax.text(W/4., H, quad_labels[1], color='k', fontsize=10)
        ax.text(W/2., H, quad_labels[2], color='k', fontsize=10)
        ax.text(W/4.*3, H, quad_labels[3], color='k', fontsize=10)

def sample_mesh_emitter(emitter_type: str, emitter_index: int, emitter_dict: dict, max_plate: int, if_clip_concave_normals: bool=False):
    assert emitter_type == 'lamp', 'no support for windows for now'
    # lamp, vertices, faces = self.os.lamp_list[emitter_index]
    lamp, vertices, faces = emitter_dict[emitter_type][emitter_index]
    intensity = lamp['emitter_prop']['intensity'] # (3,)
    # center = lamp['emitter_prop']['box3D_world']['center'] # (3,)

    # >>>> sample lamp
    v1 = vertices[faces[:, 0]-1, :]
    v2 = vertices[faces[:, 1]-1, :]
    v3 = vertices[faces[:, 2]-1, :]

    lpts = 1.0 / 3.0 * (v1 + v2 + v3)
    e1 = v2 - v1
    e2 = v3 - v1
    lpts_normal = np.cross(e1, e2)

    # [DEBUG] get rid of upper faces
    # faces = faces[lpts_normal[:, 1]<0]
    # from lib.utils_OR.utils_OR_mesh import writeMesh
    # writeMesh('tmp_mesh.obj', vertices, faces)
    # v1 = vertices[faces[:, 0]-1, :]
    # v2 = vertices[faces[:, 1]-1, :]
    # v3 = vertices[faces[:, 2]-1, :]
    # lpts = 1.0 / 3.0 * (v1 + v2 + v3)
    # e1 = v2 - v1
    # e2 = v3 - v1
    # lpts_normal = np.cross(e1, e2)

    lpts_area = 0.5 * np.sqrt(np.sum(
        lpts_normal * lpts_normal, axis=1, keepdims = True))
    lpts_normal = lpts_normal / np.maximum(2 * lpts_area, 1e-6)

    if if_clip_concave_normals:
        center = np.mean(vertices, axis=0, keepdims = True)
        normal_flip = (np.sum(lpts_normal * (lpts - center), axis=1, keepdims=True) < 0) # [TODO] ZQ is trying to deal with concave faces. Better ideas?
        normal_flip = normal_flip.astype(np.float32)
        lpts_normal = -lpts_normal * normal_flip + (1 - normal_flip) * lpts_normal

    plate_num = lpts.shape[0]

    if plate_num > max_plate:
        prob = float(max_plate)  / float(plate_num)
        # select_ind = np.random.choice([0, 1], size=(plate_num), p=[1-prob, prob])
        # lpts = lpts[select_ind == 1]
        # lpts_normal = lpts_normal[select_ind == 1]
        # lpts_area = lpts_area[select_ind == 1]
        select_ind = np.random.choice(np.arange(plate_num), size=max_plate, replace=False)
        lpts = lpts[select_ind]
        lpts_normal = lpts_normal[select_ind]
        lpts_area = lpts_area[select_ind]
    else:
        prob = 1

    return {
        'lpts': lpts, 
        'lpts_normal': lpts_normal, 
        'lpts_area': lpts_area, 
        'lpts_intensity': intensity, 
        'lpts_prob': prob, 
    }

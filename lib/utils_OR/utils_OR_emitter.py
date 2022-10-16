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


def __load_emitter_dat(self, N_ambient_rep: str):
    '''
    obsolete: originally .dat files are dumped for each frame, containing both world and per-frame info
    '''
    assert N_ambient_rep in ['1ambient', '2ambient', '3SG-SkyGrd'], 'Support for other approx to be added.'

    light_dat_list_of_lists = []
    for frame_id in self.frame_id_list:
        light_dir = self.rendering_root / self.meta_split / self.scene_name / ('light_%d'%frame_id)
        light_files = sorted(glob.glob(str(light_dir / ('light_%s_*.dat'%N_ambient_rep))))
        if if_save_storage:
            box_files = sorted(glob.glob((str(light_dir / 'box*.dat').replace('DiffLight', ''))))
        else:
            box_files = sorted(glob.glob((str(light_dir / 'box*.dat'))))
        assert len(light_files) == len(box_files)
        assert len(light_files) != 0
        light_dat_lists = {'objs': [], 'windows': []}
        for light_id, (light_file, box_file) in enumerate(zip(light_files, box_files)):
            with open(light_file, 'rb') as fIn:
                light_dat = pickle.load(fIn)
            with open(box_file, 'rb') as fIn:
                box_dat = pickle.load(fIn)
            light_dat.update(box_dat)
            # print(light_file, light_dat, light_dat['isWindow'])
            if light_dat['isWindow']:
                light_dat['N_ambient_rep'] = N_ambient_rep
                light_dat_lists['windows'].append((light_file, light_dat))
            else:
                light_dat_lists['objs'].append((light_file, light_dat))

            # print('-----', frame_id, light_id)
            # print(light_dat.keys())
        
        light_dat_list_of_lists.append(light_dat_lists) # per-frame emitter data (world+cam+local)

        light_dat_lists_world = merge_light_dat_list_of_lists_to_global(light_dat_list_of_lists) # only keeps global emitter data from all frames (_world)
        # print('=====>', )
        # print(light_dat_lists_world)

    return light_dat_list_of_lists, light_dat_lists_world

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
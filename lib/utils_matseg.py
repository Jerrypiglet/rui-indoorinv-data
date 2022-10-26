import numpy as np

def get_map_aggre_map(objMask):
    cad_map = objMask[:, :, 0]
    mat_idx_map = objMask[:, :, 1]        
    obj_idx_map = objMask[:, :, 2] # 3rd channel: object INDEX map

    mat_aggre_map = np.zeros_like(cad_map)
    cad_ids = np.unique(cad_map)
    num_mats = 1
    for cad_id in cad_ids:
        cad_mask = cad_map == cad_id
        mat_index_map_cad = mat_idx_map[cad_mask]
        mat_idxes = np.unique(mat_index_map_cad)

        obj_idx_map_cad = obj_idx_map[cad_mask]
        if_light = list(np.unique(obj_idx_map_cad))==[0]
        if if_light:
            mat_aggre_map[cad_mask] = 0
            continue

        # mat_aggre_map[cad_mask] = mat_idx_map[cad_mask] + num_mats
        # num_mats = num_mats + max(mat_idxs)
        cad_single_map = np.zeros_like(cad_map) 
        cad_single_map[cad_mask] = mat_idx_map[cad_mask]
        for i, mat_idx in enumerate(mat_idxes):
    #         mat_single_map = np.zeros_like(cad_map)
            mat_aggre_map[cad_single_map==mat_idx] = num_mats
            num_mats += 1

    return mat_aggre_map, num_mats-1

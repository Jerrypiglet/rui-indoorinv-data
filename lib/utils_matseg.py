import numpy as np

def get_map_aggre_map(objMask):
    '''
    im: ![](https://i.imgur.com/CwSFgOt.png) mainDiffLight_xml1/scene0024_02/im_sdr/im_3.png
    
    three channels, values and vis example: 
        
        [   0    1   59  378 1099 1399 1548 1968 2557 2708 2757 3619]
        [0 1 2 3]
        [ 0  1  3  4  5  6  7  8 10 13 14 15 16 17 18 22 23 24]
        
        ![view 1](https://i.imgur.com/mq9PRu2.png)
        ![view 2](https://i.imgur.com/Vz8Rg85.png)
    '''
    cad_map = objMask[:, :, 0] # 1 channel: (global-over entire OR) cad model id map
    mat_idx_map = objMask[:, :, 1] # 2nd channel: (relative) material INDEX map, INSIDE each cad (e.g. a cad can be all pixels of chairs; then the material index map could be 0 for chair back mat, 1 for chair arm mat, 2 for chair seat mat, etc.)
    obj_idx_map = objMask[:, :, 2] # 3rd channel: (global-scene specific) object id map

    mat_aggre_map = np.zeros_like(cad_map)
    cad_ids = np.unique(cad_map)
    mat_count = 1
    
    raw_tuple_dict = {0: ()}
    
    for cad_id in cad_ids:
        cad_mask = cad_map == cad_id
        mat_count_map_cad = mat_idx_map[cad_mask]
        mat_idxes = np.unique(mat_count_map_cad)

        obj_idx_map_cad = obj_idx_map[cad_mask]
        if_light = list(np.unique(obj_idx_map_cad))==[0]
        if if_light:
            mat_aggre_map[cad_mask] = 0
            continue

        # mat_aggre_map[cad_mask] = mat_idx_map[cad_mask] + mat_count
        # mat_count = mat_count + max(mat_idxs)
        cad_single_map = np.zeros_like(cad_map) 
        cad_single_map[cad_mask] = mat_idx_map[cad_mask]
        for i, mat_idx in enumerate(mat_idxes):
    #         mat_single_map = np.zeros_like(cad_map)
            mat_aggre_map[cad_single_map==mat_idx] = mat_count
            
            obj_idx_map_mat = obj_idx_map[cad_single_map==mat_idx]
            obj_idxes = list(np.unique(obj_idx_map_mat)) # one material may span over multiple objects (e.g. chair seat material over many chairs)
            # print({_: np.sum(obj_idx_map_mat==_) for _ in obj_idxes})
            # import ipdb; ipdb.set_trace()
            # assert len(obj_idxes)==1
            
            # raw_tuple_dict[mat_count] = (cad_id, mat_idx, obj_idxes)
            raw_tuple_dict[mat_count] = (cad_id, mat_idx)
            
            mat_count += 1

    return mat_aggre_map, mat_count-1, raw_tuple_dict

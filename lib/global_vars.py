PATH_HOME_dict = {
    'apple': '/Users/jerrypiglet/Documents/Projects/rui-indoorinv-data', 
    'mm1': '/home/ruizhu/Documents/Projects/rui-indoorinv-data', 
    'r4090': '/home/ruizhu/Documents/Projects/rui-indoorinv-data', 
}
OR_RAW_ROOT_dict = {
    'apple': '/Users/jerrypiglet/Documents/Projects/data/Openrooms_RAW', 
    'mm1': '/newfoundland2/ruizhu/siggraphasia20dataset', 
    'r4090': '/data/Openrooms_RAW', 
}
INV_NERF_ROOT_dict = {
    'apple': '/Users/jerrypiglet/Documents/Projects/inv-nerf', 
    'mm1': '/home/ruizhu/Documents/Projects/inv-nerf', 
    'r4090': '',
}
MONOSDF_ROOT_dict = {
    'apple': '/Users/jerrypiglet/Documents/Projects/monosdf', 
    'mm1': '/home/ruizhu/Documents/Projects/monosdf', 
    'r4090': '',
}

OR_MODALITY_FRAMENAME_DICT = {
    'default': 
    {
        'im_hdr':  'im_%d.hdr', 
        'im_sdr':  ['/data/ruizhu/OR-pngs', 'im_%d.png'], 
        'albedo':  'imbaseColor_%d.png', 
        'roughness':  'imroughness_%d.png', 
        'depth':  'imdepth_%d.dat', 
        'normal':  'imnormal_%d.png', 
        'seg':  'immask_%d.png', 
        'semseg':  'imsemLabel_%d.npy', 
        'matseg':  'imcadmatobj_%d.dat', 
        
    }, 
    'apple':
        {
            'im_hdr': 'im_hdr/im_%d.hdr', 
            'im_sdr': 'im_sdr/im_%d.png', 
            'albedo': 'albedo/imbaseColor_%d.png', 
            'roughness': 'roughness/imroughness_%d.png', 
            'depth': 'depth/imdepth_%d.dat', 
            'normal': 'normal/imnormal_%d.png', 
            'seg': 'mask/immask_%d.png', 
            'semseg': 'semseg/imsemLabel_%d.npy', 
            'matseg': 'matseg/imcadmatobj_%d.dat', 
        }

}

def query_host(dic: dict, host: str):
    if host in dic:
        return dic[host]
    else:
        return dic['default']

mi_variant_dict = {
    'apple': 'llvm_ad_rgb', 
    'mm1': 'cuda_ad_rgb', 
    'r4090': 'cuda_ad_rgb', 
}
from pathlib import Path
import numpy as np
np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)
from typing import Tuple
import os
import cv2
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
from PIL import Image
import struct
import h5py
import random
from skimage.measure import block_reduce 
from skimage.measure import block_reduce 

'''
utils take from Yinhao
'''

def load_matrix(path: Path, if_inverse_y: bool=False) -> np.ndarray:
    """Read a 1d or 2d matrix from a file (as saved by np.savetxt)"""
    m = np.loadtxt(str(path)).astype(np.float32)
    if if_inverse_y:
        if m.shape==(3, 4):
            m = np.hstack((m[:3, :3] @ np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]), m[:3, 3:4]))
        elif m.shape==(4, 4):
            m = np.vstack((
                np.hstack((m[:3, :3] @ np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]), m[:3, 3:4])), 
                m[3:4, :]
            ))
        else:
            assert False, 'wrong input matrix dimension when if_inverse_y=True!'
    return m

def load_img(path: Path, expected_shape: tuple=(), ext: str='png', target_HW: Tuple[int, int]=(), resize_method: str='area') -> np.ndarray:
    '''
    Load an image from a file, trying to maintain its datatype (gray, 16bit, rgb)
    set target_HW to some shape to resize after loading
    '''
    if not Path(path).exists():
        raise FileNotFoundError(path)
        
    assert path.suffix[1:] == ext
    assert ext in ['png', 'jpg', 'hdr', 'npy', 'exr']
    if ext in ['png', 'jpg']:
        im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    elif ext in ['exr']:
        im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)[:, :, :3]
    elif ext in ['hdr']:
        im = cv2.imread(str(path), -1)
    elif ext in ['npy']:
        im = np.load(str(path))

    # cv2.imread returns None when it cannot read the file
    if im is None:
        raise RuntimeError(f"Failed to load {path}")

    if len(im.shape) == 3 and im.shape[2] == 3 and ext in ['png', 'jpg', 'hdr', 'exr']:
        # Color image, convert BGR to RGB
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    if expected_shape != ():
        assert tuple(im.shape) == expected_shape, '%s != %s'%(str(tuple(im.shape)), str(expected_shape))

    if target_HW != ():
        im = resize_img(im, target_HW, resize_method)

    return im

def convert_write_png(hdr_image_path, png_image_path, scale=1., im_key='im_', if_mask=True, im_hdr=None):
    # Read HDR image
    if im_hdr is None:
        im_hdr = load_img(Path(hdr_image_path), ext=str(hdr_image_path).split('.')[1]) * scale

    if if_mask:
        seg_path = png_image_path.replace(im_key, 'immask_')
        seg = load_img(Path(seg_path))[:, :, 0] / 255. # [0., 1.]
        seg_area = np.logical_and(seg > 0.49, seg < 0.51).astype(np.float32)
        seg_env = (seg < 0.1).astype(np.float32)
        seg_obj = (seg > 0.9) 
        im_hdr_scaled, hdr_scale = scale_HDR(im_hdr, seg[..., np.newaxis], fixed_scale=True, if_print=True, if_clip_01=False)
    else:
        im_hdr_scaled = im_hdr * scale
    
    im_SDR = np.clip(im_hdr_scaled**(1.0/2.2), 0., 1.)
    im_SDR_uint8 = (255. * im_SDR).astype(np.uint8)
    Image.fromarray(im_SDR_uint8).save(str(png_image_path))

def load_HDR(path: Path, expected_shape: tuple=(), target_HW: Tuple[int, int]=()) -> np.ndarray:
    if not path.exists():
        if str(path).endswith('.hdr'):
            if not Path(str(path).replace('.hdr', '.rgbe')).exists():
                path = Path(str(path).replace('.hdr', '.rgbe'))
        elif str(path).endswith('.rgbe'):
            if not Path(str(path).replace('.rgbe', '.hdr')).exists():
                path = Path(str(path).replace('.rgbe', '.hdr'))
        else:
            raise FileNotFoundError(path)
            
    im = cv2.imread(str(path), -1)

    if im is None:
        raise RuntimeError(f"Failed to load {path}")
        
    im = im[:, :, ::-1]

    if expected_shape != ():
        assert tuple(im.shape) == expected_shape

    if target_HW != ():
        im = resize_img(im, target_HW, 'area') # fixed to be 'area' for HDR images

    return im

def to_nonHDR(im, extra_scale=1.):
    total_scale = 1.
    im = im * extra_scale
    total_scale *= extra_scale
    im_not_hdr = np.clip((im)**(1.0/2.2), 0., 1.)
    return im_not_hdr, total_scale

def resize_img(im: np.array, target_HW: Tuple[int, int]=(), resize_method: str='area') -> np.array:
    interpolation_dict = {'area': cv2.INTER_AREA, 'nearest': cv2.INTER_NEAREST}
    im_shape_ori = im.shape
    im = cv2.resize(im, (target_HW[1], target_HW[0]), interpolation = interpolation_dict[resize_method])
    assert len(im.shape) == len(im_shape_ori), 'mismatch of shape dimensions before/after resize_img: %s (%d D) VS %s (%d D) '%(str(im_shape_ori), len(im_shape_ori), str(im.shape), len(im.shape))
    return im

def scale_HDR(hdr, seg, fixed_scale=True, scale_input=None, if_return_scale_only=False, if_print=False, if_clip_to_01=False):
    intensityArr = (hdr * seg).flatten()
    intensityArr.sort()
    im_height, im_width = hdr.shape[:2]

    if scale_input is None:
        if not fixed_scale:
            scale = (0.95 - 0.1 * random.random() )  / np.clip(intensityArr[int(0.95 * im_width * im_height * 3) ], 0.1, None)
        else:
            scale = (0.95 - 0.05)  / np.clip(intensityArr[int(0.95 * im_width * im_height * 3) ], 0.1, None)
    else:
        scale = scale_input

    if if_return_scale_only:
        return scale

    hdr = scale * hdr

    # print('-----', np.amax(hdr), np.amin(hdr), np.median(hdr), np.mean(hdr))
    # print('---', scale, np.sum(hdr>1.)/(im_height*im_width*3.), np.amax(hdr))
    if if_clip_to_01:
        hdr = np.clip(hdr, 0., 1.)

    return hdr, scale 

def load_binary(path: Path, expected_shape: tuple=(), target_HW: Tuple[int, int]=(), resize_method: str='area', channels: int=1, dtype: np.dtype=np.float32) -> np.ndarray:
    '''
    return depth/seg map of (H, W, (channels))
    '''
    assert dtype in [np.float32, np.int32], 'Invalid binary type outside (np.float32, np.int32)!'
    if not Path(path).exists():
        raise FileNotFoundError(path)
    with open(path, 'rb') as fIn:
        hBuffer = fIn.read(4)
        height = struct.unpack('i', hBuffer)[0]
        wBuffer = fIn.read(4)
        width = struct.unpack('i', wBuffer)[0]
        dBuffer = fIn.read(4 * channels * width * height )
        if dtype == np.float32:
            decode_char = 'f'
        elif dtype == np.int32:
            decode_char = 'i'
        im = np.asarray(struct.unpack(decode_char * channels * height * width, dBuffer), dtype=dtype)

    im = im.reshape([height, width, channels] )
    im = np.squeeze(im)

    if expected_shape != ():
        assert tuple(im.shape[:2]) == expected_shape

    if target_HW != ():
        if dtype == np.int32:
            assert resize_method == 'nearest'
        if resize_method == 'area':
            assert dtype == np.float32
        im = resize_img(im, target_HW, resize_method)

    return im

def load_h5(path: Path) -> np.ndarray:
    if not Path(path).exists():
        raise FileNotFoundError(path)
    hf = h5py.File(path, 'r')
    im = np.array(hf.get('data' ) )
    return im 

# def convert_lighting_axis_local_to_global_np_torch(rL, lighting_SG_torch, normal_torch):
#     '''
#     rL: rL = renderingLayer(imWidth = SG_params['env_col'], imHeight = SG_params['env_row'], isCuda=False)
#     lighting_SG_torch: (N, SG_num, 6)
#     normal_torch: (3, H, W)
#     '''

def load_envmap(path: Path, env_height=8, env_width=16, env_row = 120, env_col=160, SG_num=12):
    # print('>>>>load_envmap', Path)
    
    if not Path(path).exists():
        raise FileNotFoundError(path)

    # path_8x16 = path.replace('.hdr', '_8x16.hdr')

    # if Path(path_8x16).exists():
    #     if_dump = False
    #     env_heightOrig, env_widthOrig = 16, 32
    # else:
    #     if_dump = True
    if '_8x16' in path:
        env_heightOrig, env_widthOrig = 8, 16
    else:
        env_heightOrig, env_widthOrig = 16, 32

    assert( (env_heightOrig / env_height) == (env_widthOrig / env_width) )
    assert( (env_heightOrig >= env_height) and (env_widthOrig >= env_width) )
    assert( env_heightOrig % env_height == 0)

    env = cv2.imread(str(path), -1 ) 
    assert env is not None

    env = env.reshape(env_row, env_heightOrig, env_col,
        env_widthOrig, 3) # (1920, 5120, 3) -> (120, 16, 160, 32, 3)
    env = np.ascontiguousarray(env.transpose([4, 0, 2, 1, 3] ) ) # -> (3, 120, 160, 16, 32)

    scale = env_heightOrig / env_height
    if scale > 1:
        env = block_reduce(env, block_size = (1, 1, 1, 2, 2), func = np.mean )

    envInd = np.ones([1, 1, 1], dtype=np.float32 )
    return env, envInd

def vis_envmap(envmap, downsample_ratio: int=10, downsize_ratio_hw: int=1, downsize_ratio_env: int=1, hdr_scale: int=1, upscale: int=1) -> np.ndarray:
    '''
    envmap shape: (3, 120, 160, 8, 16)
    '''
    if downsize_ratio_env > 1 or downsize_ratio_hw > 1:
        envmap = block_reduce(envmap, block_size = (1, downsize_ratio_hw, downsize_ratio_hw, downsize_ratio_env, downsize_ratio_env), func = np.mean )
    H_grid, W_grid, h, w = envmap.shape[1:]
    assert H_grid % downsample_ratio == 0
    assert W_grid % downsample_ratio == 0
    xx, yy = np.meshgrid(np.arange(0, H_grid, downsample_ratio), np.arange(0, W_grid, downsample_ratio))
    a = envmap[:, xx.T, yy.T, :, :] * hdr_scale
    b = a.transpose(1, 3, 2, 4, 0).reshape(H_grid*h//downsample_ratio, W_grid*w//downsample_ratio, 3)
    if upscale > 1:
        from skimage.transform import resize
        b = resize(b, (b.shape[0]*upscale, b.shape[1]*upscale))
        # print(b.shape, np.amax(b), np.amin(b), np.mean(b))
        # print('-->', b_Nx.shape, np.amax(b_Nx), np.amin(b_Nx), np.mean(b_Nx))
        h, w = h*upscale, w*upscale

    for ii in range(H_grid//downsample_ratio):
        for jj in range(W_grid//downsample_ratio):
            b[ii*h, :] = 1
            b[:, jj*w] = 1

    return b

def read_cam_params(camFile: Path) -> list:
    assert camFile.exists()
    with open(str(camFile), 'r') as camIn:
    #     camNum = int(camIn.readline().strip() )
        cam_data = camIn.read().splitlines()
    cam_num = int(cam_data[0])
    cam_params = np.array([x.split(' ') for x in cam_data[1:]]).astype(np.float32)
    assert cam_params.shape[0] == cam_num * 3
    cam_params = np.split(cam_params, cam_num, axis=0) # [[origin, lookat, up], ...]
    return cam_params

def normalize_v(x) -> np.ndarray:
    return x / np.linalg.norm(x)

def resize_intrinsics(K: np.ndarray, scale_factor: Tuple[float, float]) -> np.ndarray:
    """
    Scale intrinsics according to image resize factor

    Args:
        K: a single camera intrinsics matrix of shape 3 x 3.
        scale_factor: scale_factor for height and width, target size / src size
    """
    assert tuple(K.shape) == (3, 3)

    K = K.copy()
    K[0] *= scale_factor[1]  # width
    K[1] *= scale_factor[0]  # height
    return K

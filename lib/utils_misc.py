from termcolor import colored
from datetime import datetime
import numpy as np
import logging
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import random
import string
import torch

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('True', 'yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('False', 'no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expectedl; got: %s'%v)

def get_key(d, key: str):
    v = d.get(key)
    assert v is not None, 'key: %s does not exist in dict!'%key
    return v

def get_list_of_keys(d, key_list: list, type_list: list=[]) -> list:
    value_list = []
    if type_list != []:
        assert len(type_list) == len(key_list)

    for _, key in enumerate(key_list):
        v = d.get(key)
        assert v is not None, 'key: %s does not exist in dict!'%key
        if type_list != []:
            assert type(v) == type_list[_], '[key: %s] type of value does not match expected: %s VS %s!'%(key, str(type(v)), str(type_list[_]))
        value_list.append(v)

    return value_list
    #  if len(value_list)!= 1 else value_list[0]

def check_list_of_tensors_size(list_of_tensors: list, tensor_size: tuple):
    for tensor in list_of_tensors:
        assert tuple(tensor.shape) == tensor_size, 'wrong tensor size: %s loaded VS %s required'%(str(tuple(tensor.shape)), str(tensor_size))

def get_datetime():
    # today = date.today()
    now = datetime.now()
    d1 = now.strftime("%Y%m%d-%H%M%S")
    return d1

def basic_logger(name='basic_logger'):
    logger = logging.getLogger(name)
    return logger

# Training
def red(text):
    return colored(text, 'yellow', 'on_red')

def white_red(text):
    coloredd = colored(text, 'white', 'on_red')
    return coloredd

def print_red(text):
    print(red(text))

# Data
def white_blue(text):
    coloredd = colored(text, 'white', 'on_blue')
    return coloredd

def white_magenta(text):
    coloredd = colored(text, 'white', 'on_magenta')
    return coloredd

def blue_text(text):
    coloredd = colored(text, 'blue')
    return coloredd

def print_white_blue(text):
    print(white_blue(text))

# Logging
def green(text):
    coloredd = colored(text, 'blue', 'on_green')
    return coloredd

def green_text(text):
    coloredd = colored(text, 'green')
    return coloredd

def print_green(text):
    print(green(text))

def yellow(text):
    coloredd = colored(text, 'blue', 'on_yellow')
    return coloredd

def yellow_text(text):
    coloredd = colored(text, 'yellow')
    return coloredd

# Model
def magenta(text):
    coloredd = colored(text, 'white', 'on_magenta')
    return coloredd

def print_magenta(text):
    print(magenta(text))

def get_datetime():
    # today = date.today()
    now = datetime.now()
    d1 = now.strftime("%Y%m%d-%H%M%S")
    return d1
    return not any(l)

def gen_random_str(length=5):
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))

def vis_disp_colormap(disp_array_, file=None, normalize=True, min_and_scale=None, valid_mask=None, cmap_name='jet'):
    disp_array = disp_array_.copy()
    cm = plt.get_cmap(cmap_name) # the larger the hotter
    if valid_mask is not None:
        assert valid_mask.shape==disp_array.shape
        assert valid_mask.dtype==bool
    else:
        valid_mask = np.ones_like(disp_array).astype(bool)
    if valid_mask.size == 0:
        valid_mask = np.ones_like(disp_array).astype(bool)
        
    if normalize:
        if min_and_scale is None:
            depth_min = np.amin(disp_array[valid_mask])
            disp_array -= depth_min
            depth_scale = 1./(1e-6+np.amax(disp_array[valid_mask]))
            disp_array = disp_array * depth_scale
            min_and_scale = [depth_min, depth_scale]
        else:
            disp_array -= min_and_scale[0]
            disp_array = disp_array * min_and_scale[1]

    disp_array = np.clip(disp_array, 0., 1.)
    disp_array = (cm(disp_array)[:, :, :3] * 255).astype(np.uint8)
    
    if file is not None:
        from PIL import Image, ImageFont, ImageDraw
        disp_Image = Image.fromarray(disp_array)
        disp_Image.save(file)
    else:
        return disp_array, min_and_scale

def colorize(gray, palette):
    # gray: numpy array of the label and 1*3N size list palette
    color = Image.fromarray(gray.astype(np.uint8)).convert('P')
    color.putpalette(palette)
    return color

def get_device(host: str, device_id: int=-1):
    assert host in ['apple', 'mm1', 'qc', 'liwen'], 'Unsupported host: %s!'%host
    device = 'cpu'
    if host == 'apple':
        if torch.backends.mps.is_built() and torch.backends.mps.is_available():
            device = 'mps'
    else:
        if torch.cuda.is_available():
            if device_id == -1:
                device = 'cuda'
            else:
                device = 'cuda:%d'%device_id

    if device == 'cpu':
        print(yellow('[WARNING] rendering could be slow because device is cpu at %s'%host))
    return device
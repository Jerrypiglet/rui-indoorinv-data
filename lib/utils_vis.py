import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import colorsys
from PIL import Image

def _get_colors(num_colors):
    colors=[]
    for i in np.arange(0., 360., 360. / num_colors):
        hue = i/360.
        lightness = (50 + np.random.rand() * 10)/100.
        saturation = (90 + np.random.rand() * 10)/100.
        colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))
    return colors

def vis_index_map(index_map, num_colors=-1):
    """
    input: [H, W], np.uint8, with indexs from [0, 1, 2, 3, ...] where 0 is no object
    return: [H, W], np.float32, RGB ~ [0., 1.]
    """
    if num_colors == -1:
        num_colors = np.amax(index_map)
        # num_colors = 50
    colors = _get_colors(num_colors)
    index_map_vis = np.zeros((index_map.shape[0], index_map.shape[1], 3))
    for color_idx, color in enumerate(colors):
        mask = index_map == color_idx
        index_map_vis[mask] = color
    return index_map_vis

def reindex_output_map(index_map, invalid_index):
    index_map_reindex = np.zeros_like(index_map)
    index_map_reindex[index_map==invalid_index] = 0

    for new_index, index in enumerate(list(np.unique(index_map[index_map!=invalid_index]))):
        index_map_reindex[index_map==index] = new_index + 1

    return index_map_reindex

def vis_disp_colormap(disp_array, file=None, normalize=True, min_and_scale=None, valid_mask=None, cmap_name='jet'):
    # disp_array = cv2.applyColorMap(disp_array, cv2.COLORMAP_JET)
    # disp_array = cv2.applyColorMap(disp_array, get_mpl_colormap('jet'))
    cm = plt.get_cmap(cmap_name) # the larger the hotter
    # disp_array = disp_array[:, :, :3]
    # print('-', disp_array.shape)
    if valid_mask is not None:
        assert valid_mask.shape==disp_array.shape
        assert valid_mask.dtype==np.bool
    else:
        valid_mask = np.ones_like(disp_array).astype(np.bool)
    
    if normalize:
        if min_and_scale is None:
            depth_min = np.amin(disp_array[valid_mask])
            # print(np.amax(disp_array), np.amin(disp_array))
            disp_array -= depth_min
            depth_scale = 1./(1e-6+np.amax(disp_array[valid_mask]))
            # print(depth_min, depth_scale)
            disp_array = disp_array * depth_scale
            # print(np.amax(disp_array), np.amin(disp_array))
            min_and_scale = [depth_min, depth_scale]
        else:
            disp_array -= min_and_scale[0]
            disp_array = disp_array * min_and_scale[1]
    disp_array = np.clip(disp_array, 0., 1.)
    # print('--', disp_array.shape)
    disp_array = (cm(disp_array)[:, :, :3] * 255).astype(np.uint8)
    # print('---', disp_array.shape)
    
    # print('+++++', np.amax(disp_array), np.amin(disp_array))
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

def color_map_color(values, vmin=0, vmax=1, cmap_name='jet', ):
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap(cmap_name)  # PiYG
    colors = cmap(norm(abs(values)))[:, :3]  # will return rgba, we take only first 3 so we get rgb
    # colors = matplotlib.colors.rgb2hex(colors)
    return colors

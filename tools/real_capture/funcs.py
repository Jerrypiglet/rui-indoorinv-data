import cv2
import numpy as np
import rawpy
from pathlib import Path
import os
import lensfunpy

def get_files(path):
    path = Path(path)
    files = []
    for p in sorted(os.listdir(path)):
        p = path / p
        if p.is_file() and not p.stem.startswith('.'):
            files.append(p)
    return files

def read_process_img(
    p,
    mod,
#     crop_x=150,  # Canon Setup
#     crop_y=100,
    crop_x=100,  # Sony Setup
    crop_y=50,
    img_max=np.float32(2**14-1),
):
    # Images should be linear rgb for tiff(float) images
    # https://lensfun.github.io/calibration-tutorial/lens-vignetting.html
    with rawpy.imread(str(p)) as raw:
        image = raw.raw_image.copy()
        black = np.reshape(np.array(raw.black_level_per_channel, dtype=image.dtype), (2, 2))
        black = np.tile(black, (image.shape[0]//2, image.shape[1]//2))
        image = np.maximum(image, black) - black
        image = cv2.demosaicing(image, code=cv2.COLOR_BAYER_BG2BGR)
        image = image / (raw.white_level - black)[...,np.newaxis].astype(np.float32)
    did_apply = mod.apply_color_modification(image)
    image = np.clip(image, 0, 1)
    return image[crop_y:-crop_y, crop_x:-crop_x]

def process_bracket(_):
# for index in tqdm(range(len(image_paths)//len(times))):
#     index = 83
    index, params = _
    exr_output_dir = params['exr_output_dir']
    pose_output_dir = params['pose_output_dir']
    png_output_dir = params['png_output_dir']
    times = params['times']
    image_paths = params['image_paths']
    times_max_index = params['times_max_index']
    times_min_index = params['times_min_index']
    exposure_target = params['exposure_target']
    white_balance = params['white_balance']
    apply_median_blur = params['apply_median_blur']
    mapx = params['mapx']
    mapy = params['mapy']
    exr_resize_shape = params['exr_resize_shape']
    png_exposure_scale = params['png_exposure_scale']
    pose_resize_scale = params['pose_resize_scale']
    # height = params['height']
    # width = params['width']
    mod = params['mod']
    
    
    # db = lensfunpy.Database()
    # # Sony Setting
    # cam = db.find_cameras("Sony", "ILCE-7M3")[0]
    # lens = db.find_lenses(cam, "Sony", "FE 24-105mm")[0]
    # focal_length, aperture, distance = 24, 20, 0.5
    # # Canon Setting
    # # cam = db.find_cameras("Canon", "Canon EOS 5D Mark III")[0]
    # # lens = db.find_lenses(cam, "Canon", "Canon EF 24-70mm f/2.8L II USM")[0]
    # # focal_length, aperture, distance = 24, 19, 2.2

    # mod = lensfunpy.Modifier(lens, cam.crop_factor, width, height)
    # mod.initialize(focal_length, aperture, distance, scale=1.0, pixel_format=np.float32)

    
    exr_path = exr_output_dir / f"img_{str(index).zfill(4)}.exr"
    pose_path = pose_output_dir / f"img_{str(index).zfill(4)}.png"
    png_path = png_output_dir / f"img_{str(index).zfill(4)}.png"
    # Get the correct image index
    index *= len(times)
    
    # Read and correct images
    images = []
    for i, time in enumerate(times):
        images.append(read_process_img(str(image_paths[index]), mod=mod))
        index += 1
    images = np.array(images)
    
    # Calculate hat function for averaging the values
    weights = np.minimum(images, 1.0 - images)
    weights[times_max_index][images[times_max_index]<0.5] = 1.0
    weights[times_min_index][images[times_min_index]>0.5] = 1.0
    weights /= np.sum(weights, axis=0)
    
    # Scale the images to same exposure
    images_scaled = images * (exposure_target / times)[:, np.newaxis, np.newaxis, np.newaxis]
    # Combine the images
    imageHDR = np.sum(images_scaled * weights, axis=0) * white_balance
    if apply_median_blur:
        imageHDR = cv2.medianBlur(imageHDR, ksize=3)
        
    # Undistort hdr image
    imageHDR = cv2.remap(imageHDR, mapx, mapy, cv2.INTER_LINEAR)
    
    # Write exr image for reconstruction
    output_img_resized = cv2.resize(
        imageHDR, 
        dsize=exr_resize_shape, 
        interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(exr_path), output_img_resized)
    png_image = np.clip(output_img_resized * png_exposure_scale, a_min=0, a_max=1) **(1/2.2)
    png_image = (png_image * 255).astype(np.uint8)
    cv2.imwrite(str(png_path), png_image)
    
    # Write png pose image for colmap poses
    pose_image = cv2.resize(
        imageHDR, 
        dsize=(exr_resize_shape[0]*pose_resize_scale, exr_resize_shape[1]*pose_resize_scale), 
        interpolation=cv2.INTER_AREA)
    pose_image = np.clip(pose_image * png_exposure_scale, a_min=0, a_max=1) **(1/2.2)
    pose_image = (pose_image * 255).astype(np.uint8)
    cv2.imwrite(str(pose_path), pose_image)
import shutil
import glob
from tqdm import tqdm
import numpy as np
np.set_printoptions(suppress=True)
import imageio

from pathlib import Path
import mitsuba as mi
from lib.utils_misc import blue_text, yellow, red
from lib.utils_io import convert_write_png
from lib.utils_io import normalize_v
from lib.utils_io import resize_intrinsics

class renderer_mitsubaScene_3D():
    '''
    A class used to visualize/render Mitsuba scene in XML format
    '''
    def __init__(
        self, 
        scene_rendering_path: Path, 
        im_params_dict: dict, 
    ):
        self.scene_rendering_path = scene_rendering_path
        self.im_params_dict = im_params_dict
        
    def render_im(self):
        self.spp = self.im_params_dict.get('spp', 1024)
        if_render = 'y'
        im_files = sorted(glob.glob(str(self.scene_rendering_path / 'Image' / '*_*.exr')))
        if len(im_files) > 0:
            if_render = input(red("%d *_*.exr files found at %s. Re-render? [y/n]"))
        if if_render in ['N', 'n']:
            print(yellow('ABORTED rendering by Mitsuba'))
            return
        else:
            shutil.rmtree(str(self.scene_rendering_path / 'Image'))
            self.scene_rendering_path / 'Image'.mkdir(parents=True, exist_ok=True)

        print(blue_text('Rendering RGB to... by Mitsuba: %s')%str(self.scene_rendering_path / 'Image'))
        for i, (origin, lookatvector, up) in tqdm(enumerate(self.origin_lookatvector_up_list)):
            sensor = self.get_sensor(origin, origin+lookatvector, up)
            image = mi.render(self.mi_scene, spp=self.spp, sensor=sensor)
            im_rendering_path = str(self.scene_rendering_path / 'Image' / ('%03d_0001.exr'%i))
            # im_rendering_path = str(self.scene_rendering_path / 'Image' / ('im_%d.rgbe'%i))
            mi.util.write_bitmap(str(im_rendering_path), image)
            '''
            load exr: https://mitsuba.readthedocs.io/en/stable/src/how_to_guides/image_io_and_manipulation.html?highlight=load%20openexr#Reading-an-image-from-disk
            '''

            # im_rgbe = cv2.imread(str(im_rendering_path), -1)
            # dest_path = str(im_rendering_path).replace('.rgbe', '.hdr')
            # cv2.imwrite(dest_path, im_rgbe)
            
            convert_write_png(hdr_image_path='', png_image_path=str(im_rendering_path).replace('.exr', '.png'), if_mask=False, im_hdr=np.array(image))

        print(blue_text('DONE.'))
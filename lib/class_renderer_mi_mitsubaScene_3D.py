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

from .class_rendererBase import rendererBase

class renderer_mi_mitsubaScene_3D(rendererBase):
    '''
    A class used to render Mitsuba scene with Mitsuba
    '''
    def __init__(
        self, 
        mitsuba_scene, 
        modality_list, 
        *args, **kwargs, 
    ):
        rendererBase.__init__(
            self, 
            mitsuba_scene, 
            modality_list, 
            *args, **kwargs
        )

    @property
    def valid_modalities(self):
        return [
            'im', 
            # 'seg', 
            # 'albedo', 'roughness', 
            'depth', 'normal', 
            # 'lighting_envmap', 
        ]

    def render(self):
        for _ in self.modality_list:
            if _ == 'im': self.render_im()
        
    def render_im(self):
        self.spp = self.im_params_dict.get('spp', 1024)
        folder_name, render_folder_path = self.render_modality_check('im')

        print(blue_text('Rendering RGB to... by Mitsuba: %s')%str(render_folder_path))
        for i, (origin, lookatvector, up) in tqdm(enumerate(self.os.origin_lookatvector_up_list)):
            sensor = self.get_sensor(origin, origin+lookatvector, up)
            image = mi.render(self.os.mi_scene, spp=self.spp, sensor=sensor)
            im_rendering_path = str(render_folder_path / ('%03d_0001.exr'%i))
            # im_rendering_path = str(render_folder_path / ('im_%d.rgbe'%i))
            mi.util.write_bitmap(str(im_rendering_path), image)
            '''
            load exr: https://mitsuba.readthedocs.io/en/stable/src/how_to_guides/image_io_and_manipulation.html?highlight=load%20openexr#Reading-an-image-from-disk
            '''

            # im_rgbe = cv2.imread(str(im_rendering_path), -1)
            # dest_path = str(im_rendering_path).replace('.rgbe', '.hdr')
            # cv2.imwrite(dest_path, im_rgbe)
            
            convert_write_png(hdr_image_path='', png_image_path=str(im_rendering_path).replace('.exr', '.png'), if_mask=False, im_hdr=np.array(image))

        print(blue_text('DONE.'))

    def get_sensor(self, origin, target, up):
        from mitsuba import ScalarTransform4f as T
        return mi.load_dict({
            'type': 'perspective',
            'fov': np.arctan(self.os.K[0][2]/self.os.K[0][0])/np.pi*180.*2.,
            'fov_axis': 'x',
            'to_world': T.look_at(
                origin=mi.ScalarPoint3f(origin.flatten()),
                target=mi.ScalarPoint3f(target.flatten()),
                up=mi.ScalarPoint3f(up.flatten()),  
            ),
            'sampler': {
                'type': 'independent',
                'sample_count': int(self.spp), 
            },
            'film': {
                'type': 'hdrfilm',
                'width': self.os.im_W_load,
                'height': self.os.im_H_load,
                'rfilter': {
                    'type': 'tent',
                },
                'pixel_format': 'rgb',
            },
        })


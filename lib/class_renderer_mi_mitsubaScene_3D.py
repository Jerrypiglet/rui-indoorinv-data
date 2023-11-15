import shutil
import glob
from tqdm import tqdm
import numpy as np
np.set_printoptions(suppress=True)

from pathlib import Path
import mitsuba as mi
from lib.utils_misc import blue_text, white_blue
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
        if_skip_check: bool=False,
        *args, **kwargs, 
    ):
        rendererBase.__init__(
            self, 
            mitsuba_scene=mitsuba_scene, 
            modality_list=modality_list, 
            if_skip_check=if_skip_check, 
            *args, **kwargs
        )
        self.spp = self.im_params_dict.get('spp', 1024)

    @property
    def valid_modalities(self):
        return [
            'im', 
            'albedo', 
            'depth', 'geo_normal', 'sh_normal', 
            'uv', 'position', 
            'shape_index', 
            # 'seg', 
            # 'roughness', 
            # 'lighting_envmap', 
        ]

    def render(self, if_force: bool=False):
        for modality in self.modality_list:
            if modality in ['im', 'depth', 'geo_normal', 'sh_normal', 'uv', 'position', 'shape_index', 'albedo']: 
                self.render_2d(modality, if_force=if_force)
                
    def render_2d(self, modality: str='im', if_force: bool=False):
        
        folder_name, render_folder_path = self.render_modality_check(modality, if_force=if_force, file_name_appendix='_mi')

        print(white_blue('[%s] Rendering %s to... by [Mitsuba] (spp %d)): %s')%(self.__class__.__name__, modality, self.spp, str(render_folder_path)))
        
        for i, (origin, lookatvector, up) in tqdm(enumerate(self.os.origin_lookatvector_up_list)):
            sensor = self.get_sensor(origin, origin+lookatvector, up, modality=modality)
            integrator = self.get_integrator(modality=modality)
            image = mi.render(self.os.mi_scene, sensor=sensor, integrator=integrator) 
            im_rendering_path = str(render_folder_path / ('%03d_0001_mi.exr'%self.os.frame_id_list[i]))
            # im_rendering_path = str(render_folder_path / ('%03d_0001_mi_BlenderExport.exr'%self.os.frame_id_list[i]))
            print('Writing rendering to: %s...'%im_rendering_path)
            mi.util.write_bitmap(str(im_rendering_path), image)
            '''
            load exr: https://mitsuba.readthedocs.io/en/stable/src/how_to_guides/image_io_and_manipulation.html?highlight=load%20openexr#Reading-an-image-from-disk
            '''

            # im_rgbe = cv2.imread(str(im_rendering_path), -1)
            # dest_path = str(im_rendering_path).replace('.rgbe', '.hdr')
            # cv2.imwrite(dest_path, im_rgbe)
            
            '''
            visualization/convert to SDR as png
            '''
            im = np.array(image)
            if modality in ['im']:
                convert_write_png(hdr_image_path='', png_image_path=str(im_rendering_path).replace('.exr', '.png'), if_mask=False, im_hdr=im)
            elif modality in ['albedo']:
                convert_write_png(hdr_image_path='', png_image_path=str(im_rendering_path).replace('.exr', '.png'), if_mask=False, im_hdr=im, if_gamma_22=True)
            elif modality in ['geo_normal','sh_normal']:
                assert len(im.shape) == 3 and im.shape[2] == 3
                assert np.amax(np.linalg.norm(im, axis=-1) - 1.) < 1e-4, np.amax(np.linalg.norm(im, axis=-1))
                convert_write_png(hdr_image_path='', png_image_path=str(im_rendering_path).replace('.exr', '.png'), if_mask=False, im_hdr=im, if_gamma_22=False)
            elif modality in ['depth']:
                assert len(im.shape) == 3 and im.shape[2] == 1
                im = im[:, :, 0]
                from lib.utils_vis import vis_disp_colormap
                vis_disp_colormap(im, file=str(im_rendering_path).replace('.exr', '.png'), normalize=True, min_and_scale=None, valid_mask=None, cmap_name='jet')
            elif modality in ['uv']:
                assert len(im.shape) == 3 and im.shape[2] == 2 # min: 0., max: 1.
                assert np.amax(im) <= 1. and np.amin(im) >= 0.
                im = np.concatenate([im, np.zeros_like(im[:, :, 0:1])], axis=2)
                convert_write_png(hdr_image_path='', png_image_path=str(im_rendering_path).replace('.exr', '.png'), if_mask=False, im_hdr=im, if_gamma_22=False)
            elif modality in ['position']:
                assert len(im.shape) == 3 and im.shape[2] == 3
                im = (im - np.amin(im, axis=(0, 1))) / (np.amax(im, axis=(0, 1)) - np.amin(im, axis=(0, 1)))
                convert_write_png(hdr_image_path='', png_image_path=str(im_rendering_path).replace('.exr', '.png'), if_mask=False, im_hdr=im, if_gamma_22=False)
            elif modality in ['shape_index']:
                assert len(im.shape) == 3 and im.shape[2] == 1
                im = im[:, :, 0].astype(np.int32)
                from lib.utils_vis import vis_index_map
                vis_index_map(im, file=str(im_rendering_path).replace('.exr', '.png'), num_colors=-1)
            else:
                import ipdb; ipdb.set_trace()

        print(blue_text('DONE.'))
        
    def get_sensor(self, origin, target, up, modality='im'):
        from mitsuba import ScalarTransform4f as T
        print('[Mitsuba sensor] origin', origin.flatten().tolist(), 'fov_x', np.arctan(self.os.K[0][2]/self.os.K[0][0])/np.pi*180.*2.)
        assert (np.linalg.norm(up) - 1.) < 1e-5
        assert modality in self.valid_modalities
        
        # aov with Mitsuba: https://mitsuba.readthedocs.io/en/latest/src/generated/plugins_integrators.html#arbitrary-output-variables-integrator-aov
        
        rfilter = {
            'im': 'tent', 
            # 'seg', 
            # 'albedo', 'roughness', 
            'depth': 'box', 
            'geo_normal': 'box',  
            'sh_normal': 'box',  
            'albedo': 'box',
            'uv': 'box', 
            'position': 'box', 
            'shape_index': 'box', 
        }[modality]
        
        mi_sensor_dict = {
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
                'sample_count': int(self.spp) if modality == 'im' else 1, 
            },
            'film': {
                'type': 'hdrfilm',
                'width': self.im_params_dict['im_W_load'],
                'height': self.im_params_dict['im_H_load'],
                'rfilter': {
                    'type': rfilter,
                },
                'pixel_format': 'rgb',
                'component_format': 'float32',
            },
        }
        sensor = mi.load_dict(mi_sensor_dict)
        return sensor
    
    def get_integrator(self, modality='im'):
        # https://mitsuba.readthedocs.io/en/latest/src/generated/plugins_integrators.html#arbitrary-output-variables-integrator-aov
        
        if modality in ['im']:
            mi_integrator_dict = {
                'type': 'path',
                'max_depth': 65
            }
        elif modality in ['depth', 'geo_normal', 'sh_normal', 'uv', 'position', 'shape_index', 'albedo']:
            mi_integrator_dict = {
                'type': 'aov',
                'aovs': {
                    'geo_normal': 'nn:geo_normal', 
                    'sh_normal': 'nnsh:sh_normal', 
                    'depth': 'dd.y:depth', 
                    'uv': 'uv:uv', 
                    'position': 'position:position', 
                    'shape_index': 'shape_index:shape_index', 
                    'albedo': 'albedo:albedo', 
                    }[modality], 
            }
        else:
            raise NotImplementedError
        
        return  mi.load_dict(mi_integrator_dict)
            
        
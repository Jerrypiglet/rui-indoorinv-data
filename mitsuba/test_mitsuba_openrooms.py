import sys
sys.path.append('../lib')

# Import the library using the alias "mi"
import mitsuba as mi
# Set the variant of the renderer
mi.set_variant('cuda_ad_rgb')

import xml.etree.ElementTree as et
from utils_OR.utils_OR_xml import gen_random_str, transformToXml

# # Load a scene
# scene = mi.load_dict(mi.cornell_box())
# # Render the scene
# img = mi.render(scene)
# # Write the rendered image to an EXR file
# mi.Bitmap(img).write('cbox.exr')


'''
    loading an OpenRooms scene and transform to Mitsuba 0.6.0 (not 3.0.0)
    https://mitsuba.readthedocs.io/en/latest/src/key_topics/scene_format.html#scene-xml-file-format
'''

xml_file = '/home/ruizhu/Documents/Projects/renderOpenRooms/public_re_2/scenes/xml1/scene0552_00_more/mainDiffLight.xml'
tree = et.parse(xml_file)
root  = tree.getroot()
root.attrib.pop('verion', None)
# scene = mi.load_file("../scenes/cbox.xml")
root.set('version', '0.6.0')

# remove microfacet bsdfs FOR NOW (incompatible with mitsuba) # [TODO] fix bsdf
for bsdf in root.findall('bsdf'):
    root.remove(bsdf)
for shape in root.findall('shape'):
    bsdf_refs = shape.findall('ref')
    for bsdf_ref in bsdf_refs:
        if bsdf_ref.get('name') == 'bsdf':
            shape.remove(bsdf_ref)     

# fix duplicate shape ids
shape_ids = []
for shape in root.findall('shape'):
    shape_id = shape.get('id')
    if shape_id not in shape_ids:
        shape_ids.append(shape_id)
    else:
        shape_id += gen_random_str(5)
        shape.set('id', shape_id)
    shape_ids.append(shape_id)

# sampler: adaptive -> independent
sampler = root.findall('sensor')[0].findall('sampler')[0]
sampler.set('type', 'independent')

xmlString = transformToXml(root)
with open('scene.xml', 'w') as xmlOut:
    xmlOut.write(xmlString )

# render
scene = mi.load_file("scene.xml")
image = mi.render(scene, spp=256)
mi.util.write_bitmap("my_first_render.png", image)
mi.util.write_bitmap("my_first_render.exr", image)
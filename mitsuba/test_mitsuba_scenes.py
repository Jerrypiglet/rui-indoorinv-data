import sys
# Import the library using the alias "mi"
import mitsuba as mi
# Set the variant of the renderer
mi.set_variant('llvm_ad_rgb')
# mi.set_variant('cuda_ad_rgb')

'''
load this part scenes compatible with Mitsuba 3.0.0
'''

'''
Render.
Should be consistent with OpenRooms renderings in layout:
'''
scene = mi.load_file("data/scenes/kitchen/scene_v3.xml")
image = mi.render(scene, spp=64)
mi.util.write_bitmap("rendering_kitchen.png", image)
mi.util.write_bitmap("rendering_kitchen.exr", image)
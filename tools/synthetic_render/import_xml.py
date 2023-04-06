import bpy
import os

bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath('./test.blend'))
bpy.ops.wm.open_mainfile(filepath=os.path.abspath('./test.blend'))
bpy.ops.import_scene.mitsuba(filepath='test.xml', override_scene=True)
bpy.ops.wm.save_mainfile()
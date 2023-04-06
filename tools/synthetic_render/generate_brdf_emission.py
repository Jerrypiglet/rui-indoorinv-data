import argparse, sys, os
import json
import numpy as np
import bpy

import os
split = os.getenv('SPLIT')

DEBUG = False


# render setup resolution
RESOLUTION_X = 640#512
RESOLUTION_Y = 320#512
SAMPLES = 128

# output setup
COLOR_DEPTH = 16
FORMAT = 'OPEN_EXR'

# camera pose file
POSE_FILE = '{}.npy'.format(split)
RESULTS_PATH = POSE_FILE.replace('.npy','')

# modals to render
MODALS = ['DiffCol','GlossCol','Emit','IndexMA','Depth','Normal']#,'Alpha','IndexOB']
AOV_MODALS = ['Roughness']#,'Metallic']


out_dir=RESULTS_PATH
fp = RESULTS_PATH

location_euler = np.load(POSE_FILE)
# number of training views
VIEWS = location_euler.shape[0]

def listify_matrix(matrix):
    matrix_list = []
    for row in matrix:
        matrix_list.append(list(row))
    return matrix_list

if not os.path.exists(fp):
    os.makedirs(fp)

# Data to store in JSON file
out_data = {
    'camera_angle_x': bpy.context.scene.camera.data.angle_x#bpy.data.objects['Camera'].data.angle_x,
}

scene = bpy.context.scene


# Set up renderer params
scene = bpy.data.scenes["Scene"]
scene.render.engine = 'CYCLES'
scene.render.use_motion_blur = False
scene.cycles.device = 'GPU'

#scene.render.film_transparent = False#True
scene.view_layers[0].cycles.use_denoising = True

scene.cycles.samples = SAMPLES

cycles_prefs = bpy.context.preferences.addons['cycles'].preferences
cycles_prefs.compute_device_type = 'CUDA'
cycles_prefs.get_devices()

# assigne material id
for i,mat in enumerate(bpy.data.materials):
    mat.pass_index=(i+1)
    
# link aov output
for aov_modal in AOV_MODALS:
    for mat in bpy.data.materials:
        tree = mat.node_tree
        if tree is None:
            continue
        for node in tree.nodes:
            if 'Bsdf' not in node.bl_idname:
                continue
            if (node.bl_idname =='ShaderNodeBsdfGlossy' or node.bl_idname =='ShaderNodeBsdfGlass') and node.distribution == 'SHARP' and aov_modal=='Roughness':
                modal_value = 0.0
                buffer_node = tree.nodes.new('ShaderNodeValue')
                from_socket = buffer_node.outputs['Value']
                from_socket.default_value = modal_value
            elif aov_modal not in node.inputs.keys():
                modal_value = 0.0
                buffer_node = tree.nodes.new('ShaderNodeValue')
                from_socket = buffer_node.outputs['Value']
                from_socket.default_value = modal_value
            else:
                socket = node.inputs[aov_modal]
                if len(socket.links) == 0:
                    if node.bl_idname == 'ShaderNodeBsdfDiffuse' and aov_modal == 'Roughness':
                        modal_value = 1.0
                    else:
                        modal_value = socket.default_value
                    buffer_node = tree.nodes.new('ShaderNodeValue')
                    from_socket = buffer_node.outputs['Value']
                    from_socket.default_value = modal_value
                    tree.links.new(from_socket,socket)
                else:
                    from_socket=socket.links[0].from_socket

            aov_node = tree.nodes.new('ShaderNodeOutputAOV')
            aov_node.name = aov_modal
            tree.links.new(from_socket,aov_node.inputs['Value'])

# Background
scene.render.dither_intensity = 0.0
scene.render.film_transparent = True#False
scene.render.resolution_x = RESOLUTION_X
scene.render.resolution_y = RESOLUTION_Y
scene.render.resolution_percentage = 100

cam = bpy.context.scene.camera#scene.objects['Camera']

obj_idx = 1
for obj in bpy.context.scene.objects:
    if obj.type in ('MESH'):
        obj.pass_index=obj_idx
        obj_idx += 1

scene.render.image_settings.file_format = FORMAT
scene.render.image_settings.color_depth = str(COLOR_DEPTH)

# Set pass
scene.view_layers["ViewLayer"].use_pass_normal = True
scene.view_layers["ViewLayer"].use_pass_z = True
#scene.view_layers["ViewLayer"].use_pass_object_index = True
scene.view_layers["ViewLayer"].use_pass_material_index = True
scene.view_layers["ViewLayer"].use_pass_diffuse_color = True
scene.view_layers["ViewLayer"].use_pass_emit = True
scene.view_layers["ViewLayer"].use_pass_glossy_color = True

scene.use_nodes = True

for aov_modal in AOV_MODALS:
    bpy.ops.scene.view_layer_add_aov()
    scene.view_layers["ViewLayer"].aovs[-1].name = aov_modal
    scene.view_layers["ViewLayer"].aovs[-1].type = "VALUE"
    
tree = scene.node_tree
links = tree.links
for n in tree.nodes:
    print(n)
    tree.nodes.remove(n)
    
render_layers = tree.nodes.new('CompositorNodeRLayers')


# Create output folder
modals = MODALS + AOV_MODALS
modal_file_outputs = []
# modal_dirs = []
for modal in modals:
    modal_dir = os.path.join(out_dir,modal)
    os.makedirs(modal_dir,exist_ok=True)
    modal_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
    modal_file_output.label = modal
    links.new(render_layers.outputs[modal],modal_file_output.inputs[0])
    modal_file_output.base_path = modal_dir
    modal_file_outputs.append(modal_file_output)
    # modal_dirs.append(modal_dir)

rotation_mode = 'XYZ'


for di, device in enumerate(cycles_prefs.devices):
    device.use = (di == 0)
    
out_data['frames'] = []

for i in range(VIEWS):
    filename = '{0:03d}'.format(int(i))
    scene.render.filepath = os.path.join(
            out_dir,
            'Image',
            filename+'_0001')
    
    cam.location = location_euler[i,0]
    cam.rotation_euler[2] = location_euler[i,1,2]
    cam.rotation_euler[0] = location_euler[i,1,0]
   
    for modal_file_output in modal_file_outputs:
        modal_file_output.file_slots[0].path = filename + '_'
    
    if DEBUG:
        break
    else:
        bpy.ops.render.render(write_still=True)  # render still

    frame_data = {
        'file_path': scene.render.filepath,
        'transform_matrix': listify_matrix(cam.matrix_world)
    }
    out_data['frames'].append(frame_data)

with open(fp + '/' + 'transforms.json', 'w') as out_file:
    json.dump(out_data, out_file, indent=4)
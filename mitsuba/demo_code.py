import torch

import mitsuba

def torch2mi(x):
    """coordinate conversion from synthetic nerf to mistuba"""
    ret = x[:,[0,2,1]]
    ret = torch.tensor([[1,1,-1]],device=x.device)*ret
    return ret
def mi2torch(x):
    """coordinate conversion from mistuba to synthetic nerf"""
    ret = torch.tensor([[1,1,-1]],device=x.device)*x
    return ret[:,[0,2,1]]

mitsuba.set_variant('cuda_ad_rgb') # gpu auto differetial rgb
# scene stored in xml file, has the structure:
#<scene version="3.0.0">
#    <shape type="obj">
#        <string name="filename" value="room.obj"/>
#    </shape>
#</scene>
scene = mitsuba.load_file('scene.xml')

xs_mi = mitsuba.Point3f(torch2mi(xs))
ds_mi = mitsuba.Vector3f(torch2mi(ds))
# ray origin, direction, t_max
rays_mi = mitsuba.Ray3f(xs_mi,ds_mi,mitsuba.Float(6.0))
ret = scene.ray_intersect(rays_mi)
# returned structure contains intersection location, nomral, ray step, ...
positions = mi2torch(ret.p.torch())
normals = mi2torch(ret.n.torch())
ts  = ret.t.torch()
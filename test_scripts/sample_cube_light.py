def sample_cube_light(light_min,light_max,position,S1):
    """
    Args:
        light_min, light_max: min/max range of the bounding box
        position: Bx3 position
        S1: number of samples to query
    Return:
        BXS1X3 sampled directions
        BxS1 inverse pdf
        BxS1 MIS weights
    """
    device = light_min.device
    B = position.shape[0]
    light_whd = light_max-light_min

    # randomly select a face of the cube weighted by their area
    W,H,D = light_whd
    cube_pdf = torch.stack([W*H,W*H,W*D,W*D,H*D,H*D])
    light_area = cube_pdf.sum()
    cube_pdf /= light_area
    cube_cdf = torch.cat([torch.zeros_like(cube_pdf[:1]),cube_pdf.cumsum(0)])
    face = torch.searchsorted(cube_cdf.contiguous(), torch.rand(B,S1,device=device).clamp_min(1e-12),right=False)-1

    # assign corresonding bounding box axies and origin
    light_origin = torch.ones(B,S1,3,device=device)*light_min[None,None]
    light_normal = torch.zeros(B,S1,3,device=device)
    light_uv = torch.zeros(2,B,S1,3,device=device)

    mask = (face==0)
    ii,jj = torch.where(mask)
    if mask.any():
        light_normal[ii,jj,2] = -1
        light_uv[0,ii,jj,0] = light_whd[0]
        light_uv[1,ii,jj,1] = light_whd[1]

    mask = (face==1)
    ii,jj = torch.where(mask)
    if mask.any():
        light_origin[ii,jj,2] += light_whd[2]
        light_normal[ii,jj,2] = 1
        light_uv[0,ii,jj,0] = light_whd[0]
        light_uv[1,ii,jj,1] = light_whd[1]

    mask = (face==2)
    ii,jj = torch.where(mask)
    if mask.any():
        light_normal[ii,jj,1] = -1
        light_uv[0,ii,jj,0] = light_whd[0]
        light_uv[1,ii,jj,2] = light_whd[2]

    mask = (face==3)
    ii,jj = torch.where(mask)
    if mask.any():
        light_origin[ii,jj,1] += light_whd[1]
        light_normal[ii,jj,1] = 1
        light_uv[0,ii,jj,0] = light_whd[0]
        light_uv[1,ii,jj,2] = light_whd[2]

    mask = (face==4)
    ii,jj = torch.where(mask)
    if mask.any():
        light_normal[ii,jj,0] = -1
        light_uv[0,ii,jj,1] = light_whd[1]
        light_uv[1,ii,jj,2] = light_whd[2]

    mask = (face==5)
    ii,jj = torch.where(mask)
    if mask.any():
        light_origin[ii,jj,0] += light_whd[0]
        light_normal[ii,jj,0] = 1
        light_uv[0,ii,jj,1] = light_whd[1]
        light_uv[1,ii,jj,2] = light_whd[2]

    # radnomly sample on each surface
    u,v = torch.rand(2,B,S1,1,device=device)
    pos = light_origin + light_uv[0]*u + light_uv[1]*v
    Lnew = pos-position[:,None]

    # 1/pdf = cos_theta/r^2 * Area
    rr = (Lnew*Lnew).sum(-1)
    Lnew = NF.normalize(Lnew,dim=-1)
    cos_theta = (-light_normal*Lnew).sum(-1).relu()

    pdf_inv = cos_theta/rr.clamp_min(1e-12)*light_area
    pdf = rr/(light_area*cos_theta)
    pdf[(light_area*cos_theta)<1e-12] = 0.0

    return Lnew,pdf_inv,(pdf*S1).pow(2)
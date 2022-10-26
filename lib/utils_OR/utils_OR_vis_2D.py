def in_frame(p, width, height):
    if p[0]>0 and p[0]<width and p[1]>0 and p[1]<height:
        return True
    else:
        return False

def draw_projected_bdb3d(draw, bdb2D_from_3D, front_flags=None, color=(255, 255, 255), width=5):
    bdb2D_from_3D = [tuple(item) for item in bdb2D_from_3D]
    if front_flags is None:
        front_flags = [True] * len(bdb2D_from_3D)
    assert len(front_flags) == len(bdb2D_from_3D)

    for idx_list in [[0, 1, 2, 3, 0], [4, 5, 6, 7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]:
        draw_lines_notalloutside_image(draw, bdb2D_from_3D, idx_list, front_flags, color=color, width=width)

def draw_lines_notalloutside_image(draw, v_list, idx_list, front_flags, color=(255, 255, 255), width=5):
    assert len(v_list) == len(front_flags)
    for i in range(len(idx_list)-1):
        if front_flags[idx_list[i]] and front_flags[idx_list[i+1]]:
            draw.line([v_list[idx_list[i]], v_list[idx_list[i+1]]], width=width, fill=color)

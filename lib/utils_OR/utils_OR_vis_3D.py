from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import numpy as np
import matplotlib.pyplot as plt


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


def vis_cube_plt(Xs, ax, color=None, linestyle='-', label=None, if_face_idx_text=False, if_vertex_idx_text=False, text_shift=[0., 0., 0.], fontsize_scale=1., linewidth=1.):
    # https://stackoverflow.com/questions/11140163/plotting-a-3d-cube-a-sphere-and-a-vector-in-matplotlib
    index1 = [0, 1, 2, 3, 0, 4, 5, 6, 7, 4]
    index2 = [[1, 5], [2, 6], [3, 7]]
    index3 = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [
        1, 2, 6, 5], [2, 3, 7, 6], [0, 3, 7, 4]]  # faces
    if color is None:
        color = list(np.random.choice(range(256), size=3) / 255.)
    #   print(color)
    ax.plot3D(Xs[index1, 0], Xs[index1, 1], Xs[index1, 2],
              color=color, linestyle=linestyle, linewidth=linewidth)
    for index in index2:
        ax.plot3D(Xs[index, 0], Xs[index, 1], Xs[index, 2],
                  color=color, linestyle=linestyle, linewidth=linewidth)
    if label is not None:
        # ax.text3D(Xs[0, 0]+text_shift[0], Xs[0, 1]+text_shift[1], Xs[0, 2]+text_shift[2], label, color=color, fontsize=10*fontsize_scale)
        ax.text3D(Xs.mean(axis=0)[0], Xs.mean(axis=0)[1], Xs.mean(
            axis=0)[2], label, color=color, fontsize=10*fontsize_scale)
    if if_vertex_idx_text:
        for vertex_idx, V in enumerate(Xs):
            ax.text3D(V[0]+text_shift[0], V[1]+text_shift[1], V[2]+text_shift[2],
                      str(vertex_idx), color=color, fontsize=10*fontsize_scale)
    if if_face_idx_text:
        for face_idx, index in enumerate(index3):
            X_center = Xs[index, :].mean(0)
            # print(X_center.shape)
            ax.text3D(X_center[0]+text_shift[0], X_center[1]+text_shift[1], X_center[2] +
                      text_shift[2], str(face_idx), color='grey', fontsize=30*fontsize_scale)
            ax.scatter3D(X_center[0], X_center[1],
                         X_center[2], color=[0.8, 0.8, 0.8], s=30)
            for cross_index in [[index[0], index[2]], [index[1], index[3]]]:
                ax.plot3D(Xs[cross_index, 0], Xs[cross_index, 1], Xs[cross_index, 2], color=[
                          0.8, 0.8, 0.8], linestyle='--', linewidth=1)


def vis_axis(ax):
    for vec, tag, tag_loc in zip([([0, 1], [0, 0], [0, 0]), ([0, 0], [0, 1], [0, 0]), ([0, 0], [0, 0], [0, 1])], [r'$X_w$', r'$Y_w$', r'$Z_w$'], [[1, 0, 0], [0, 1, 0], [0, 0, 1]]):
        a = Arrow3D(vec[0], vec[1], vec[2], mutation_scale=20,
                    lw=1, arrowstyle="->", color="k")
        ax.text3D(tag_loc[0], tag_loc[1], tag_loc[2], tag)
        ax.add_artist(a)


def vis_axis_xyz(ax, x, y, z, origin=[0., 0., 0.], suffix='_w', colors=['k']*3):
    if isinstance(colors, str):
        colors = [colors] * 3
    for color, vec, tag, tag_loc in zip(
            colors, 
            [([origin[0], (origin+x)[0]], [origin[1], (origin+x)[1]], [origin[2], (origin+x)[2]]),
            ([origin[0], (origin+y)[0]], [origin[1], (origin+y)[1]], [origin[2], (origin+y)[2]]),
            ([origin[0], (origin+z)[0]], [origin[1], (origin+z)[1]], [origin[2], (origin+z)[2]])], [r'$X%s$' % suffix, r'$Y%s$' % suffix, r'$Z%s$' % suffix], [origin+x, origin+y, origin+z]
        ):
        a = Arrow3D(vec[0], vec[1], vec[2], mutation_scale=20,
                    lw=1, arrowstyle="->", color=color)
        ax.text3D(tag_loc[0], tag_loc[1], tag_loc[2], tag, color=color)
        ax.add_artist(a)

# https://stackoverflow.com/a/63625222
# Functions from @Mateen Ulhaq and @karlo


def set_axes_equal(ax: plt.Axes):
    """Set 3D plot axes to equal scale.

    Make axes of 3D plot have equal scale so that spheres appear as
    spheres and cubes as cubes.  Required since `ax.axis('equal')`
    and `ax.set_aspect('equal')` don't work on 3D.
    """
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    _set_axes_radius(ax, origin, radius)


def _set_axes_radius(ax, origin, radius):
    x, y, z = origin
    ax.set_xlim3d([x - radius, x + radius])
    ax.set_ylim3d([y - radius, y + radius])
    ax.set_zlim3d([z - radius, z + radius])

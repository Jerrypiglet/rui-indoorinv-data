import numpy as np

def bb_intersection_over_union(boxA, boxB, if_return_areas=False):
    """
    Taken from https://gist.github.com/meyerjo/dd3533edc97c81258898f60d8978eddc
    both boxes should be in format [x1, y1, x2, y2]
    """

    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    # if interArea == 0:
    #     return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    if not if_return_areas:
        return iou
    else:
        return iou, (interArea, boxAArea, boxBArea)

# intersection function
def isect_line_plane_v3(p0, p1, p_co, p_no, epsilon=1e-6):
    """
    # https://stackoverflow.com/questions/5666222/3d-line-plane-intersection
    p0, p1: Define the line.
    p_co, p_no: define the plane:
        p_co Is a point on the plane (plane coordinate).
        p_no Is a normal vector defining the plane direction;
             (does not need to be normalized).

    Return a Vector or None (when the intersection can't be found).
    """

    u = sub_v3v3(p1, p0)
    dot = dot_v3v3(p_no, u)

    if abs(dot) > epsilon:
        # The factor of the point between p0 -> p1 (0 - 1)
        # if 'fac' is between (0 - 1) the point intersects with the segment.
        # Otherwise:
        #  < 0.0: behind p0.
        #  > 1.0: infront of p1.
        w = sub_v3v3(p0, p_co)
        fac = -dot_v3v3(p_no, w) / dot
        u = mul_v3_fl(u, fac)
        output = add_v3v3(p0, u)
        return np.stack(output).reshape((3, 1))
    else:
        # The segment is parallel to plane.
        return None

# ----------------------
# generic math functions

def add_v3v3(v0, v1):
    return (
        v0[0] + v1[0],
        v0[1] + v1[1],
        v0[2] + v1[2],
        )


def sub_v3v3(v0, v1):
    return (
        v0[0] - v1[0],
        v0[1] - v1[1],
        v0[2] - v1[2],
        )


def dot_v3v3(v0, v1):
    return (
        (v0[0] * v1[0]) +
        (v0[1] * v1[1]) +
        (v0[2] * v1[2])
        )


def len_squared_v3(v0):
    return dot_v3v3(v0, v0)


def mul_v3_fl(v0, f):
    return (
        v0[0] * f,
        v0[1] * f,
        v0[2] * f,
        )

    
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def angle_between_2d(v2, v1):
    """ Returns the [clockwise] angle in [degrees] between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1).reshape((2,))
    v2_u = unit_vector(v2).reshape((2,))
    # return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    cosTh = np.dot(v1_u, v2_u)
    sinTh = np.cross(v1_u, v2_u)
    return np.rad2deg(np.arctan2(sinTh,cosTh))

def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix
    
# def rotation_matrix_vs(A,B):
# # a and b are in the form of numpy array
#     A,B = A.flatten(), B.flatten()

#     ax = A[0]
#     ay = A[1]
#     az = A[2]

#     bx = B[0]
#     by = B[1]
#     bz = B[2]

#     au = A/(np.sqrt(ax*ax + ay*ay + az*az))
#     bu = B/(np.sqrt(bx*bx + by*by + bz*bz))

#     R=np.array([[bu[0]*au[0], bu[0]*au[1], bu[0]*au[2]], [bu[1]*au[0], bu[1]*au[1], bu[1]*au[2]], [bu[2]*au[0], bu[2]*au[1], bu[2]*au[2]] ])

#     return(R)

def distance_points_plane(v1, v2, u) :
    u = np.array(u, ndmin=2)
    v = np.vstack((v1, v2))
    vv = np.dot(v, v.T) # shape (2, 2)
    uv = np.dot(u, v.T) # shape (n ,2)
    ab = np.dot(np.linalg.inv(vv), uv.T) # shape(2, n)
    w = u - np.dot(ab.T, v)
    return np.sqrt(np.sum(w**2, axis=1)) # shape (n,)
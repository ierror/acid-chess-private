import math

import numpy as np
from cv2 import cv2
from scipy import spatial


def order_points(pts):
    # sort the points based on their x-coordinates
    x_sorted = pts[np.argsort(pts[:, 0]), :]

    # grab the left-most and right-most points from the sorted
    # x-coodinate points
    left_most = x_sorted[:2, :]
    right_most = x_sorted[2:, :]

    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    left_most = left_most[np.argsort(left_most[:, 1]), :]
    (tl, bl) = left_most
    # now that we have the top-left coordinate, use it as an
    # anchor to calculate the Euclidean distance between the
    # top-left and right-most points; by the Pythagorean
    # theorem, the point with the largest distance will be
    # our bottom-right point

    D = spatial.distance.cdist(tl[np.newaxis], right_most, "euclidean")[0]
    (br, tr) = right_most[np.argsort(D)[::-1], :]
    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    return np.array([tl, tr, br, bl], dtype="float32")


def four_point_transform(image, pts):
    pts = order_points(np.array(pts))
    p = [pts[0], pts[1], pts[3], pts[2]]

    rows, cols = image.shape[0:2]
    # image center
    u0 = (cols) / 2.0
    v0 = (rows) / 2.0

    # widths and heights of the projected image
    w1 = spatial.distance.euclidean(p[0], p[1])
    w2 = spatial.distance.euclidean(p[2], p[3])

    h1 = spatial.distance.euclidean(p[0], p[2])
    h2 = spatial.distance.euclidean(p[1], p[3])

    w = max(w1, w2)
    h = max(h1, h2)

    # visible aspect ratio
    ar_vis = float(w) / float(h)

    # make numpy arrays and append 1 for linear algebra
    m1 = np.array((p[0][0], p[0][1], 1)).astype('float32')
    m2 = np.array((p[1][0], p[1][1], 1)).astype('float32')
    m3 = np.array((p[2][0], p[2][1], 1)).astype('float32')
    m4 = np.array((p[3][0], p[3][1], 1)).astype('float32')

    # calculate the focal disrance
    k2 = np.dot(np.cross(m1, m4), m3) / np.dot(np.cross(m2, m4), m3)
    k3 = np.dot(np.cross(m1, m4), m2) / np.dot(np.cross(m3, m4), m2)

    n2 = k2 * m2 - m1
    n3 = k3 * m3 - m1

    n21 = n2[0]
    n22 = n2[1]
    n23 = n2[2]

    n31 = n3[0]
    n32 = n3[1]
    n33 = n3[2]

    f = math.sqrt(np.abs((1.0 / (n23 * n33)) * (
            (n21 * n31 - (n21 * n33 + n23 * n31) * u0 + n23 * n33 * u0 * u0) + (
            n22 * n32 - (n22 * n33 + n23 * n32) * v0 + n23 * n33 * v0 * v0))))

    A = np.array([[f, 0, u0], [0, f, v0], [0, 0, 1]]).astype('float32')

    At = np.transpose(A)
    Ati = np.linalg.inv(At)
    Ai = np.linalg.inv(A)

    # calculate the real aspect ratio
    ar_real = math.sqrt(np.dot(np.dot(np.dot(n2, Ati), Ai), n2) / np.dot(np.dot(np.dot(n3, Ati), Ai), n3))

    if ar_real < ar_vis:
        W = int(w)
        H = int(W / ar_real)
    else:
        H = int(h)
        W = int(ar_real * H)

    pts1 = np.array(p).astype('float32')
    pts2 = np.float32([[0, 0], [W, 0], [0, H], [W, H]])

    # project the image with the new w/h
    M = cv2.getPerspectiveTransform(pts1, pts2)

    return cv2.warpPerspective(image, M, (W, H))

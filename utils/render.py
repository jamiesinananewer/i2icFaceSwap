# coding: utf-8

__author__ = 'cleardusk'

import sys

sys.path.append('..')

import cv2
import numpy as np


from CythonBuilding import RenderPipeline
from utils.functions import plot_image
from .tddfa_util import _to_ctype

'''cfg = {
    'intensity_ambient': 0.3,
    'color_ambient': (1, 1, 1),
    'intensity_directional': 0.6,
    'color_directional': (1, 1, 1),
    'intensity_specular': 0.1,
    'specular_exp': 5,
    'light_pos': (0, 0, 5),
    'view_pos': (0, 0, 5)
}'''
cfg = {
    'intensity_ambient': 1,
    'color_ambient': (1, 1, 1),
    'intensity_directional': 0.6,
    'color_directional': (1, 1, 1),
    'intensity_specular': 1,
    'specular_exp': 5,
    'light_pos': (0, 0, 5),
    'view_pos': (1000, 0, 500000)
}

render_app = RenderPipeline(**cfg)


def render(img, ver_lst, tri, vertex_alphas, alpha=1.0, show_flag=False, wfp=None, with_bg_flag=True, texture = None):
    if with_bg_flag:
        overlap = img.copy()
    else:
        overlap = np.zeros_like(img)

    for ver_ in ver_lst:
        ver = _to_ctype(ver_.T)  # transpose
        overlap = render_app(ver, tri, vertex_alphas, overlap, texture = texture)

    if with_bg_flag:
        res = cv2.addWeighted(img, 1 - alpha, overlap, alpha, 0)
    else:
        res = overlap

    if wfp is not None:
        cv2.imwrite(wfp, res)
        print(f'Save visualization result to {wfp}')

    if show_flag:
        plot_image(res)

    return res

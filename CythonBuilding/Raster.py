from . import _init_paths

import numpy as np
import SwapCython

def rasterize_alpha(vertices, triangles, colors, vertex_alphas, max_alpha = 1.0, bg = None,
              reverse=False, color_correct = True):
    if bg is not None:
        height, width, channel = bg.shape
    else:
        assert height is not None and width is not None and channel is not None
        bg = np.zeros((height, width, channel), dtype=np.uint8)

    buffer = np.zeros((height, width), dtype=np.float32) - 1e8

    if colors.dtype != np.float32:
        colors = colors.astype(np.float32)
    
    if color_correct:
        SwapCython.rasterize_color_correct(bg, vertices, triangles, colors, buffer, vertex_alphas, triangles.shape[0], 
                          height, width, channel, max_alpha = max_alpha, reverse=reverse, sigma_correct=True)
    else:
        SwapCython.rasterize_alpha(bg, vertices, triangles, colors, buffer, vertex_alphas, triangles.shape[0], 
                          height, width, channel, max_alpha = max_alpha, reverse=reverse)
    return bg

def rasterize(vertices, triangles, colors, bg = None,
              reverse=False):
    if bg is not None:
        height, width, channel = bg.shape
    else:
        assert height is not None and width is not None and channel is not None
        bg = np.zeros((height, width, channel), dtype=np.uint8)

    buffer = np.zeros((height, width), dtype=np.float32) - 1e8

    if colors.dtype != np.float32:
        colors = colors.astype(np.float32)
    
    CythonBuild.rasterize(bg, vertices, triangles, colors, buffer, triangles.shape[0], 
                          height, width, channel, reverse=reverse)
    return bg

def get_normal(vertices, triangles):
    normal = np.zeros_like(vertices, dtype=np.float32)
    SwapCython.get_normal(normal, vertices, triangles, vertices.shape[0], triangles.shape[0])
    return normal


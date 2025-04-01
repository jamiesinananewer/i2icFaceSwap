import sys
import argparse
import imageio
import cv2
from tqdm import tqdm
import yaml
import numpy as np
import os

from FaceBoxes import FaceBoxes
from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
from TDDFA import TDDFA
from TDDFA_ONNX import TDDFA_ONNX
from utils.render import render
from utils.functions import *
from utils.uv import uv_tex, get_colors
from utils.serialization import ser_to_obj

#Within this script contains the option to create obj files.
#This allows for these models to be imported into 3D rendering
#software such as blender to be played around with.


'''def load_image_params(img, tddfa, face_boxes, dense_flag = True):

    

    boxes = face_boxes(img)

    n = len(boxes)

    if n == 0:
        print(f'No face detected, exit')
        sys.exit(-1)
    print(f'Detect {n} faces')

    img_param_lst, img_roi_box_lst = tddfa(img, boxes)

    img_ver_lst = tddfa.recon_vers(img_param_lst, img_roi_box_lst, dense_flag = dense_flag)

    return img_param_lst, img_roi_box_lst, img_ver_lst'''







if __name__ == '__main__':

    config_path = 'configs/mb1_120x120.yml'

    cfg = yaml.load(open(config_path), Loader=yaml.SafeLoader)

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    os.environ['OMP_NUM_THREADS'] = '4'

    face_boxes = FaceBoxes_ONNX()
    tddfa = TDDFA_ONNX(**cfg)
    
    
    source_img = cv2.imread('examples/inputs/denzel.jpg')

    source_params_lst, source_roi_box_lst, source_ver_lst = load_image_params(source_img, tddfa, face_boxes)

    textures = []
    for ver in source_ver_lst:
        tex = get_colors(source_img, ver)
        textures.append(tex)
    texture = textures[0].astype(np.float32) / 255.0
    print(f'ver = {source_ver_lst[0][0]}')
    print(f'ver len = {len(source_ver_lst[0][0])}')

    #ncolors = 2
    #texture = color_grid(source_ver_lst, ncolors=ncolors)
    #ratio_list = [0.42, 0.35, 0.101, 0.119]
    #texture = color_grid_ratios(source_ver_lst, ratio_list=ratio_list)
    
    #cutoff_list = [16113, 13427, 4122, 4703]
    #texture = color_grid_indices(source_ver_lst, cutoff_list = cutoff_list)
    #print(f'textures = {texture}')
    print(f'tex len = {len(texture)}')

    

    target_img = cv2.imread('examples/inputs/willem.jpg')
    target_params_lst, target_roi_box_lst, target_ver_lst = load_image_params(target_img, tddfa, face_boxes)

    #target_params_lst = video_data[179]['param_lst']
    #target_roi_box_lst = video_data[179]['roi_box_lst']


    swap_face = {
        'params_lst': source_params_lst,
        'roi_box_lst': target_roi_box_lst[-1]
    }

    swap_face['params_lst'][:12] = target_params_lst[:12] #swap pose
    swap_face['params_lst'][52:62] = target_params_lst[52:62] #swap expression
    swap_params_lst = swap_face['params_lst']
    swap_ver_lst = tddfa.recon_vers([swap_params_lst[0]], [target_roi_box_lst[0]], dense_flag = True)

    bound_ver = compute_boundary_vertices(tddfa.tri)
    dists = compute_distance_to_boundary(swap_ver_lst[0].T, bound_ver)
    alphas = compute_alpha_from_distance(dists, sigma=40)
    alphas[:16113] = 1.0
    #source_bgr = source_img[... , ::-1] #RGB to BGR

    render(target_img, swap_ver_lst, tddfa.tri, vertex_alphas=alphas, alpha = 1.0, show_flag=False, with_bg_flag=True,wfp = 'examples/results/swap_test_nobg_culled.jpg', texture=texture)
    #ser_to_obj(target_img, source_ver_lst, tddfa.tri, height = target_img.shape[0], colors = texture, wfp = f'examples/results/face_parse_{len(cutoff_list)}_colors_indices2.obj')
    #ser_to_obj(target_img, source_ver_lst, tddfa.tri, height = target_img.shape[0], colors = texture, wfp = f'examples/results/denzel.obj')


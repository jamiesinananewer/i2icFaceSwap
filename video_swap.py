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
from utils.uv import get_colors




if __name__ == "__main__":
    print('Loading 3DMM Model...')
    config_path = 'configs/mb1_120x120.yml'

    cfg = yaml.load(open(config_path), Loader=yaml.SafeLoader)

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    os.environ['OMP_NUM_THREADS'] = '4'

    face_boxes = FaceBoxes_ONNX()
    tddfa = TDDFA_ONNX(**cfg)


    img_path = 'examples/inputs/denzel.jpg'
    source_img = cv2.imread(img_path)

    print("\nLoading Image Parameters...")
    img_params_lst, img_roi_box_lst, img_ver_lst = load_image_params(source_img, tddfa, face_boxes)

    textures = []
    for ver in img_ver_lst:
        tex = get_colors(source_img, ver)
        textures.append(tex)

    texture = textures[0].astype(np.float32) / 255.0

    #blur_textures = []
    '''for ver in img_ver_lst:
        
        blur_tex = edge_feather(ver.T, texture, tddfa.tri)
        blur_textures.append(blur_tex)'''

    #blur_texture = blur_textures[0].astype(np.float32)
    #source_texture = uv_tex(source_img, img_ver_lst, tddfa.tri, show_flag=False, wfp=None)

    print("\nImage Processing Results:")
    print(f"Number of Faces Detected: {len(img_params_lst)}")
    print(f"Shape of img_ver_lst[0]: {img_ver_lst[0].shape if img_ver_lst else 'No faces detected'}")
    
    

    vid_path = 'examples/inputs/videos/inception.mp4'
    # Load video
    video = imageio.get_reader(vid_path)
    fps = video.get_meta_data()['fps']
    output_path = f'{vid_path}_swapped_cillian.mp4'
    
    boundary_verts = compute_boundary_vertices(tddfa.tri)
    boundary_verts = boundary_verts[boundary_verts >= 16113]

    ver_tran = img_ver_lst[0].T
    dists = compute_distance_to_boundary_kdtree(ver_tran, boundary_verts)

    vertex_alphas = compute_alpha_from_distance(dists, sigma=100)
    vertex_alphas = vertex_alphas.astype(np.float32)
    #print(f'vertex alphas :')
    #for i, ver_al in enumerate(vertex_alphas):
        #print (f'vertex {i}: alpha = {vertex_alphas[i]}')

    print("\nLoading Video Parameters...")
    video_data = load_video_params(video, tddfa, face_boxes)

    #create frame data for source face swapped onto target face
    swap_video_data = swap_faces_3D(video_data, img_params_lst, tddfa)

    
    writer = imageio.get_writer('examples/results/videos/inception_denzel_swapped_corrected_exp_only_maxalph07.mp4', fps = fps)
    
    print('Writing swapped face(s) to video...')
    for i, frame in tqdm(enumerate(video)):
        if i == i:
            frame_bgr = frame[... , ::-1] #RGB to BGR

            param_lst = swap_video_data[i]['param_lst']
            roi_box_lst = swap_video_data[i]['roi_box_lst']
            ver_lst = swap_video_data[i]['ver_lst']

            bgr_copy = frame_bgr.copy()
            tri = tddfa.tri
            
            res = render(bgr_copy, ver_lst, tddfa.tri, vertex_alphas, alpha=0.7, texture=texture, show_flag=False, with_bg_flag=True)

            writer.append_data(res[... , ::-1]) #BGR to RGB
    
    print('Video written successfully')
    



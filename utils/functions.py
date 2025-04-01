# coding: utf-8

__author__ = 'cleardusk'
# with additions from Jamie :)

import sys
import numpy as np
import cv2
from math import sqrt
from scipy.spatial import cKDTree
from tqdm import tqdm
import matplotlib.pyplot as plt

# Define common color constants.
RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)


def get_suffix(filename):
    """
    Extracts and returns the suffix (extension) from a filename.
    
    Args:
        filename (str): A filename (e.g., "a.jpg").
        
    Returns:
        str: The file extension starting with the dot (e.g., ".jpg") or an empty string if no dot is found.
    """
    # Find the last occurrence of a period in the filename.
    pos = filename.rfind('.')
    # If no period is found, return an empty string.
    if pos == -1:
        return ''
    return filename[pos:]


def crop_img(img, roi_box):
    """
    Crops the input image based on a region-of-interest (ROI) box.
    
    Args:
        img (numpy.ndarray): The input image array.
        roi_box (iterable): A four-element list/tuple (sx, sy, ex, ey) representing the ROI coordinates.
        
    Returns:
        numpy.ndarray: The cropped image.
    """
    # Get the original image dimensions.
    h, w = img.shape[:2]
    
    # Round the ROI coordinates.
    sx, sy, ex, ey = [int(round(_)) for _ in roi_box]
    # Compute the desired output dimensions.
    dh, dw = ey - sy, ex - sx
    
    # Initialize the output array with zeros; support color or grayscale.
    if len(img.shape) == 3:
        res = np.zeros((dh, dw, 3), dtype=np.uint8)
    else:
        res = np.zeros((dh, dw), dtype=np.uint8)
    
    # Adjust start indices and offsets if the ROI exceeds image boundaries.
    if sx < 0:
        sx, dsx = 0, -sx
    else:
        dsx = 0
    if ex > w:
        ex, dex = w, dw - (ex - w)
    else:
        dex = dw
    if sy < 0:
        sy, dsy = 0, -sy
    else:
        dsy = 0
    if ey > h:
        ey, dey = h, dh - (ey - h)
    else:
        dey = dh
    
    # Copy the region from the original image to the result.
    res[dsy:dey, dsx:dex] = img[sy:ey, sx:ex]
    return res


def calc_hypotenuse(pts):
    """
    Calculates a scaled hypotenuse length based on the bounding box of the input points.
    
    Args:
        pts (numpy.ndarray): A 2xN array of points.
        
    Returns:
        float: One third of the hypotenuse of the bounding box encompassing the points.
    """
    # Calculate bounding box from the given points.
    bbox = [min(pts[0, :]), min(pts[1, :]), max(pts[0, :]), max(pts[1, :])]
    # Determine the center of the bounding box.
    center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
    # Calculate the radius as half the maximum width/height.
    radius = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2
    # Update the bounding box to be a square centered at the center.
    bbox = [center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius]
    # Compute the hypotenuse length of the square.
    llength = sqrt((bbox[2] - bbox[0]) ** 2 + (bbox[3] - bbox[1]) ** 2)
    return llength / 3


def parse_roi_box_from_landmark(pts):
    """
    Calculates a region-of-interest (ROI) box based on facial landmarks.
    
    Args:
        pts (numpy.ndarray): A 2xN array of landmark points.
        
    Returns:
        list: A list [x1, y1, x2, y2] representing the ROI box.
    """
    # Compute bounding box of landmarks.
    bbox = [min(pts[0, :]), min(pts[1, :]), max(pts[0, :]), max(pts[1, :])]
    # Determine the center of the bounding box.
    center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
    # Compute the radius based on the maximum dimension.
    radius = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2
    # Create a square ROI box centered at the landmark center.
    bbox = [center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius]
    # Calculate the hypotenuse of the square.
    llength = sqrt((bbox[2] - bbox[0]) ** 2 + (bbox[3] - bbox[1]) ** 2)
    center_x = (bbox[2] + bbox[0]) / 2
    center_y = (bbox[3] + bbox[1]) / 2
    
    # Construct the ROI box based on the hypotenuse.
    roi_box = [0] * 4
    roi_box[0] = center_x - llength / 2
    roi_box[1] = center_y - llength / 2
    roi_box[2] = roi_box[0] + llength
    roi_box[3] = roi_box[1] + llength

    return roi_box


def parse_roi_box_from_bbox(bbox):
    """
    Generates an ROI box from a given bounding box with a fixed scaling factor.
    
    Args:
        bbox (iterable): A list or tuple of [left, top, right, bottom].
        
    Returns:
        list: A list [x1, y1, x2, y2] representing the new ROI box.
    """
    # Unpack the original bounding box.
    left, top, right, bottom = bbox[:4]
    # Compute an old size metric based on width and height.
    old_size = (right - left + bottom - top) / 2
    # Calculate the center of the bounding box.
    center_x = right - (right - left) / 2.0
    center_y = bottom - (bottom - top) / 2.0 + old_size * 0.14
    # Determine new size by scaling.
    size = int(old_size * 1.58)

    # Build the new ROI box.
    roi_box = [0] * 4
    roi_box[0] = center_x - size / 2
    roi_box[1] = center_y - size / 2
    roi_box[2] = roi_box[0] + size
    roi_box[3] = roi_box[1] + size

    return roi_box


def plot_image(img):
    """
    Displays the input image using matplotlib.
    
    Args:
        img (numpy.ndarray): The input image array.
    """
    # Get image dimensions.
    height, width = img.shape[:2]
    # Set up the plot figure with a computed size.
    plt.figure(figsize=(12, height / width * 12))
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.axis('off')
    # Show the image (convert BGR to RGB for correct colors).
    plt.imshow(img[..., ::-1])
    plt.show()


def draw_landmarks(img, pts, style='fancy', wfp=None, show_flag=False, **kwargs):
    """
    Draws facial landmarks on the input image using matplotlib.
    
    Args:
        img (numpy.ndarray): The input image.
        pts (list or numpy.ndarray): Landmark points. If not a list, it will be wrapped as one.
        style (str): Drawing style; default is 'fancy'.
        wfp (str): Optional file path to save the output visualization.
        show_flag (bool): Whether to display the plot.
        **kwargs: Additional keyword arguments (e.g., color, dense_flag, markeredgecolor).
    """
    # Get the image dimensions.
    height, width = img.shape[:2]
    plt.figure(figsize=(12, height / width * 12))
    plt.imshow(img[..., ::-1])
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.axis('off')
    
    # Extract additional flags and parameters.
    dense_flag = kwargs.get('dense_flag')
    
    # Ensure pts is in a list format.
    if not isinstance(pts, (tuple, list)):
        pts = [pts]
    
    # Loop over each set of landmarks.
    for i in range(len(pts)):
        if dense_flag:
            # Draw dense points with small markers.
            plt.plot(pts[i][0, ::6], pts[i][1, ::6], 'o', markersize=0.4, color='c', alpha=0.7)
        else:
            # Set drawing parameters.
            alpha = 0.8
            markersize = 4
            lw = 1.5
            color = kwargs.get('color', 'w')
            markeredgecolor = kwargs.get('markeredgecolor', 'black')
            # Predefined landmark indices for facial parts.
            nums = [0, 17, 22, 27, 31, 36, 42, 48, 60, 68]
            
            # Draw connections for closed features (eyes, mouth).
            plot_close = lambda i1, i2: plt.plot([pts[i][0, i1], pts[i][0, i2]],
                                                 [pts[i][1, i1], pts[i][1, i2]],
                                                 color=color, lw=lw, alpha=alpha - 0.1)
            plot_close(41, 36)
            plot_close(47, 42)
            plot_close(59, 48)
            plot_close(67, 60)
            
            # Draw the lines and markers for the rest of the landmarks.
            for ind in range(len(nums) - 1):
                l, r = nums[ind], nums[ind + 1]
                plt.plot(pts[i][0, l:r], pts[i][1, l:r], color=color, lw=lw, alpha=alpha - 0.1)
                plt.plot(pts[i][0, l:r], pts[i][1, l:r], marker='o', linestyle='None', markersize=markersize,
                         color=color, markeredgecolor=markeredgecolor, alpha=alpha)
    
    # Save to file if a path is provided.
    if wfp is not None:
        plt.savefig(wfp, dpi=150)
        print(f'Save visualization result to {wfp}')
    # Show the image if required.
    if show_flag:
        plt.show()


def cv_draw_landmark(img_ori, pts, box=None, color=GREEN, size=1):
    """
    Draws landmarks on an image using OpenCV.
    
    Args:
        img_ori (numpy.ndarray): The original image.
        pts (numpy.ndarray): Landmark points.
        box (iterable, optional): A bounding box [left, top, right, bottom] to draw.
        color (tuple): Color for the landmarks.
        size (int): Circle radius for landmarks.
        
    Returns:
        numpy.ndarray: The image with drawn landmarks and optional bounding box.
    """
    # Create a copy of the original image.
    img = img_ori.copy()
    n = pts.shape[1]
    
    # Draw each landmark point.
    if n <= 106:
        for i in range(n):
            cv2.circle(img, (int(round(pts[0, i])), int(round(pts[1, i]))), size, color, -1)
    else:
        sep = 1
        for i in range(0, n, sep):
            cv2.circle(img, (int(round(pts[0, i])), int(round(pts[1, i]))), size, color, 1)
    
    # If a bounding box is provided, draw it.
    if box is not None:
        left, top, right, bottom = np.round(box).astype(np.int32)
        left_top = (left, top)
        right_top = (right, top)
        right_bottom = (right, bottom)
        left_bottom = (left, bottom)
        cv2.line(img, left_top, right_top, BLUE, 1, cv2.LINE_AA)
        cv2.line(img, right_top, right_bottom, BLUE, 1, cv2.LINE_AA)
        cv2.line(img, right_bottom, left_bottom, BLUE, 1, cv2.LINE_AA)
        cv2.line(img, left_bottom, left_top, BLUE, 1, cv2.LINE_AA)
    
    return img


def color_grid(ver_lst, ncolors=9):
    """
    Assigns colors from a preset palette to vertices, segmenting them into ncolors groups.
    
    Args:
        ver_lst (list): List containing vertex data.
        ncolors (int): Maximum number of colors to use (capped at 9).
        
    Returns:
        list: A list of colors assigned to each vertex.
    """
    # Limit ncolors to a maximum of 9.
    ncolors = min(ncolors, 9)
    
    # Predefined list of 9 RGB colors normalized to [0,1].
    preset_colors = [
        np.array([255.0, 0.0, 0.0]) / 255.0,    # Red
        np.array([0.0, 255.0, 0.0]) / 255.0,      # Green
        np.array([0.0, 0.0, 255.0]) / 255.0,      # Blue
        np.array([255.0, 255.0, 0.0]) / 255.0,    # Yellow
        np.array([0.0, 255.0, 255.0]) / 255.0,    # Cyan
        np.array([255.0, 0.0, 255.0]) / 255.0,    # Magenta
        np.array([255.0, 165.0, 0.0]) / 255.0,    # Orange
        np.array([128.0, 0.0, 128.0]) / 255.0,    # Purple
        np.array([0.0, 128.0, 128.0]) / 255.0     # Teal
    ]
    
    # Use only the first ncolors from the preset list.
    palette = preset_colors[:ncolors]
    
    # Determine the total number of vertices.
    ln = len(ver_lst[0][0])
    textures = [None] * ln

    # Calculate segment size to divide vertices evenly.
    segment_size = ln / ncolors

    # Assign a color to each vertex based on its segment.
    for i in range(ln):
        segment_index = int(i // segment_size)
        # Ensure the index is within palette range.
        if segment_index >= ncolors:
            segment_index = ncolors - 1
        textures[i] = palette[segment_index]

    return textures


def color_grid_ratios(ver_lst, ratio_list):
    """
    Assigns colors to vertices based on provided ratio weights.
    
    Args:
        ver_lst (list): List containing vertex data.
        ratio_list (list): List of ratios determining the color segmentation.
        
    Returns:
        list: A list of colors (BGR normalized) assigned to each vertex.
    """
    # Limit the number of colors to at most 9.
    ncolors = min(len(ratio_list), 9)
    ratio_list = ratio_list[:ncolors]
    
    # Normalize the ratio list so that their sum equals 1.
    total = sum(ratio_list)
    if total == 0:
        raise ValueError("Sum of ratio_list must be > 0")
    ratio_list = [r / total for r in ratio_list]
    
    # Predefined list of 9 colors in BGR normalized to [0,1].
    preset_colors = [
        np.array([0.0, 0.0, 255.0]) / 255.0,    # Red in BGR
        np.array([0.0, 255.0, 0.0]) / 255.0,      # Green
        np.array([255.0, 0.0, 0.0]) / 255.0,      # Blue in BGR
        np.array([0.0, 255.0, 255.0]) / 255.0,    # Yellow in BGR
        np.array([255.0, 255.0, 0.0]) / 255.0,    # Cyan in BGR
        np.array([255.0, 0.0, 255.0]) / 255.0,    # Magenta
        np.array([0.0, 165.0, 255.0]) / 255.0,    # Orange in BGR
        np.array([128.0, 0.0, 128.0]) / 255.0,    # Purple
        np.array([128.0, 128.0, 0.0]) / 255.0     # Teal in BGR
    ]
    
    # Use only the first ncolors from the preset list.
    palette = preset_colors[:ncolors]
    
    # Determine the total number of vertices.
    ln = len(ver_lst[0][0])
    textures = [None] * ln

    # Compute cumulative ratios.
    cumulative = np.cumsum(ratio_list)

    # Assign a color based on the vertex's normalized position.
    for i in range(ln):
        pos = (i + 1) / ln
        seg_idx = 0
        while seg_idx < ncolors and pos > cumulative[seg_idx]:
            seg_idx += 1
        if seg_idx >= ncolors:
            seg_idx = ncolors - 1
        textures[i] = palette[seg_idx]

    return textures


def color_grid_indices(ver_lst, cutoff_list):
    """
    Assigns colors to vertices based on cumulative index cutoffs.
    
    Args:
        ver_lst (list): List containing vertex data.
        cutoff_list (list): List of cutoff indices for segmenting vertices.
        
    Returns:
        list: A list of colors (BGR normalized) assigned to each vertex.
    """
    # Limit the number of segments to at most 9.
    nsegments = min(len(cutoff_list), 9)
    cutoff_list = cutoff_list[:nsegments]
    
    # Predefined list of 9 colors in BGR normalized to [0,1].
    preset_colors = [
        np.array([0.0, 0.0, 255.0]) / 255.0,    # Red in BGR
        np.array([0.0, 255.0, 0.0]) / 255.0,      # Green
        np.array([255.0, 0.0, 0.0]) / 255.0,      # Blue in BGR
        np.array([0.0, 255.0, 255.0]) / 255.0,    # Yellow in BGR
        np.array([255.0, 255.0, 0.0]) / 255.0,    # Cyan in BGR
        np.array([255.0, 0.0, 255.0]) / 255.0,    # Magenta
        np.array([0.0, 165.0, 255.0]) / 255.0,    # Orange in BGR
        np.array([128.0, 0.0, 128.0]) / 255.0,    # Purple
        np.array([128.0, 128.0, 0.0]) / 255.0     # Teal in BGR
    ]
    
    # Use only the first nsegments colors.
    palette = preset_colors[:nsegments]
    
    # Determine the total number of vertices.
    ln = len(ver_lst[0][0])
    
    # Compute cumulative boundaries from cutoff_list.
    cum_boundaries = np.cumsum(cutoff_list)
    textures = [None] * ln
    
    # Assign colors based on which cumulative boundary the vertex index falls under.
    for i in range(ln):
        seg_idx = np.searchsorted(cum_boundaries, i + 1, side='right')
        if seg_idx >= nsegments:
            seg_idx = nsegments - 1
        textures[i] = palette[seg_idx]
    
    return textures


def norm_vertices(vertices):
    """
    Normalizes vertex coordinates to a scaled range.
    
    Args:
        vertices (numpy.ndarray): Array of vertex coordinates.
        
    Returns:
        numpy.ndarray: Normalized vertices.
    """
    # Shift vertices so that the minimum is at zero.
    vertices -= vertices.min(0)[None, :]
    # Scale vertices to [0,1].
    vertices /= vertices.max()
    # Scale to a range of 2.
    vertices *= 2
    # Center the vertices.
    vertices -= vertices.max(0)[None, :] / 2
    return vertices


def convert_type(obj):
    """
    Converts the input object to a numpy array of type float32.
    
    Args:
        obj (tuple or list or numpy.ndarray): Input data.
        
    Returns:
        numpy.ndarray: Converted numpy array with shape (1, N) if the input is a tuple or list;
                       otherwise returns the original object.
    """
    if isinstance(obj, (tuple, list)):
        return np.array(obj, dtype=np.float32)[None, :]
    return obj


def compute_boundary_vertices(triangles):
    """
    Computes the boundary vertices given a list of triangles.
    
    Each triangle is represented by a list of three vertex indices. An edge is considered a boundary
    edge if it appears only once among all triangles.
    
    Args:
        triangles (iterable): An iterable of triangles (each triangle is an iterable of 3 indices).
        
    Returns:
        numpy.ndarray: Array of unique boundary vertex indices (dtype=int32).
    """
    # Build a dictionary to count how many times each edge appears.
    edge_count = {}
    for tri in triangles:
        # For each triangle, consider its three edges.
        edges = [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]
        for (a, b) in edges:
            # Sort the edge so that (a,b) and (b,a) are identical.
            edge = tuple(sorted((a, b)))
            edge_count[edge] = edge_count.get(edge, 0) + 1
    # Identify boundary edges that appear only once.
    boundary_edges = [edge for edge, count in edge_count.items() if count == 1]
    # Collect unique boundary vertices from these edges.
    boundary_vertices = set()
    for edge in boundary_edges:
        boundary_vertices.update(edge)
    return np.array(list(boundary_vertices), dtype=np.int32)


def compute_distance_to_boundary(vertices, boundary_indices):
    """
    Computes the Euclidean distance from each vertex to the nearest boundary vertex.
    
    Args:
        vertices (numpy.ndarray): An (N, 3) array of vertex coordinates.
        boundary_indices (iterable): Indices of vertices that are on the boundary.
        
    Returns:
        numpy.ndarray: A 1D array of distances (length N) for each vertex.
    """
    # Extract boundary vertices from the full vertex set.
    boundary_vertices = vertices[boundary_indices]  # shape: (num_boundary, 3)
    # Compute differences between each vertex and all boundary vertices using broadcasting.
    diff = vertices[:, None, :] - boundary_vertices[None, :, :]  # shape: (N, B, 3)
    # Calculate Euclidean distances.
    dists = np.linalg.norm(diff, axis=2)  # shape: (N, B)
    # Get the minimum distance for each vertex.
    min_dists = np.min(dists, axis=1)     # shape: (N,)
    return min_dists


def compute_distance_to_boundary_kdtree(vertices, boundary_indices):
    """
    Computes the Euclidean distance from each vertex to the nearest boundary vertex using a KD-tree.
    
    Args:
        vertices (numpy.ndarray): An (N, 3) array of vertex coordinates.
        boundary_indices (iterable): Indices of boundary vertices.
        
    Returns:
        numpy.ndarray: A 1D array of distances (length N) for each vertex.
    """
    # Extract boundary vertices.
    boundary_vertices = vertices[boundary_indices]
    # Build a KD-tree for fast nearest-neighbor lookup.
    tree = cKDTree(boundary_vertices)
    # Query the KD-tree for the distance to the nearest boundary vertex.
    dists, _ = tree.query(vertices)
    return dists


def compute_alpha_from_distance(distances, sigma):
    """
    Computes an alpha (transparency) value for each vertex based on its distance using a Gaussian falloff.
    
    Args:
        distances (numpy.ndarray): Array of distances for each vertex.
        sigma (float): The sigma parameter for the Gaussian function.
        
    Returns:
        numpy.ndarray: Array of alpha values (clipped between 0 and 1).
    """
    # Compute Gaussian falloff for each distance.
    alphas = 1 - np.exp(- (distances ** 2) / (2 * sigma ** 2))
    # Clip the values to ensure they are between 0 and 1.
    return np.clip(alphas, 0, 1)


def select_target_faces(img, boxes):
    """
    Displays the first frame with detected faces and allows the user to select multiple target faces.
    
    Args:
        img (numpy.ndarray): The input BGR image where faces are detected.
        boxes (list): List of bounding boxes for detected faces.
        
    Returns:
        list: List of indices of selected faces.
    """
    # Create a copy of the image for drawing.
    img_copy = img.copy()
    print("Boxes:", boxes)
    print("Boxes Shape:", np.shape(boxes))  # Debug: Check the shape of boxes
    
    # Draw bounding boxes and print their coordinates.
    for i, box in enumerate(boxes):
        x_min, y_min, x_max, y_max = box[:4]
        print(f'Face {i}: ({x_min},{y_min}) , ({x_max},{y_max})')
        # Uncomment below lines to draw rectangles and labels if needed.
        # cv2.rectangle(img_copy, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        # cv2.putText(img_copy, f"{i}", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Display the image with labeled faces (optional).
    # cv2.imshow("Select Target Faces (Enter indices separated by commas)", img_copy)
    # cv2.waitKey(1)
    
    # Prompt the user to select target faces.
    while True:
        try:
            # For automated testing, "all" is chosen. Switch to input for face selection.
            input_str = "all"
            #input_str = input(f'Please select face index(es) for swapping (type {'all'} for all):')
            if input_str.lower() == "all":
                selected_indices = list(range(len(boxes)))  # Select all faces.
                break
            else:
                selected_indices = [int(idx) for idx in input_str.split(",") if 0 <= int(idx) < len(boxes)]
                if selected_indices:
                    break
                else:
                    print("Invalid indices. Please enter valid numbers separated by commas.")
        except ValueError:
            print("Invalid input. Please enter numbers separated by commas.")
    
    cv2.destroyAllWindows()
    return selected_indices


def load_image_params(img, tddfa, face_boxes, dense_flag=True):
    """
    Loads and computes image parameters for face alignment/detection.
    
    Args:
        img (numpy.ndarray): The input image.
        tddfa (object): The face alignment/detection model.
        face_boxes (function): Function to detect face boxes.
        dense_flag (bool): Flag to determine whether to compute dense 3D reconstructions.
        
    Returns:
        tuple: (img_param_lst, img_roi_box_lst, img_ver_lst)
    """
    # Detect face boxes in the image.
    boxes = face_boxes(img)
    n = len(boxes)
    if n == 0:
        print('No face detected, exit')
        sys.exit(-1)
    print(f'Detect {n} faces')
    
    # Compute face parameters and ROI boxes using tddfa.
    img_param_lst, img_roi_box_lst = tddfa(img, boxes)
    # Reconstruct vertex list based on computed parameters.
    img_ver_lst = tddfa.recon_vers(img_param_lst, img_roi_box_lst, dense_flag=dense_flag)
    
    return img_param_lst, img_roi_box_lst, img_ver_lst


def load_video_params(video, tddfa, face_boxes, dense_flag=True):
    """
    Processes a video frame-by-frame to compute face parameters for each frame.
    
    Args:
        video (iterable): An iterable of video frames.
        tddfa (object): The face alignment/detection model.
        face_boxes (function): Function to detect face boxes in a frame.
        dense_flag (bool): Flag for dense 3D reconstructions.
        
    Returns:
        list: A list of dictionaries containing per-frame face data.
    """
    pre_ver_lst = None
    video_data = []
    
    # Process each frame in the video.
    for i, frame in tqdm(enumerate(video)):
        # Convert RGB frame to BGR.
        frame_bgr = frame[..., ::-1]
        
        if i == 0:
            # Detect faces in the first frame.
            boxes = face_boxes(frame_bgr)
            print(f"{len(boxes)} faces found in first frame")
            if len(boxes) == 0:
                print("Warning: No faces detected in the first frame!")
                return []  # Exit if no faces found.
            # Select target faces.
            selected_indices = select_target_faces(frame_bgr, boxes)
            selected_boxes = [boxes[idx] for idx in selected_indices]
            # Compute parameters for selected faces.
            param_lst, roi_box_lst = tddfa(frame_bgr, selected_boxes)
            ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)
        else:
            param_lst, roi_box_lst = [], []
            # Track each selected face using previous vertex estimates.
            for pre_ver in pre_ver_lst:
                param, roi_box = tddfa(frame_bgr, [pre_ver], crop_policy='landmark')
                param_lst.append(param[0])
                roi_box_lst.append(roi_box[0])
            # If tracking fails, re-detect faces.
            new_boxes = face_boxes(frame_bgr) if len(roi_box_lst) == 0 else []
            if new_boxes:
                print("Tracking failed, re-detecting selected faces...")
                selected_boxes = [new_boxes[idx] for idx in selected_indices]
                param_lst, roi_box_lst = tddfa(frame_bgr, selected_boxes)
            ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)
        
        pre_ver_lst = ver_lst.copy()
        # Store per-frame data.
        frame_data = {
            'frame_no': i,
            'param_lst': param_lst,
            'roi_box_lst': roi_box_lst,
            'ver_lst': ver_lst
        }
        video_data.append(frame_data)
    
    return video_data


def swap_faces_3D(video_data, img_params_lst, tddfa, dense_flag=True):
    """
    Creates a modified version of the video data by swapping face parameters so that the source face
    follows the pose and expressions of the target face.
    
    Args:
        video_data (list): List of dictionaries containing per-frame face data.
        img_params_lst (list): Source face parameter list from the source image.
        tddfa (object): The face alignment/detection model.
        dense_flag (bool): Flag for dense 3D reconstruction.
        
    Returns:
        list: Modified video data with swapped face parameters.
    """
    swap_video_data = []
    
    # Process each frame in the video.
    for frame_data in video_data:
        frame_no = frame_data['frame_no']
        swap_frame_data = {'frame_no': frame_no, 'param_lst': [], 'roi_box_lst': [], 'ver_lst': []}
        
        # Loop over each target face in the frame.
        for target_face_idx in range(len(frame_data['param_lst'])):
            target_param_lst = frame_data['param_lst'][target_face_idx]
            target_roi_box_lst = frame_data['roi_box_lst'][target_face_idx]
            
            # Create new parameters by combining source face shape with target's pose and expression.
            swap_param_lst = np.array(img_params_lst[0]).copy()
            swap_param_lst[:12] = target_param_lst[:12]  # Swap pose.
            swap_param_lst[52:62] = target_param_lst[52:62]  # Swap expression.
            
            swap_frame_data['param_lst'].append(swap_param_lst)
            swap_frame_data['roi_box_lst'].append(target_roi_box_lst)
            
            # Reconstruct the vertices using the new swapped parameters.
            swap_ver_lst = tddfa.recon_vers([swap_param_lst], [target_roi_box_lst], dense_flag=dense_flag)
            swap_frame_data['ver_lst'].append(swap_ver_lst[0])
        
        # Optionally store texture information if needed.
        # swap_frame_data['texture'] = source_tex.copy()
        
        swap_video_data.append(swap_frame_data)
    
    return swap_video_data



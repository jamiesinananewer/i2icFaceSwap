# cython: language_level=3

import numpy as np
cimport numpy as np
cimport cython
from cpython.object cimport PyObject
from libcpp.vector cimport vector
from libcpp cimport bool


# Declare external c++ functions from header script
cdef extern from "swap.h":

    cdef struct ColorStats:
        vector[float] mean
        vector[float] stdev

    ColorStats _compute_color_statistics(const unsigned char* image, const bool* mask, 
                                    int h, int w, int c);    
    
    ColorStats _color_stats(const float* image, int H, int W, int C, const bool* mask)

    void _color_correction(const float* tgt_img, const float* src_img, const bool* mask,
                        int h, int w,  int c, float epsilon, float* corr_img)

    PyObject* py_compute_boundary_vertices(const int* triangles, int ntri)

    PyObject* py_compute_alpha_from_distance (const vector[float]& distances, float sigma)

    void _rasterize(
    unsigned char *image, float *vertices, int *triangles, float *colors, float *depth_buffer,
    int ntri, int h, int w, int c, float alpha, bool reverse)

    void _rasterize_and_correct(
        unsigned char *image,       # Background image buffer [0,255] (input/output)
        float* vertices,            # Vertex array (size: 3 * n_vertices)
        int* triangles,             # Triangle indices (size: 3 * ntri)
        float* colors,              # Per-vertex colors (size: 3 * n_vertices)
        float* depth_buffer,        # Depth buffer (size: h*w)
        float* vertex_alphas,       # Per-vertex alpha values (size: n_vertices)
        int ntri,                   # Number of triangles
        int h, int w, int c,        # Image dimensions (c must equal 3)
        float max_alpha,            # Maximum alpha multiplier for blending
        bool reverse,                # If true, flip vertical axis
        bool sigma_correct
    )

    PyObject* py_compute_distance_to_boundary_from_arrays(const float* vertices_data, int num_vertices,
                                                        const int* boundary_data, int num_boundary)

    void _rasterize_alpha_blur(unsigned char *image, float *vertices, int *triangles, float *colors, float *depth_buffer, 
                            float *vertex_alphas, int ntri, int h, int w, int c, float max_alpha, bool reverse)

    void _get_normal(float *ver_normal, float *vertices, int *triangles, int nver, int ntri)




# Python functions that call C++ functions


def compute_boundary_vertices(np.ndarray[np.int32_t, ndim=1] triangles):
    """
    Accepts a 1D numpy array of triangles (flattened, with 3 indices per triangle)
    and returns a Python list of boundary vertex indices.
    """
    cdef int ntri = triangles.shape[0] // 3
    return <object>py_compute_boundary_vertices(<const int*>triangles.data, ntri)

def compute_distance_to_boundary(np.ndarray[np.float32_t, ndim=2] vertices,
                                    np.ndarray[np.int32_t, ndim=1] boundary_indices):
    
    cdef int num_vertices = vertices.shape[0]
    cdef int num_boundary = boundary_indices.shape[0]

    return <object>py_compute_distance_to_boundary_from_arrays(
        <const float*>vertices.data, num_vertices,
        <const int*>boundary_indices.data, num_boundary)

def compute_alpha_from_distance(np.ndarray[np.float32_t, ndim=1] distances, float sigma):

    cdef int num_dists = distances.shape[0]

    #convert raw NumPy data to c++ vector<float>

    cdef vector[float] vec_dists = vector[float]()
    cdef int i
    cdef const float* data_ptr = <const float*>distances.data

    #populate c++ vector by iterating through distances
    for i in range(num_dists):
        vec_dists.push_back(data_ptr[i])
    
    #cdef vector[float] vec_dists = vector[float](<const float*> distances.data, <const float*> distances.data + num_dists)

    return <object>py_compute_alpha_from_distance(vec_dists, sigma)

def color_stats(np.ndarray[float, ndim=3] image not None,
                np.ndarray[bool, ndim=2] mask not None):
    """
    Calls _color_stats_ptr with the raw data pointers from image and mask.
    Returns two 1D numpy arrays for mean and standard deviation.
    """
    cdef int H = image.shape[0]
    cdef int W = image.shape[1]
    cdef int C = image.shape[2]
    if C != 3:
         raise ValueError("Image must have exactly 3 channels (RGB)")
    if mask.shape[0] != H or mask.shape[1] != W:
         raise ValueError("Mask dimensions must match image dimensions")

    cdef ColorStats cs = _color_stats(<float*> image.data, H, W, C, <bool*> mask.data)

    # Allocate output numpy arrays (assume size is 3 for both mean and stdev)
    cdef np.ndarray[float, ndim=1] mean_arr = np.empty(cs.mean.size(), dtype=np.float32)
    cdef np.ndarray[float, ndim=1] stdev_arr = np.empty(cs.stdev.size(), dtype=np.float32)
    cdef float* mean_ptr = <float*> mean_arr.data
    cdef float* stdev_ptr = <float*> stdev_arr.data
    cdef size_t idx, n

    n = cs.mean.size()
    for idx in range(n):
         mean_ptr[idx] = cs.mean[idx]
    n = cs.stdev.size()
    for idx in range(n):
         stdev_ptr[idx] = cs.stdev[idx]

    return mean_arr, stdev_arr


def color_correction(np.ndarray[float, ndim=3] tgt_img not None,
                     np.ndarray[float, ndim=3] src_img not None,
                     np.ndarray[bool, ndim=2] mask not None,
                     float epsilon=1e-6):
    """
    Performs color correction on tgt_img using src_img and mask.
    Both tgt_img and src_img must be 3D NumPy arrays (H x W x 3) with float values in [0, 1].
    Mask must be a 2D boolean array of shape (H, W).
    Returns a new corrected image as a NumPy array.
    """
    cdef int h = tgt_img.shape[0]
    cdef int w = tgt_img.shape[1]
    cdef int c = tgt_img.shape[2]
    if c != 3:
         raise ValueError("Image must have exactly 3 channels (RGB)")
    if mask.shape[0] != h or mask.shape[1] != w:
         raise ValueError("Mask dimensions must match image dimensions")
    
    # Allocate an output array (copy of tgt_img shape)
    cdef np.ndarray[float, ndim=3] corr_img = np.empty_like(tgt_img)
    
    _color_correction(<float*>tgt_img.data, <float*>src_img.data, <bool*>mask.data,
                           h, w, c, epsilon, <float*>corr_img.data)
    
    return corr_img

    

def compute_color_statistics(np.ndarray[const unsigned char, ndim = 3, mode = "c"] image not None,
                        np.ndarray[const bool, ndim =2, mode = "c"] mask not None,
                        int h, int w, int c):

    cdef ColorStats cs = _compute_color_statistics(
        <const unsigned char*> np.PyArray_DATA(image),
        <const bool*> np.PyArray_DATA(mask),
        h, w, c
    )

    #convert C++ vector[float] to python list for mean and stdev
    cdef list py_mean = []
    cdef list py_stdev = []
    cdef int i

    #assume cs.mean.size() & cs.stdev.size() give sizes of vectors
    for i in range(cs.mean.size()):
        py_mean.append(cs.mean[i])

    for i in range(cs.stdev.size()):
        py_stdev.append(cs.stdev[i])

    return py_mean, py_stdev


@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def rasterize_color_correct(np.ndarray[unsigned char, ndim=3, mode = "c"] image not None,
              np.ndarray[float, ndim=2, mode = "c"] vertices not None,
              np.ndarray[int, ndim=2, mode="c"] triangles not None,
              np.ndarray[float, ndim=2, mode = "c"] colors not None,
              np.ndarray[float, ndim=2, mode = "c"] depth_buffer not None,
              np.ndarray[float, ndim = 1, mode = "c"] vertex_alphas not None,
              int ntri, int h, int w, int c, float max_alpha = 1, bool reverse = False, bool sigma_correct = False
              ):
    _rasterize_and_correct(
        <unsigned char*> np.PyArray_DATA(image), <float*> np.PyArray_DATA(vertices),
        <int*> np.PyArray_DATA(triangles),
        <float*> np.PyArray_DATA(colors),
        <float*> np.PyArray_DATA(depth_buffer),
        <float*> np.PyArray_DATA(vertex_alphas),
        ntri, h, w, c, max_alpha, reverse, sigma_correct)


@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def rasterize_alpha(np.ndarray[unsigned char, ndim=3, mode = "c"] image not None,
              np.ndarray[float, ndim=2, mode = "c"] vertices not None,
              np.ndarray[int, ndim=2, mode="c"] triangles not None,
              np.ndarray[float, ndim=2, mode = "c"] colors not None,
              np.ndarray[float, ndim=2, mode = "c"] depth_buffer not None,
              np.ndarray[float, ndim = 1, mode = "c"] vertex_alphas not None,
              int ntri, int h, int w, int c, float max_alpha = 1, bool reverse = False
              ):
    _rasterize_alpha_blur(
        <unsigned char*> np.PyArray_DATA(image), <float*> np.PyArray_DATA(vertices),
        <int*> np.PyArray_DATA(triangles),
        <float*> np.PyArray_DATA(colors),
        <float*> np.PyArray_DATA(depth_buffer),
        <float*> np.PyArray_DATA(vertex_alphas),
        ntri, h, w, c, max_alpha, reverse)

@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def rasterize(np.ndarray[unsigned char, ndim=3, mode = "c"] image not None,
              np.ndarray[float, ndim=2, mode = "c"] vertices not None,
              np.ndarray[int, ndim=2, mode="c"] triangles not None,
              np.ndarray[float, ndim=2, mode = "c"] colors not None,
              np.ndarray[float, ndim=2, mode = "c"] depth_buffer not None,
              int ntri, int h, int w, int c, float alpha = 1, bool reverse = False
              ):
    _rasterize(
        <unsigned char*> np.PyArray_DATA(image), <float*> np.PyArray_DATA(vertices),
        <int*> np.PyArray_DATA(triangles),
        <float*> np.PyArray_DATA(colors),
        <float*> np.PyArray_DATA(depth_buffer),
        ntri, h, w, c, alpha, reverse)



def rasterize_alpha_blu(np.ndarray[np.uint8_t, ndim=3] image,
                         np.ndarray[np.float32_t, ndim=2] vertices,
                         np.ndarray[np.int32_t, ndim=1] triangles,
                         np.ndarray[np.float32_t, ndim=2] colors,
                         np.ndarray[np.float32_t, ndim=2] depth_buffer,
                         np.ndarray[np.float32_t, ndim=1] vertex_alphas,
                         float max_alpha,
                         bint reverse):
    """
    Renders the 3D face model with per-vertex alpha blending.
    
    Parameters:
      image: output image (height x width x channels) as uint8
      vertices: 2D array of vertices with shape (n_vertices, 3) (x, y, depth)
      triangles: 1D flattened array of triangle indices (3 per triangle)
      colors: 2D array of vertex colors with shape (n_vertices, c)
      depth_buffer: 2D array with shape (height, width) holding depth values
      vertex_alphas: 1D array of per-vertex alpha values
      max_alpha: maximum alpha multiplier (scales the computed alpha)
      reverse: if True, flips the image vertically
    """
    cdef int ntri = triangles.shape[0] // 3
    cdef int h = image.shape[0]
    cdef int w = image.shape[1]
    cdef int c = image.shape[2]
    
    # Get raw pointers to the data
    cdef unsigned char* image_ptr = <unsigned char*> image.data
    cdef float* vertices_ptr = <float*> vertices.data
    cdef int* triangles_ptr = <int*> triangles.data
    cdef float* colors_ptr = <float*> colors.data
    cdef float* depth_buffer_ptr = <float*> depth_buffer.data
    cdef float* vertex_alphas_ptr = <float*> vertex_alphas.data

    # Call the external C++ function declared in our header
    _rasterize_alpha_blur(image_ptr, vertices_ptr, triangles_ptr, colors_ptr,
                         depth_buffer_ptr, vertex_alphas_ptr, ntri, h, w, c,
                         max_alpha, reverse)
    return None


def get_normal(np.ndarray[float, ndim=2, mode = "c"] ver_normal not None,
                   np.ndarray[float, ndim=2, mode = "c"] vertices not None,
                   np.ndarray[int, ndim=2, mode="c"] triangles not None,
                   int nver, int ntri):
    _get_normal(
        <float*> np.PyArray_DATA(ver_normal), <float*> np.PyArray_DATA(vertices), <int*> np.PyArray_DATA(triangles),
        nver, ntri)
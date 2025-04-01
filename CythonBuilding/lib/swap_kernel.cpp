#include "swap.h"
#include <iostream>
#include <vector>       //use dynamic arrays
#include <array>
#include <cmath>
#include <cassert>
#include <limits>
#include <unordered_map>    //provides hash tables for dictionaries
#include <unordered_set>    //set data structure to store unique items
#include <utility>          //useful functions
#include <algorithm>        //useful functions
#include <stdexcept>
#include <Python.h>         //Python API functions for Cython







////////////////////////////////////
////////////RASTERIZATION///////////
////////////////////////////////////

//utility function: finding mean intensity and stdev for color channels in bounding region of image
/*struct ColorStats {
    std::vector<float> mean;    //mean per color channel
    std::vector<float> stdev;   //stdev per color channel
};*/

ColorStats _color_stats(const float* image, int H, int W, int C, const bool* mask) {
    if (C != 3)
         throw std::invalid_argument("Image must have exactly 3 channels (RGB)");

    ColorStats cs;
    cs.mean = std::vector<float>(3, 0.0f);
    cs.stdev = std::vector<float>(3, 0.0f);
    int count = 0;

    // First pass: accumulate the sums for each channel.
    for (int i = 0; i < H; ++i) {
         for (int j = 0; j < W; ++j) {
              // The mask is stored in row-major order: index = i*W + j.
              if (mask[i * W + j]) {
                  ++count;
                  // For the image, pixels are stored in row-major order, and each pixel has C channels.
                  // Calculate the starting index of the pixel.
                  int idx = (i * W + j) * C;
                  cs.mean[0] += image[idx];     // Red channel
                  cs.mean[1] += image[idx + 1];   // Green channel
                  cs.mean[2] += image[idx + 2];   // Blue channel
              }
         }
    }
    
    if (count == 0)
         throw std::runtime_error("Mask has no true values.");

    // Compute the mean for each channel.
    cs.mean[0] /= count;
    cs.mean[1] /= count;
    cs.mean[2] /= count;

    // Second pass: accumulate variance for each channel.
    std::vector<float> variance(3, 0.0f);
    for (int i = 0; i < H; ++i) {
         for (int j = 0; j < W; ++j) {
              if (mask[i * W + j]) {
                  int idx = (i * W + j) * C;
                  float diff0 = image[idx] - cs.mean[0];
                  float diff1 = image[idx + 1] - cs.mean[1];
                  float diff2 = image[idx + 2] - cs.mean[2];
                  variance[0] += diff0 * diff0;
                  variance[1] += diff1 * diff1;
                  variance[2] += diff2 * diff2;
              }
         }
    }
    
    cs.stdev[0] = std::sqrt(variance[0] / count);
    cs.stdev[1] = std::sqrt(variance[1] / count);
    cs.stdev[2] = std::sqrt(variance[2] / count);

    return cs;
}

void _color_correction(const float* tgt_img, const float* src_img, const bool* mask,
                        int h, int w,  int c, float epsilon, float* corr_img) {
    if (c != 3) {
        throw std::invalid_argument("Image must have exactly 3 channels (RGB)");
    }

    //compute stats for target and source images
    ColorStats tgt_stats = _color_stats(tgt_img, h, w, c, mask);
    ColorStats src_stats = _color_stats(src_img, h, w, c, mask);

    

    //copy target img to corr_img
    int total = h*w*c;
    for (int idx = 0; idx < total; ++idx) {
        corr_img[idx] = tgt_img[idx];
    }

    //apply color correction to each pixel in image marked True in mask
    //new_value = (tgt_stdev / src_stdev)* src_value + (tgt_mean - (tgt_stdev/src_stdev)*src_mean)
    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
            if (mask[i*w + j]) {
                int base = (i*w + j) * c;
                for (int c = 0; c < 3; ++c) {
                    float ratio;
                    if (src_stats.stdev[c] < epsilon) {
                        ratio = 1.0f;
                    }
                    else if (src_stats.stdev[c] >= epsilon) {
                        ratio = tgt_stats.stdev[c] / src_stats.stdev[c];
                    }

                    float new_val = ratio*src_img[base + c] + 
                                    (tgt_stats.mean[c] - ratio*src_stats.mean[c]);
                    
                    //clip new value to [0,1]
                    if (new_val < 0.0f) {
                        new_val = 0.0f;
                    }
                    else if (new_val > 1.0f) {
                        new_val = 1.0f;
                    }

                    corr_img[base + c] = new_val;
                }
            }
        }
    }
}

ColorStats _compute_color_statistics(const unsigned char* image, const bool* mask, 
                                    int h, int w, int c) {
    
    ColorStats stats;
    stats.mean.resize(c, 0.0f);
    stats.stdev.resize(c, 0.0f);
    int count = 0;

    //first pass: compute sums, converting to float as needed
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            
            int idx = i*w + j;

            if (mask[idx]) {
                
                count++;
                
                for (int k = 0; k < c; k++) {
                    int pixel_idx = idx*c + k;

                    //convert unsigned char to float
                    stats.mean[k] += image[pixel_idx] / 255.0f;
                }
            }
        }
    }

    if (count == 0) return stats; //don't divide by 0

    //compute mean for each channel
    for (int k = 0; k < c; k++) {
        stats.mean[k] /= count;

    }

    //second pass: compute variance
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            
            int idx = i*w +j;

            if (mask[idx]) {
                for (int k = 0; k <c; k++) {
                    int pixel_idx = idx*c + k;
                    float pixel_val = image[pixel_idx] / 255.0f;
                    float diff = pixel_val - stats.mean[k];
                    stats.stdev[k] += diff*diff;
                }
            }
        }
    }

    //compute stdev for each colour channel
    for (int k = 0; k < c; k++) {
        stats.stdev[k] = std::sqrt(stats.stdev[k] / count);
    }

    return stats;
}





//utility function: find barycentric weight for pixels relative to triangle points
void get_point_weight(float *weight, Point p, Point p0, Point p1, Point p2) {
    // vectors
    Point v0, v1, v2;
    v0 = p2 - p0;
    v1 = p1 - p0;
    v2 = p - p0;

    // dot products
    float dot00 = v0.dot(v0); //v0.x * v0.x + v0.y * v0.y //np.dot(v0.T, v0)
    float dot01 = v0.dot(v1); //v0.x * v1.x + v0.y * v1.y //np.dot(v0.T, v1)
    float dot02 = v0.dot(v2); //v0.x * v2.x + v0.y * v2.y //np.dot(v0.T, v2)
    float dot11 = v1.dot(v1); //v1.x * v1.x + v1.y * v1.y //np.dot(v1.T, v1)
    float dot12 = v1.dot(v2); //v1.x * v2.x + v1.y * v2.y//np.dot(v1.T, v2)

    // barycentric coordinates
    float inverDeno;
    if (dot00 * dot11 - dot01 * dot01 == 0)
        inverDeno = 0;
    else
        inverDeno = 1 / (dot00 * dot11 - dot01 * dot01);

    float u = (dot11 * dot02 - dot01 * dot12) * inverDeno;
    float v = (dot00 * dot12 - dot01 * dot02) * inverDeno;

    // weight
    weight[0] = 1 - u - v;
    weight[1] = v;
    weight[2] = u;
}



void _rasterize_and_correct(
    unsigned char *image,       // Background image buffer [0,255] (input/output)
    float* vertices,            // Vertex array (size: 3 * n_vertices)
    int* triangles,             // Triangle indices (size: 3 * ntri)
    float* colors,              // Per-vertex colors (size: 3 * n_vertices)
    float* depth_buffer,        // Depth buffer (size: h*w)
    float* vertex_alphas,       // Per-vertex alpha values (size: n_vertices)
    int ntri,                   // Number of triangles
    int h, int w, int c,        // Image dimensions (c must equal 3)
    float max_alpha,            // Maximum alpha multiplier for blending
    bool reverse,               // If true, flip vertical axis
    bool sigma_correct          // If true, correct using stdev; if false, use only mean differences
) {
    // Internal default parameters.
    const float epsilon = 1e-6f;
    const float alpha_threshold = 0.1f;  // Adjust as needed

    int total = h * w * c;
    int npixels = h * w;

    // Allocate temporary buffers.
    float* target_img = new float[total];   // Background image converted to [0,1]
    float* src_raster  = new float[total];    // Face model colors (rasterized)
    float* alpha_map   = new float[npixels];  // Interpolated per-pixel alpha values
    // Initialize src_raster and alpha_map.
    for (int i = 0; i < total; ++i)
         src_raster[i] = 0.0f;
    for (int i = 0; i < npixels; ++i)
         alpha_map[i] = 0.0f;

    // Convert input image (unsigned char [0,255]) to float [0,1].
    for (int i = 0; i < total; ++i)
         target_img[i] = image[i] / 255.0f;

    // Allocate a mask (bool array) for pixels where the face model is applied.
    bool* mask = new bool[npixels];
    for (int i = 0; i < npixels; ++i)
         mask[i] = false;

    // Rasterize the face model.
    for (int t = 0; t < ntri; ++t) {
         int tri_p0_ind = triangles[3 * t];
         int tri_p1_ind = triangles[3 * t + 1];
         int tri_p2_ind = triangles[3 * t + 2];

         Point p0, p1, p2;
         p0.x = vertices[3 * tri_p0_ind];
         p0.y = vertices[3 * tri_p0_ind + 1];
         float p0_depth = vertices[3 * tri_p0_ind + 2];

         p1.x = vertices[3 * tri_p1_ind];
         p1.y = vertices[3 * tri_p1_ind + 1];
         float p1_depth = vertices[3 * tri_p1_ind + 2];

         p2.x = vertices[3 * tri_p2_ind];
         p2.y = vertices[3 * tri_p2_ind + 1];
         float p2_depth = vertices[3 * tri_p2_ind + 2];

         int x_min = std::max((int)ceil(std::min(p0.x, std::min(p1.x, p2.x))), 0);
         int x_max = std::min((int)floor(std::max(p0.x, std::max(p1.x, p2.x))), w - 1);
         int y_min = std::max((int)ceil(std::min(p0.y, std::min(p1.y, p2.y))), 0);
         int y_max = std::min((int)floor(std::max(p0.y, std::max(p1.y, p2.y))), h - 1);
         if (x_max < x_min || y_max < y_min)
              continue;
              
         for (int y = y_min; y <= y_max; ++y) {
              for (int x = x_min; x <= x_max; ++x) {
                   Point p; p.x = (float)x; p.y = (float)y;
                   float weight[3];
                   get_point_weight(weight, p, p0, p1, p2);
                   if (weight[0] > 0 && weight[1] > 0 && weight[2] > 0) {
                        float p_depth = weight[0] * p0_depth + weight[1] * p1_depth + weight[2] * p2_depth;
                        int pix_index = y * w + x;
                        if (p_depth > depth_buffer[pix_index]) {
                             float interp_alpha = weight[0] * vertex_alphas[tri_p0_ind] +
                                                  weight[1] * vertex_alphas[tri_p1_ind] +
                                                  weight[2] * vertex_alphas[tri_p2_ind];
                             interp_alpha *= max_alpha;
                             interp_alpha = std::min(1.0f, std::max(0.0f, interp_alpha));
                             
                             alpha_map[pix_index] = interp_alpha;
                             if (interp_alpha >= alpha_threshold)
                                  mask[pix_index] = true;
                                  
                             int base = pix_index * c;
                             for (int ch = 0; ch < c; ++ch) {
                                  float p0_color = colors[c * tri_p0_ind + ch];
                                  float p1_color = colors[c * tri_p1_ind + ch];
                                  float p2_color = colors[c * tri_p2_ind + ch];
                                  float face_color = weight[0] * p0_color + weight[1] * p1_color + weight[2] * p2_color;
                                  // Record the face model color.
                                  src_raster[base + ch] = face_color;
                             }
                             depth_buffer[pix_index] = p_depth;
                        }
                   }
              }
         }
    }

    // Now perform color correction.
    // We allocate an output corrected image buffer.
    float* corr_img = new float[total];
    if (sigma_correct) {
         // Use the existing full color correction.
         _color_correction(target_img, src_raster, mask, h, w, c, epsilon, corr_img);
    }
    else {
         // Compute only mean correction.
         // Compute per-channel means for target and source over masked pixels.
         ColorStats tgt_stats = _color_stats(target_img, h, w, c, mask);
         ColorStats src_stats = _color_stats(src_raster, h, w, c, mask);
         for (int i = 0; i < npixels; ++i) {
              int base = i * c;
              if (mask[i]) {
                   for (int ch = 0; ch < c; ++ch) {
                        // Instead of scaling by stdev ratio, we just shift by the mean difference.
                        corr_img[base + ch] = src_raster[base + ch] + (tgt_stats.mean[ch] - src_stats.mean[ch]);
                   }
              }
              else {
                   for (int ch = 0; ch < c; ++ch) {
                        corr_img[base + ch] = target_img[base + ch];
                   }
              }
         }
    }

    // Final blending:
    // For each pixel in the mask, blend the background with the corrected face color using the per-pixel alpha.
    for (int i = 0; i < npixels; ++i) {
         if (mask[i]) {
              int base = i * c;
              for (int ch = 0; ch < c; ++ch) {
                   target_img[base + ch] = (1 - alpha_map[i]) * target_img[base + ch] + alpha_map[i] * corr_img[base + ch];
              }
         }
    }

    // Convert target_img back to uint8 [0,255] and update the output image.
    for (int i = 0; i < total; ++i) {
         image[i] = (unsigned char)(target_img[i] * 255);
    }

    // Clean up temporary buffers.
    delete[] target_img;
    delete[] src_raster;
    delete[] alpha_map;
    delete[] mask;
    delete[] corr_img;
}

void _rasterize_alpha_blur(
    unsigned char *image,       // output image buffer
    float *vertices,            // vertices array
    int *triangles,             // triangle vertex indices aRRAY
    float *colors,              // per vertex colour array
    float *depth_buffer,        // depth buffer for z-buffering
    float *vertex_alphas,       // per vertex alpha values for blurring
    int ntri,                   // number of triangles
    int h, int w, int c,        // image height, width, number of colour channels
    float max_alpha,             // maximum alpha multiplier (for alpha blending)
    bool reverse) {             // flag for vertical inversion of image coordinates
    
    //loop encounters and temp variables
    int x, y, k;
    int tri_p0_ind, tri_p1_ind, tri_p2_ind;
    Point p0, p1, p2, p;
    int x_min, x_max, y_min, y_max;
    float p_depth, p0_depth, p1_depth, p2_depth;
    float p_color, p0_color, p1_color, p2_color;
    float weight[3];

    //loop over every triangle
    for (int i = 0; i < ntri; i++) {
        
        //extract vertex indices for the triangle
        tri_p0_ind = triangles[3*i];
        tri_p1_ind = triangles[3*i + 1];
        tri_p2_ind = triangles[3*i + 2];

        //get vertex positions (x, y) and depth values from the vetices array
        p0.x = vertices[3*tri_p0_ind];
        p0.y = vertices[3*tri_p0_ind + 1];
        p0_depth = vertices[3*tri_p0_ind + 2]; 

        p1.x = vertices[3*tri_p1_ind];
        p1.y = vertices[3*tri_p1_ind + 1];
        p1_depth = vertices[3*tri_p1_ind + 2]; 

        p2.x = vertices[3*tri_p2_ind];
        p2.y = vertices[3*tri_p2_ind + 1];
        p2_depth = vertices[3*tri_p2_ind + 2];
        
        //compute triangle bounding box using image dimensions
        x_min = std::max((int) ceil(std::min(p0.x, std::min(p1.x, p2.x))), 0);
        x_max = std::min((int) floor(std::max(p0.x, std::max(p1.x, p2.x))), w - 1);

        y_min = std::max((int) ceil(std::min(p0.y, std::min(p1.y, p2.y))), 0);
        y_max = std::min((int) floor(std::max(p0.y, std::max(p1.y, p2.y))), h - 1);

        if (x_max < x_min || y_max < y_min)
            continue;

        //loop over each pixel in bounding box
        for (y = y_min; y <= y_max; y++) {
            for (x = x_min; x <= x_max; x++) {

                p.x = float(x);
                p.y = float(y);

                //compute barycentric weights for current pixel, relative to triangle
                get_point_weight(weight, p, p0, p1, p2);

                //determine if pixel is inside triangle (check weights are > 0)
                if (weight[0] > 0 && weight[1] > 0 && weight[2] > 0) {

                    //interpolate depth at pixel using weights
                    p_depth = weight[0]*p0_depth + weight[1]*p1_depth + weight[2]*p2_depth;

                    //update only if new depth is closer (using z-buffer)
                    if (p_depth > depth_buffer[y*w + x]) {

                        //Interpolate per-pixel alpha from per-vertex alpha values
                        float interp_alpha = weight[0] * vertex_alphas[tri_p0_ind] +
                                                weight[1] * vertex_alphas[tri_p1_ind] +
                                                weight[2] * vertex_alphas[tri_p2_ind];

                        //multiply by max_alpha
                        interp_alpha *= max_alpha;

                        //clip alpha to [0, 1]
                        interp_alpha = std::min(1.0f, std::max(0.0f, interp_alpha));

                        //for each color channel, interpolate and blend the color based on alpha
                        for (k = 0; k < c; k++) {
                            p0_color = colors[c * tri_p0_ind + k];
                            p1_color = colors[c * tri_p1_ind + k];
                            p2_color = colors[c * tri_p2_ind + k];

                            p_color = weight[0]*p0_color + weight[1]*p1_color + weight[2]*p2_color;
                            
                            //blend computed color with the background pixel using interpolated alpha
                            if (reverse) {
                                image[(h - 1 - y) * w * c + x * c + k] = (unsigned char)(
                                    (1 - interp_alpha)*image[(h - 1 - y)*w*c + x*c + k] + interp_alpha*255*p_color);
                            } else {
                                image[y*w*c + x*c + k] = (unsigned char)(
                                    (1- interp_alpha)*image[y*w*c + x*c + k] + interp_alpha*255*p_color);
                                
                            }
                        }
                        //lastly, update depth buffer with new depth value
                        depth_buffer[y*w + x] = p_depth;

                    }
                }
            }
        } 

    }
}




void _get_normal(float *ver_normal, float *vertices, int *triangles, int nver, int ntri) {
    int tri_p0_ind, tri_p1_ind, tri_p2_ind;
    float v1x, v1y, v1z, v2x, v2y, v2z;

    // get tri_normal
//    float tri_normal[3 * ntri];
    float *tri_normal;
    tri_normal = new float[3 * ntri];
    for (int i = 0; i < ntri; i++) {
        tri_p0_ind = triangles[3 * i];
        tri_p1_ind = triangles[3 * i + 1];
        tri_p2_ind = triangles[3 * i + 2];

        // counter clockwise order
        v1x = vertices[3 * tri_p1_ind] - vertices[3 * tri_p0_ind];
        v1y = vertices[3 * tri_p1_ind + 1] - vertices[3 * tri_p0_ind + 1];
        v1z = vertices[3 * tri_p1_ind + 2] - vertices[3 * tri_p0_ind + 2];

        v2x = vertices[3 * tri_p2_ind] - vertices[3 * tri_p0_ind];
        v2y = vertices[3 * tri_p2_ind + 1] - vertices[3 * tri_p0_ind + 1];
        v2z = vertices[3 * tri_p2_ind + 2] - vertices[3 * tri_p0_ind + 2];


        tri_normal[3 * i] = v1y * v2z - v1z * v2y;
        tri_normal[3 * i + 1] = v1z * v2x - v1x * v2z;
        tri_normal[3 * i + 2] = v1x * v2y - v1y * v2x;

    }

    // get ver_normal
    for (int i = 0; i < ntri; i++) {
        tri_p0_ind = triangles[3 * i];
        tri_p1_ind = triangles[3 * i + 1];
        tri_p2_ind = triangles[3 * i + 2];

        for (int j = 0; j < 3; j++) {
            ver_normal[3 * tri_p0_ind + j] += tri_normal[3 * i + j];
            ver_normal[3 * tri_p1_ind + j] += tri_normal[3 * i + j];
            ver_normal[3 * tri_p2_ind + j] += tri_normal[3 * i + j];
        }
    }

    // normalizing
    float nx, ny, nz, det;
    for (int i = 0; i < nver; ++i) {
        nx = ver_normal[3 * i];
        ny = ver_normal[3 * i + 1];
        nz = ver_normal[3 * i + 2];

        det = sqrt(nx * nx + ny * ny + nz * nz);
        if (det <= 0) det = 1e-6;
        ver_normal[3 * i] = nx / det;
        ver_normal[3 * i + 1] = ny / det;
        ver_normal[3 * i + 2] = nz / det;
    }

    delete[] tri_normal;
}


//this is the original function from 3DDFA
void _rasterize(
    unsigned char *image, float *vertices, int *triangles, float *colors, float *depth_buffer,
    int ntri, int h, int w, int c, float alpha, bool reverse) {
int x, y, k;
int tri_p0_ind, tri_p1_ind, tri_p2_ind;
Point p0, p1, p2, p;
int x_min, x_max, y_min, y_max;
float p_depth, p0_depth, p1_depth, p2_depth;
float p_color, p0_color, p1_color, p2_color;
float weight[3];

for (int i = 0; i < ntri; i++) {
    tri_p0_ind = triangles[3 * i];
    tri_p1_ind = triangles[3 * i + 1];
    tri_p2_ind = triangles[3 * i + 2];

    p0.x = vertices[3 * tri_p0_ind];
    p0.y = vertices[3 * tri_p0_ind + 1];
    p0_depth = vertices[3 * tri_p0_ind + 2];
    p1.x = vertices[3 * tri_p1_ind];
    p1.y = vertices[3 * tri_p1_ind + 1];
    p1_depth = vertices[3 * tri_p1_ind + 2];
    p2.x = vertices[3 * tri_p2_ind];
    p2.y = vertices[3 * tri_p2_ind + 1];
    p2_depth = vertices[3 * tri_p2_ind + 2];

    x_min = std::max((int) ceil(std::min(p0.x, std::min(p1.x, p2.x))), 0);
    x_max = std::min((int) floor(std::max(p0.x, std::max(p1.x, p2.x))), w - 1);

    y_min = std::max((int) ceil(std::min(p0.y, std::min(p1.y, p2.y))), 0);
    y_max = std::min((int) floor(std::max(p0.y, std::max(p1.y, p2.y))), h - 1);

    if (x_max < x_min || y_max < y_min) {
        continue;
    }

    for (y = y_min; y <= y_max; y++) {
        for (x = x_min; x <= x_max; x++) {
            p.x = float(x);
            p.y = float(y);

            // call get_point_weight function once
            get_point_weight(weight, p, p0, p1, p2);

            // and judge is_point_in_tri by below line of code
            if (weight[2] > 0 && weight[1] > 0 && weight[0] > 0) {
                get_point_weight(weight, p, p0, p1, p2);
                p_depth = weight[0] * p0_depth + weight[1] * p1_depth + weight[2] * p2_depth;

                if ((p_depth > depth_buffer[y * w + x])) {
                    for (k = 0; k < c; k++) {
                        p0_color = colors[c * tri_p0_ind + k];
                        p1_color = colors[c * tri_p1_ind + k];
                        p2_color = colors[c * tri_p2_ind + k];

                        p_color = weight[0] * p0_color + weight[1] * p1_color + weight[2] * p2_color;
                        if (reverse) {
                            image[(h - 1 - y) * w * c + x * c + k] = (unsigned char) (
                                    (1 - alpha) * image[(h - 1 - y) * w * c + x * c + k] + alpha * 255 * p_color);
//                                image[(h - 1 - y) * w * c + x * c + k] = (unsigned char) (255 * p_color);
                        } else {
                            image[y * w * c + x * c + k] = (unsigned char) (
                                    (1 - alpha) * image[y * w * c + x * c + k] + alpha * 255 * p_color);
//                                image[y * w * c + x * c + k] = (unsigned char) (255 * p_color);
                        }
                    }

                    depth_buffer[y * w + x] = p_depth;
                }
            }
        }
    }
}
}
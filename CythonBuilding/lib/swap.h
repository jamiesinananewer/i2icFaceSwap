#include <Python.h>
#include <vector>


#ifndef RASTERIZE_H
#define RASTERIZE_H

#ifdef __cplusplus

extern "C" {

#endif
class Point3D {
    public:
        float x;
        float y;
        float z;
    
    public:
        Point3D() : x(0.f), y(0.f), z(0.f) {}
        Point3D(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
    
        void initialize(float x_, float y_, float z_){
            this->x = x_; this->y = y_; this->z = z_;
        }
    
        Point3D cross(Point3D &p){
            Point3D c;
            c.x = this->y * p.z - this->z * p.y;
            c.y = this->z * p.x - this->x * p.z;
            c.z = this->x * p.y - this->y * p.x;
            return c;
        }
    
        float dot(Point3D &p) {
            return this->x * p.x + this->y * p.y + this->z * p.z;
        }
    
        Point3D operator-(const Point3D &p) {
            Point3D np;
            np.x = this->x - p.x;
            np.y = this->y - p.y;
            np.z = this->z - p.z;
            return np;
        }
    
    };
    
    class Point {
    public:
        float x;
        float y;
    
    public:
        Point() : x(0.f), y(0.f) {}
        Point(float x_, float y_) : x(x_), y(y_) {}
        float dot(Point p) {
            return this->x * p.x + this->y * p.y;
        }
    
        Point operator-(const Point &p) {
            Point np;
            np.x = this->x - p.x;
            np.y = this->y - p.y;
            return np;
        }
    
        Point operator+(const Point &p) {
            Point np;
            np.x = this->x + p.x;
            np.y = this->y + p.y;
            return np;
        }
    
        Point operator*(float s) {
            Point np;
            np.x = s * this->x;
            np.y = s * this->y;
            return np;
        }
    };


struct ColorStats {
    std::vector<float> mean;    //mean per color channel
    std::vector<float> stdev;   //stdev per color channel
};

ColorStats _compute_color_statistics(const unsigned char* image, const bool* mask, 
                                    int h, int w, int c);    

ColorStats _color_stats(const float* image, int H, int W, int C, const bool* mask);

void _color_correction(const float* tgt_img, const float* src_img, const bool* mask,
    int h, int w,  int c, float epsilon, float* corr_img);

PyObject* py_compute_boundary_vertices(const int* triangles, int ntri);



PyObject* py_compute_distance_to_boundary_from_arrays(const float* vertices_data, int num_vertices,
                                                        const int* boundary_data, int num_boundary);


PyObject* py_compute_alpha_from_distance (const std::vector<float>& distances, float sigma);

void _rasterize_alpha_blur(unsigned char *image, float *vertices, int *triangles, float *colors, float *depth_buffer, 
                            float *vertex_alphas, int ntri, int h, int w, int c, float max_alpha, bool reverse);

void _rasterize(
    unsigned char *image, float *vertices, int *triangles, float *colors, float *depth_buffer,
    int ntri, int h, int w, int c, float alpha, bool reverse);

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
    bool reverse,                // If true, flip vertical axis
    bool sigma_correct
);
void _get_normal(float *ver_normal, float *vertices, int *triangles, int nver, int ntri);

#ifdef __cplusplus

}

#endif

#endif //RASTERIZE_H
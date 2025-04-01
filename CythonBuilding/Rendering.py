# coding: utf-8
import sys

sys.path.append('..')
import numpy as np
from .Raster import rasterize_alpha, rasterize, get_normal
from scipy.spatial import cKDTree
_norm = lambda arr: arr / np.sqrt(np.sum(arr ** 2, axis=1))[:, None]


def norm_vertices(vertices):
    vertices -= vertices.min(0)[None, :]
    vertices /= vertices.max()
    vertices *= 2
    vertices -= vertices.max(0)[None, :] / 2
    return vertices

def convert_type(obj):
    if isinstance(obj, tuple) or isinstance(obj, list):
        return np.array(obj, dtype=np.float32)[None, :]
    return obj

class RenderPipeline(object):
    def __init__(self, **kwargs):
        self.intensity_ambient = convert_type(kwargs.get('intensity_ambient', 0.3))
        self.intensity_directional = convert_type(kwargs.get('intensity_directional', 0.6))
        self.intensity_specular = convert_type(kwargs.get('intensity_specular', 0.1))
        self.specular_exp = kwargs.get('specular_exp', 5)
        self.color_ambient = convert_type(kwargs.get('color_ambient', (1, 1, 1)))
        self.color_directional = convert_type(kwargs.get('color_directional', (1, 1, 1)))
        self.light_pos = convert_type(kwargs.get('light_pos', (0, 0, 5)))
        self.view_pos = convert_type(kwargs.get('view_pos', (0, 0, 5)))

    def update_light_pos(self, light_pos):
        self.light_pos = convert_type(light_pos)

    def __call__(self, vertices, triangles, vertex_alphas, bg, texture=None):
        normal = get_normal(vertices, triangles)

        # --- Backface Culling ---
        # triangles is assumed to be an (M, 3) integer array.
        # For each triangle, get its vertex indices.
        tri = triangles  # alias for clarity

        # Get the 3D positions of the vertices for each triangle.
        v0 = vertices[tri[:, 0]]
        v1 = vertices[tri[:, 1]]
        v2 = vertices[tri[:, 2]]
        
        # Compute the centroid of each triangle.
        centroids = (v0 + v1 + v2) / 3.0
        
        # Compute an approximate face normal by averaging vertex normals.
        face_normals = (normal[tri[:, 0]] + normal[tri[:, 1]] + normal[tri[:, 2]]) / 3.0
        face_normals = _norm(face_normals)  # normalize the face normals

        # Compute the view vector for each triangle: from the centroid to the camera.
        # self.view_pos is shaped like (1, 3); broadcast it to match centroids.
        view_vec = _norm(self.view_pos - centroids)

        # Compute the dot product between the face normals and the view vector.
        dots = np.sum(face_normals * view_vec, axis=1)
        
        # Determine which triangles are front-facing (dot product > 0).
        # You can adjust the threshold if needed.
        valid = dots > 0
        triangles = triangles[valid]

        #only expression vertices
        #triangles = triangles[np.all(triangles <= 16113, axis=1)]
        # --------------------------------



        # 2. lighting
        light = np.zeros_like(vertices, dtype=np.float32)
        # ambient component
        if self.intensity_ambient > 0:
            light += self.intensity_ambient * self.color_ambient

        vertices_n = norm_vertices(vertices.copy())
        if self.intensity_directional > 0:
            # diffuse component
            direction = _norm(self.light_pos - vertices_n)
            cos = np.sum(normal * direction, axis=1)[:, None]
            # cos = np.clip(cos, 0, 1)
            #  todo: check below
            light += self.intensity_directional * (self.color_directional * np.clip(cos, 0, 1))

            # specular component
            if self.intensity_specular > 0:
                v2v = _norm(self.view_pos - vertices_n)
                reflection = 2 * cos * normal - direction
                spe = np.sum((v2v * reflection) ** self.specular_exp, axis=1)[:, None]
                spe = np.where(cos != 0, np.clip(spe, 0, 1), np.zeros_like(spe))
                light += self.intensity_specular * self.color_directional * np.clip(spe, 0, 1)
        light = np.clip(light, 0, 1)

        

        # If a texture is provided, modulate its per-vertex colors by the computed alpha.
        if texture is not None:
            # texture is (N, 3); multiply each vertex color by its alpha value.
            texture = texture #* vertex_alpha[:, None]
        else:
            # Alternatively, you could apply alpha to the lighting component:
            light = light #* vertex_alpha[:, None]
        # ----------------------------------------------

        # 2. rasterization, [0, 1]
        if texture is None:
            print(f'ver = {vertices}')
            render_img = rasterize_alpha(vertices, triangles, light, vertex_alphas, bg=bg, max_alpha=1.0)
            return render_img
        else:
            texture *= light
            render_img = rasterize_alpha(vertices, triangles, texture, vertex_alphas, bg=bg, max_alpha=1.0)
            return render_img


def main():
    pass


if __name__ == '__main__':
    main()

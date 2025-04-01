import open3d as o3d

# Load the OBJ file
n = 4
obj_file = f"examples\\results\\denzel.obj"  # Replace with your OBJ file path
mesh = o3d.io.read_triangle_mesh(obj_file)

# Check if the file loaded correctly
if mesh.is_empty():
    print("Error: Failed to load the OBJ file.")
else:
    print("OBJ file loaded successfully!")

# Create a visualization window
o3d.visualization.draw_geometries([mesh], window_name="OBJ Viewer")


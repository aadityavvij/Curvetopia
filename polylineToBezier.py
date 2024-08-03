# import csv
# import numpy as np
# import svgwrite

# def read_csv(file_path):
#     polylines = []
#     current_polyline = []
#     with open(file_path, 'r') as file:
#         reader = csv.reader(file)
#         previous_x, previous_y = None, None
#         for row in reader:
#             if len(row) >= 4:  # Ensure the row has at least four columns
#                 try:
#                     x, y = float(row[2]), float(row[3])  # Read only the 3rd and 4th columns
#                     # Determine if we should start a new polyline
#                     if previous_x is not None and previous_y is not None:
#                         if abs(x - previous_x) > 10 or abs(y - previous_y) > 10:
#                             if current_polyline:
#                                 polylines.append(current_polyline)
#                                 current_polyline = []
#                     current_polyline.append((x, y))
#                     previous_x, previous_y = x, y
#                 except ValueError as e:
#                     print(f"Skipping invalid row: {row} - Error: {e}")
#             else:
#                 if current_polyline:  # End of a polyline segment
#                     polylines.append(current_polyline)
#                     current_polyline = []
#         if current_polyline:  # Add the last polyline if it exists
#             polylines.append(current_polyline)
#     return polylines

# def catmull_rom_to_bezier(p0, p1, p2, p3):
#     """Convert Catmull-Rom to cubic Bézier control points."""
#     b0 = p1
#     b1 = p1 + (p2 - p0) / 6
#     b2 = p2 - (p3 - p1) / 6
#     b3 = p2
#     return b0, b1, b2, b3

# def polyline_to_bezier(polyline):
#     bezier_curves = []
#     for i in range(len(polyline) - 1):
#         p0 = np.array(polyline[i - 1]) if i > 0 else np.array(polyline[i])
#         p1 = np.array(polyline[i])
#         p2 = np.array(polyline[i + 1])
#         p3 = np.array(polyline[i + 2]) if i < len(polyline) - 2 else np.array(polyline[i + 1])
        
#         b0, b1, b2, b3 = catmull_rom_to_bezier(p0, p1, p2, p3)
#         bezier_curves.append((b0, b1, b2, b3))
#     return bezier_curves

# def generate_svg(polylines, output_file):
#     dwg = svgwrite.Drawing(output_file, profile='tiny')
#     for polyline in polylines:
#         if len(polyline) > 1:  # Ensure there are at least two points to form a line
#             bezier_curves = polyline_to_bezier(polyline)
#             for b0, b1, b2, b3 in bezier_curves:
#                 path_data = f'M {b0[0]},{b0[1]} C {b1[0]},{b1[1]}, {b2[0]},{b2[1]}, {b3[0]},{b3[1]}'
#                 dwg.add(dwg.path(d=path_data, stroke="black", fill="none"))
#     dwg.save()

# # Example usage:
# polylines = read_csv('./examples/occlusion1.csv')
# generate_svg(polylines, 'output.svg')


import csv
import numpy as np
import svgwrite
from scipy.spatial import distance

def read_csv(file_path):
    polylines = []
    current_polyline = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        previous_x, previous_y = None, None
        for row in reader:
            if len(row) >= 4:  # Ensure the row has at least four columns
                try:
                    x, y = float(row[2]), float(row[3])  # Read only the 3rd and 4th columns
                    # Determine if we should start a new polyline
                    if previous_x is not None and previous_y is not None:
                        if abs(x - previous_x) > 10 or abs(y - previous_y) > 10:
                            if current_polyline:
                                polylines.append(current_polyline)
                                current_polyline = []
                    current_polyline.append((x, y))
                    previous_x, previous_y = x, y
                except ValueError as e:
                    print(f"Skipping invalid row: {row} - Error: {e}")
            else:
                if current_polyline:  # End of a polyline segment
                    polylines.append(current_polyline)
                    current_polyline = []
        if current_polyline:  # Add the last polyline if it exists
            polylines.append(current_polyline)
    return polylines

def rdp(points, epsilon):
    """Simplify polyline using the Ramer-Douglas-Peucker algorithm."""
    if len(points) < 3:
        return points

    def point_line_distance(point, start, end):
        """Calculate the distance from a point to a line segment."""
        if (start == end).all():
            return distance.euclidean(point, start)
        return np.abs(np.cross(end-start, start-point) / np.linalg.norm(end-start))

    start, end = points[0], points[-1]
    dmax = 0
    index = 0

    for i in range(1, len(points) - 1):
        d = point_line_distance(np.array(points[i]), np.array(start), np.array(end))
        if d > dmax:
            index = i
            dmax = d

    if dmax > epsilon:
        left = rdp(points[:index+1], epsilon)
        right = rdp(points[index:], epsilon)
        return left[:-1] + right
    else:
        return [start, end]

def catmull_rom_to_bezier(p0, p1, p2, p3):
    """Convert Catmull-Rom to cubic Bézier control points."""
    b0 = p1
    b1 = p1 + (p2 - p0) / 6
    b2 = p2 - (p3 - p1) / 6
    b3 = p2
    return b0, b1, b2, b3

def polyline_to_bezier(polyline):
    bezier_curves = []
    for i in range(len(polyline) - 1):
        p0 = np.array(polyline[i - 1]) if i > 0 else np.array(polyline[i])
        p1 = np.array(polyline[i])
        p2 = np.array(polyline[i + 1])
        p3 = np.array(polyline[i + 2]) if i < len(polyline) - 2 else np.array(polyline[i + 1])
        
        b0, b1, b2, b3 = catmull_rom_to_bezier(p0, p1, p2, p3)
        bezier_curves.append((b0, b1, b2, b3))
    return bezier_curves

def generate_svg(polylines, output_file, epsilon=0.2):
    dwg = svgwrite.Drawing(output_file, profile='tiny')
    for polyline in polylines:
        if len(polyline) > 1:  # Ensure there are at least two points to form a line
            simplified_polyline = rdp(polyline, epsilon)
            bezier_curves = polyline_to_bezier(simplified_polyline)
            for b0, b1, b2, b3 in bezier_curves:
                path_data = f'M {b0[0]},{b0[1]} C {b1[0]},{b1[1]}, {b2[0]},{b2[1]}, {b3[0]},{b3[1]}'
                dwg.add(dwg.path(d=path_data, stroke="black", fill="none"))
    dwg.save()

# Example usage:
polylines = read_csv('./examples/occlusion1.csv')
generate_svg(polylines, 'output.svg', epsilon=0.2)

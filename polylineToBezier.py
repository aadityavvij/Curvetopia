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


# import csv
# import numpy as np
# import svgwrite
# from scipy.spatial import distance

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

# def adaptive_rdp(points, base_epsilon, damping_factor=0.3):
#     """Simplify polyline using an adaptive Ramer-Douglas-Peucker algorithm."""
#     if len(points) < 3:
#         return points

#     def point_line_distance(point, start, end):
#         """Calculate the distance from a point to a line segment."""
#         if (start == end).all():
#             return distance.euclidean(point, start)
#         return np.abs(np.cross(end - start, start - point) / np.linalg.norm(end - start))

#     def local_epsilon(start, end):
#         """Determine local epsilon based on segment complexity."""
#         segment_length = np.linalg.norm(end - start)
#         return base_epsilon * (1 + damping_factor * segment_length)

#     start, end = points[0], points[-1]
#     dmax = 0
#     index = 0

#     for i in range(1, len(points) - 1):
#         d = point_line_distance(np.array(points[i]), np.array(start), np.array(end))
#         adaptive_epsilon = local_epsilon(np.array(start), np.array(end))
#         if d > dmax and d > adaptive_epsilon:
#             index = i
#             dmax = d

#     if dmax > base_epsilon:
#         left = adaptive_rdp(points[:index+1], base_epsilon, damping_factor)
#         right = adaptive_rdp(points[index:], base_epsilon, damping_factor)
#         return left[:-1] + right
#     else:
#         return [start, end]

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

# def generate_svg(polylines, output_file, epsilon=0.2, damping_factor=0.3):
#     dwg = svgwrite.Drawing(output_file, profile='tiny')
#     for polyline in polylines:
#         if len(polyline) > 1:  # Ensure there are at least two points to form a line
#             simplified_polyline = adaptive_rdp(polyline, epsilon, damping_factor)
#             bezier_curves = polyline_to_bezier(simplified_polyline)
#             for b0, b1, b2, b3 in bezier_curves:
#                 path_data = f'M {b0[0]},{b0[1]} C {b1[0]},{b1[1]}, {b2[0]},{b2[1]}, {b3[0]},{b3[1]}'
#                 dwg.add(dwg.path(d=path_data, stroke="black", fill="none"))
#     dwg.save()

# # Example usage:
# polylines = read_csv('./examples/frag0.csv')
# generate_svg(polylines, 'output.svg', epsilon=0.2, damping_factor=0.3)


import csv
import math
import numpy as np
import svgwrite
from scipy.spatial import distance

def read_csv(file_path, distance_threshold=10):
    polylines = []
    current_polyline = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        previous_point = None
        for row in reader:
            if len(row) >= 4:  # Ensure the row has at least four columns
                try:
                    x, y = float(row[2]), float(row[3])  # Read only the 3rd and 4th columns
                    current_point = (x, y)
                    if previous_point is not None:
                        # Determine if we should start a new polyline based on the distance threshold
                        if abs(current_point[0] - previous_point[0]) > distance_threshold or abs(current_point[1] - previous_point[1]) > distance_threshold:
                            if current_polyline:
                                polylines.append(current_polyline)
                                current_polyline = []
                    current_polyline.append(current_point)
                    previous_point = current_point
                except ValueError as e:
                    print(f"Skipping invalid row: {row} - Error: {e}")
            else:
                if current_polyline:  # End of a polyline segment
                    polylines.append(current_polyline)
                    current_polyline = []
        if current_polyline:  # Add the last polyline if it exists
            polylines.append(current_polyline)
    return polylines

def calculate_angle(p1, p2, p3):
    """Calculate the angle between the line segments p1p2 and p2p3."""
    a = (p1[0] - p2[0], p1[1] - p2[1])
    b = (p3[0] - p2[0], p3[1] - p2[1])
    dot_product = a[0] * b[0] + a[1] * b[1]
    mag_a = math.sqrt(a[0] ** 2 + a[1] ** 2)
    mag_b = math.sqrt(b[0] ** 2 + b[1] ** 2)
    if mag_a * mag_b == 0:
        return 0
    cosine_angle = dot_product / (mag_a * mag_b)
    cosine_angle = max(min(cosine_angle, 1.0), -1.0)  # Clamp the value to the range [-1, 1]
    angle = math.acos(cosine_angle)
    return math.degrees(angle)

def split_polyline_at_sharp_turns(polylines, threshold_angle=30, interval=5):
    """Split polylines at sharp turns with an angle less than the threshold."""
    new_polylines = []
    for polyline in polylines:
        current_polyline = []
        i = 0
        while i < len(polyline):
            current_polyline.append(polyline[i])
            if i > interval and (i + interval) < len(polyline):
                angle = calculate_angle(polyline[i-interval], polyline[i], polyline[i+interval])
                if angle < threshold_angle:
                    new_polylines.append(current_polyline[:-1])
                    current_polyline = current_polyline[-2:]
            i += 1
        new_polylines.append(current_polyline)
    return new_polylines

def adaptive_rdp(points, base_epsilon):
    """Simplify polyline using an adaptive Ramer-Douglas-Peucker algorithm."""
    if len(points) < 3:
        return points

    def point_line_distance(point, start, end):
        """Calculate the distance from a point to a line segment."""
        if (start == end).all():
            return distance.euclidean(point, start)
        return np.abs(np.cross(end - start, start - point) / np.linalg.norm(end - start))

    def local_epsilon(start, end):
        """Determine local epsilon based on segment complexity."""
        segment_length = np.linalg.norm(end - start)
        return base_epsilon * (1 + segment_length)

    start, end = points[0], points[-1]
    dmax = 0
    index = 0

    for i in range(1, len(points) - 1):
        d = point_line_distance(np.array(points[i]), np.array(start), np.array(end))
        adaptive_epsilon = local_epsilon(np.array(start), np.array(end))
        if d > dmax and d > adaptive_epsilon:
            index = i
            dmax = d

    if dmax > base_epsilon:
        left = adaptive_rdp(points[:index+1], base_epsilon)
        right = adaptive_rdp(points[index:], base_epsilon)
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
            simplified_polyline = adaptive_rdp(polyline, epsilon)
            bezier_curves = polyline_to_bezier(simplified_polyline)
            for b0, b1, b2, b3 in bezier_curves:
                path_data = f'M {b0[0]},{b0[1]} C {b1[0]},{b1[1]}, {b2[0]},{b2[1]}, {b3[0]},{b3[1]}'
                dwg.add(dwg.path(d=path_data, stroke="black", fill="none"))
    dwg.save()

# Example usage:
# polylines = read_csv('./examples/occlusion1.csv')
polylines1 = read_csv('./examples/isolated.csv')
polylines2 = split_polyline_at_sharp_turns(polylines1, threshold_angle=130, interval=4)
generate_svg(polylines2, 'output.svg', epsilon=0.08)

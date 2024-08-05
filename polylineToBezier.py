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

'''
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

def perpendicular_distance(point, line_start, line_end):
    # Line vector
    line_vec = np.array(line_end) - np.array(line_start)
    line_vec_length = np.linalg.norm(line_vec)
    
    if line_vec_length == 0:
        return np.linalg.norm(np.array(point) - np.array(line_start))
    
    line_vec = line_vec / line_vec_length
    
    point_vec = np.array(point) - np.array(line_start)
    proj_len = np.dot(point_vec, line_vec)
    proj_point = np.array(line_start) + proj_len * line_vec
    return np.linalg.norm(np.array(point) - proj_point)

def is_almost_straight_line(points, threshold=10.0):
    if len(points) < 3:
        return True  # A line with less than 3 points is already straight
    
    p1 = points[0]
    p2 = points[-1]
    
    for point in points[1:-1]:
        distance = perpendicular_distance(point, p1, p2)
        if distance > threshold:
            return False
    return True

def simplify_polylines(polylines, threshold=10.0):
    simplified_polylines = []
    for polyline in polylines:
        if is_almost_straight_line(polyline, threshold):
            simplified_polylines.append([polyline[0], polyline[-1]])
        else:
            simplified_polylines.append(polyline)
    return simplified_polylines

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
            simplified_polyline = simplify_polylines(polyline)
            bezier_curves = polyline_to_bezier(simplified_polyline)
            for b0, b1, b2, b3 in bezier_curves:
                path_data = f'M {b0[0]},{b0[1]} C {b1[0]},{b1[1]}, {b2[0]},{b2[1]}, {b3[0]},{b3[1]}'
                dwg.add(dwg.path(d=path_data, stroke="black", fill="none"))
    dwg.save()

# Example usage:
# polylines = read_csv('./examples/occlusion1.csv')
polylines1 = read_csv('./examples/frag0.csv')
# polylines2 = split_polyline_at_sharp_turns(polylines1, threshold_angle=130, interval=4)
generate_svg(polylines1, 'output.svg', epsilon=0.08)
'''

import numpy as np
import matplotlib.pyplot as plt
import math
from collections import deque

def read_csv_(csv_path):
    np_path_XYs = np.genfromtxt(csv_path, delimiter=',')
    path_XYs = []
    for i in np.unique(np_path_XYs[:, 0]):
        npXYs = np_path_XYs[np_path_XYs[:, 0] == i][:, 1:]  # Select all points for the path i, ignore the path_id
        XYs = npXYs[:, 1:]  # Ignore the segment_id
        path_XYs.append(XYs)
    return path_XYs

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

def plot(paths_XYs, title, ax):
    colours = ['red', 'green', 'blue', 'yellow', 'purple']  # Define some colors for plotting
    for path_index, XYs in enumerate(paths_XYs):
        c = colours[path_index % len(colours)]
        ax.plot(XYs[:, 0], XYs[:, 1], c=c, linewidth=2)
    ax.set_aspect('equal')
    ax.set_title(title)

def split_polyline_at_sharp_turns(polylines, threshold_angle=130, interval=3):
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
                    new_polylines.append(np.array(current_polyline[:-1]))
                    current_polyline = current_polyline[-2:]
                    i += interval
                else:
                    i += 1
            else:
                i += 1
        new_polylines.append(np.array(current_polyline))
    return new_polylines

def point_line_distance(p1, p2, p):
    """Calculate the perpendicular distance from point p to the line formed by points p1 and p2."""
    if np.array_equal(p1, p2):
        return np.linalg.norm(p - p1)
    return np.abs(np.cross(p2-p1, p1-p)) / np.linalg.norm(p2-p1)

def simplify_to_straight_lines(polylines, distance_threshold=11.0):
    """Simplify polylines by reducing almost straight segments to their endpoints."""
    simplified_polylines = []
    for polyline in polylines:
        if len(polyline) < 3:
            simplified_polylines.append(polyline)
            continue
        p1 = polyline[0]
        p2 = polyline[-1]
        distances = [point_line_distance(p1, p2, p) for p in polyline[1:-1]]
        max_distance = max(distances)
        if max_distance < distance_threshold:
            simplified_polylines.append(np.array([p1, p2]))
        else:
            simplified_polylines.append(polyline)
    return simplified_polylines

def euclidean_distance(point1, point2):
    """Calculate the Euclidean distance between two points."""
    return np.linalg.norm(np.array(point1) - np.array(point2))

def are_almost_parallel(vector1, vector2, tolerance=0.1):
    """Check if two vectors are almost parallel."""
    unit_vector1 = vector1 / np.linalg.norm(vector1)
    unit_vector2 = vector2 / np.linalg.norm(vector2)
    dot_product = np.dot(unit_vector1, unit_vector2)
    return abs(abs(dot_product) - 1) < tolerance

def combine_straight_polylines(polylines, epsilon=5):
    # Separate two-point polylines from others
    two_point_polylines = [polyline for polyline in polylines if polyline.shape[0] == 2]
    non_straight_polylines = [polyline for polyline in polylines if polyline.shape[0] != 2]

    # List to keep track of polylines to be checked
    polylines_to_check = two_point_polylines[:]

    # List to keep track of combined polylines
    combined_polylines = []

    while polylines_to_check:
        current_polyline = polylines_to_check.pop(0)
        combined = False

        for i in range(len(polylines_to_check)):
            polyline_to_check = polylines_to_check[i]
            p1_start, p1_end = current_polyline[0], current_polyline[-1]
            p2_start, p2_end = polyline_to_check[0], polyline_to_check[-1]

            # Check if any of the endpoints are close and combine accordingly
            if euclidean_distance(p1_start, p2_start) < epsilon:
                combined_polyline = np.vstack((current_polyline[::-1], polyline_to_check[1:]))
                polylines_to_check.pop(i)
                polylines_to_check.append(combined_polyline)
                combined = True
                break
            elif euclidean_distance(p1_start, p2_end) < epsilon:
                combined_polyline = np.vstack((current_polyline[::-1], polyline_to_check[::-1][1:]))
                polylines_to_check.pop(i)
                polylines_to_check.append(combined_polyline)
                combined = True
                break
            elif euclidean_distance(p1_end, p2_start) < epsilon:
                combined_polyline = np.vstack((current_polyline, polyline_to_check[1:]))
                polylines_to_check.pop(i)
                polylines_to_check.append(combined_polyline)
                combined = True
                break
            elif euclidean_distance(p1_end, p2_end) < epsilon:
                combined_polyline = np.vstack((current_polyline, polyline_to_check[::-1][1:]))
                polylines_to_check.pop(i)
                polylines_to_check.append(combined_polyline)
                combined = True
                break

        # If the current polyline was not combined with any other, add it to the result
        if not combined:
            combined_polylines.append(current_polyline)

    # Add non-straight polylines to the final result
    combined_polylines.extend(non_straight_polylines)

    return combined_polylines


def combine_polylines(polyline1, polyline2, tolerance_distance=5.0, tolerance_parallel=0.1):
    """Combine two polylines if their endpoints are close and they are almost parallel."""
    polyline1 = np.array(polyline1)
    polyline2 = np.array(polyline2)
    
    ends1 = [polyline1[0], polyline1[-1]]
    ends2 = [polyline2[0], polyline2[-1]]
    
    for end1 in ends1:
        for end2 in ends2:
            distance = euclidean_distance(end1, end2)
            if distance < tolerance_distance:
                if np.array_equal(end1, polyline1[0]) and np.array_equal(end2, polyline2[0]):
                    vector1 = polyline1[1] - polyline1[0]
                    vector2 = polyline2[1] - polyline2[0]
                elif np.array_equal(end1, polyline1[0]) and np.array_equal(end2, polyline2[-1]):
                    vector1 = polyline1[1] - polyline1[0]
                    vector2 = polyline2[-2] - polyline2[-1]
                elif np.array_equal(end1, polyline1[-1]) and np.array_equal(end2, polyline2[0]):
                    vector1 = polyline1[-2] - polyline1[-1]
                    vector2 = polyline2[1] - polyline2[0]
                elif np.array_equal(end1, polyline1[-1]) and np.array_equal(end2, polyline2[-1]):
                    vector1 = polyline1[-2] - polyline1[-1]
                    vector2 = polyline2[-2] - polyline2[-1]
                else:
                    continue  # Skip if the combination is not handled

                if are_almost_parallel(vector1, vector2, tolerance_parallel):
                    if np.array_equal(end1, polyline1[0]):
                        polyline1 = np.flip(polyline1, axis=0)
                    if np.array_equal(end2, polyline2[-1]):
                        polyline2 = np.flip(polyline2, axis=0)
                    combined_polyline = np.concatenate((polyline1, polyline2[1:]), axis=0)
                    return combined_polyline
    return None

def combine_all_polylines(polylines, tolerance_distance=5.0, tolerance_parallel=0.1):
    """Combine all polylines based on distance and parallelism criteria."""
    polylines_to_check = list(polylines)
    combined_polylines = []

    while polylines_to_check:
        polyline1 = polylines_to_check.pop(0)
        combined = False
        for i, polyline2 in enumerate(polylines_to_check):
            result = combine_polylines(polyline1, polyline2, tolerance_distance, tolerance_parallel)
            if result is not None:
                polylines_to_check.pop(i)  # Remove polyline2 as it has been combined
                polylines_to_check.insert(0, result)  # Recheck with the newly combined polyline
                combined = True
                break
        if not combined:
            combined_polylines.append(polyline1)  # No combination was made, add the polyline as is

    return combined_polylines



csv_path = "./examples/isolated.csv"

output_data1 = read_csv_(csv_path)
output_data2 = read_csv_(csv_path)

output_data2 = split_polyline_at_sharp_turns(output_data2)

output_data2 = simplify_to_straight_lines(output_data2)

output_data2 = combine_all_polylines(output_data2)


output_data2 = simplify_to_straight_lines(output_data2)
output_data2 = combine_straight_polylines(output_data2)

fig, axs = plt.subplots(1, 2, tight_layout=True, figsize=(16, 8))
plot(output_data1, 'Original Data', axs[0])
plot(output_data2, 'Processed Data', axs[1])
plt.show()


import numpy as np
import matplotlib.pyplot as plt
import math
from collections import deque, defaultdict
from skimage.measure import find_contours, approximate_polygon
from copy import deepcopy

def first(output_data2):
    output_data2 = split_polyline_at_sharp_turns(output_data2, interval=1)
    output_data2 = split_polyline_at_sharp_turns(output_data2)
    output_data2 = combine_all_polylines(output_data2)
    output_data2 = simplify_to_straight_lines(output_data2)
    output_data2 = combine_all_polylines(output_data2)
    output_data2 = simplify_to_straight_lines(output_data2)
    output_data2 = combine_straight_polylines(output_data2)
    output_data2, shape_count = process_paths(output_data2)
    return output_data2, shape_count


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
                    new_polylines.append(np.array(current_polyline))
                    current_polyline = current_polyline[-1:]
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

def simplify_to_straight_lines(polylines, distance_threshold=11):
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




def is_approx_circle(XYs, tolerance=0.1, min_points=50):
    if len(XYs) < min_points:
        return False
    centroid = np.mean(XYs, axis=0)
    distances = np.linalg.norm(XYs - centroid, axis=1)
    return np.std(distances) / np.mean(distances) < tolerance

def make_circle(XYs, num_points=100):
    centroid = np.mean(XYs, axis=0)
    radius = np.mean(np.linalg.norm(XYs - centroid, axis=1))
    angles = np.linspace(0, 2 * np.pi, num_points)
    circle_points = np.array([
        centroid[0] + radius * np.cos(angles),
        centroid[1] + radius * np.sin(angles)
    ]).T
    return circle_points

def is_approx_ellipse(XYs, tolerance=0.15, min_points=50):
    if len(XYs) < min_points:
        return False
    centroid = np.mean(XYs, axis=0)
    distances = np.linalg.norm(XYs - centroid, axis=1)
    major_axis_length = max(distances)
    minor_axis_length = min(distances)
    eccentricity = np.sqrt(1 - (minor_axis_length**2 / major_axis_length**2))
    return 0 < eccentricity < tolerance

def make_ellipse(XYs, num_points=100):
    centroid = np.mean(XYs, axis=0)
    distances = np.linalg.norm(XYs - centroid, axis=1)
    major_axis_length = max(distances)
    minor_axis_length = min(distances)
    angles = np.linspace(0, 2 * np.pi, num_points)
    ellipse_points = np.array([
        centroid[0] + major_axis_length * np.cos(angles),
        centroid[1] + minor_axis_length * np.sin(angles)
    ]).T
    return np.vstack([ellipse_points, ellipse_points[0]])

def is_approx_regular_polygon(XYs, angle_tolerance=10, length_tolerance=1):
    # Number of sides is the number of unique points minus 1 (closing point)
    num_sides = len(XYs) - 1
    if num_sides < 3 or num_sides > 10:
        return False
    
    # Calculate vectors for each side
    vectors = np.diff(np.vstack([XYs, XYs[0]]), axis=0)
    
    # Calculate side lengths
    lengths = np.linalg.norm(vectors, axis=1)
    
    # Check if all side lengths are approximately equal
    if not np.allclose(lengths, lengths[0], rtol=length_tolerance):
        return False
    
    # Calculate angles between consecutive sides
    angles = []
    for i in range(num_sides):
        angle = np.arccos(np.clip(np.dot(vectors[i], vectors[(i+1) % num_sides]) / 
                                  (np.linalg.norm(vectors[i]) * np.linalg.norm(vectors[(i+1) % num_sides])), -1.0, 1.0))
        angles.append(np.degrees(angle))
    
    # The expected internal angle for a regular polygon
    expected_angle = 180 - 360 / num_sides
    
    # Check if all angles are approximately equal to the expected internal angle
    return np.allclose(angles, expected_angle, atol=angle_tolerance)

def is_approx_star_shape(XYs, distance_tolerance=1, angle_tolerance=10):
    # Ensure the polyline is closed
    if not np.array_equal(XYs[0], XYs[-1]):
        return False
    
    # Remove the closing point to work with the main points
    points = XYs[:-1]
    
    # Ensure there are exactly 10 points (5 outer, 5 inner)
    if len(points) != 10:
        return False
    
    # Calculate the centroid (center) of the points
    center = np.mean(points, axis=0)
    
    # Calculate distances of points from the center
    distances = np.linalg.norm(points - center, axis=1)
    
    # Sort points based on angle relative to the center to ensure correct ordering
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    sorted_indices = np.argsort(angles)
    sorted_points = points[sorted_indices]
    sorted_distances = distances[sorted_indices]
    
    # Classify into outer and inner points based on distance
    outer_points = sorted_points[sorted_distances >= np.median(sorted_distances)]
    inner_points = sorted_points[sorted_distances < np.median(sorted_distances)]
    
    # Check if all outer points are approximately equidistant from the center
    if not np.allclose(np.linalg.norm(outer_points - center, axis=1), np.median(sorted_distances), rtol=distance_tolerance):
        return False
    
    # Check if all inner points are approximately equidistant from the center
    if not np.allclose(np.linalg.norm(inner_points - center, axis=1), np.median(sorted_distances), rtol=distance_tolerance):
        return False
    
    angles = []
    num_points = 10
    for i in range(num_points - 1):
        vector_i = points[i] - center
        vector_next = points[(i + 1) % num_points] - center
        
        # Calculate the angle between vector_i and vector_next
        angle = np.arccos(np.clip(np.dot(vector_i, vector_next) /
                                 (np.linalg.norm(vector_i) * np.linalg.norm(vector_next)), -1.0, 1.0))
        angles.append(np.degrees(angle))
    
    # Check if all angles are approximately equal
    angles = np.array(angles)
    mean_angle = np.mean(angles)
    
    return np.allclose(angles, mean_angle, atol=angle_tolerance)

def classify_shape(XYs):
    if len(XYs) == 2:
        if(np.allclose(XYs[0], XYs[1], atol=0.001)):
            return "none"
        return "straight line"
    elif is_approx_circle(XYs):
        return "circle"
    elif is_approx_ellipse(XYs):
        return "ellipse"
    elif is_approx_rectangle(XYs):
        return "rectangle"
    elif is_approx_regular_polygon(XYs):
        return "regular_polygon"
    elif is_approx_star_shape(XYs):
        return "star"
    else:
        return "other"

def process_paths(paths_XYs, circle_tolerance=0.1, rect_angle_tolerance=10, min_points=50):
    shape_count = defaultdict(int)
    processed_paths = []
    
    for XYs in paths_XYs:
        shape = classify_shape(XYs)
        if(shape!="none"):
            shape_count[shape] += 1
        
        if shape == "circle":
            processed_paths.append(make_circle(XYs))
        elif shape == "ellipse":
            processed_paths.append(make_ellipse(XYs))
        elif shape == "rectangle":
            processed_paths.append(make_rectangle(XYs))
        elif shape == "none":
            pass
        else:
            processed_paths.append(XYs)
    
    return processed_paths, shape_count

def is_approx_rectangle(XYs, angle_tolerance=10):
    if len(XYs) != 5:
        return False
    # Calculate vectors for each side
    vectors = np.diff(np.vstack([XYs, XYs[0]]), axis=0)
    angles = []
    for i in range(4):
        angle = np.arccos(np.clip(np.dot(vectors[i], vectors[(i+1) % 4]) / 
                                  (np.linalg.norm(vectors[i]) * np.linalg.norm(vectors[(i+1) % 4])), -1.0, 1.0))
        angles.append(np.degrees(angle))
    angles = np.array(angles)
    return np.allclose(angles, 90, atol=angle_tolerance) or np.allclose(angles, 270, atol=angle_tolerance)

def make_rectangle(XYs):
    XYs = XYs[0:4]
    centroid = np.mean(XYs, axis=0)
    vectors = np.diff(np.vstack([XYs, XYs[0]]), axis=0)
    lengths = [np.linalg.norm(v) for v in vectors]
    width = np.mean([lengths[0], lengths[2]])
    height = np.mean([lengths[1], lengths[3]])
    
    # Calculate the four corners of the rectangle
    rectangle_points = np.array([
        [centroid[0] - width / 2, centroid[1] - height / 2],
        [centroid[0] + width / 2, centroid[1] - height / 2],
        [centroid[0] + width / 2, centroid[1] + height / 2],
        [centroid[0] - width / 2, centroid[1] + height / 2],
        [centroid[0] - width / 2, centroid[1] - height / 2]
    ])
    return rectangle_points

def add_symmetry_lines(ax, XYs):
    shape = classify_shape(XYs)
    if shape == "circle":
        centroid = np.mean(XYs, axis=0)
        radius = np.linalg.norm(XYs[0] - centroid)
        angle = -(np.pi/4)
        line_x = [centroid[0] - radius * np.cos(angle), centroid[0] + radius * np.cos(angle)]
        line_y = [centroid[1] - radius * np.sin(angle), centroid[1] + radius * np.sin(angle)]
        ax.plot(line_x, line_y, 'k--', linewidth=1)
    elif shape == "ellipse":
        centroid = np.mean(XYs, axis=0)
        major_axis = (XYs[np.argmax(np.linalg.norm(XYs - centroid, axis=1))] - centroid)
        minor_axis = np.array([-major_axis[1], major_axis[0]])
        for axis in [major_axis, minor_axis]:
            line_x = [centroid[0] - axis[0], centroid[0] + axis[0]]
            line_y = [centroid[1] - axis[1], centroid[1] + axis[1]]
            ax.plot(line_x, line_y, 'k--', linewidth=1)
    elif shape == "star":
        symmetry_lines = find_star_symmetry_lines(XYs)
        for line in symmetry_lines:
            ax.plot(line[:, 0], line[:, 1], c='black', linestyle='--', linewidth=1)
    elif shape == "rectangle":
        centroid = np.mean(XYs[:-1], axis=0)
        x_min, x_max = np.min(XYs[:, 0]), np.max(XYs[:, 0])
        y_min, y_max = np.min(XYs[:, 1]), np.max(XYs[:, 1])
        ax.plot([x_min, x_max], [centroid[1], centroid[1]], 'k--', linewidth=1)  # Horizontal symmetry line
        ax.plot([centroid[0], centroid[0]], [y_min, y_max], 'k--', linewidth=1)  # Vertical symmetry line
        ax.plot([x_min, x_max], [y_min, y_max], 'k--', linewidth=1)  # Diagonal symmetry line
    elif shape == "regular_polygon":
        centroid = np.mean(XYs[:-1], axis=0)
        for vertex in XYs[:-1]:
            ax.plot([centroid[0], vertex[0]], [centroid[1], vertex[1]], 'k--', linewidth=1)

def find_star_symmetry_lines(XYs):
    symmetry_lines = []
    for i in range(0, len(XYs), 2):
        vertex1 = XYs[i]
        vertex2 = XYs[(i+5)%10]
        
        # Create a line from the center to each vertex
        line = np.vstack([vertex1, vertex2])
        symmetry_lines.append(line)
    
    return symmetry_lines

def plot_with_symmetry_lines(paths_XYs, title, ax):
    # Plot the paths using the existing plot function
    plot(paths_XYs, title, ax)
    for XYs in paths_XYs:
        add_symmetry_lines(ax, XYs)
    
    ax.set_aspect('equal')
    ax.set_title(title)

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
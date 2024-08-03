import csv
import numpy as np
import svgwrite
from scipy.spatial import distance
import xml.etree.ElementTree as ET
import math

def read_svg_lines(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    lines = []

    for elem in root.findall('.//{http://www.w3.org/2000/svg}line'):
        x1 = float(elem.attrib['x1'])
        y1 = float(elem.attrib['y1'])
        x2 = float(elem.attrib['x2'])
        y2 = float(elem.attrib['y2'])
        lines.append(((x1, y1), (x2, y2), elem))

    return tree, root, lines

def are_close(a, b, tol=1):
    return abs(a - b) < tol

def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def find_rectangles(lines, tol=1):
    rectangles = []
    used_lines = set()

    for i, (p1, p2, l1) in enumerate(lines):
        if i in used_lines:
            continue
        for j, (p3, p4, l2) in enumerate(lines):
            if j in used_lines or j <= i:
                continue
            for k, (p5, p6, l3) in enumerate(lines):
                if k in used_lines or k <= j:
                    continue
                for m, (p7, p8, l4) in enumerate(lines):
                    if m in used_lines or m <= k:
                        continue

                    # Check if we have a rectangle
                    d1 = distance(p1, p2)
                    d2 = distance(p3, p4)
                    d3 = distance(p5, p6)
                    d4 = distance(p7, p8)

                    if (are_close(d1, d3, tol) and are_close(d2, d4, tol)):
                        points = {p1, p2, p3, p4, p5, p6, p7, p8}
                        if len(points) == 4:
                            rectangles.append((p1, p2, p3, p4))
                            used_lines.update([i, j, k, m])
                            break

    return rectangles, used_lines

def replace_lines_with_rectangles(root, lines, rectangles, used_lines):
    for i, line in enumerate(lines):
        if i not in used_lines:
            continue
        root.remove(line[2])

    for rect in rectangles:
        x_coords = [p[0] for p in rect]
        y_coords = [p[1] for p in rect]
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        width = max_x - min_x
        height = max_y - min_y

        rect_elem = ET.Element('rect', {
            'x': str(min_x),
            'y': str(min_y),
            'width': str(width),
            'height': str(height),
            'style': 'fill:none;stroke:black;stroke-width:1'
        })
        root.append(rect_elem)

def process_svg(file_path, output_path):
    tree, root, lines = read_svg_lines(file_path)
    rectangles, used_lines = find_rectangles(lines)
    replace_lines_with_rectangles(root, lines, rectangles, used_lines)
    tree.write(output_path)


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

def is_almost_straight_line(polyline, threshold=10.5):
    """Check if the polyline is almost a straight line."""
    start, end = np.array(polyline[0]), np.array(polyline[-1])
    for point in polyline:
        point = np.array(point)
        if np.linalg.norm(np.cross(end-start, start-point) / np.linalg.norm(end-start)) > threshold:
            return False
    return True

def generate_svg(polylines, output_file, epsilon=0.2, threshold=10.5):
    dwg = svgwrite.Drawing(output_file, profile='tiny')
    for polyline in polylines:
        if len(polyline) > 1:  # Ensure there are at least two points to form a line
            if is_almost_straight_line(polyline, threshold):
                # Draw as a straight line
                dwg.add(dwg.line(start=polyline[0], end=polyline[-1], stroke=svgwrite.rgb(0, 0, 0, '%')))
            else:
                # Draw as Bezier curves
                simplified_polyline = rdp(polyline, epsilon)
                bezier_curves = polyline_to_bezier(simplified_polyline)
                for b0, b1, b2, b3 in bezier_curves:
                    path_data = f'M {b0[0]},{b0[1]} C {b1[0]},{b1[1]}, {b2[0]},{b2[1]}, {b3[0]},{b3[1]}'
                    dwg.add(dwg.path(d=path_data, stroke="black", fill="none"))
    dwg.save()

# Example usage:
file_path = './examples/isolated.csv'
polylines = read_csv(file_path)
generate_svg(polylines, 'output.svg', epsilon=0.2, threshold=10.5)

# Usage
# input_file = 'output.svg'
# output_file = 'output.svg'
# process_svg(input_file, output_file)
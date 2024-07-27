import numpy as np
import matplotlib.pyplot as plt
import svgwrite

def read_csv(csv_path):
    np_path_XYs = np.genfromtxt(csv_path, delimiter=',')
    path_XYs = []
    for i in np.unique(np_path_XYs[:, 0]):
        npXYs = np_path_XYs[np_path_XYs[:, 0] == i][:, 1:]
        XYs = []
        for j in np.unique(npXYs[:, 0]):
            XY = npXYs[npXYs[:, 0] == j][:, 1:]
            XYs.append(XY)
        path_XYs.append(XYs)
    return path_XYs

def plot(paths_XYs):
    fig, ax = plt.subplots(tight_layout=True, figsize=(8, 8))
    for i, XYs in enumerate(paths_XYs):
        for XY in XYs:
            ax.plot(XY[:, 0], XY[:, 1], linewidth=2)
    ax.set_aspect('equal')
    plt.show()

def is_line(XY, tolerance=1e-2):
    x0, y0 = XY[0]
    x1, y1 = XY[-1]
    line_eq = lambda x: y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return np.all(np.abs(XY[:, 1] - line_eq(XY[:, 0])) < tolerance)

def is_circle(XY, tolerance=1e-2):
    center = np.mean(XY, axis=0)
    radius = np.linalg.norm(XY[0] - center)
    return np.all(np.abs(np.linalg.norm(XY - center, axis=1) - radius) < tolerance)

def is_rectangle(XY, tolerance=1e-2):
    if len(XY) != 4:
        return False
    return is_line(XY[:2], tolerance) and is_line(XY[1:3], tolerance) and is_line(XY[2:], tolerance) and is_line(XY[[3, 0]], tolerance)

def has_reflection_symmetry(XY, tolerance=1e-2):
    mid_idx = len(XY) // 2
    return np.allclose(XY[:mid_idx], XY[:mid_idx - 1:-1], atol=tolerance)

def has_rotational_symmetry(XY, n_rotations=4, tolerance=1e-2):
    center = np.mean(XY, axis=0)
    angle = 2 * np.pi / n_rotations
    rotation_matrix = lambda theta: np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    for i in range(1, n_rotations):
        rotated_XY = (rotation_matrix(i * angle) @ (XY - center).T).T + center
        if not np.allclose(np.sort(XY, axis=0), np.sort(rotated_XY, axis=0), atol=tolerance):
            return False
    return True

def complete_curve(XY, method='linear'):
    if method == 'linear':
        return XY  # Placeholder for actual completion logic
    return XY

def polylines2svg(paths_XYs, svg_path):
    W, H = 0, 0
    for path_XYs in paths_XYs:
        for XY in path_XYs:
            W, H = max(W, np.max(XY[:, 0])), max(H, np.max(XY[:, 1]))
    padding = 0.1
    W, H = int(W + padding * W), int(H + padding * H)
    dwg = svgwrite.Drawing(svg_path, profile='tiny')
    group = dwg.g()
    for path_XYs in paths_XYs:
        for XY in path_XYs:
            path_data = [("M", (XY[0, 0], XY[0, 1]))]
            for j in range(1, len(XY)):
                path_data.append(("L", (XY[j, 0], XY[j, 1])))
            if not np.allclose(XY[0], XY[-1]):
                path_data.append(("Z", None))
            group.add(dwg.path(d=path_data))
    dwg.add(group)
    dwg.save()

# Example usage
paths = read_csv('examples/occlusion1.csv')
plot(paths)
polylines2svg(paths, 'occlusion1.svg')

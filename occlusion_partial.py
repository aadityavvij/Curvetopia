import numpy as np
import matplotlib.pyplot as plt

# Function to read and parse the CSV file
def read_csv_(csv_path):
    np_path_XYs = np.genfromtxt(csv_path, delimiter=',')
    path_XYs = []
    unique_combinations = np.unique(np_path_XYs[:, :2], axis=0)  # Get unique combinations of path_id and segment_id

    for path_id, segment_id in unique_combinations:
        npXYs = np_path_XYs[(np_path_XYs[:, 0] == path_id) & (np_path_XYs[:, 1] == segment_id)][:, 2:]  # Select all points for the path_id and segment_id, ignore the first two columns
        path_XYs.append(npXYs)

    return path_XYs

# Function to plot the paths
def plot(paths_XYs, title, ax):
    colours = ['red', 'green', 'blue', 'yellow', 'purple']  # Define some colors for plotting
    for path_index, XYs in enumerate(paths_XYs):
        c = colours[path_index % len(colours)]
        ax.plot(XYs[:, 0], XYs[:, 1], c=c, linewidth=2)
    ax.set_aspect('equal')
    ax.set_title(title)

# Function to find overlapping segments in two circular paths
def find_overlap(path1, path2, atol=0.1):
    len1, len2 = len(path1), len(path2)
    overlap_indices = []

    for i in range(len1):
        for j in range(len2):
            # Check if the points are close enough to be considered overlapping
            if np.allclose(path1[i], path2[j], atol=atol):
                # Check for forward overlap
                overlap_start1, overlap_start2 = i, j
                overlap_end1, overlap_end2 = i, j

                while np.allclose(path1[overlap_end1 % len1], path2[overlap_end2 % len2], atol=atol):
                    overlap_end1 = (overlap_end1 + 1) % len1
                    overlap_end2 = (overlap_end2 + 1) % len2

                    if overlap_end1 == overlap_start1 or overlap_end2 == overlap_start2:
                        break

                if abs(overlap_end1 - overlap_start1) > 1:
                    overlap_indices.append(((overlap_start1, overlap_end1), (overlap_start2, overlap_end2)))
                    return overlap_indices

                # Check for reverse overlap
                overlap_end1, overlap_end2 = i, j

                while np.allclose(path1[overlap_end1 % len1], path2[(overlap_start2 - (overlap_end1 - overlap_start1)) % len2], atol=atol):
                    overlap_end1 = (overlap_end1 + 1) % len1

                    if overlap_end1 == overlap_start1:
                        break

                if abs(overlap_end1 - overlap_start1) > 1:
                    overlap_indices.append(((overlap_start1, overlap_end1), (overlap_start2, overlap_start2 - (overlap_end1 - overlap_start1))))
                    return overlap_indices

    return overlap_indices

# Function to check if two vectors are almost parallel
def are_almost_parallel(vector1, vector2, tolerance=0.1):
    unit_vector1 = vector1 / np.linalg.norm(vector1)
    unit_vector2 = vector2 / np.linalg.norm(vector2)
    dot_product = np.dot(unit_vector1, unit_vector2)
    return abs(abs(dot_product) - 1) < tolerance

# Function to check if an angle is sharp at a specific point in the path
def is_sharp_angle(path, index):
    if index == 0 or index == len(path) - 1:
        return False  # Cannot calculate an angle for the first or last point
    vector1 = path[index] - path[index - 1]
    vector2 = path[index + 1] - path[index]
    return not are_almost_parallel(vector1, vector2)

# Function to remove an overlapping portion and replace it with a straight line
def remove_overlap_with_straight_line(path, overlap_start, overlap_end):
    len_path = len(path)
    
    if overlap_start < overlap_end:
        # Normal case: Remove points between overlap_start and overlap_end
        new_path = np.concatenate([path[:overlap_start], path[overlap_end:]])
    else:
        # Circular case: Remove points between overlap_start and end of path, then from start of path to overlap_end
        new_path = np.concatenate([path[:overlap_end], path[overlap_start:]])
    
    return new_path

# Function to process paths by removing overlapping portions with sharp angles
def process_paths(paths_XYs, angle_threshold=130):
    for i in range(len(paths_XYs)):
        for j in range(i + 1, len(paths_XYs)):
            overlaps = find_overlap(paths_XYs[i], paths_XYs[j])
            for (overlap1, overlap2) in overlaps:
                start1, end1 = overlap1
                start2, end2 = overlap2
                
                sharp1 = is_sharp_angle(paths_XYs[i], start1) or is_sharp_angle(paths_XYs[i], end1 - 1)
                sharp2 = is_sharp_angle(paths_XYs[j], start2) or is_sharp_angle(paths_XYs[j], end2 - 1)
                
                if sharp1 and not sharp2:
                    paths_XYs[i] = remove_overlap_with_straight_line(paths_XYs[i], start1, end1)
                elif sharp2 and not sharp1:
                    paths_XYs[j] = remove_overlap_with_straight_line(paths_XYs[j], start2, end2)
                # If both are sharp or neither are sharp, we could choose one arbitrarily or add additional logic.
    
    return paths_XYs


# Example usage
csv_path = "./examples/occlusion1.csv"

output_data1 = read_csv_(csv_path)
output_data2 = read_csv_(csv_path)
output_data2 = process_paths(output_data2)

fig, axs = plt.subplots(1, 2, tight_layout=True, figsize=(16, 8))
plot(output_data1, 'Original Data', axs[0])
plot(output_data2, 'Processed Data', axs[1])
plt.show()

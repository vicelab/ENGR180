import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# Constants
np.random.seed(42)  # For reproducibility
max_offset = 0.24
num_samples = 200
grid_size = 10
cell_size = 10

# Create figure and axis
fig, ax = plt.subplots(figsize=(8, 8))

# Draw grid lines
for x in range(grid_size + 1):
    ax.plot([x, x], [0, grid_size], 'k-', linewidth=0.5)
    ax.plot([0, grid_size], [x, x], 'k-', linewidth=0.5)

# Define original line points
line_points = np.array([
    [1.5, 1.5],
    [1.5, 3.5],
    [2.5, 4.5],
#    [3.5, 4.5],
    [5.5, 4.5],
    [5.5, 8.5],
    [9.5, 4.5]
])
# Function to compute cell center
def cell_center(point):
    cell = np.floor(point)
    return cell + 0.5


# Function to compute cell center
def cell_center(point):
    return np.floor(point) + 0.5

# Compute cell centers for each point
cell_centers = np.array([cell_center(pt) for pt in line_points])

# Function to sample points along a segment
def sample_cells(p0, p1, num_samples=num_samples):
    """
    Sample points along a straight line between two points while maintaining the original routing.
    """
    samples = np.linspace(p0, p1, num_samples)
    centers = np.array([cell_center(pt) for pt in samples])

    # Convert to structured NumPy array to maintain order
    #unique_centers = np.unique(centers, axis=0)  # Ensures ordered unique points
    #return unique_centers
    return centers

# Collect control points
control_points = [cell_centers[0]]

for i in range(len(line_points) - 1):
    segment_samples = sample_cells(line_points[i], line_points[i+1])

    for pt in segment_samples:
        if not np.allclose(pt, control_points[-1]):
            control_points.append(pt)

control_points = np.array(control_points)

# Function to perturb a point while keeping it in the same cell
def perturb_point(pt, max_offset=max_offset):
    cell = cell_center(pt)
    perturbation = np.random.uniform(-max_offset, max_offset, size=2)
    new_pt = cell + perturbation  # Keep it within the same grid cell
    return np.clip(new_pt, cell - max_offset, cell + max_offset)

perturbed_points = np.array([perturb_point(pt) for pt in control_points])

# Piecewise fitting using Cubic Spline
t_control = np.linspace(0, 1, len(perturbed_points))
cs_x = CubicSpline(t_control, perturbed_points[:, 0])
cs_y = CubicSpline(t_control, perturbed_points[:, 1])

# Generate smooth fitted curve with more points for smoothness
t_fine = np.linspace(0, 1, 10)  # Increased points for smoother curve
spline_x = cs_x(t_fine)
spline_y = cs_y(t_fine)

# Generate the smooth fitted curve
t_fine = np.linspace(0, 1, num_samples)
spline_x = cs_x(t_fine)
spline_y = cs_y(t_fine)

# Plot the original red vector line and vertices
# VERTEX POINTS >> uncomment to see result
#ax.scatter(line_points[:, 0], line_points[:, 1], color='#F2300F', marker='d', s=50, zorder=3, label="Vertices")
# CONNECTED VERTEX POINTS >> uncomment to see result
#ax.plot(line_points[:, 0], line_points[:, 1], color='#54D8B1', linestyle=':', alpha=0.8, linewidth=2, label="Cell Centroid Line Segment")

# Plot the control points (after perturbation)
# CENTROIDS OF PERTURBED POINTS >> uncomment to see result
#ax.plot(perturbed_points[:, 0], perturbed_points[:, 1], color='#F21AFF', marker='X', alpha=0.7, markersize=6, linewidth=0, label="Control Points")

def calculate_segment_length(points, cell_size):
    total_length = 0
    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i+1]
        segment_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2) * cell_size
        total_length += segment_length
    return total_length

# Calculate length of "River Path"
river_path_length = calculate_segment_length(np.column_stack((spline_x, spline_y)), 10)
print(f"The total length of the River Path is: {river_path_length} meters")

# Calculate length of "Cell Centroid Line Segment"
centroid_line_length = calculate_segment_length(line_points, 10)
print(f"The total length of the Cell Centroid Line Segment is: {centroid_line_length} meters")

# Plot the piecewise fitted smooth curve (green)
ax.plot(spline_x, spline_y, color='#3B9Ab2', linestyle='-', linewidth=2, label="River Path")


# Formatting plot appearance
ax.set_xlim(0, grid_size)
ax.set_ylim(0, grid_size)
ax.set_xticks(range(grid_size + 1))
ax.set_yticks(range(grid_size + 1))
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
ax.set_title("A River Runs Through It (1992) 100m x 100m")
ax.legend()
ax.text(0.5, 0.25, f'The total length of the River Path is: {river_path_length} meters', fontsize=10, color='#446455', fontweight='bold')
plt.show()
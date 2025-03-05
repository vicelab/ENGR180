import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, box
import matplotlib.patches as patches

def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite Voronoi regions to finite regions.
    Adapted from https://stackoverflow.com/a/20678647.
    """
    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()
    center = vor.points.mean(axis=0)

    if radius is None:
        radius = vor.points.ptp().max() * 2

    # Map each point to its ridges
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct each region
    for p1, region_index in enumerate(vor.point_region):
        vertices = vor.regions[region_index]
        if all(v >= 0 for v in vertices):
            new_regions.append(vertices)
            continue

        # Reconstruct an infinite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                continue  # already in region

            # Compute the missing endpoint
            t = vor.points[p2] - vor.points[p1]
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal vector
            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius
            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # Order region vertices in a counterclockwise direction
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = [v for _, v in sorted(zip(angles, new_region))]
        new_regions.append(new_region)

    return new_regions, np.asarray(new_vertices)

# --- Parameters for Random Points ---
n = 18  # number of random points
extent = 11  # spatial extent for x and y (points will be in [0, extent])

# Generate n random points in the xy-plane
#np.random.seed(400)  # For reproducibility; remove or change seed as needed
points = np.random.rand(n, 2) * extent

# --- Compute Voronoi Diagram (Thiessen Polygons) ---
vor = Voronoi(points)
regions, vertices = voronoi_finite_polygons_2d(vor)

# --- Define a Bounding Box for Clipping ---
margin = 0.1 * extent
min_x, max_x = points[:, 0].min() - margin, points[:, 0].max() + margin
min_y, max_y = points[:, 1].min() - margin, points[:, 1].max() + margin
bounding_box = box(min_x, min_y, max_x, max_y)

# --- Convert Regions to Polygons and Clip ---
polygons = []
for region in regions:
    poly = Polygon(vertices[region])
    poly = poly.intersection(bounding_box)
    polygons.append(poly)

# --- Define Dense Grid Lines (Spacing of 0.2) ---
xgrid = np.arange(min_x, max_x + 0.2, 0.2)
ygrid = np.arange(min_y, max_y + 0.2, 0.2)

# --- Plot 1: Random Points with Dense Grid Lines ---
plt.figure(figsize=(6, 6))
plt.scatter(points[:, 0], points[:, 1], color='blue')
for x_val in xgrid:
    plt.axvline(x=x_val, color='LightGray', linestyle='--', linewidth=0.5)
for y_val in ygrid:
    plt.axhline(y=y_val, color='LightGray', linestyle='--', linewidth=0.5)
plt.title("Random Points for Thiessen Polygons")
plt.xlabel("X")
plt.ylabel("Y")
plt.xlim(min_x, max_x)
plt.ylim(min_y, max_y)
plt.show()

# --- Plot 2: Random Points, Thiessen Polygons, and Dense Grid Lines ---
plt.figure(figsize=(6, 6))
ax = plt.gca()

# Plot dense grid lines
for x_val in xgrid:
    plt.axvline(x=x_val, color='LightGray', linestyle='--', linewidth=0.5)
for y_val in ygrid:
    plt.axhline(y=y_val, color='LightGray', linestyle='--', linewidth=0.5)

# Plot Thiessen (Voronoi) polygons
for poly in polygons:
    if not poly.is_empty:
        patch = patches.Polygon(np.array(poly.exterior.coords), fill=True, edgecolor='black', alpha=0.4)
        ax.add_patch(patch)

# Plot random points
plt.scatter(points[:, 0], points[:, 1], color='red')
plt.title("Thiessen Polygons")
plt.xlabel("X")
plt.ylabel("Y")
plt.xlim(min_x, max_x)
plt.ylim(min_y, max_y)
plt.show()
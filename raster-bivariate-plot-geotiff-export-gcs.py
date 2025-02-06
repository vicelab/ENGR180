from IPython import get_ipython
from IPython.display import display
# %%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # required for 3D plotting
from scipy.stats import multivariate_normal
# Install rasterio if not already installed
!pip install rasterio
import rasterio
from rasterio.transform import from_origin

# ---------------------------
# Step 1: Define the geographic grid centered at (-120, 37)
# ---------------------------
center_lon = -120   # 120 degrees west
center_lat = 37     # 37 degrees north
half_extent = 10    # degrees extent in each direction
resolution = 0.125  # cell size of 1/8 degree

# Define grid boundaries so that the grid is centered at (center_lon, center_lat)
x_min = center_lon - half_extent
x_max = center_lon + half_extent
y_min = center_lat - half_extent
y_max = center_lat + half_extent

# Create coordinate arrays
x = np.arange(x_min, x_max, resolution)
y = np.arange(y_min, y_max, resolution)
X, Y = np.meshgrid(x, y)

# ---------------------------
# Step 2: Define and evaluate the bivariate normal distribution
# ---------------------------
# Center the distribution at the geographic center
mean = [center_lon, center_lat]
# Use a covariance matrix with moderate spread (in degrees)
cov = [[9, 0.9], [0.9, 3]]  #this has ellipse type shape in diagonal orientation

rv = multivariate_normal(mean, cov)
# Stack coordinates and evaluate the probability density function (PDF)
pos = np.dstack((X, Y))
Z = rv.pdf(pos)

# ---------------------------
# Step 3: 3D Visualization with backdrop shadows
# ---------------------------
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Main 3D surface
surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.9)

# Add shadows:
# Place the x shadow at the background (x=x_max)
ax.contourf(X, Y, Z, zdir='x', offset=x_max, cmap='viridis', alpha=0.5)
# Place the y shadow at y=y_min
ax.contourf(X, Y, Z, zdir='y', offset=y_min, cmap='viridis', alpha=0.5)

ax.set_title('3D Visualization of Bivariate Normal Distribution with Shadows')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_zlabel('Probability Density')

# Adjust limits to include shadows
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_zlim(0, np.max(Z))

fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
plt.show()

# ---------------------------
# Step 4: 2D Visualization of the Bivariate Distribution
# ---------------------------
plt.figure(figsize=(8, 6))
plt.contourf(X, Y, Z, cmap='viridis')
plt.colorbar(label='Probability Density')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('2D View of Bivariate Normal Distribution')
plt.show()

# ---------------------------
# Step 5: Convert to raster data and save as GeoTIFF
# ---------------------------
# Define the geotransform for a geographic coordinate system.
# The from_origin function takes the top-left corner (x_min, y_max) along with cell size.
transform = from_origin(x_min, y_max, resolution, resolution)

#change path and filename -->
with rasterio.open(
    'my_exported_raster.tif', 'w',
    driver='GTiff',
    height=Z.shape[0],
    width=Z.shape[1],
    count=1,
    dtype=Z.dtype,
    crs='EPSG:4326',  # GCS coordinate system (WGS84)
    transform=transform,
) as dst:
    dst.write(Z, 1)

print("More Raster Fun!'")

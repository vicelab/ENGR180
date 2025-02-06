from IPython import get_ipython
from IPython.display import display
# %%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # required for 3D plotting
from scipy.stats import multivariate_normal
# Install rasterio if not already installed
!pip install rasterio
# Import rasterio
import rasterio
# Import from_origin
from rasterio.transform import from_origin


# ---------------------------
# Step 1: Create a grid and define the bivariate distribution
# ---------------------------

# Define grid boundaries and resolution
x_min, x_max = -10, 10
y_min, y_max = -10, 10
resolution = 0.1  # grid cell size

# Create coordinate arrays
x = np.arange(x_min, x_max, resolution)
y = np.arange(y_min, y_max, resolution)
X, Y = np.meshgrid(x, y)

# Define parameters for a bivariate normal distribution
mean = [0, 0]
cov = [[5, 0], [0, 5]]  # Variance of 5 in both directions, no covariance

# Create a multivariate normal distribution object
rv = multivariate_normal(mean, cov)

# Evaluate the distribution over the grid
# The .dstack stacks X and Y along the third dimension, yielding an array of (x, y) pairs.
pos = np.dstack((X, Y))
Z = rv.pdf(pos)

# ---------------------------
# Visualize the distribution in 2D
# ---------------------------

plt.figure(figsize=(8, 6))
plt.contourf(X, Y, Z, cmap='viridis')
plt.colorbar(label='Probability Density')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('2D View of Bivariate Normal Distribution')
plt.show()

print("Raster fun!")
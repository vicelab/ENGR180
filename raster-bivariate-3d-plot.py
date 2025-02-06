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
cov = [[2, 0], [0, 2]]  # CHANGE Variance and Covariance HERE

# Create a multivariate normal distribution object
rv = multivariate_normal(mean, cov)

# Evaluate the distribution over the grid.
# .dstack stacks X and Y along the third dimension yielding (x, y) pairs.
pos = np.dstack((X, Y))
Z = rv.pdf(pos)

# ---------------------------
# Step 2: Visualize the distribution in 3D with backdrop shadows
# ---------------------------

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the main 3D surface
surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.9)

# Add shadows to the backdrop walls:
# - Project the surface onto the x=x_min wall.
ax.contourf(X, Y, Z, zdir='x', offset=x_min, cmap='viridis', alpha=0.5)
# - Project the surface onto the y=y_min wall.
ax.contourf(X, Y, Z, zdir='y', offset=y_max, cmap='viridis', alpha=0.5)

# Set titles and labels
ax.set_title('3D Visualization of Bivariate Normal Distribution with Shadows')
ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_zlabel('Probability Density')

# Adjust limits to accommodate the shadows
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_zlim(0, np.max(Z))

fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
plt.show()
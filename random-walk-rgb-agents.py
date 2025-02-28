#creates RGB heatmap of agents that take random walks
import numpy as np
import random
import matplotlib.pyplot as plt

# Initialize grid size and parameters
# Good combos are 1,120,120 progressing to 200,120000,120
num_walks = 200 # each walk has agent for each RGB channel
num_steps = 120000 # number of steps by each agent
grid_size = 1200 # size of grid to be traversed by agent

# Function to perform a single random walk
def random_walk(start_x, start_y, steps):
    x, y = start_x, start_y
    path = [(x, y)]

    for _ in range(steps):
        move = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])  # Random move (dx, dy)
        x, y = x + move[0], y + move[1]

        # Stay within bounds
        x = max(0, min(grid_size - 1, x))
        y = max(0, min(grid_size - 1, y))

        path.append((x, y))

    return path

# Function to perform multiple random walks and generate visit counts
def generate_walk_heatmap(num_walks, num_steps):
    visit_counts = np.zeros((grid_size, grid_size), dtype=int)
    for _ in range(num_walks):
        start_x = random.randint(0, grid_size - 1)
        start_y = random.randint(0, grid_size - 1)
        walk = random_walk(start_x, start_y, num_steps)

        # Update visit counts
        for x, y in walk:
            visit_counts[x, y] += 1
    return visit_counts

# Generate heatmaps for three runs
red_heatmap = generate_walk_heatmap(num_walks, num_steps)
green_heatmap = generate_walk_heatmap(num_walks, num_steps)
blue_heatmap = generate_walk_heatmap(num_walks, num_steps)

# Normalize heatmaps to the range [0, 1]
red_heatmap = red_heatmap / red_heatmap.max()
green_heatmap = green_heatmap / green_heatmap.max()
blue_heatmap = blue_heatmap / blue_heatmap.max()

# Create an RGB composite image
rgb_image = np.zeros((grid_size, grid_size, 3), dtype=float)
rgb_image[..., 0] = red_heatmap  # Red channel
rgb_image[..., 1] = green_heatmap  # Green channel
rgb_image[..., 2] = blue_heatmap  # Blue channel

# Plot the RGB composite image
plt.figure(figsize=(8, 8))
plt.imshow(rgb_image)
plt.title("Random Walk Heatmap of "+str(num_walks)+" RGB Agents each with "+str(num_steps)+" Steps")
plt.xlabel("X")
plt.ylabel("Y")
plt.axis('off')
plt.show()

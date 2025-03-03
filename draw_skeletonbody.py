import matplotlib.pyplot as plt

# List of points to plot
ori_points = [
    [120, 10], # Nose
    [40, 10], # R Eye
    [20, 10], # L Eye
    [30, 10], # R Ear
    [210, 10], # L Ear
    [100, 60], # L Shoulder
    [140, 60], # R Shoulder
    [ 90, 90], # L Elbow
    [150, 90], # R Elbow
    [ 80, 120], # LWrist
    [160, 120],  # RWrist 
    [100, 130], # L hip
    [140,130], # R hip
    [90, 180], # L knee
    [150,180], # R knee
    [80, 230], # L Ankle
    [160, 230], # R Ankle
]
points = []
for i ,point in enumerate(ori_points):
    if i == 1 or i == 2 or i == 3 or i == 4:
        continue
    points.append(point)

# Separate x and y coordinates for plotting
x_coords, y_coords = zip(*points)
y_coords = [-y for y in y_coords]
# Plot the points
# figure size 255 x 255
plt.figure(figsize=(25.5,25.5))
plt.scatter(x_coords,y_coords, color='blue')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Scatter Plot of Points')

# Save the plot as an image
output_path = "points_scatter_plot.png"
plt.savefig(output_path)


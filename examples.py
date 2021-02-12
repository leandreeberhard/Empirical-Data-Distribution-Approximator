# This file contains two examples: one for two-dimensional data and the other for 3-dimensional test data
# See the 3 dimensional to see how algorithm works
from distribution_approximator_new import SampleGenerator
import numpy as np
from math import ceil, sqrt
import matplotlib.pyplot as plt
import seaborn as sns

######################################################
# Example 1: 3D data
######################################################

precision = 10
delta = (1 / ceil(precision * (precision-1)))**2
density = ceil(precision/2)

# some test points representing a distribution
test_data = np.array([[1, 1.5, 3.5], [1, 2, 3], [5, 10, 2], [3, 3, 1], [-1, -1, -5]])

# create a new distribution sampler object using the test data
sg = SampleGenerator(test_data, precision, density, delta)

# sample some points from the estimated distribution
sample_points = sg.sample_points(n_samples=1000)

from mpl_toolkits import mplot3d
fig = plt.figure()
ax = plt.axes(projection='3d')

# plot sampled points in blue
ax.scatter3D(sample_points[:,0], sample_points[:,1], sample_points[:,2], color='Blue', alpha=0.2)

# plot data points in red
ax.scatter3D(test_data[:, 0], test_data[:, 1], test_data[:, 2], color='Red', alpha=1)

plt.show()

######################################################
# Example 2: 2D data
######################################################
# generate some normally distributed points in 2D that are not independently distributed
n = 10000
z_1 = np.random.normal(loc=0, scale=1, size=n)
z = np.random.normal(loc=0, scale=1, size=n)
z_2 = np.sign(z)*z_1

z_1 = z_1.reshape((n, 1))
z_2 = z_2.reshape((n, 1))

# put these new points into an array
test_data = np.concatenate((z_1, z_2), axis=1)

# plot the original distribution in blue, smoothed using a kde smoother
graph = sns.jointplot(x=test_data[:, 0], y=test_data[:, 1], kind="kde", space=0, color='red')
plt.show()

# set parameters
precision = 100
delta = (1 / ceil(precision * (precision-1)))**2
density = ceil(precision/2)

# create a new distribution sampler object using the test data
sg = SampleGenerator(test_data, precision, density, delta)

# check that there are less subcubes with mass than points
print(f'There are {len(list(sg.true_masses.keys()))} subcubes for {test_data.shape[0]} points')
print(f'There are {len(set(sg.true_masses.values()))} distinct masses.')

# generate a new sample from the estimated distribution
sampled_points = sg.sample_points(10000)

# plot newly sampled points by themselves
sns.jointplot(x=sampled_points[:, 0], y=sampled_points[:, 1], kind="kde", space=0, color='blue')
plt.show()

# plot the newly sampled points onto the same plot as the original points
graph.x = sampled_points[:, 0]
graph.y = sampled_points[:, 1]

graph.plot_joint(sns.kdeplot, color='blue')

plt.show()
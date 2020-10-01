import numpy as np
import pandas as pd
import scipy.stats as sp
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from generate_histogram_approximation import *
from generate_rotating_plot import *
from itertools import product


# make a 3d scatter plot
# arguments:
# x,y,z all vectors of the same length
def scatter3D(x, y, z, size=5):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c='skyblue', s=size)
    ax.view_init(30, 185)
    plt.show()


# function for the tentmap
# arguments
# x: scalar value
# s: number of pieces in the tentmap
def tentmap(x, s):
    if np.floor(x * s) % 2 == 0:
        return x * s - np.floor(x * s)
    else:
        return 1 - x * s + np.floor(x * s)


# functions that take a vector as input
def tentmap_vec(vec, s):
    return [tentmap(x, s) for x in vec]


def piece_linear_vec(vec, slopes, intercepts, breakpoints_x):
    return [piece_linear(x, slopes, intercepts, breakpoints_x) for x in vec]


size = 10000
# generate uniform points in one dimension
points_unif = np.random.uniform(size=size)

# generate uniformly spaced vector of values between 0 and 1
points_x = np.arange(0, size + 1) / size

# interactive 3D plot
scatter3D(np.repeat(0, size), np.repeat(0, size), points_unif)

# generate some valid slops and intercepts
m_tilde1 = [.8, .2]
c_tilde1 = [0, 1/2, 1]
n1 = len(m_tilde1) - 1

# x-breakpoints of the piecewise linear map as the cumulative sums of the weights
breakpoints_x1 = [sum(m_tilde1[0:i]) for i in range(0, n1 + 2)]
# generate slopes and intercepts as input for piece_linear
s1 = [(c_tilde1[i] - c_tilde1[i - 1]) / (breakpoints_x1[i] - breakpoints_x1[i - 1]) for i in range(1, n1 + 2)]
int1 = [
    (c_tilde1[i - 1] * breakpoints_x1[i] - c_tilde1[i] * breakpoints_x1[i - 1]) / (breakpoints_x1[i] - breakpoints_x1[i - 1])
    for i in range(1, n1 + 2)]

linear_out1 = np.array([piece_linear(x, slopes=s1, intercepts=int1, breakpoints_x=breakpoints_x1) for x in points_x])

plt.plot(points_x, linear_out1)



# second PWL function
m_tilde2 = [.3, .7]
c_tilde2 = [0, 1/2, 1]
n2 = len(m_tilde2) - 1

# x-breakpoints of the piecewise linear map as the cumulative sums of the weights
breakpoints_x2 = [sum(m_tilde2[0:i]) for i in range(0, n2 + 2)]
# generate slopes and intercepts as input for piece_linear
s2 = [(c_tilde2[i] - c_tilde2[i - 1]) / (breakpoints_x2[i] - breakpoints_x2[i - 1]) for i in range(1, n2 + 2)]
int2 = [
    (c_tilde2[i - 1] * breakpoints_x2[i] - c_tilde2[i] * breakpoints_x2[i - 1]) / (breakpoints_x2[i] - breakpoints_x2[i - 1])
    for i in range(1, n2 + 2)]

linear_out2 = np.array([piece_linear(x, slopes=s2, intercepts=int2, breakpoints_x=breakpoints_x2) for x in points_x])

plt.plot(points_x, linear_out2)


# plot the input points and the points after being transformed
plt.plot(points_x, np.repeat(1, size + 1), c='blue', linewidth=5, alpha=.7)

# regular uniform distribution
plt.hist(points_x, density=True, bins=3)

# piecewise-linear plot
pwl, ax = plt.subplots()
ax.scatter(points_x, linear_out, c='blue', s=2)
# horizontal lines
xline1 = mlines.Line2D(xdata=[-1, .5], ydata=[1/3, 1/3], color='black')
ax.add_line(xline1)
xline2 = mlines.Line2D(xdata=[-1, .6], ydata=[2/3, 2/3], color='black')
ax.add_line(xline2)
# vertical lines
yline1 = mlines.Line2D(xdata=[.5, .5], ydata=[-1, 1/3], color='black')
ax.add_line(yline1)
yline2 = mlines.Line2D(xdata=[.6, .6], ydata=[-1, 2/3], color='black')
ax.add_line(yline2)
# change ticks
plt.yticks(ticks=[0, 1/3, 2/3, 1], labels=["0", "1/3", "2/3", "1"])
plt.xticks(ticks=[0, .5, .6, 1], labels=["0", "0.5", "0.6", "1"])
plt.show()

# histogram of pushforward
pwl_hist = plt.hist(linear_out, density=True, bins=3)
# change ticks
plt.xticks(ticks=[0, 1/3, 2/3, 1], labels=["0", "1/3", "2/3", "1"])
plt.yticks(ticks=[0, 3/10, 6/5, 3/2], labels=["0", "3/10", "6/5", "3/2"])
plt.show()

# interactive 3D plot
scatter3D(points_x, linear_out, np.repeat(0, size + 1))

tent_out = np.array([tentmap(x, 4) for x in points_x])
plt.scatter(x=points_x, y=tent_out)

# take some points in 3D
size = 10
points_x = np.arange(0, size + 1) / size

points_xy = np.array(list(product(points_x, points_x)))
points_xyz = np.array(list(product(points_x, points_x, points_x)))

scatter3D(points_xyz[:, 0], points_xyz[:, 1], points_xyz[:, 2])

# 2D tentmap with k=4
tent_xy = pd.DataFrame(
    {'x': points_x, 'y': [tentmap(x, 4) for x in points_x]}
)

plt.plot(tent_xy['x'], tent_xy['y'], c='blue')

# 2D tentmap with the piecewise linear map
tent_pwl_xy = pd.DataFrame(
    {'x': points_x, 'y': [piece_linear(tentmap(x, 4), slopes=s1, intercepts=int1, breakpoints_x=breakpoints_x) for x in points_x]}
)

plt.plot(tent_pwl_xy['x'], tent_pwl_xy['y'], c='blue')







# 3D tentmap with k=4
tent_xyz = pd.DataFrame(
    {'x': points_x, 'y': [tentmap(x, 4) for x in points_x], 'z': [tentmap(x, 16) for x in points_x]})

scatter3D(tent_xyz['x'], tent_xyz['y'], tent_xyz['z'])

# make a video of the rotating plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# X, Y, Z = axes3d.get_test_data(0.05)
# s = ax.plot_surface(X, Y, Z, cmap=cm.jet)


s = ax.scatter(tent_xyz['x'], tent_xyz['y'], tent_xyz['z'], c='blue', s=2)
# plt.axis('off') # remove axes for visual appeal

# A list of 100 angles between 0 and 360
angles = np.linspace(0,360,101)[:-1]

# create an animated gif (20ms between frames)
rotanimate(ax, angles,'sawtooth3d.gif',delay=20)


# 3D tentmap with the same piecewise linear map everywhere
tent_pwl_xyz = pd.DataFrame({'x': piece_linear_vec(points_x, slopes=s1, intercepts=int1, breakpoints_x=breakpoints_x),
                             'y': piece_linear_vec(tentmap_vec(piece_linear_vec(points_x, s1, int1, breakpoints_x), 2),
                                                   s1, int1, breakpoints_x),
                             'z': piece_linear_vec(tentmap_vec(
                                 piece_linear_vec(tentmap_vec(piece_linear_vec(points_x, s1, int1, breakpoints_x), 2),
                                                  s1, int1, breakpoints_x), 4), s1, int1, breakpoints_x)})

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(tent_pwl_xyz['x'], tent_pwl_xyz['y'], tent_pwl_xyz['z'], alpha=.05, c='blue', s=5)
plt.show()








## spike maps
c1 = .6
c2 = .8
s = 4

# equivalent constructions

# representation using only max
spikemap = np.maximum(0, s/2 * (-(1/(c2-c1)) * (np.maximum(0, 2 * points_x - c2 - c1) + np.maximum(0, -2 * points_x + c2 + c1)) + 1))
plt.plot(points_x, spikemap)


# select the correct map with the spikemap
s_bar = np.amax([s1, s2]) * 2

c1 = 0
c2 = .5
spikemap1 = np.maximum(0, s_bar/2 * (-(1/(c2-c1)) * (np.maximum(0, 2 * points_x - c2 - c1) + np.maximum(0, -2 * points_x + c2 + c1)) + 1))
plt.plot(points_x, spikemap1)

c1 = .5
c2 = 1
spikemap2 = np.maximum(0, s_bar/2 * (-(1/(c2-c1)) * (np.maximum(0, 2 * points_x - c2 - c1) + np.maximum(0, -2 * points_x + c2 + c1)) + 1))
plt.plot(points_x, spikemap2)


tent_pwl_out1 = [piece_linear(tentmap(x, 4), slopes=s1, intercepts=int1, breakpoints_x=breakpoints_x1) for x in points_x]
tent_pwl_out2 = [piece_linear(tentmap(x, 4), slopes=s2, intercepts=int2, breakpoints_x=breakpoints_x2) for x in points_x]

plt.plot(points_x, tent_pwl_out1, c='blue')
plt.plot(points_x, tent_pwl_out2, c='orange')


plt.plot(points_x, tent_pwl_out1 + spikemap1, c='blue')
plt.plot(points_x, tent_pwl_out2 + spikemap2, c='orange')


selected = np.maximum(tent_pwl_out1 + spikemap1, tent_pwl_out2 + spikemap2)
plt.plot(points_x, selected - spikemap1 - spikemap2, c='blue')

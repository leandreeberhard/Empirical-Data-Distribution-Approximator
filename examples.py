# test the whole chain

# number of subintervals to divide each axis into
precision = 20
# quantization rate. Smaller = more precise
# must satisfy delta < 1/(precision * (precision - 1))
delta = (1 / ceil(precision * (precision-1)))**2
density = ceil(precision/2)

test_data = np.array([[1, 1.5, 3.5], [1, 2, 3], [5, 10, 2], [3, 3, 1], [-1, -1, -5]])

scaled_data, scale_factors = scale_data(test_data)
true_masses = calculate_true_masses(scaled_data, precision)
quantized_masses, quantized_scaled_marginals_dict = calculate_quantized_masses(true_masses, delta, precision)
quantized_breakpoints_dict = calculate_quantized_breakpoints(quantized_scaled_marginals_dict, delta, precision)

# test the approximator
test_input = np.random.uniform(size=1000)

test_output = [approximator(x, density, precision, quantized_breakpoints_dict, quantized_scaled_marginals_dict) for x in test_input]
test_output = np.array(test_output)
# reverse the transformation from the beginning
test_output_scaled = undo_scaling(test_output, scale_factors)

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(test_output_scaled[:,0], test_output_scaled[:,1], test_output_scaled[:,2], color='Blue', alpha=0.2)
ax.scatter3D(test_data[:,0], test_data[:,1], test_data[:,2], color='Red', alpha=1)




# use the quantized masses and breakpoints to define piecewise-linear maps
from generate_histogram_approximation import *
import matplotlib.pyplot as plt

i = (10,)
k = 2
# masses
masses = [quantized_scaled_marginals_dict[2][key] for key in quantized_scaled_marginals_dict[2].keys() if key[:k-1] == i]
# breakpoints
c = quantized_breakpoints_dict[2][i]

dist = generate_histogram_dist(precision-1, c, masses, 2000000)

hist_out = plt.hist(dist, bins = c)
n_in_bins = hist_out[0]
[n_in_bins[i] / sum(n_in_bins) for i in range(len(n_in_bins))]





# plot test
import seaborn as sns
z_1 = np.random.normal(loc=0, scale=1, size=n)
z = np.random.normal(loc=0, scale=1, size=n)
z_2 = np.sign(z)*z_1

z_1 = z_1.reshape((10000,1))
z_2 = z_2.reshape((10000,1))

# save the graph for later
graph = sns.jointplot(x=z_1[:,0], y=z_2[:,0], kind="kde", space=0, color='blue')





# test the algorithm
test_data = np.concatenate((z_1, z_2), axis=1)



precision = 100
# must satisfy delta < 1/(precision * (precision - 1))
delta = (1 / ceil(precision * (precision-1)))**2

scaled_data, scale_factors = scale_data(test_data)
true_masses = calculate_true_masses(scaled_data, precision)
quantized_masses, quantized_scaled_marginals_dict = calculate_quantized_masses(true_masses, delta, precision)
quantized_breakpoints_dict = calculate_quantized_breakpoints(quantized_scaled_marginals_dict, delta, precision)


# test the approximator
test_input = np.random.uniform(size=10000)

test_output = [approximator(x, density, precision, quantized_breakpoints_dict, quantized_scaled_marginals_dict) for x in test_input]
test_output = np.array(test_output)
# reverse the transformation from the beginning
test_output_scaled = undo_scaling(test_output, scale_factors)


# plot estimated distribution onto the same plot as before
graph.x = test_output_scaled[:,0]
graph.y = test_output_scaled[:,1]

graph.plot_joint(sns.kdeplot, color='red')



# sns.jointplot(x=test_output_scaled[:,0], y=test_output_scaled[:,1], kind="kde", space=0, color='red')

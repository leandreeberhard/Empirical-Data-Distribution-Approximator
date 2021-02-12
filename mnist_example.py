######################################################
# Example 3: MNIST Numbers
######################################################
from distribution_approximator_new import SampleGenerator
import numpy as np
from math import ceil, sqrt
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pickle

# import MNIST data
path = "/Users/leandereberhard/Downloads/digit-recognizer"

mnist_data = pd.read_csv(path+'/train.csv', header=0)

data_array = np.array(mnist_data)

# given all the fours in the MNIST training set, we want to try to generate a new four

# numbers to include
numbers_incl = {2}
fours = np.array([r[1:] for r in data_array if r[0] in numbers_incl])
label = np.array([r[0] for r in data_array if r[0] in numbers_incl])

fours.shape

# plot an example
plt.imshow(np.reshape(fours[196], (28, 28)), interpolation='nearest', cmap='gray', vmin=0, vmax=255)

# reduce the dimension of the data with PCA
# scale the data
data_scaler = StandardScaler()

scaled_data = data_scaler.fit_transform(fours)

pca_fitter = PCA(n_components=50)

pca_reduced = pca_fitter.fit_transform(scaled_data)


'''
#plot 3d data
fig = plt.figure()
ax = plt.axes(projection='3d')

colors = {3: 'r', 4: 'g'}
colors_mapped = [colors[v] for v in label]


# plot sampled points in blue
ax.scatter3D(pca_reduced[:,0], pca_reduced[:,1], pca_reduced[:,2], c=colors_mapped, alpha=0.2)

# plot data points in red
ax.scatter3D(new_four[:, 0], new_four[:, 1], new_four[:, 2], color='Red', alpha=1)
'''

# variance explained
print(sum(pca_fitter.explained_variance_ratio_))

# apply inverse transformation
pca_recovered = pca_fitter.inverse_transform(pca_reduced)

# apply inverse scaling
pca_recovered_scaled = data_scaler.inverse_transform(pca_recovered)

# plot some recovered numbers
rand_ind = np.random.choice(pca_recovered_scaled.shape[0])
plt.imshow(np.reshape(pca_recovered_scaled[rand_ind], (28, 28)), interpolation='nearest', cmap='gray', vmin=0, vmax=255)








precision = 50
delta = (1 / ceil(precision * (precision-1)))**2
density = ceil(precision/2)

# train the generator
sg = SampleGenerator(pca_reduced, precision, density, delta)

# number of subcubes with mass
len(list(sg.true_masses.keys()))
set(sg.true_masses.values())

# generate a new sample
new_four = sg.sample_points(1)

# reshaped = np.reshape(new_four, (new_four.shape[1],))

'''
# experiment
new_four = np.random.sample(3) * 6

new_four = [1e6,1e6,1e6]
'''

# apply inverse PCA transformation
pca_recovered_four = pca_fitter.inverse_transform(new_four)

# apply inverse scaling
pca_recovered_scaled_four = data_scaler.inverse_transform(pca_recovered_four)

reshaped = np.reshape(pca_recovered_scaled_four, (784))



# round each value to the nearest integer
rounded = [round(x) for x in list(reshaped)]

plt.imshow(np.reshape(rounded, (28, 28)), interpolation='nearest', cmap='gray', vmin=0, vmax=255)
plt.show()

# make sure we are not just adding noise to a single example in the training set
def vector_distance(vec1, vec2):
    sum = 0
    for i in range(len(vec1)):
        sum += (vec1[i] - vec2[i])**2
    return sqrt(sum)

closest = min(pca_recovered_scaled, key = lambda x: vector_distance(x, rounded))
plt.imshow(np.reshape(closest, (28, 28)), interpolation='nearest', cmap='gray', vmin=0, vmax=255)
plt.show()


# save the generated four

# pickle.dump(new_four, open(f"generated_four_precision{precision}", "wb"))
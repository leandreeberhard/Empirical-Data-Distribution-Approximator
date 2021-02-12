# takes a nxd numpy array as input containing n data points of dimension d
import numpy as np
from math import ceil, floor
import itertools as it

# approximates the empirical distribution of the points in the array

# Step 1: scale the distribution so that the all the points are in the interval [0,1]
# Scale each coordinate separately. Save each coefficient so that the transformation can be undone at the end
# Linearly maps the maximum in each column to 1 and the minimum to 0
def linear_scaler(x, slope, intercept):
    return slope * x + intercept


def scale_data(array):
    # scale_factors stores the original min and max values for each column
    scale_factors = []
    scaled_data = np.zeros(array.shape)

    for i in range(array.shape[1]):
        current_col = array[:, i]

        max_val = max(current_col)
        min_val = min(current_col)

        scale_factors.append((min_val, max_val))

        # calculate slope and intercept of the desired linear transformation
        if max_val != min_val:
            slope = 1 / (max_val - min_val)
            intercept = -1 * slope * min_val

        else:
            slope = 1
            intercept = -max_val

        scaled_data[:, i] = [linear_scaler(x, slope, intercept) for x in current_col]

    return scaled_data, scale_factors


def undo_scaling(scaled_array, scale_factors):
    descaled_data = np.zeros(scaled_array.shape)

    for i in range(scaled_array.shape[1]):
        min_val, max_val = scale_factors[i]

        if min_val != max_val:
            slope = max_val - min_val
            intercept = min_val
        else:
            slope = 1
            intercept = max_val

        descaled_data[:, i] = [linear_scaler(x, slope, intercept) for x in scaled_array[:, i]]

    return descaled_data


# pick scaled masses and breakpoints Step 1: calculate the true masses in each subcube according to the input data
# considered as an empirical distribution only store the masses for the subcubes containing a point; otherwise,
# there is automatically a mass of 0 in that subcube
def calculate_true_masses(scaled_array, precision):

    # initialize the mass in each subcube to be 0
    true_masses = {}

    # dimension of the data
    n_points, full_dim = scaled_array.shape

    # classify each point into the correct subcube and add the mass to that subcube
    # generate the index of the subcube that each point belongs to
    for point in scaled_array:
        # initialize the index vector for the point
        generated_coordinate = [0 for _ in range(full_dim)]
        for i in range(full_dim):
            if point[i] == 0:
                generated_coordinate[i] = 1
            else:
                generated_coordinate[i] = ceil(precision * point[i])

        # convert to tuple
        generated_coordinate = tuple(generated_coordinate)

        # add the mass of the point to the correct subcube
        if generated_coordinate in true_masses.keys():
            true_masses[generated_coordinate] += 1 / n_points
        else:
            true_masses[generated_coordinate] = 1 / n_points

    return true_masses


# Step 2: calculate quantized masses and breakpoints

# previous_dim_ind should be a vector of length at most full_dim-1

# returns a vector of length precision that contains masses corresponding to the marginal mass contained in the next
# dimension with respect to previous_dim_ind

# corresponds to m_{mathbf{i}_k}

def get_marginals(true_masses, previous_dim_ind, precision):
    full_dim = len(list(true_masses.keys())[0])
    # dimension of previous_dim_ind
    sub_dim = len(previous_dim_ind)

    # make sure the requirement is satisfied
    assert sub_dim <= full_dim-1

    # initialize list of marginal weights
    marginals = [0 for _ in range(precision)]

    # sum over all subcubes containing mass in true_masses
    for i in range(1, precision+1):
        for subcube_ind in true_masses.keys():
            if subcube_ind[:sub_dim+1] == tuple(previous_dim_ind) + (i,):
                # add the mass from the matching cube to the correct entry in the marginal vector
                marginals[i-1] += true_masses[subcube_ind]

    return marginals


# returns index of the maximum
# if there is more than one index matching the maximum, return a random index
def rand_argmax(masses):
    max_mass = max(masses)

    # all indices with the maximum mass
    max_ind = [i for i in range(len(masses)) if masses[i] == max_mass]

    # return a random index
    return np.random.choice(max_ind)


# calculate quantized weights from a vector of masses
def get_quantized_masses(masses, precision, delta):
    normalization_factor = sum(masses)

    # interval from which mass is borrowed
    borrow_ind = rand_argmax(masses)

    # case 1: there is no mass at all in the provided vector of masses, so approximate a uniform distribution
    if normalization_factor == 0:
        quantized_masses = [0 for _ in masses]

        # set quantized mass in each interval except for the borrowed interval
        for i in set(range(precision)) - {borrow_ind}:
            quantized_masses[i] = delta * ceil(1 / (delta * precision))

        # set mass in borrowed interval
        quantized_masses[borrow_ind] = 1 - sum(quantized_masses)

    # case 2: there is some mass in the provided vector of masses
    else:
        scaled_masses = [m / normalization_factor for m in masses]
        quantized_masses = [0 for _ in masses]

        # set quantized masses for each interval except borrow_ind
        for i in set(range(precision)) - {borrow_ind}:
            if masses[i] == 0:
                quantized_masses[i] = delta
            else:
                quantized_masses[i] = delta * ceil(1/delta * scaled_masses[i])

        # set mass in final interval
        quantized_masses[borrow_ind] = 1 - sum(quantized_masses)

    return quantized_masses


##################################################
# Approximation of distribution using quantized masses and breakpoints
##################################################

# function for the tentmap
# arguments
# x: scalar value
# s: number of pieces in the tentmap
def tentmap(x, s):
    if floor(x * s) % 2 == 0:
        return x * s - floor(x * s)
    else:
        return 1 - x * s + floor(x * s)


# piecewise linear map, for a single value x
# masses is a list of length precision
# breakpoints is a list of length precision+1
def piecewise_linear(x, precision, breakpoints, masses):
    # x-breakpoints of the piecewise linear map as the cumulative sums of the weights
    breakpoints_x = [sum(masses[0:i]) for i in range(0, precision + 1)]
    # make sure the last breakpoint is exactly 1; can be lower due to rounding errors
    breakpoints_x[len(breakpoints_x)-1] = 1

    # generate slopes and intercepts to be used in the point slope form equation
    slopes = [(breakpoints[i] - breakpoints[i - 1]) / (breakpoints_x[i] - breakpoints_x[i - 1]) for i in range(1, precision+1)]
    intercepts = [(breakpoints[i - 1] * breakpoints_x[i] - breakpoints[i] * breakpoints_x[i - 1]) / (
                breakpoints_x[i] - breakpoints_x[i - 1]) for i in range(1, precision + 1)]

    # calculate the value of the pwl function based on the slopes and intercepts
    if x <= 0:
        return 0
    elif x >= 1:
        return max(breakpoints)
    else:
        for i in range(1, len(breakpoints_x)):
            if breakpoints_x[i - 1] < x <= breakpoints_x[i]:
                return slopes[i - 1] * x + intercepts[i - 1]


# density controls the number of teeth per tentmap, corresponds to s in the thesis
# default value of density = ceil(precision/2)
def approximator(x, true_masses, precision, density, delta):
    full_dim = len(list(true_masses.keys())[0])

    # the breakpoints for all the piecewise-linear maps are the same
    breakpoints = [i/precision for i in range(precision+1)]

    # stores the values of the tentmap/pwl approximator
    phi = [0 for _ in range(full_dim)]

    # vector that stores the index of the previous dimension
    previous_dim_ind = ()

    print(f'Coordinate 1 of {full_dim}', end='\r')
    # calculate phi for the first coordinate
    # masses are the marginals in the first dimension
    masses = get_marginals(true_masses, previous_dim_ind, precision)

    # calculate the quantized masses
    quantized_masses = get_quantized_masses(masses, precision, delta)

    # calculate the first output coordinate
    phi[0] = piecewise_linear(x, precision, breakpoints, quantized_masses)

    # set the index of the first coordinate (the index of the upper breakpoint of the interval phi[0] falls into)
    if phi[0] == 0:
        previous_dim_ind += (1,)
    else:
        previous_dim_ind += (ceil(precision * phi[0]),)

    # calculate phi for all other coordinates
    for k in range(2, full_dim+1):
        print(f'Coordinate {k} of {full_dim}', end='\r')
        # get current breakpoints and masses based on the previously calculated values of phi
        masses = get_marginals(true_masses, previous_dim_ind, precision)
        quantized_masses = get_quantized_masses(masses, precision, delta)

        # scale the coordinate generated in the previous step
        scaled_input = precision * phi[k-2] - (previous_dim_ind[len(previous_dim_ind)-1] - 1)
        scaled_input = tentmap(scaled_input, 2 * density)

        # set current coordinate of the approximator
        phi[k-1] = piecewise_linear(scaled_input, precision, breakpoints, quantized_masses)

        # add the coordinate calculated from this dimension
        if phi[k-1] == 0:
            previous_dim_ind += (1,)
        else:
            previous_dim_ind += (ceil(precision * phi[k-1]),)

    return phi


# class containing the functions
class SampleGenerator:
    def __init__(self, input_data, precision=None, density=None, delta=None):
        if precision is None:
            self.precision = 20
        else:
            self.precision = precision

        if density is None:
            self.density = ceil(self.precision/2)
        else:
            self.density = density

        if delta is None:
            self.delta = 1/(self.precision * (self.precision-1))**2
        else:
            self.delta = delta

        self.input_data = input_data

        print('Rescaling data...')
        self.scaled_data, self.scale_factors = scale_data(self.input_data)
        print('Calculating true masses...')
        self.true_masses = calculate_true_masses(self.scaled_data, self.precision)
        print('DONE')

    # run this function to generate a new sample from the distribution
    def sample_points(self, n_samples=1):
        # a point from a uniform distribution
        samples = np.random.uniform(size=n_samples)
        dim = len(list(self.true_masses.keys())[0])
        approx = np.zeros(shape=(n_samples, dim))

        for i in range(n_samples):
            print(f'Generating sample {i+1} of {n_samples}')
            approx[i, :] = approximator(samples[i], self.true_masses, self.precision, self.density, self.delta)
            print('\r\r')

        print('DONE')

        # rescale the generated points back to the original scale of the data
        return undo_scaling(approx, self.scale_factors)



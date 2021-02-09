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
        slope = 1 / (max_val - min_val)
        intercept = -1 * slope * min_val

        scaled_data[:, i] = [linear_scaler(x, slope, intercept) for x in current_col]

    return scaled_data, scale_factors


def undo_scaling(scaled_array, scale_factors):
    descaled_data = np.zeros(scaled_array.shape)

    for i in range(scaled_array.shape[1]):
        min_val, max_val = scale_factors[i]
        slope = max_val - min_val
        intercept = min_val

        descaled_data[:, i] = [linear_scaler(x, slope, intercept) for x in scaled_array[:, i]]

    return descaled_data



# pick scaled masses and breakpoints


# Step 1: calculate the true masses in each subcube according to the input data considered as an empirical distribution
# i = 1,...,precision
# corresponds to C_i in thesis
def is_in_subinterval(x, i, precision):
    if i == 1:
        return 0 <= x <= 1/precision
    else:
        return (i-1)/precision < x <= i/precision

# i is a vector of length k of indices
# x is a vector of length k containing a data point
# corresponds to C_{i_k} in thesis
def is_in_subcube(x, i, precision):
    if len(x) != len(i):
        print('Error: x and i must have the same length')
        return None
    else:
        return all(is_in_subinterval(x[j], i[j], precision) for j in range(len(x)))


def calculate_true_masses(scaled_array, precision):
    # subcube indices: returns list of length precision**dimension_of_data
    def indices_to_check():
        return it.product(*[[i for i in range(1,precision+1)] for _ in range(scaled_array.shape[1])])

    true_masses = dict((i, None) for i in indices_to_check())

    for i in indices_to_check():
        rows = [scaled_array[i,:] for i in range(scaled_array.shape[0])]
        true_masses[i] = sum([is_in_subcube(point, i, precision) for point in rows]) / scaled_array.shape[0]

    return true_masses


#######################################################
# USE THIS VERSION !!!
#######################################################
# try to assign the points to the correct subcubes immediately
# classify separately for each coordinate
def calculate_true_masses(scaled_array, precision):
    subcube_indices = it.product(*[[i for i in range(1, precision+1)] for _ in range(scaled_array.shape[1])])

    # initialize the mass in each subcube to be 0
    true_masses = dict((i, 0) for i in subcube_indices)

    # classify each point into the correct subcube and add the mass to that subcube
    for row_ind in range(scaled_array.shape[0]):
        generated_index = [None for _ in range(scaled_array.shape[1])]
        for col_ind in range(scaled_array.shape[1]):
            if scaled_array[row_ind, col_ind] == 0:
                generated_index[col_ind] = 1
            else:
                generated_index[col_ind] = ceil(scaled_array[row_ind, col_ind] * precision)

        # add mass to correct subcube
        true_masses[tuple(generated_index)] += 1 / scaled_array.shape[0]

    return true_masses


# Step 2: calculate quantized masses and breakpoints

# dim must be some integer between 1 and the dimension of true_masses (i.e. the dimension of the original data)
# corresponds to tilde{m}
def get_marginals(dim, true_masses, precision):
    full_dim = len(list(true_masses.keys())[0])

    marginal_indices = it.product(*[[i for i in range(1, precision+1)] for _ in range(dim)])
    marginals = dict((i, 0) for i in marginal_indices)

    for i in marginals.keys():
        sum_indices = it.product(*[[i for i in range(1, precision+1)] for _ in range(full_dim - dim)])
        marginals[i] = 0
        for j in sum_indices:
            marginals[i] += true_masses[i + j]

    return marginals

# get rescaled marginals so that weights sum up to 1 over the last coordinate
# corresponds to tilde{n}
def get_scaled_marginals(unscaled_marginals, precision):
    dim = len(list(unscaled_marginals.keys())[0])
    scaling_factor = get_marginals(dim-1, unscaled_marginals, precision)

    scaled_marginals = dict((i, 0) for i in unscaled_marginals.keys())

    for i in unscaled_marginals.keys():
        if scaling_factor[i[:len(i)-1]] == 0:
            scaled_marginals[i] = 1/precision
        else:
            scaled_marginals[i] = unscaled_marginals[i] / scaling_factor[i[:len(i)-1]]

    return scaled_marginals



# returns the dictionary key with the maximum value
# CHANGE TO CHOOSE RANDOM INDEX IF THERE ARE MULTIPLE ARGMAXs
def dict_argmax(dictionary):
    return max(dictionary.keys(), key=lambda key: dictionary[key])

# returns (dictionary of scaled masses, dictionary of dictionaries quantized scaled masses for each dimension)
def calculate_quantized_masses(true_masses, delta, precision):
    full_dim = len(list(true_masses.keys())[0])

    # initialize scaling factors for the quantized marginals
    previous_dim_quantized_marginals = {(): 1}

    # initialize dictionary to store the scaled quantized marginals for each dimension
    quantized_scaled_marginals_dict = {}

    for k in range(1, full_dim+1):
        marginals = get_marginals(k, true_masses, precision)
        scaled_marginals = get_scaled_marginals(marginals, precision)

        previous_dim_ind = it.product(*[[i for i in range(1, precision+1)] for _ in range(k-1)])

        # initialize the quantized scaled marginal weights
        quantized_marginals = dict((key, 0) for key in marginals.keys())
        # initialize the quantized scaled marginal weights
        quantized_scaled_marginals = dict((key, 0) for key in marginals.keys())

        for i in previous_dim_ind:

            current_slice = dict((key, marginals[key]) for key in marginals.keys() if key[:k - 1] == i)

            # borrow mass from the interval with the largest mass, for each coordinate in the previous dimension
            borrow_ind = dict_argmax(current_slice)

            # start with all the mass in the interval with the most true mass
            quantized_scaled_marginals[borrow_ind] = 1

            # set quantized masses for all intervals except the one from which we borrow mass
            for j in set(current_slice.keys()) - {borrow_ind}:
                if scaled_marginals[j] == 0:
                    quantized_weight = delta
                else:
                    quantized_weight = delta * ceil(1/delta * scaled_marginals[j])
                # write the quantized mass
                quantized_scaled_marginals[j] = quantized_weight
                # subtract the borrowed mass
                quantized_scaled_marginals[borrow_ind] -= quantized_weight

            # set the scaled quantized masses for all intervals
            for j in current_slice.keys():
                quantized_marginals[j] = quantized_scaled_marginals[j] * previous_dim_quantized_marginals[j[:k-1]]

        # set the quantized marginals for the next iteration of the loop
        previous_dim_quantized_marginals = quantized_marginals

        # save the quantized_scaled_marginals for dimension k (used for the breakpoints later)
        quantized_scaled_marginals_dict[k] = quantized_scaled_marginals

    return quantized_marginals, quantized_scaled_marginals_dict




# Step 2: Pick the quantized breakpoints using quantized_scaled_marginals_dict from calculate_quantized_masses

# returns dictionary with the quantized breakpoints for each dimension k, with (precision+1)**(k-1) entries for each k
def calculate_quantized_breakpoints(quantized_scaled_marginals_dict, delta, precision):

    full_dim = max(quantized_scaled_marginals_dict.keys())

    # initialize dictionary of quantized output breakpoints
    quantized_breakpoints_dict = dict((k, {}) for k in range(1, full_dim+1))

    for k in range(1, full_dim+1):
        previous_dim_ind = it.product(*[[i for i in range(1, precision + 1)] for _ in range(k - 1)])

        for i in previous_dim_ind:
            # g is an interger that the breakpoint c depends on
            c_list = [0] + [0 for _ in range(precision)]
            c = 0
            sum_g = 0

            for j in range(1, precision+1):
                # used to avoid rounding mistakes
                g_intermediate = 1/(delta * quantized_scaled_marginals_dict[k][i+(j,)])
                g = floor(g_intermediate / precision * j - g_intermediate * c) - sum_g
                sum_g += g
                c = delta * quantized_scaled_marginals_dict[k][i+(j,)] * sum_g + c

                # save the breakpoint
                c_list[j] = c

            # store these breakpoints
            quantized_breakpoints_dict[k][i] = c_list

    return quantized_breakpoints_dict


# use these masses and breakpoints in a piecewise-linear map
# accepts vector of breakpoints 0 = c_0 < c_1 < ... < c_n
# x must be x <= c_n
def find_breakpoint_interval(x, breakpoints):
    if x == 0:
        return 1
    else:
        def key(bp, x):
            if bp - x < 0:
                return 1
            else:
                return bp - x
        return breakpoints.index(min(breakpoints, key=lambda bp: key(bp, x)))



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
    slopes = [(breakpoints[i] - breakpoints[i - 1]) / (breakpoints_x[i] - breakpoints_x[i - 1]) for i in range(1, precision + 1)]
    intercepts = [(breakpoints[i - 1] * breakpoints_x[i] - breakpoints[i] * breakpoints_x[i - 1]) / (
                breakpoints_x[i] - breakpoints_x[i - 1]) for i in range(1, precision + 1)]

    # calculate the value of the pwl function based on the slopes and intercepts
    if x < 0:
        return 0
    elif x >= 1:
        return max(breakpoints)
    else:
        for i in range(1, len(breakpoints_x)):
            if breakpoints_x[i - 1] < x <= breakpoints_x[i]:
                return slopes[i - 1] * x + intercepts[i - 1]


# density controls the number of teeth per tentmap, corresponds to s in the thesis
# default value of density = ceil(precision/2)
def approximator(x, precision, density, quantized_breakpoints_dict, quantized_scaled_marginals_dict):
    full_dim = max(quantized_breakpoints_dict.keys())

    # stores which interval each element of phi is located in
    previous_indices = ()

    # stores the values of the tentmap/pwl approximator
    phi = [0 for _ in range(full_dim)]

    # calculate phi for the first coordinate
    breakpoints = quantized_breakpoints_dict[1][previous_indices]
    masses = [quantized_scaled_marginals_dict[1][key] for key in quantized_scaled_marginals_dict[1].keys() if key[:0] == previous_indices]

    phi[0] = piecewise_linear(x, precision, breakpoints, masses)

    # set the index of the first coordinate (the index of the upper breakpoint of the interval phi[0] falls into)
    previous_indices += (find_breakpoint_interval(phi[0], breakpoints),)

    # calculate phi for all other coordinates
    for k in range(2, full_dim+1):

        # get current breakpoints and masses based on the previously calculated values of phi
        breakpoints = quantized_breakpoints_dict[k][previous_indices]
        masses = [quantized_scaled_marginals_dict[k][key] for key in quantized_scaled_marginals_dict[k].keys() if key[:k-1] == previous_indices]

        # scale the input based on the breakpoints in the current dimension and apply a tentmap
        breakpoint_upper = breakpoints[find_breakpoint_interval(phi[0], breakpoints)]
        breakpoint_lower = breakpoints[find_breakpoint_interval(phi[0], breakpoints) - 1]

        scaled_input = (phi[k-2] - breakpoint_lower) / (breakpoint_upper - breakpoint_lower)
        scaled_input = tentmap(scaled_input, 2)

        # current value of the approximator
        phi[k-1] = piecewise_linear(tentmap(scaled_input, density), precision, breakpoints, masses)

        # add the coordinate calculated from this dimension
        previous_indices += (find_breakpoint_interval(phi[k-1], breakpoints),)

    return phi


# class containing the functions
class DistributionSampler():
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
        print('Calculating quantized masses...')
        self.quantized_masses, self.quantized_scaled_marginals_dict = calculate_quantized_masses(self.true_masses, self.delta, self.precision)
        print('Calculating quantized breakpoints...')
        self.quantized_breakpoints_dict = calculate_quantized_breakpoints(self.quantized_scaled_marginals_dict, self.delta, self.precision)
        print('DONE')

    # run this function to generate a new sample from the distribution
    def sample_points(self, n_points=1):
        # a point from a uniform distribution
        samples = np.random.uniform(size=n_points)
        approx = np.array([approximator(x, self.precision, self.density, self.quantized_breakpoints_dict, self.quantized_scaled_marginals_dict) for x in samples], ndmin=2)
        return undo_scaling(approx, self.scale_factors)



import numpy as np
import scipy.stats as sp
import matplotlib.pyplot as plt


# bound on Wasserstein Distance
# arguments:
# n : integer giving the number of breakpoints
# m : vector of length n+1 giving the total mass in each interval [(i-1)/(n+1) ,i/(n+1)] of the true distribution
# m_tilde : vector of length n+1 giving the quantized masses
# c_tilde : vector of length n+1 giving the quantized breakpoints

def wasserstein_upper(n, m, m_tilde, c_tilde):
    # add 0 as the first quantized breakpoint
    # c_tilde = np.concatenate([[0],c_tilde])
    true_breakpoints = np.arange(1, n + 2) / (n + 1)

    # vector of terms to sum over
    summands = m * true_breakpoints - .5 * c_tilde[1:(n + 2)] * m - .5 * c_tilde[0:(n + 1)] * m_tilde
    return (sum(summands))


# piecewise linear map, for a single value x
def piece_linear(x, slopes, intercepts, breakpoints_x):
    if x < 0:
        return 0
    elif x >= 1:
        return 1
    for i in range(1, len(breakpoints_x)):
        if breakpoints_x[i - 1] <= x < breakpoints_x[i]:
            return slopes[i - 1] * x + intercepts[i - 1]


# calculates the mass in the interval for a given distribution
def calc_mass_in_interval(i, n, dist_points):
    if i == 1:
        return sum(np.logical_and(dist_points >= [(i - 1) / (n + 1)], dist_points <= [i / (n + 1)]) / len(dist_points))
    else:
        return sum(np.logical_and(dist_points > [(i - 1) / (n + 1)], dist_points <= [i / (n + 1)]) / len(dist_points))


# main algorithm that generates a vector of quantized weights and breakpoints
# Arguments
# dist_points : vector containing a sample from the true distribution
# n : number of breakpoints
# delta : sets the fineness of the quantization

def calc_quantized_breakpoints_masses(dist_points, n, delta=.00001):
    m_true = [calc_mass_in_interval(i, n=n, dist_points=dist_points) for i in range(1, (n + 2))]
    c_true = np.arange(0, n + 2) / (n + 1)

    # make sure there aren't any regions with 0 mass
    if min(m_true) == 0:
        print("Error: regions with 0 mass")
        return None

    # make sure the quantization rate will not cause any issues
    if delta > 1 / (n + 1) * min(m_true):
        delta_denom = (n + 1) * 1 / min(m_true)
        if delta > 1 / np.ceil(delta_denom):
            delta = 1 / np.ceil(delta_denom)
            print(f"Delta decreased to {delta}")
        else:
            print("Decrease delta")

    # initialize mass vectors
    k = np.zeros(shape=n + 2)
    m_tilde = np.zeros(shape=n + 1)

    # initialize breakpoint vectors
    g = np.zeros(shape=n + 2)
    c_tilde = np.zeros(shape=n + 2)

    # initial values
    k[0] = 0
    k[n + 1] = 1 / delta
    g[0] = 0

    # pick the integers according to the algorithm
    for i in range(1, n + 1):
        # Pick an integer for k[i] such that k[i-1] < k[i] < 1/delta
        # and delta(k[i] - k[i-1]) >= m[i] is as close as possible

        # equivalent condition that satisfies this
        k[i] = np.ceil(1 / delta * m_true[i - 1]) + k[i - 1]
        # index on m_tilde starts at 0 for the first interval
        m_tilde[i - 1] = delta * (k[i] - k[i - 1])

        # Pick an integer for g[i] such that g[i] > -sum_{j=1}^{i-1} g_j
        # and c[i] <= i/(n+1)

        # equivalent condition for this
        g[i] = np.floor(1 / (delta * m_tilde[i - 1]) * (c_true[i] - c_tilde[i - 1])) - sum(g[0:i])
        # recursive formula for c_i; requires less computations
        c_tilde[i] = delta * m_tilde[i - 1] * sum(g[0:(i + 1)]) + c_tilde[i - 1]

    # calculate the final mass based on the initialized value for k
    m_tilde[n] = delta * (k[n + 1] - k[n])

    # pick g[n+1]
    g[n + 1] = np.floor(1 / (delta * m_tilde[n]) * (c_true[n + 1] - c_tilde[n])) - sum(g[0:(n + 1)])
    # recursive formula for c_i; requires less computations
    c_tilde[n + 1] = delta * m_tilde[n] * sum(g[0:(n + 2)]) + c_tilde[n]

    return k, g, m_tilde, c_tilde, m_true, c_true


# generate points from the approximating histogram distribution given the output of calc_quantized_breakpoints_masses
# Arguments:
# n : number of breakpoints
# c_tilde : estimated breakpoint locations; output from calc_quantized_breakpoints_masses
# m_tilde : estimated masses; output from calc_quantized_breakpoints_masses
# size : number of samples to generate

def generate_histogram_dist(n, c_tilde, m_tilde, size=1000000):
    points_unif = np.random.uniform(size=size)

    # x-breakpoints of the piecewise linear map as the cumulative sums of the weights
    breakpoints_x = [sum(m_tilde[0:i]) for i in range(0, n + 2)]

    # generate slopes and intercepts as input for piece_linear
    slopes = [(c_tilde[i] - c_tilde[i - 1]) / (breakpoints_x[i] - breakpoints_x[i - 1]) for i in range(1, n + 2)]
    intercepts = [(c_tilde[i - 1] * breakpoints_x[i] - c_tilde[i] * breakpoints_x[i - 1]) / (
                breakpoints_x[i] - breakpoints_x[i - 1]) for i in range(1, n + 2)]

    return [piece_linear(x, slopes, intercepts, breakpoints_x) for x in points_unif]


# reallocate uniformly distributed points using the piecewise-linear function specified by c_tilde and m_tilde
def reallocate_mass(points_unif, c_tilde, m_tilde, n):
    breakpoints = [sum(m_tilde[0:i]) for i in range(0, n + 2)]
    slopes = [(c_tilde[i] - c_tilde[i - 1]) / (breakpoints[i] - breakpoints[i - 1]) for i in range(1, n + 2)]
    intercepts = [
        (c_tilde[i - 1] * breakpoints[i] - c_tilde[i] * breakpoints[i - 1]) / (breakpoints[i] - breakpoints[i - 1]) for
        i in range(1, n + 2)]
    return [piece_linear(x, slopes, intercepts, breakpoints) for x in points_unif]

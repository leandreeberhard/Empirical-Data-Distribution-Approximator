The goal of this project to sample points from an empirical data source using only the samples, without any knowledge of the underlying distribution. This is similar to generative modeling, where new distinct samples resembling the given data are generated. Where this differs from bootstrap sampling is that the newly generated data points are not identical to the given data samples.

# Running the Code
Running the code is easy and can be done in three lines of code.

First, you need to import the main project file using the command
`from distribution_approximator import SampleGenerator` .

Then, create a new SampleGenerator object using your data source stored as a numpy array, with the rows containing each data point and the columns containing the entries of the data points. There are three parameters that can optionally be set:
* `precision`: Controls the number of regions the sample space is divided into. Larger values mean the estimated distribution will more closely match the given empirical distribution. Beware of overfitting when this is too high.
* `density`: Controls how densly packed the points in each region are distributed. This parameters is the number of peaks in the tentmap (see below).
* `delta`: Controls the minimal mass that can be assigned to a region of the sample space. Setting this too low will lead to a high probability of non-representative samples. Make sure not to set `delta` to be lower than `1/(precision*(precision-1))` 

`sg = SampleGenerator(data_array, precision=None, density=None, delta=None)`

Finally, we can generate new samples using the following command.
* `n_points`: sets the number of new samples to generate
`samples = sg.sample_points(n_points=1000)`



# Examples
You can generate the following two examples by running `examples.py`.


The two main ingredients of the approximator are the following:
# Piecewise-Linear Function
A piecewise-linear function is a from the interval [0,1] to an interval [0,c], where c is any positive real number with associated input breakpoints such that the function is linear between each pair of input breakpoints. The output breakpoints are the corresponding breakpoints determined by the slopes of these linear pieces and the input breakpoints.

# Tentmap / Sawtooth Function
Tentmap functions and sawtooth functions are two names for the same thing. Esentially, a sawtooth function oscilates up and down on the interval [0,1] in a straight line, where we specify the number of pieces that we want. The higher the number of pieces, the more oscilations we have. This allows us to fill higher-dimensional space.

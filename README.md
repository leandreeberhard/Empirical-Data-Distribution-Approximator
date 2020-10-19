# tentmap_plotter
Generate Plots of Tentmaps and Piecewise-Linear Maps in Python according to my Master's Thesis. The goal of this repo is to be able to estimate a distribution of data with a histogram distribution using the piecewise-linear and sawtooth/tentmap approach. Here is an explanation of the two main functions we will plot.

# Piecewise-Linear Function
A piecewise-linear function is a from the interval [0,1] to an interval [0,c], where c is any positive real number with associated input breakpoints such that the function is linear between each pair of input breakpoints. The output breakpoints are the corresponding breakpoints determined by the slopes of these linear pieces and the input breakpoints.

# Tentmap / Sawtooth Function
Tentmap functions and sawtooth functions are two names for the same thing. Esentially, a sawtooth function oscilates up and down on the interval [0,1] in a straight line, where we specify the number of pieces that we want. The higher the number of pieces, the more oscilations we have. This allows us to fill higher-dimensional space.

The file generate_hisogram_approximation.py contains the main functions.

# Running the code

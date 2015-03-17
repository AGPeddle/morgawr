# morgawr
Spectral Method Solver for Various Common 1-D PDEs

Author: Adam G. Peddle
Current version: 1.0
Date: 17 March 2015

Morgawr is a program for solving various (1+1) dimensional
Partial Differential Equations using a standard Fourier
Spectral Method for the spatial derivatives. All solutions are then
assumed to be singly periodic in the spatial domain.

Invocation of the program requires a proper config file, an
example of which is included in this repository.
Calling takes the form:

python3 morgawr controlFileName.ctr

Morgawr runs only with Python3 and is not backwards-compatible.
Morgawr depends on Numpy and Scipy.

At the moment, there are no known bugs in the Adams-Bashforth3
and Runge-Kutta4 methods. The Matrix Exponential is as of this writing
only functional for linear, homogeneous problems. In general, Morgawr 
permits the use of arbitrary forcing and initial conditions, which must be
declared in valid numpy-compatible code in the config file.

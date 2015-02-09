#!/usr/bin/env/ python3
"""
Morgawr, named for the sea monster ostensibly sighted off the coasts of Devon and Cornwall, is
a program to implement various PDE's, in particular the Rotating Shallow Water Equations (RSWE)
for the purposes of experimenting with different solvers (such as PinT methods) and possibly
for teaching or exposition.

Invoking Morgawr requires specifying the control file, which contains all necessary information
for the program to run, e.g. python3 morgawr.py control_file_name.ctr. Morgawr is not
(necessarily) backwards compatible with Python versions prior to 3.4.

Author: Adam G. Peddle
Contact: ap553@exeter.ac.uk
Version: 0.1
"""

import numpy as np
import scipy
from numpy import fft
from numpy.linalg import inv
import matplotlib.pyplot as plt
import time
import pylab
import json
import logging
import collections.abc
import time
import sys

class Control(dict):
    """
    Simple container class to hold relevant parameters for the control
    and execution of the code. Primarly populated from the control file.

    v1.0 AGP    01 Dec 2014
    """
    def __init__(self, controlFileName):

        with open(controlFileName) as controlFile:
            controlData = json.load(controlFile)

            super(Control,self).__setitem__('Lx',controlData['Control']['domainWidth'])
            super(Control,self).__setitem__('Lt',controlData['Control']['endTime'])
            super(Control,self).__setitem__('N_x',controlData['Control']['N_steps_x'])
            super(Control,self).__setitem__('N_t',controlData['Control']['N_steps_t'])
            super(Control,self).__setitem__('conv_tol',controlData['Control']['conv_tol'])
            super(Control,self).__setitem__('iter_num',controlData['Control']['iter_num'])
            super(Control,self).__setitem__('Solver',controlData['Control']['Solver'])
            super(Control,self).__setitem__('Equation',controlData['Control']['Equation'])

            physicsParams = ('diffusionCoeff',\
                               'viscosity',\
                               'celerity',\
                               'InitialConditions',\
                               'Forcing')

            for item in physicsParams:
                try:
                    super(Control,self).__setitem__(item,controlData['Physics'][item])
                except KeyError:
                    super(Control,self).__setitem__(item,False)

            super(Control,self).__setitem__('logLevel',controlData['Output']['loggingLevel'])
            super(Control,self).__setitem__('outFileStem',controlData['Output']['outFileStem'])
            super(Control,self).__setitem__('nPlots',controlData['Output']['nPlots'])

    def __getitem__(self, name):
        if super(Control, self).__getitem__(name) or name == 'Forcing':
            return super(Control, self).__getitem__(name)
        else:
            logging.error("Required parameter '{}' not present in control file".format(name))
            raise AttributeError("Required parameter '{}' not present in control file".format(name))


class Geometry:

    def __init__(self, Nx, Lx):
        self.Nx = Nx
        self.Lx = Lx
        self.k = fft.fftfreq(Nx,Lx/Nx)*2.*np.pi

        self.delta_x = self.Lx/float(self.Nx)
        self.x_grid = np.arange(float(Nx))
        self.x_grid[0] = 0.
        for k in range(1,Nx):
            self.x_grid[k] = self.x_grid[k-1] + Lx/float(Nx)


class U_xx_memo(collections.abc.Callable):
    """
    Callable class to implement the second spatial derivative of U
    Employs memoisation to reduce computational overhead
    Uses 2nd order central differences

    v1.0 AGP    13 Jan 2015

    Obsolete if using Fourier method, although memoisation could still be
    applied in principle...
    """
    def __init__(self):
        self.memo = {}

    def __call__(self,i,j,h,state):
        if (i,j) not in self.memo:
            self.memo[i,j] = (state[i-1,j] - 2*state[i,j] + state[i+1,j])/(h**2)
        return self.memo[i,j]


class Equations:
    """
    Class to contain implemented equations (e.g. Burgers, Diffusion, etc.) such that they may
    be declared in the control file and fetched by getattr. Allows general code framework to
    be applied for many different types of equation.

    v1.0 AGP 30 Jan 2015
    """

    def forcing(t_in, control, geometry):
        """
        Implements the user-specified forcing (dependent on x and/or t). User inputs must
        be valid NumPy expressions.

        v1.0 AGP    09 Feb 2015
        """
        if control['Forcing']:
            x = geometry.x_grid
            t = np.ones(control['N_x'],dtype=complex)
            t *= t_in
            force = eval(control['Forcing'])
            return fft.fft(force)
        else:
            return np.zeros(control['N_x'],dtype=complex)

    @staticmethod
    def burgers(u_hat, t, control, geometry):
        """
        Implements the 1-D burgers equation in a form which can be handled by the Fourier pseudo-spectral
        method.

        u_{t} = nu*u_{xx} - u*u_{t}, nu is viscosity

        v1.0 AGP    30 Jan 2015
        """
        #u_hat = fft.fft(u)
        #ux = fft.ifft(1j*geometry.k*u_hat)
        #uxx = fft.ifft(-geometry.k*geometry.k*u_hat)
        #return np.array(-u*ux + control['viscosity']*uxx)

        uhx = 1j*geometry.k*u_hat
        u = fft.ifft(u_hat)
        ux = fft.ifft(uhx)
        uhux = fft.fft(u*ux)
        uxx = -geometry.k*geometry.k*u_hat
        return (-uhux + control['viscosity']*uxx + forcing(t, control, geometry))

    @staticmethod
    def diffusion(u_hat, t, control, geometry):
        """
        Implementation of 1-D diffusion equation, again for Fourier pseudo-spectral methods.

        u_{t} = D*u_{xx}

        v1.0 AGP    31 Jan 2015
        """
        #u_hat = fft.fft(u)
        #uxx = fft.ifft(-geometry.k*geometry.k*u_hat)
        #return np.array(control['diffusionCoeff']*uxx)

        uxx = -geometry.k*geometry.k*u_hat
        return control['diffusionCoeff']*uxx + forcing(t, control, geometry)

    @staticmethod
    def advection(u_hat, t, control, geometry):
        """
        Implementation of simple linear advection with wave speed c.

        u_{t} = -c*u_{x}

        v1.0 AGP    09 Feb 2015
        """
        ux = 1j*geometry.k*u_hat
        return -control['celerity']*ux + Equations.forcing(t, control, geometry)

class Empty:
    """
    Kludgey empty class to permit me to remove built-in methods from solver/governing equation
    ones when returning error messages, as done in main() for the solver choice and in the
    methods of the Solver class for the governing equation choice, i.e.:

    print(set(dir(cls)) - set(dir(Empty)))

    v1.0 AGP    31 Jan 2015
    """
    pass


class Solvers:
    """
    Holds the implemented solvers to permit them to be easily and safely changed in the control
    file.

    v1.0 AGP    31 Jan 2015
    """
    @staticmethod
    def AdamsBashforth3(control, state, geometry):
        """
        Adams-Bashforth 3rd-order solver. Handles spatial derivatives with the Fourier
        pseudo-spectral method.

        v1.0 AGP    13 Jan 2015
        """
        logging.info("**********Beginning Adams-Bashforth-3 Computation**********")
        k = control['Lt']/control['N_t']
        t = 0

        try:
            govEq = getattr(Equations,control['Equation'])
        except AttributeError:
            logging.critical("Invalid Governing Equation Choice")
            print('\n Invalid Governing Equation Choice! Implemented equations are: \n' + str(set(dir(Equations)) - set(dir(Empty))) + '\n')
            raise

        #First step by Euler's Method (a.k.a AdamsBashforth1)
        j = 0
        start = time.time()
        state[:,1] = state[:,0] + k*govEq(state[:,0], t, control, geometry)
        end = time.time()
        logging.debug("Initial Euler step completed in {:.8f} seconds".format(end-start))
        t += k

        #Second step by AB2:
        j = 1
        start = time.time()
        state[:,j+1] = state[:,j] + k*(3*govEq(state[:,j], t, control, geometry) - govEq(state[:,j-1], t-k, control, geometry))
        t += k

        end = time.time()
        logging.debug("AB2 step completed in {:.8f} seconds".format(end-start))

        #AB3 for the rest of it
        for j in range(2,control['N_t']-1):
            start = time.time()
            state[:,j+1] = state[:,j] + k*(23./12.*govEq(state[:,j], t, control, geometry) - \
                16./12.*govEq(state[:,j-1], t - k, control, geometry) + 5./12.*govEq(state[:,j-2], t-2*k, control, geometry))
            t += k

            end = time.time()
            logging.debug("Timestep {:>4} completed in {:.8f} seconds".format(j,end-start))

        logging.info("**********Adams-Bashforth Computation Complete**********")
        return state

    @staticmethod
    def RungeKutta4(control, state, geometry):
        """
        Runge-Kutta 4th-order solver. Handles spatial derivatives with the Fourier
        pseudo-spectral method.

        v1.0 AGP    31 Jan 2015
        """

        logging.info("**********Beginning Runge-Kutta-4 Computation**********")
        k = control['Lt']/control['N_t']
        t = 0

        try:
            govEq = getattr(Equations,control['Equation'])
        except AttributeError:
            logging.critical("Invalid Governing Equation Choice")
            print('\n Invalid Governing Equation Choice! Implemented equations are: \n' + str(set(dir(Equations)) - set(dir(Empty))) + '\n')
            raise

        for j in range(control['N_t']-1):
            start = time.time()

            k1 = k*govEq(state[:,j], t, control, geometry)
            k2 = k*govEq(state[:,j] + 0.5*k1, t + 0.5*k, control, geometry)
            k3 = k*govEq(state[:,j] + 0.5*k2, t + 0.5*k, control, geometry)
            k4 = k*govEq(state[:,j] + k3, t + k, control, geometry)
            state[:,j+1] = state[:,j] + (k1 + 2.0*(k2 + k3) + k4)/6.0
            t += k

            end = time.time()
            logging.debug("Timestep {:>4} completed in {:.8f} seconds".format(j,end-start))

        logging.info("**********Runge-Kutta Computation Complete**********")
        return state

def scaling(control):
    """
    Will be used for scaling as the code becomes a real RSWE code and this
    becomes necessary. Until then, placeholder.

    v0.0 AGP    12 Jan 2015
    """
    pass
    logging.debug('Scaling not yet implemented')

def initialise(controlFileName):
    """
    Handles the initialisation of the code including the creation of the control object
    and the state object.

    v1.0 AGP 12 Jan 2015
    """

    #Create control structure with values from control file
    control = Control(controlFileName)

    with open(control['outFileStem'] + "_log.info", 'w'):
        pass #Clear an existing logfile
    #Then set up the logging
    try: #Check that the loglevel was set properly in the control file
        log_level = getattr(logging, control["logLevel"])
        logReset = False
    except AttributeError as ex:
        control["logLevel"] = logging.INFO
        logReset = True

    logging.basicConfig(filename = control['outFileStem'] + "_log.info", level = control['logLevel'], \
        format='%(asctime)s %(message)s', datefmt='%d/%m/%Y %H:%M:%S' )

    if logReset: logging.warning('WARNING: Logging Level set to INFO')
    logging.info('Control structure created successfully')
    logging.getLogger().addHandler(logging.StreamHandler())

    scaling(control) #Perform scaling (not yet implemented)

    geometry = Geometry(control['N_x'], control['Lx']) #Create geometry structure

    state = np.zeros(shape=(control['N_x'],control['N_t']), dtype = complex) #Create state structure for u
    #state[:,0] = 1.-np.cos(geometry.x_grid) #Initial conditions...in the future can come from user input

    x = geometry.x_grid
    state[:,0] = eval(control['InitialConditions'])
    state[:,0] = fft.fft(state[:,0])

    logging.info("Initialisation Complete")

    return control, state, geometry

def output(control, state, geometry):
    """
    Implements the output in the form of relevant plots and datasets (eventually).
    Logging is not implemented through this function. Rather, this is the post-processing
    phase of the computation.

    v1.0 AGP    12 Jan 2015
    """

    plotInterval = int(control['N_t']/control['nPlots'])

    for x in range(control['N_t']):
        state[:,x] = fft.ifft(state[:,x])

    fig = plt.figure(1)
    labels = []
    ctr = 0
    for n in range(control['N_t']):
        if n%plotInterval == 0:
            plt.plot(geometry.x_grid,np.real(state[:,n]))
            labels.append('u'+str(ctr))
            ctr += 1
    plt.title('Profiles')
    plt.xlabel('x')
    plt.ylabel('u')
    plt.legend(labels)
    plt.savefig(control['outFileStem'] + "_" + control['Equation'] + "_" + control['Solver'] + ".png")
    logging.info("Plot saved to " + control['outFileStem'] + "_" + control['Equation'] + "_" + control['Solver'] + ".png")
    plt.show()
    logging.info('Output completed successfully')

def main(controlFileName):
    """
    Main function.

    v1.0 AGP    12 Jan 2015
        Calls to initialise, output, and choice of solvers
    """

    controlFileName = ''.join(controlFileName)
    control, state, geometry = initialise(controlFileName)

    try:
        solver = getattr(Solvers,control['Solver'])
    except AttributeError:
        logging.critical("Invalid Solver Choice")
        print('\n Invalid Solver Choice! Implemented solvers are: \n' + str(set(dir(Solvers)) - set(dir(Empty))) + '\n')
        raise
    state = solver(control, state, geometry)

    output(control, state, geometry)
    logging.info("Computation Completed Successfully!")

if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except FileNotFoundError:
        print("""Control File not found. Please specify the control file.
Proper calling is: python3 morgawr.py control_file_name.ctr""")




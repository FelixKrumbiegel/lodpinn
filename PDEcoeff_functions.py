import numpy as np
import Geo
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, RegularGridInterpolator
import os
from matplotlib import gridspec
import math


class PDEcoeff_tan:
    def __init__(self, domain):
        self.domain = domain

    def __call__(self, x, eps):
        N = np.size(x, axis=0)
        interp_coeff = np.zeros((N*np.size(eps)))
        for i in range(np.size(eps)):
            interp_coeff[i*N:(i+1)*N] = np.squeeze(1./np.prod(2 -
                                                              np.sin(2*np.pi*(np.tan(np.pi*x[:, [0]]/eps[i]))), axis=1))
        return interp_coeff.flatten()[:, None]


class PDEcoeff_cont:
    def __init__(self, domain, N=10, mineps=0.1, maxeps=1., scale=0.01):
        self.domain = domain
        self.N = N
        self.scale = scale
        self.mineps = mineps
        self.maxeps = maxeps
        self.f = lambda x, y, N: np.cos(
            N*x**2*y-2*N*x**2) * np.sin(N/2*y**3*x+N*x**2) + np.cos(N*x**3-2*N*y**2)

    def __call__(self, x, eps):
        N = np.size(x, axis=0)
        interp_coeff = np.zeros((N*np.size(eps)))
        for i in range(np.size(eps)):
            fun = self.f(x[:, [0]], x[:, [1]], 1/eps[i]/self.scale)
            interp_coeff[i*N:(i+1)*N] = np.squeeze(self.mineps + (self.maxeps -
                                                                  self.mineps)*(fun-np.min(fun))/(np.max(fun)-np.min(fun)))
        return interp_coeff.flatten()[:, None]


class PDEcoeff_battery_heat:
    def __init__(self, domain, ccc=236.3, acc=398.71, mineps=10, maxeps=50):
        self.domain = domain
        self.ccc = ccc
        self.acc = acc
        self.n = 10
        self.scale = 0.01
        self.mineps = mineps
        self.maxeps = maxeps
        self.f = lambda x, y, n: np.cos(
            n*x**2*y-2*n*x**2) * np.sin(n/2*y**3*x+n*x**2) + np.cos(n*x**3-2*n*y**2)
        self.rep = 4
        self.repP = 4
        self.interpolation_points = np.linspace(
            domain[-1, 0], domain[-1, 1], 10*self.repP*self.rep+self.repP)  # 40*4+4

    def __call__(self, x, coeff_am):
        N = np.size(x, axis=0)
        interp_coeff = np.zeros((N*np.size(coeff_am)))
        for i in range(np.size(coeff_am)):
            fun = self.f(x[:, [0]], x[:, [1]], 1/coeff_am[i]/self.scale)
            rand = np.squeeze(self.mineps + (self.maxeps-self.mineps)
                              * (fun-np.min(fun))/(np.max(fun)-np.min(fun)))
            tr_filter_am = np.concatenate((np.tile(np.concatenate(
                ([0]*self.repP, [1]*4*self.repP, [0]*self.repP, [1]*4*self.repP), 0), (self.rep)), [0]*self.repP), 0)
            tr_filter = np.concatenate((np.tile(np.concatenate(([self.ccc]*self.repP, [0]*4*self.repP, [
                                       self.acc]*self.repP, [0]*4*self.repP), 0), (self.rep)), [self.ccc]*self.repP), 0)
            coeff_tr0 = interp1d(self.interpolation_points,
                                 tr_filter_am, kind="previous")
            coeff_am0 = interp1d(self.interpolation_points,
                                 tr_filter, kind="previous")
            interp_coeff[i*N:(i+1)*N] = np.squeeze(coeff_tr0(x[:, -1])
                                                   * rand + coeff_am0(x[:, -1]))
        return interp_coeff.flatten()[:, None]


class PDEcoeff_battery:
    def __init__(self, domain, ccc=236.3, acc=398.71):
        self.domain = domain
        self.ccc = ccc
        self.acc = acc
        self.interpolation_points = np.linspace(
            domain[-1, 0], domain[-1, 1], 40*4+4)

    def __call__(self, x, coeff_am):
        N = np.size(x, axis=0)
        interp_coeff = np.zeros((N*np.size(coeff_am)))
        for i in range(np.size(coeff_am)):
            tr = np.concatenate((np.tile(np.concatenate(
                ([self.ccc]*4, [10*coeff_am[i]]*16, [self.acc]*4, [10*coeff_am[i]]*16), 0), (4)), [self.ccc]*4), 0)
            coeff_interpolator = interp1d(
                self.interpolation_points, tr, kind="previous")
            interp_coeff[i*N:(i+1) *
                         N] = np.squeeze(coeff_interpolator(x[:, -1]))
        return interp_coeff.flatten()[:, None]


class PDEcoeff_random:
    def __init__(self, domain, mat1=0.2, mat2=1., p=[0.2, 0.3], nchoice=25, mat1_ind=None, mat2_ind=None):
        np.random.seed(73)
        self.domain = domain
        self.mat1 = mat1
        self.mat2 = mat2
        self.p = p
        self.nchoice = nchoice
        all_choices = range(nchoice)
        if np.any(mat1_ind) == None and np.any(mat2_ind) == None:
            assert (mat1_ind == None)
            assert (mat2_ind == None)
            self.mat1_ind = np.random.choice(
                range(nchoice), int(p[0]*nchoice), replace=False)
            self.mat2_ind = np.random.choice(np.setdiff1d(
                all_choices, self.mat1_ind), int(p[1]*nchoice), replace=False)
        else:
            self.mat1_ind = mat1_ind
            self.mat2_ind = mat2_ind
        self.eps_ind = np.setdiff1d(np.setdiff1d(
            all_choices, self.mat1_ind), self.mat2_ind)
        self.interpolation_points_x = np.linspace(
            domain[0, 0], domain[0, 1], nchoice)
        self.interpolation_points_y = np.linspace(
            domain[-1, 0], domain[-1, 1], nchoice)

    def __call__(self, x, eps):
        N = np.size(x, axis=0)
        interp_coeff = np.zeros((N*np.size(eps)))
        for i in range(np.size(eps)):
            coefficient_rand = np.ones((self.nchoice,))
            coefficient_rand[self.mat1_ind] = self.mat1
            coefficient_rand[self.mat2_ind] = self.mat2
            coefficient_rand[self.eps_ind] = eps[i]
            coeff_interpolator = interp1d(
                self.interpolation_points_x, coefficient_rand, kind="nearest")
            interp_coeff[i*N:(i+1)*N] = np.squeeze(coeff_interpolator(x[:, 0]))
            if np.size(x, 1) == 2:
                coeff_interpolator = interp1d(
                    self.interpolation_points_y, coefficient_rand, kind="nearest")
                interp_coeff[i*N:(i+1)*N] = 0.5*(interp_coeff[i*N:(i+1)
                                                              * N] + np.squeeze(coeff_interpolator(x[:, -1])))
        return interp_coeff.flatten()[:, None]


class PDEcoeff_random_pvec:
    def __init__(self, domain, nchoice=[5, 5]):
        np.random.seed(73)
        self.domain = domain
        self.nchoice = nchoice
        self.interpolation_points_x = np.linspace(0., 1., int(nchoice[0]))
        self.interpolation_points_y = np.linspace(0., 1., int(nchoice[1]))

    def __call__(self, x, p):
        N = np.size(x, axis=0)
        interp_coeff = np.zeros((N*np.size(p, 0)))
        for i in range(np.size(p, 0)):
            Z = np.random.choice(p[i], size=self.nchoice)[:, None]
            self.interp = RegularGridInterpolator((self.interpolation_points_x, self.interpolation_points_y),
                                                  np.reshape(Z, [int(self.nchoice[0]), int(self.nchoice[1])]), method="linear")
            interp_coeff[i*N:(i+1)*N] = self.interp(x)
        return interp_coeff.flatten()[:, None]


class PDEcoeff_random_1p:
    def __init__(self, domain, nchoice=[5, 5]):
        np.random.seed(73)
        self.domain = domain
        self.nchoice = nchoice
        if np.size(domain, axis=0) == 1:
            interpolation_points_x = np.linspace(
                0., 1., int(nchoice[0]), endpoint=True)
            self.interpFunc = lambda z:  interp1d(
                interpolation_points_x, z, kind="linear")
        else:
            interpolation_points_x = np.linspace(
                0., 1., int(nchoice[0]), endpoint=True)
            interpolation_points_y = np.linspace(
                0., 1., int(nchoice[1]), endpoint=True)
            self.interpFunc = lambda z: RegularGridInterpolator((interpolation_points_x, interpolation_points_y),
                                                                np.reshape(z, [int(self.nchoice[0]), int(self.nchoice[1])], "F"), method="linear")

    def __call__(self, x, p):
        N = np.size(x, axis=0)
        interp_coeff = np.zeros((N*np.size(p, 0)))
        for i in range(np.size(p, 0)):
            Z = p[i]
            self.interp = self.interpFunc(Z)
            interp_coeff[i*N:(i+1)*N] = self.interp(x).squeeze()
        return interp_coeff.flatten()[:, None]


def plot_coeff(fun, x, eps, cols, fig):
    N = np.size(eps, 0)
    rows = int(math.ceil(N / cols))
    gs = gridspec.GridSpec(rows, int(cols))
    if np.size(x, axis=1) == 1:
        ax = fig.add_subplot()
        coeff = fun(x, eps)
        ax.plot(x, coeff)
        ax.set_xlabel('x')
        ax.set_ylabel('A')
    elif np.size(x, axis=1) == 2:
        for i in range(N):
            ax = fig.add_subplot(gs[i])
            coeff = fun(x, [eps[i]])
            N = int(np.sqrt(np.size(x, 0)))
            X = np.reshape(x[:, 0], [N, N])
            Y = np.reshape(x[:, 1], [N, N])
            Z = np.reshape(coeff, [N, N])
            im = ax.pcolormesh(X, Y, Z)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title("p "+str(i))
            fig.colorbar(im)
    # fig.tight_layout()


def plotandSaveCoeff(path, fun, x, eps, index, ax, fig, label):
    coeff = fun(x, eps)
    if np.size(x, axis=1) == 1:
        ax.plot(x, coeff)
        ax.set_xlabel('x')
        ax.set_ylabel('A')
    elif np.size(x, axis=1) == 2:
        im = ax.scatter(x[:, 0], x[:, 1], c=coeff)  # ,norm='log')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        fig.colorbar(im)
        plt.savefig(path+"coeffient for "+label +
                    " parameter p="+str(index).replace(".", "_"))
    plt.close()

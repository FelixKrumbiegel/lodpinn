import numpy as np
import util
import matplotlib.pyplot as plt
import Geo
import PDEcoeff_functions
from scipy.sparse import block_diag
import timeit
import LOD


class FE_Nodal(Geo.GEO):
    def __init__(self, geo, PDEcoeff, coeff_parameters, test_parameters, withData=False, np_dtype=np.float32):
        super().__init__(geo.domain, geo.N_coarse, geo.N_fine_per_element,
                         geo.ell, geo.np_dtype, geo.elementwise)
        self.geo = geo
        self.dtype = np_dtype
        self.withData = withData
        self.PDEcoeff = PDEcoeff
        self.coeff_parameters = coeff_parameters
        self.test_parameters = test_parameters
        self.N_coeff = np.size(coeff_parameters, 0)
        self.N_test = np.size(test_parameters, 0)

        self.N_v_ppc_coarse = self.N_v_pp_coarse * self.N_coeff
        self.N_v_ppc_fine = self.N_v_pp_fine * self.N_coeff

        self.I = self.interpolator(
            self.N_coarse, self.N_fine_per_element, self.a)
        self.Icompl = self.compl_interpolator(
            self.N_coarse, self.N_fine_per_element, self.a)

        if self.withData:
            self.TrainData = self.generateData(coeff_parameters)
            self.TestData = self.generateData(test_parameters)

    def generateData(self, coeff_parameters, index="all"):
        if str(index) == "all":
            if self.elementwise:
                C = []
                for _, coeff in enumerate(coeff_parameters):
                    A = self.PDEcoeff(self.Mids, [coeff]).flatten()
                    C_tmp, _ = LOD.lodCorrector(self, A)
                    C.append(C_tmp.toarray())
            else:
                C = []
                for _, coeff in enumerate(coeff_parameters):
                    A = self.PDEcoeff(self.Mids, [coeff]).flatten()
                    _, C_tmp = LOD.lodCorrector(self, A)
                    C.append(C_tmp.toarray())
            return C

    def basisFunctions(self, index_BF="all"):
        if str(index_BF) == "all":
            BF = []
            if self.dim == 1:
                for i in range(self.N_vertices_coarse[0]):
                    BF.append(np.prod(np.maximum(
                        1. - (1/self.H)*np.abs(self.vertices_fine-self.H*i), 0.), axis=1)[:, None])
            elif self.dim == 2:
                for j in range(self.N_vertices_coarse[1]):
                    for i in range(self.N_vertices_coarse[0]):
                        index_BF = np.array([i, j])
                        BF.append(np.prod(np.maximum(
                            1. - np.abs(self.vertices_fine/self.H-index_BF), 0.), axis=1)[:, None])
        else:
            BF = np.prod(np.maximum(
                1. - (1/self.H)*np.abs(self.vertices_fine-self.H*index_BF), 0.), axis=1)[:, None]
        return BF

    def shapeFunctions1D(self, x, H, H_index_ShF):
        supp = ((x-H_index_ShF) <= H) * ((x-H_index_ShF) >= 0)
        return np.concatenate(((1-1/H*(x-H_index_ShF))*supp, 1/H*(x-H_index_ShF)*supp), 1)

    def shapeFunctions(self, index_element="all"):
        if index_element == "all":
            shF = []
            if self.dim == 1:
                for i in range(self.N_coarse[0]):
                    shF.append(self.shapeFunctions1D(
                        self.vertices_fine, self.H, self.H*i))
            elif self.dim == 2:
                for j in range(self.N_coarse[1]):
                    for i in range(self.N_coarse[0]):
                        shFx = np.tile(self.shapeFunctions1D(
                            self.vertices_fine[:, [0]], self.H[0], self.H[0]*i), (1, 2))
                        shFy = np.repeat(self.shapeFunctions1D(
                            self.vertices_fine[:, [1]], self.H[1], self.H[1]*j), 2, 1)
                        shF.append(shFx[:, [0, 1, 3, 2]] * shFy)
        else:
            if self.dim == 1:
                shF = self.shapeFunctions1D(
                    self.vertices_fine, self.H, self.H*index_element)
            elif self.dim == 2:
                if np.size(index_element) == 2:
                    shFx = np.tile(self.shapeFunctions1D(
                        self.vertices_fine[:, [0]], self.H[0], self.H[0]*index_element[0]), (1, 2))
                    shFy = np.repeat(self.shapeFunctions1D(
                        self.vertices_fine[:, [1]], self.H[1], self.H[1]*index_element[1]), 2, 1)
                else:
                    nodesId = self.nodesID2elem_coarse[index_element, :]
                    shFx = np.tile(self.shapeFunctions1D(self.vertices_fine[:, [
                                   0]], self.H[0], self.vertices_coarse[nodesId[0], 0]), (1, 2))
                    shFy = np.repeat(self.shapeFunctions1D(self.vertices_fine[:, [
                                     1]], self.H[1], self.vertices_coarse[nodesId[0], 1]), 2, 1)
                shF = shFx[:, [0, 1, 3, 2]] * shFy
        return shF

    def assembleFEMatrices(self, coeff_parameters, coeff_index="all"):
        if coeff_index == "all":
            S_fine = []
            M_fine = []
            for _, coeff in enumerate(coeff_parameters):
                A = self.PDEcoeff(self.Mids, [coeff]).flatten()
                S_fine_tmp = util.assembleGlobalMatrices(util.localStiffness(
                    self.N_fine, a=self.a), self.N_fine, 'cg', weight=A).toarray()
                M_fine_tmp = util.assembleGlobalMatrices(util.localMass(
                    self.N_fine, a=self.a), self.N_fine, 'cg').toarray()
                S_fine.append(S_fine_tmp)
                M_fine.append(M_fine_tmp)
        else:
            A = self.PDEcoeff(
                self.Mids, [coeff_parameters[coeff_index]]).flatten()
            S_fine = util.assembleGlobalMatrices(util.localStiffness(
                self.N_fine, a=self.a), self.N_fine, 'cg', weight=A).toarray()
            M_fine = util.assembleGlobalMatrices(util.localMass(
                self.N_fine, a=self.a), self.N_fine, 'cg').toarray()
        return S_fine, M_fine

    def systemMatrixDOF(self, coeff_parameters, index_element):
        S_fine = []
        for _, coeff in enumerate(coeff_parameters):
            A = self.PDEcoeff(self.Mids, coeff).flatten()
            S_fine_tmp = util.assembleGlobalMatrices(util.localStiffness(
                self.N_fine, a=self.a), self.N_fine, 'cg', weight=A)
            S_fine.append(
                S_fine_tmp[np.ix_(self.dofF[index_element], self.dofF[index_element])])
        S_fine = block_diag(S_fine)
        return S_fine.toarray()

    def interpolator(self, NC, NR, a):
        NF = NC*NR
        C = util.cg2dg(NF)
        M = util.assembleGlobalMatrices(util.localMass(NF, a=a), NF, 'dg')
        P = util.dgProlongation(NC, NR)
        MInv = util.assembleGlobalMatrices(
            util.localMassInv(NC, a=a), NC, 'dg')
        E = util.averaging(NC)
        I = E @ MInv @ P.T @ M @ C
        return I.toarray()

    def compl_interpolator(self, NC, NR, a):
        I = self.interpolator(NC, NR, a)
        P = util.cgProlongation(NC, NR)
        identity = np.eye(np.prod(NC*NR+1))
        Icompl = identity - P@I
        return Icompl

    def plot_BF(self, index_BF, ax):
        bf = self.basisFunctions(index_BF="all")
        if self.dim == 1:
            ax.plot(self.vertices_fine, bf[index_BF], label="bf")
            ax.set_title("nodal BF")
        elif self.dim == 2:
            ax.scatter(
                self.vertices_fine[:, 0], self.vertices_fine[:, 1], c=bf[index_BF], label="bf")
            ax.set_title("nodal BF")

    def plot_shapeFunctions(self, index_element, index_ShF, ax):
        bf = self.shapeFunctions(index_element)
        if self.dim == 1:
            ax.plot(self.vertices_fine, bf[:, index_ShF], label="bf")
            ax.set_title("nodal shF")
        elif self.dim == 2:
            ax.scatter(
                self.vertices_fine[:, 0], self.vertices_fine[:, 1], c=bf[:, index_ShF], label="bf")
            ax.set_title("nodal shF")

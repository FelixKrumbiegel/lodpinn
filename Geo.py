import numpy as np
import util
import matplotlib.pyplot as plt


class GEO:
    def __init__(self, domain, N_coarse, N_fine_per_element, ell=100, np_dtype=np.float32, elementwise=False):
        self.np_dtype = np_dtype
        self.ell = ell
        self.dim = np.size(N_coarse)
        assert (np.size(domain) == 2*self.dim)
        assert (np.size(N_fine_per_element) == self.dim)
        self.domain = domain
        self.N_coarse = N_coarse
        self.N_fine_per_element = N_fine_per_element
        self.H = (self.domain[:, 1]-self.domain[:, 0])/N_coarse
        self.h = self.H/N_fine_per_element
        self.a = domain[:, 1]
        self.N_fine = N_coarse * N_fine_per_element
        self.N_vertices_fine = self.N_fine + self.dim * [1]
        self.N_vertices_coarse = self.N_coarse + self.dim * [1]
        self.vertices_coarse = util.vertices(self.N_coarse, a=self.a)
        self.vertices_fine = util.vertices(self.N_fine, a=self.a)
        self.Mids = util.vertices(self.N_fine, mids=True, a=self.a)
        self.Mids_coarse = util.vertices(self.N_coarse, mids=True, a=self.a)
        self.nodesID2elem_coarse = util.elems(self.N_coarse)
        self.nodesID2elem_fine = util.elems(self.N_fine)
        self.elementwise = elementwise
        if elementwise:
            self.dofC, self.dofF, self.PatchC, self.PatchF = util.getPatches_elements(
                self.N_coarse, self.N_fine_per_element, ell)
            self.N_v_pp_coarse = [np.size(self.dofC[i])
                                  for i in range(np.prod(N_coarse))]
            self.N_v_pp_fine = [np.size(self.dofF[i])
                                for i in range(np.prod(N_coarse))]
            self.boundaryID_coarse, self.boundaryID_fine = util.getBoundaryPoints(
                self.nodesID2elem_coarse, self.nodesID2elem_fine, self.PatchC, self.PatchF)
            self.Xfine_all = [self.vertices_fine[self.dofF[i], :]
                              for i in range(np.prod(N_coarse))]
            self.patchDomains = [np.vstack([[self.Xfine_all[i][0, :]], [
                                           self.Xfine_all[i][-1, :]]]) for i in range(np.prod(N_coarse))]
        else:
            self.dofC, self.dofF, self.PatchC, self.PatchF = util.getPatches_nodes(
                self.N_coarse, self.N_fine_per_element, ell)
            self.N_v_pp_coarse = [np.size(self.dofC[i])
                                  for i in range(np.prod(N_coarse+1))]
            self.N_v_pp_fine = [np.size(self.dofF[i])
                                for i in range(np.prod(N_coarse+1))]
            self.boundaryID_coarse, self.boundaryID_fine = util.getBoundaryPoints(
                self.nodesID2elem_coarse, self.nodesID2elem_fine, self.PatchC, self.PatchF)
            self.Xfine_all = [self.vertices_fine[self.dofF[i], :]
                              for i in range(np.prod(N_coarse+1))]
            self.patchDomains = [np.vstack([[self.Xfine_all[i][0, :]], [
                                           self.Xfine_all[i][-1, :]]]) for i in range(np.prod(N_coarse+1))]

    def sampler(self, domain, N):
        if self.dim == 1:
            X = np.linspace(domain[0][0], domain[0][1],
                            N[0], dtype=self.np_dtype)
            w = 1./N
        else:
            x = np.linspace(domain[0, 0], domain[0, 1],
                            N[0], dtype=self.np_dtype)
            y = np.linspace(domain[1, 0], domain[1, 1],
                            N[1], dtype=self.np_dtype)
            xf, yf = np.meshgrid(x, y)
            w = 1./np.prod(N)
            X = np.concatenate(
                (xf.flatten()[:, None], yf.flatten()[:, None]), axis=1)
        return X, w

    def input_transform(self, X, p):
        print("Warning: input_transform is not yet implemented!")
        return X, p

    def plot_vertices(self, ax):
        if self.dim == 1:
            ax.plot(self.vertices_fine, np.zeros_like(
                self.vertices_fine), "o", label="h = "+str(self.h))
            ax.plot(self.Mids, np.zeros_like(self.Mids), "o", label="mids")
            ax.plot(self.vertices_coarse, np.zeros_like(
                self.vertices_coarse), "o", label="H = "+str(self.H))
            ax.set_title("global vertices")
            ax.legend()
        elif self.dim == 2:
            ax.scatter(
                self.vertices_fine[:, 0], self.vertices_fine[:, 1], label="h = "+str(self.h))
            ax.scatter(self.Mids[:, 0], self.Mids[:, 1], label="mids")
            ax.scatter(
                self.vertices_coarse[:, 0], self.vertices_coarse[:, 1], label="H = "+str(self.H))
            ax.set_title("global vertices")
            ax.legend()

    def plotPatches_nodes(self, index_BF, ell, ax):
        dofC, dofF, PatchC, PatchF = util.getPatches_nodes(
            self.N_coarse, self.N_fine_per_element, ell, index_BF)
        if self.dim == 1:
            ax.plot(self.vertices_fine, np.zeros_like(
                self.vertices_fine), "o", label="rest")
            ax.plot(self.vertices_fine[dofF], np.zeros_like(
                self.vertices_fine[dofF]), "o", label="fine")
            ax.plot(self.vertices_coarse[dofC], np.zeros_like(
                self.vertices_coarse[dofC]), "o", label="coarse")
            ax.set_title("patches (vertices-wise)")
            ax.legend()
        elif self.dim == 2:
            ax.scatter(self.vertices_fine[:, 0],
                       self.vertices_fine[:, 1], label="rest")
            ax.scatter(self.vertices_fine[dofF, 0],
                       self.vertices_fine[dofF, 1], label="fine")
            ax.scatter(
                self.vertices_coarse[dofC, 0], self.vertices_coarse[dofC, 1], label="coarse")
            ax.plot(self.vertices_coarse[index_BF, 0],
                    self.vertices_coarse[index_BF, 1], "rp", label="BF")
            ax.set_title("patches (vertices-wise)")
            ax.legend()

    def plotPatches_elements(self, index_element, ell, ax):
        dofC, dofF, PatchC, PatchF = util.getPatches_elements(
            self.N_coarse, self.N_fine_per_element, ell, index_element)
        if self.dim == 1:
            ax.plot(self.vertices_fine, np.zeros_like(
                self.vertices_fine), "o", label="rest")
            ax.plot(self.vertices_fine[dofF], np.zeros_like(
                self.vertices_fine[dofF]), "o", label="fine")
            ax.plot(self.vertices_coarse[dofC], np.zeros_like(
                self.vertices_coarse[dofC]), "o", label="coarse")
            ax.set_title("patches (elementwise)")
            ax.legend()
        elif self.dim == 2:
            ax.scatter(self.vertices_fine[:, 0],
                       self.vertices_fine[:, 1], label="rest")
            ax.scatter(self.vertices_fine[dofF, 0],
                       self.vertices_fine[dofF, 1], label="fine")
            ax.scatter(
                self.vertices_coarse[dofC, 0], self.vertices_coarse[dofC, 1], label="coarse")
            nodesId = self.nodesID2elem_coarse[index_element, :]
            ax.plot(self.vertices_coarse[nodesId[0], 0],
                    self.vertices_coarse[nodesId[0], 1], "rp")
            ax.plot(self.vertices_coarse[nodesId[1], 0],
                    self.vertices_coarse[nodesId[1], 1], "rp")
            ax.plot(self.vertices_coarse[nodesId[2], 0],
                    self.vertices_coarse[nodesId[2], 1], "rp")
            ax.plot(self.vertices_coarse[nodesId[3], 0],
                    self.vertices_coarse[nodesId[2], 1], "rp", label="element")

            ax.set_title("patches (elementwise)")
            ax.legend()

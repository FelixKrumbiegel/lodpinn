# evaluate_model
from pathlib import Path
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"    # which gpu to use
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.append(str(Path(__file__).parent.parent.parent))
import numpy as np
import tensorflow as tf
import timeit
import matplotlib.pyplot as plt
import json
import cProfile
from scipy.stats import qmc
import Geo
import PDEcoeff_functions
import LODPINN
import DiscreteSetting
import util


def evaluateModel(loadPath, evalCoeff="train_coeff"):
    with open(loadPath+"config.json", 'r') as configFile:
        config = json.load(configFile)
    # import parameters to create the same objects used in training
    coeff_fun = config["coeffFunction"]
    epsilon_min = config["epsilon_min"]
    epsilon_max = config["epsilon_max"]
    N_coeff = config["N_coeff"]
    domain = np.array(config["domain"])
    N_coarse = np.array(config["N_coarse"])
    N_fine_per_element = np.array(config["N_fine_per_element"])
    ell = config["ell"]
    if os.path.exists(loadPath+"trainParameter_Iter"+str(0)+".txt"):
        train_coeff = np.loadtxt(loadPath+"trainParameter_Iter"+str(0)+".txt")
        train_coeff = np.squeeze(train_coeff)
        if N_coeff == 1:
            train_coeff = np.array([train_coeff])
    else:
        train_coeff = np.linspace(
            epsilon_min, epsilon_max, N_coeff, dtype=np.float32)
    PDEcoeff_class = eval("PDEcoeff_functions.PDEcoeff_"+coeff_fun)
    #########################

    # evaluation coefficients
    if str(evalCoeff) == "train_coeff":
        eval_coeff = train_coeff
    else:
        eval_coeff = evalCoeff
    N_eval = np.size(eval_coeff, 0)
    print("eval coeff = ", eval_coeff)
    #########################
    if not os.path.exists(loadPath+"EvalCoeff"):
        os.makedirs(loadPath+"EvalCoeff")
    # Discretization
    Mesh = Geo.GEO(domain, N_coarse, N_fine_per_element, ell)

    if (coeff_fun == "battery") or (coeff_fun == "battery_heat"):
        PDEcoeff = PDEcoeff_class(domain, ccc=236.3, acc=398.71)
    else:
        PDEcoeff = PDEcoeff_class(domain, N_coarse*N_fine_per_element+[1])
    FE = DiscreteSetting.FE_Nodal(Mesh, PDEcoeff, train_coeff, eval_coeff)
    #########################

    # evaluation function
    def evaluate_all(index_learned, index2predict, mapping, wt_loc):
        dofFl = Mesh.dofF[index_learned]
        dofFp = Mesh.dofF[index2predict]
        dofCp = Mesh.dofC[index2predict]
        N = np.size(dofFl)
        Xfinet = np.tile(Mesh.vertices_fine[dofFl, :], (N_eval, 1))
        Ctestx = np.repeat(eval_coeff[:, None], N, 0)
        InputData = np.concatenate((Xfinet, Ctestx), axis=1)
        Gmodel, *_ = util.loadModel(loadPath, index_learned)
        Xa = Mesh.vertices_fine[dofFl, :][0, :]
        Xl = Mesh.vertices_fine[dofFl, :][-1, :]
        H = Xl-Xa
        sigma = 0.001
        X = Mesh.vertices_fine[dofFl, :]
        inp = (X-Xa)/H
        locFun = np.prod(2*np.arctan(np.sin(np.pi * inp) /
                         sigma)/np.pi, axis=1)[:, None]
        out = Gmodel(InputData).numpy()
        phi = np.zeros((np.size(Mesh.vertices_fine, 0), N_eval))
        ShF = FE.basisFunctions(np.array(
            [index_learned % (FE.N_coarse[0]+1), index_learned//(FE.N_coarse[1]+1)]))
        for i in range(N_eval):
            phim = mapping(ShF[dofFl]-locFun*out[i*N:(i+1)*N])
            phi[dofFp, i:i+1] = phim
            pG, pF = 0., 0.
        G = phi
        return G, pG, pF
    # Evaluate models and build G
    G_PINN = np.zeros((np.prod(N_coarse*N_fine_per_element+1),
                      np.prod(N_coarse+1), N_eval))
    start = timeit.default_timer()
    dim = 2
    wt_loc = False
    for index in range(np.prod(N_coarse+1)):
        if os.path.exists(loadPath+"correctedHatModel"+str(index)+".h5"):
            index_learned = index
            def mapping(x): return x
            mapping_str = "Id"
            wt_loc = True
            print("index", index,
                  " corresponds to learned basis function -> use index with mapping ", mapping_str)
        else:
            N_patches = np.size(FE.PatchC[index])
            if N_patches == 2**dim*(ell+1)**dim:  # interior patches
                index_learned = int(np.prod(N_coarse+1)//2 + 0.5)
                def mapping(x): return x
                mapping_str = "Id"
                wt_loc = True
                print("index", index, " corresponds to interior -> use index ",
                      index_learned, "with mapping ", mapping_str)
            elif N_patches == (ell+1)**dim:  # corner
                index_learned = np.prod(N_coarse+1)-1
                patch_shape = [ell+1, ell+1]*N_fine_per_element+[1]
                if np.all(FE.vertices_coarse[index, :] == [0, 0]):  # quadrant 1
                    def mapping(x): return np.flip(x)
                    mapping_str = "flip"
                # quadrant 2
                elif np.all(FE.vertices_coarse[index, :] == [Mesh.a[0], 0]):
                    def mapping(x): return np.flipud(np.reshape(
                        x, patch_shape, "F")).flatten()[:, None]
                    mapping_str = "flipud"
                elif np.all(FE.vertices_coarse[index, :] == Mesh.a):  # quadrant 3
                    def mapping(x): return x
                    mapping_str = "Id"
                # quadrant 4
                elif np.all(FE.vertices_coarse[index, :] == [0, Mesh.a[1]]):
                    def mapping(x): return np.fliplr(np.reshape(
                        x, patch_shape, "F")).flatten()[:, None]
                    mapping_str = "fliplr"
                print("index", index, " corresponds to corner -> use index ",
                      index_learned, "with mapping ", mapping_str)
            elif N_patches == 2*(ell+1)**dim:  # boundary
                index_learned = np.prod(N_coarse+1) - N_coarse[0]//2-1
                patch_shape = [(ell+1), 2*(ell+1)]*N_fine_per_element+[1]
                if FE.vertices_coarse[index, 1] == 0.:  # down
                    def mapping(x): return np.flipud(
                        np.reshape(x, patch_shape)).flatten()[:, None]
                    mapping_str = "flipud"
                elif FE.vertices_coarse[index, 0] == Mesh.a[0]:  # right
                    def mapping(x): return np.reshape(
                        x, patch_shape).transpose().flatten()[:, None]
                    mapping_str = "transpose"
                elif FE.vertices_coarse[index, 1] == Mesh.a[1]:  # up
                    def mapping(x): return x
                    mapping_str = "Id"
                elif FE.vertices_coarse[index, 0] == 0.:  # left
                    def mapping(x): return np.fliplr(np.reshape(
                        x, patch_shape).transpose()).flatten()[:, None]
                    mapping_str = "fliplr_transpose"
                print("index", index, " corresponds to boundary -> use index ",
                      index_learned, "with mapping ", mapping_str)
            elif N_patches == (ell+2)**dim:  # interior intersect corner
                index_learned = np.prod(N_coarse+1) - 2*N_coarse[0]-1
                patch_shape = [(ell+2), ell+2]*N_fine_per_element+[1]
                # quadrant 1
                if FE.vertices_coarse[index, 0] < Mesh.a[0]/2 and FE.vertices_coarse[index, 1] < Mesh.a[1]/2:
                    def mapping(x): return np.flipud(
                        np.reshape(x, patch_shape)).flatten()[:, None]
                    mapping_str = "flipud"
                # quadrant 2
                elif FE.vertices_coarse[index, 0] > Mesh.a[0]/2 and FE.vertices_coarse[index, 1] < Mesh.a[1]/2:
                    def mapping(x): return np.flip(x)
                    mapping_str = "flip"
                # quadrant 4
                elif FE.vertices_coarse[index, 0] < Mesh.a[0]/2 and FE.vertices_coarse[index, 1] > Mesh.a[1]/2:
                    def mapping(x): return x
                    mapping_str = "Id"
                # quadrant 3
                elif FE.vertices_coarse[index, 0] > Mesh.a[0]/2 and FE.vertices_coarse[index, 1] > Mesh.a[1]/2:
                    def mapping(x): return np.fliplr(
                        np.reshape(x, patch_shape)).flatten()[:, None]
                    mapping_str = "fliplr"
                wt_loc = True
                print("index", index, " corresponds to interior corner -> use index ",
                      index_learned, "with mapping ", mapping_str)
            # interior intersect boundary (only dim = 2)
            elif N_patches == 2*(ell+1)*(ell+2):
                index_learned = np.prod(N_coarse+1) - \
                    2-N_coarse[0]//2-N_coarse[0]
                patch_shape = [ell+2, 2*(ell+1)]*N_fine_per_element+[1]
                if FE.vertices_coarse[index, 1] < (Mesh.H[1]+1E-5):  # down
                    def mapping(x): return np.flipud(
                        np.reshape(x, patch_shape)).flatten()[:, None]
                    mapping_str = "flipud"
                # right
                elif FE.vertices_coarse[index, 0] > Mesh.a[0]-(Mesh.H[0]+1E-5):
                    def mapping(x): return np.reshape(
                        x, patch_shape).transpose().flatten()[:, None]
                    mapping_str = "transpose"
                # up
                elif FE.vertices_coarse[index, 1] > Mesh.a[1]-(Mesh.H[1]+1E-5):
                    def mapping(x): return x
                    mapping_str = "Id"
                elif FE.vertices_coarse[index, 0] < (Mesh.H[0]+1E-5):  # left
                    def mapping(x): return np.fliplr(np.reshape(
                        x, patch_shape).transpose()).flatten()[:, None]
                    mapping_str = "fliplr_transpose"
                wt_loc = True
                print("index", index, " corresponds to interior boundary -> use index ",
                      index_learned, "with mapping ", mapping_str)
            # boundary intersect corner (only dim = 2)
            elif N_patches == (ell+1)*(ell+2):
                index_learned = np.prod(N_coarse+1)-2-N_coarse[0]
                patch_shape = [(ell+2), ell+1]*N_fine_per_element+[1]
                if FE.vertices_coarse[index, 0] == 0.:
                    if FE.vertices_coarse[index, 1] < Mesh.a[1]/2:
                        def mapping(x): return np.flip(x)
                        mapping_str = "flip"
                    else:
                        def mapping(x): return np.fliplr(
                            np.reshape(x, patch_shape)).flatten()[:, None]
                        mapping_str = "fliplr"
                elif FE.vertices_coarse[index, 0] == Mesh.a[0]:
                    if FE.vertices_coarse[index, 1] < Mesh.a[1]/2:
                        def mapping(x): return np.flipud(
                            np.reshape(x, patch_shape)).flatten()[:, None]
                        mapping_str = "flipud"
                    else:
                        def mapping(x): return x
                        mapping_str = "Id"
                elif FE.vertices_coarse[index, 1] == 0.:
                    if FE.vertices_coarse[index, 0] > Mesh.a[0]/2:
                        def mapping(x): return np.flipud(np.reshape(
                            x, patch_shape).transpose()).flatten()[:, None]
                        mapping_str = "flipud_transpose"
                    else:
                        def mapping(x): return np.fliplr(
                            np.flipud(np.reshape(x, patch_shape).transpose())).flatten()[:, None]
                        mapping_str = "fliplrud_transpose"
                elif FE.vertices_coarse[index, 1] == Mesh.a[1]:
                    if FE.vertices_coarse[index, 0] > Mesh.a[0]/2:
                        def mapping(x): return np.reshape(
                            x, patch_shape).transpose().flatten()[:, None]
                        mapping_str = "transpose"
                    else:
                        def mapping(x): return np.fliplr(np.reshape(
                            x, patch_shape).transpose()).flatten()[:, None]
                        mapping_str = "fliplr_transpose"
                print("index", index, " corresponds to boundary corner -> use index ",
                      index_learned, "with mapping ", mapping_str)
        G_PINN[:, index, :], pG, pF = evaluate_all(
            index_learned, index, mapping, wt_loc)
        wt_loc = False
    stop = timeit.default_timer()
    print("total evaluation time: %.2f" % (stop - start), "s")
    return FE, PDEcoeff, eval_coeff, G_PINN

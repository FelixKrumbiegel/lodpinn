from pathlib import Path
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
import timeit
import matplotlib.pyplot as plt
import evaluate_model
import LOD
import PDEcoeff_functions
import DiscreteSetting
import util
import Geo
import solvers
from scipy.interpolate import interp1d


def compute_uref(Mesh, coeff):
    N_coarse = Mesh.N_coarse * 2*np.ones_like(Mesh.N_coarse)
    N_fine_per_element = Mesh.N_fine_per_element * \
        2 * np.ones_like(Mesh.N_coarse)
    ell = 1
    Mesh = Geo.GEO(Mesh.domain, N_coarse, N_fine_per_element, ell)
    A = PDEcoeff(Mesh.Mids, [coeff]).flatten()
    G_ref, _ = LOD.lodCorrector(Mesh, A)
    u_ref, *_ = solvers.solveHomElliptic(Mesh, G_ref, A, rhs)
    return Mesh, u_ref


if __name__ == '__main__':
    start = timeit.default_timer()
    showPlots = True
    epsilon_min = 5
    epsilon_max = 15
    domain = np.vstack([[0, 1], [0, 1]])
    loadPath = "results/case_domain_"+str(domain.tolist())+"_w128_d4_p"+str(
        epsilon_min)+"to"+str(epsilon_max)+"_c[8 8]_f[5 5]_l1_coeff20_nodal_withoutData/"

    print("+ Path to config file: ", loadPath)
    # load config file
    assert os.path.exists(
        loadPath), "This case has not been trained yet! Start train_model first!"
    rep = 4
    Mesh, PDEcoeff, eval_coeff, G_PINN = evaluate_model.evaluateModel(
        loadPath, evalCoeff="train_coeff")

    def rhs(x):
        interpolation_points = np.linspace(
            domain[1, 0], domain[1, 1], 40*rep+4)
        tr = np.concatenate(
            (np.tile(np.concatenate(([0.]*4, [1.]*16, [0]*4, [1.]*16), 0), (rep)), [0.]*4), 0)
        coeff_interpolator = interp1d(
            interpolation_points, tr, kind="previous")
        interp_coeff = np.squeeze(coeff_interpolator(x[:, 1]))
        return 3*interp_coeff.flatten()[:, None]

    def initial_cond(x, u0_const, ep):
        u0 = u0_const * np.ones_like(x[:, [1]])
        dirichlet_data = 0
        c = 3*(u0-dirichlet_data)/ep**2
        d = -2*(u0-dirichlet_data)/ep**3
        init_u = np.any(x[:, [1]] <= ep, axis=1).reshape([-1, 1]) * (u0 + c*x[:, [1]]
                                                                     ** 2 + d*x[:, [1]]**3) + np.any(x[:, [1]] > ep, axis=1).reshape([-1, 1])*u0
        return init_u

    u0_const = 0
    eps_ccc = 0.0074639*4/184
    c_am = 1010.119
    rho_am = 2094.302

    u0 = initial_cond(Mesh.vertices_fine, u0_const, eps_ccc)
    T = np.linspace(0, 1, Mesh.N_coeff, dtype=np.float32)

    def MassCoeff(x, c, rho):
        y = x[:, [1]]
        c_ccc = 897.8
        c_acc = 384.65
        interpolation_points = np.linspace(
            domain[1, 0], domain[1, 1], 40*rep+4)
        c_coeff = np.zeros((np.size(y)*np.size(c)))
        for i in range(np.size(c)):
            tr = np.concatenate((np.tile(np.concatenate(
                ([c_ccc]*4, [c[i]]*16, [c_acc]*4, [c[i]]*16), 0), (rep)), [c_ccc]*4), 0)
            coeff_interpolator = interp1d(
                interpolation_points, tr, kind="previous")
            c_coeff[i*np.size(y):(i+1)*(np.size(y))
                    ] = np.squeeze(coeff_interpolator(y))
        rho_ccc = 2706.77
        rho_acc = 8710.2
        rho_coeff = np.zeros((np.size(y)*np.size(rho)))
        for i in range(np.size(rho)):
            tr = np.concatenate((np.tile(np.concatenate(
                ([rho_ccc]*4, [rho[i]]*16, [rho_acc]*4, [rho[i]]*16), 0), (rep)), [rho_ccc]*4), 0)
            coeff_interpolator = interp1d(
                interpolation_points, tr, kind="previous")
            rho_coeff[i*np.size(y):(i+1)*(np.size(y))
                      ] = np.squeeze(coeff_interpolator(y))
        return (c_coeff*rho_coeff).flatten()[:, None]

    def mass_fun(x): return MassCoeff(x, [c_am], [rho_am])
    A_stiff = np.zeros((np.size(Mesh.Mids, 0), Mesh.N_coeff))
    A_mass = np.ones_like(A_stiff)
    G_LOD = np.zeros_like(G_PINN)
    train_err = []
    Mesh_ref, u_ref = [], []
    print("- Solve problem for all training time steps: i.e., ", T)
    for i in range(Mesh.N_coeff):
        coeff = eval_coeff[i]
        fig, ax = plt.subplots()
        PDEcoeff_functions.plotandSaveCoeff(loadPath, PDEcoeff, Mesh.vertices_fine, [
                                            coeff], i, ax, fig, "train")
        A = PDEcoeff(Mesh.Mids, [coeff]).flatten()
        A_m = mass_fun(Mesh.Mids).flatten()
        A_mass[:, [i]] = A_m[:, None]
        A_stiff[:, [i]] = A[:, None]
        G_tmp, _ = LOD.lodCorrector(Mesh, A)
        G_LOD[:, :, i] = G_tmp.toarray()
    indexPlot = 33
    util.plotandSaveSolution(
        loadPath, Mesh, G_LOD[:, indexPlot, i], i, "G"+str(indexPlot)+"_LOD", meshType="fine")
    util.plotandSaveSolution(
        loadPath, Mesh, G_PINN[:, indexPlot, i], i, "G"+str(indexPlot)+"_PINN", meshType="fine")
    util.plotandSaveSolution(
        loadPath, Mesh, G_LOD[:, indexPlot, i], Mesh.N_coeff//2, "G"+str(indexPlot)+"_LOD", meshType="fine")
    util.plotandSaveSolution(
        loadPath, Mesh, G_PINN[:, indexPlot, i], Mesh.N_coeff//2, "G"+str(indexPlot)+"_PINN", meshType="fine")
    fig = plt.figure(figsize=(6, 4))
    util.plotCrossSection(loadPath, Mesh, G_PINN[:, indexPlot, i],
                          Mesh.N_coeff, fig, "G"+str(indexPlot)+"_PINN", meshType="fine")
    util.plotCrossSection(loadPath, Mesh, G_LOD[:, indexPlot, i],
                          Mesh.N_coeff, fig, "G"+str(indexPlot)+"_LOD", meshType="fine")
    util.save_results(loadPath, Mesh.N_coeff//2, "G_cross", fig)
    plt.close()

    u_LOD, * \
        _ = solvers.solveHomParabolic(Mesh, G_LOD, A_stiff, A_mass, rhs, u0, T)
    u_PINN, * \
        _ = solvers.solveHomParabolic(
            Mesh, G_PINN, A_stiff, A_mass, rhs, u0, T)
    print("relative error basis functions: ", (np.linalg.norm(G_PINN - G_LOD, axis=0, keepdims=True)
                                               / np.linalg.norm(G_LOD, axis=0, keepdims=True)*100), "%")
    np.savetxt(loadPath+"BF_error_"+str(indexPlot)+"_coeff7"+".txt", (np.linalg.norm(G_PINN[:, :, 8] - G_LOD[:, :, 8], axis=0, keepdims=True)
                                                                      / np.linalg.norm(G_LOD[:, :, 8], axis=0, keepdims=True)*100))
    print("mean relative error basis functions: ", np.mean(np.linalg.norm(G_PINN - G_LOD, axis=0, keepdims=True)
                                                           / np.linalg.norm(G_LOD, axis=0, keepdims=True)*100), "%")
    print(np.sum(G_PINN)/np.prod(Mesh.N_vertices_fine)/Mesh.N_coeff)
    print(np.sum(G_LOD)/np.prod(Mesh.N_vertices_fine)/Mesh.N_coeff)
    train_err.append(np.linalg.norm(u_LOD-u_PINN)/np.linalg.norm(u_LOD)*100)
    util.plotandSaveSolution(
        loadPath, Mesh, u_PINN[:, 0], 0, "uPINN_train_t=0", meshType="fine")
    util.plotandSaveSolution(
        loadPath, Mesh, u_LOD[:, 0], 0, "uLOD_train_t=0", meshType="fine")
    util.plotandSaveSolution(loadPath, Mesh, u_PINN[:, Mesh.N_coeff//2],
                             eval_coeff[Mesh.N_coeff//2], "uPINN_train_t=halfTime", meshType="fine")
    util.plotandSaveSolution(loadPath, Mesh, u_LOD[:, Mesh.N_coeff//2],
                             eval_coeff[Mesh.N_coeff//2], "uLOD_train_t=halfTime", meshType="fine")
    util.plotandSaveSolution(
        loadPath, Mesh, u_PINN[:, -1], coeff, "uPINN_train_t=last", meshType="fine")
    util.plotandSaveSolution(
        loadPath, Mesh, u_LOD[:, -1], coeff, "uLOD_train_t=last", meshType="fine")

    train_err_avg = np.mean(train_err)
    print(" - train error per coeff: err= ", train_err, "%")
    print(" -> in average: err = %.2f" % (train_err_avg), "%")
    stop = timeit.default_timer()
    print("total time: %.2f" % (stop - start), "s")

    # Testing coefficients
    start = timeit.default_timer()
    if np.size(eval_coeff) > 1:
        N_eval = 25
        np.random.seed(37)
        eval_coeff = np.linspace(
            epsilon_min, epsilon_max, N_eval, dtype=np.float32)
        Mesh, PDEcoeff, eval_coeff, G_PINN = evaluate_model.evaluateModel(
            loadPath, evalCoeff=eval_coeff)
        T = np.linspace(0, 1, N_eval, dtype=np.float32)
        test_err = []
        G_LOD = np.zeros_like(G_PINN)
        A_stiff = np.zeros((np.size(Mesh.Mids, 0), N_eval))
        A_mass = np.ones_like(A_stiff)
        print("- Solve problem for all test time steps: i.e., ", T)
        for i in range(N_eval):
            coeff = eval_coeff[i]
            fig, ax = plt.subplots()
            PDEcoeff_functions.plotandSaveCoeff(loadPath, PDEcoeff, Mesh.vertices_fine, [
                                                coeff], i, ax, fig, "test")
            A = PDEcoeff(Mesh.Mids, [coeff]).flatten()
            A_m = mass_fun(Mesh.Mids).flatten()
            A_mass[:, [i]] = A_m[:, None]
            A_stiff[:, [i]] = A[:, None]
            G_tmp, _ = LOD.lodCorrector(Mesh, A)
            G_LOD[:, :, i] = G_tmp.toarray()

        util.plotandSaveSolution(
            loadPath, Mesh, G_LOD[:, indexPlot, i], i, "G"+str(indexPlot)+"_LODtest", meshType="fine")
        util.plotandSaveSolution(
            loadPath, Mesh, G_PINN[:, indexPlot, i], i, "G"+str(indexPlot)+"_PINNtest", meshType="fine")
        util.plotandSaveSolution(
            loadPath, Mesh, G_LOD[:, indexPlot, i], N_eval//2, "G"+str(indexPlot)+"_LODtest", meshType="fine")
        util.plotandSaveSolution(
            loadPath, Mesh, G_PINN[:, indexPlot, i], N_eval//2, "G"+str(indexPlot)+"_PINNtest", meshType="fine")
        u_LOD, * \
            _ = solvers.solveHomParabolic(
                Mesh, G_LOD, A_stiff, A_mass, rhs, u0, T)
        u_PINN, * \
            _ = solvers.solveHomParabolic(
                Mesh, G_PINN, A_stiff, A_mass, rhs, u0, T)
        print("relative error basis functions: ", (np.linalg.norm(G_PINN - G_LOD, axis=0, keepdims=True)
                                                   / np.linalg.norm(G_LOD, axis=0, keepdims=True)*100), "%")
        np.savetxt(loadPath+"BFtest_error_"+str(indexPlot)+"_coeff7"+".txt", (np.linalg.norm(G_PINN[:, :, 8] - G_LOD[:, :, 8], axis=0, keepdims=True)
                                                                              / np.linalg.norm(G_LOD[:, :, 8], axis=0, keepdims=True)*100))
        print("mean relative error basis functions: ", np.mean(np.linalg.norm(G_PINN - G_LOD, axis=0, keepdims=True)
                                                               / np.linalg.norm(G_LOD, axis=0, keepdims=True)*100), "%")
        print(np.sum(G_PINN)/np.prod(Mesh.N_vertices_fine)/N_eval)
        print(np.sum(G_LOD)/np.prod(Mesh.N_vertices_fine)/N_eval)
        test_err.append(np.linalg.norm(u_LOD-u_PINN)/np.linalg.norm(u_LOD)*100)
        indexPlot = int(np.ceil((np.prod(Mesh.N_coarse)+1)//2 + 0.5))
        util.plotandSaveSolution(
            loadPath, Mesh, u_PINN[:, 1], eval_coeff[1], "uPINN_test_t=0", meshType="fine")
        util.plotandSaveSolution(
            loadPath, Mesh, u_LOD[:, 1], eval_coeff[1], "uLOD_test_t=0", meshType="fine")
        util.plotandSaveSolution(
            loadPath, Mesh, u_PINN[:, N_eval//2], eval_coeff[N_eval//2], "uPINN_test_t=halfTime", meshType="fine")
        util.plotandSaveSolution(
            loadPath, Mesh, u_LOD[:, N_eval//2], eval_coeff[N_eval//2], "uLOD_test_t=halfTime", meshType="fine")
        util.plotandSaveSolution(
            loadPath, Mesh, u_PINN[:, -1], coeff, "uPINN_test_t=last", meshType="fine")
        util.plotandSaveSolution(
            loadPath, Mesh, u_LOD[:, -1], coeff, "uLOD_test_t=last", meshType="fine")

        test_err_avg = np.mean(test_err)
        print(" - test error per coeff: err= ", test_err, "%")
        print(" -> in average: err = %.2f" % (test_err_avg), "%")
        stop = timeit.default_timer()
        print("total time: %.2f" % (stop - start), "s")
    else:
        print("- Not enough coefficients to see generalization: No testing")

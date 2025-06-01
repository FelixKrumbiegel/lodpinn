from pathlib import Path
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.append(str(Path(__file__).parent.parent.parent))
import solvers
import Geo
import util
import PDEcoeff_functions
import LOD
import evaluate_model
import matplotlib.pyplot as plt
import timeit
import numpy as np


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
    epsilon_min = 0.01
    epsilon_max = 1.0
    domain = np.vstack([[0, 1], [0, 1]])

    loadPath = "results/case_domain_"+str(domain.tolist())+"_w128_d8_p"+str(
        epsilon_min)+"to"+str(epsilon_max)+"_c[6 6]_f[6 6]_l1_coeff40_nodal_withoutData/"

    print("+ Path to config file: ", loadPath)
    # load config file
    assert os.path.exists(
        loadPath), "This case has not been trained yet! Start train_model first!"

    # Training coefficients
    Mesh, PDEcoeff, eval_coeff, G_PINN = evaluate_model.evaluateModel(
        loadPath, evalCoeff="train_coeff")

    def rhs(x): return 1*np.ones((np.size(x, 0), 1))
    u0 = np.zeros_like(Mesh.vertices_fine[:, [1]])
    T = np.linspace(0, 1, Mesh.N_coeff, dtype=np.float32)
    train_err = []
    Mesh_ref, u_ref = [], []
    G_LOD = np.zeros_like(G_PINN)
    A_stiff = np.zeros((np.size(Mesh.Mids, 0), Mesh.N_coeff))
    A_mass = np.ones_like(A_stiff)
    print("- Solve problem for all training time steps: i.e., ", T)
    for i in range(Mesh.N_coeff):
        coeff = eval_coeff[i]
        fig, ax = plt.subplots()
        PDEcoeff_functions.plotandSaveCoeff(loadPath, PDEcoeff, Mesh.vertices_fine, [
                                            coeff], i, ax, fig, "train")
        A = PDEcoeff(Mesh.Mids, [coeff]).flatten()
        A_stiff[:, [i]] = A[:, None]
        G_tmp, _ = LOD.lodCorrector(Mesh, A)
        G_LOD[:, :, i] = G_tmp.toarray()
    u_LOD, * \
        _ = solvers.solveHomParabolic(Mesh, G_LOD, A_stiff, A_mass, rhs, u0, T)
    u_PINN, * \
        _ = solvers.solveHomParabolic(
            Mesh, G_PINN, A_stiff, A_mass, rhs, u0, T)
    print("relative error basis functions: ", (np.linalg.norm(G_PINN - G_LOD, axis=0, keepdims=True)
                                               / np.linalg.norm(G_LOD, axis=0, keepdims=True)*100), "%")
    print("mean relative error basis functions: ", np.mean(np.linalg.norm(G_PINN - G_LOD, axis=0, keepdims=True)
                                                           / np.linalg.norm(G_LOD, axis=0, keepdims=True)*100), "%")
    print(np.sum(G_PINN)/np.prod(Mesh.N_vertices_fine)/Mesh.N_coeff)
    print(np.sum(G_LOD)/np.prod(Mesh.N_vertices_fine)/Mesh.N_coeff)
    train_err.append(np.linalg.norm(u_LOD-u_PINN)/np.linalg.norm(u_LOD)*100)
    indexPlot = int(np.ceil((np.prod(Mesh.N_coarse)+1)//2 + 0.5))
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
        T = np.linspace(0, 1, N_eval, dtype=np.float32)
        Mesh, PDEcoeff, eval_coeff, G_PINN = evaluate_model.evaluateModel(
            loadPath, evalCoeff=eval_coeff)
        test_err = []
        Mesh_ref, u_ref = [], []
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
            A_stiff[:, [i]] = A[:, None]
            G_tmp, _ = LOD.lodCorrector(Mesh, A)
            G_LOD[:, :, i] = G_tmp.toarray()
        u_LOD, * \
            _ = solvers.solveHomParabolic(
                Mesh, G_LOD, A_stiff, A_mass, rhs, u0, T)
        u_PINN, * \
            _ = solvers.solveHomParabolic(
                Mesh, G_PINN, A_stiff, A_mass, rhs, u0, T)
        print("relative error basis functions: ", (np.linalg.norm(G_PINN - G_LOD, axis=0, keepdims=True)
                                                   / np.linalg.norm(G_LOD, axis=0, keepdims=True)*100), "%")
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

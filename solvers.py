import numpy as np
import util


def solveHomEllipticFEM(Mesh, A, f):
    NC = Mesh.N_coarse
    u = np.zeros((np.prod(NC+1), 1))
    S_coarse = util.assembleGlobalMatrices(
        util.localStiffness(NC, a=Mesh.a), NC, 'cg', weight=A).toarray()
    S = S_coarse
    dof = np.setdiff1d(np.arange(np.prod(NC+1)), util.boundaryNodes(NC))
    M_coarse = util.assembleGlobalMatrices(
        util.localMass(NC, a=Mesh.a), NC, 'cg').toarray()
    b = M_coarse @ f(Mesh.vertices_coarse)
    x = np.linalg.solve(S[np.ix_(dof, dof)], b[dof, :]).reshape((-1, 1))
    u[dof, :] = x
    res_vec = S_coarse@u-M_coarse @ f(Mesh.vertices_coarse)
    res = np.linalg.norm(res_vec) / np.linalg.norm(u)
    return u, res, np.reshape(res_vec / np.linalg.norm(u), Mesh.N_vertices_coarse)


def solveHomElliptic(Mesh, G, A, f):
    NC, NF = Mesh.N_coarse, Mesh.N_fine
    u = np.zeros((np.prod(NC+1), 1))
    S_fine = util.assembleGlobalMatrices(util.localStiffness(
        NF, a=Mesh.a), NF, 'cg', weight=A).toarray()
    S = G.T @ S_fine @ G
    dof = np.setdiff1d(np.arange(np.prod(NC+1)), util.boundaryNodes(NC))
    M_fine = util.assembleGlobalMatrices(
        util.localMass(NF, a=Mesh.a), NF, 'cg').toarray()
    b = G.T@M_fine @ f(Mesh.vertices_fine)
    x = np.linalg.solve(S[np.ix_(dof, dof)], b[dof, :]).reshape((-1, 1))
    u[dof, :] = x
    u = G @ u
    res_vec = S_fine@u-M_fine @ f(Mesh.vertices_fine)
    res = np.linalg.norm(res_vec) / np.linalg.norm(u)
    return u, res, np.reshape(res_vec / np.linalg.norm(u), Mesh.N_vertices_fine)


def solveHomParabolic(Mesh, G_time, A_stiff, A_mass, f, u0, time):
    NC, NF = Mesh.N_coarse, Mesh.N_fine
    boundary_dof = util.boundaryNodes(NC)
    dof = np.setdiff1d(np.arange(np.prod(NC+1)), boundary_dof)
    u_t = np.zeros((np.prod(NF+1), np.size(time)))
    u_t[:, 0:1] = u0
    for index, t in enumerate(time[1:]):
        S_fine = util.assembleGlobalMatrices(util.localStiffness(
            NF, a=Mesh.a), NF, 'cg', weight=A_stiff[:, index].flatten()).toarray()
        G = G_time[:, :, index]
        S = G.T @ S_fine @ G
        M_fine = util.assembleGlobalMatrices(util.localMass(
            NF, a=Mesh.a), NF, 'cg', weight=A_mass[:, index].flatten()).toarray()
        M = G.T @ M_fine @ G
        Mii = M[np.ix_(dof, dof)]
        Sii = S[np.ix_(dof, dof)]
        rhs = G.T@M_fine@f(Mesh.vertices_fine)
        dt = t - time[index]
        Aii = Mii/dt+Sii
        sol = np.linalg.solve(Aii, (Mii/dt)@u0[dof]+rhs[dof])
        u1 = np.zeros((np.prod(NC+1), 1))
        u1[dof] = sol
        u_t[:, index+1:index+2] = G@u1
        u0 = u1
    return u_t, 0

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"    # which gpu to use
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import matplotlib.pyplot as plt
import myDenseLayer
from scipy.stats import qmc
import optimizers
from matplotlib import gridspec
import math
import datetime
import util
import Geo
import PDEcoeff_functions
import DiscreteSetting
import tensorflow as tf


def my_custom_activation(x, coeff):
    return tf.math.tanh(coeff * x)


class LODPINN(DiscreteSetting.FE_Nodal):
    def __init__(self, fe, layers, lr, lr_sa, train_data, labels, index, withTransformInput, model, resultPath, DTYPE, mode="train", variableCoeffParam=False):
        super().__init__(fe.geo, fe.PDEcoeff, fe.coeff_parameters,
                         fe.test_parameters, fe.withData, fe.np_dtype)
        tf.random.set_seed(73)
        self.mode = mode
        self.variableCoeffParam = variableCoeffParam
        if mode == "train":
            if self.withData:
                print("Training is performed with data")
            else:
                print("Training is performed without data")
            self.lr = lr
            self.lr_sa = lr_sa
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
            self.optimizer_weights = tf.keras.optimizers.Adam(
                learning_rate=lr_sa)
            self.kernel_initializer = tf.keras.initializers.glorot_normal(
                seed=73)
            self.bias_initializer = tf.keras.initializers.glorot_normal(
                seed=73)
            self.activation = tf.keras.activations.tanh  # my_custom_activation
            self.layerType = tf.keras.layers.Dense  # myDenseLayer.myDense
            self.withTransformInput = withTransformInput
        self.layers = layers
        self.dtype = DTYPE
        self.resultPath = resultPath
        if resultPath and not os.path.exists(resultPath):
            os.makedirs(resultPath)
        if self.dim == 2:
            if np.size(index) == 2:
                self.index = index[0]*self.N_coarse[1]+index[1]
            else:
                self.index = index
        elif self.dim == 1:
            self.index = index
        # inputs and parameters restricted to element and dof
        dofC, dofF, PatchC, PatchF = self.dofC[self.index], self.dofF[
            self.index], self.PatchC[self.index], self.PatchF[self.index]
        self.N_patches = np.size(PatchC)
        N_v_pp_coarse = self.N_v_pp_coarse[self.index]
        N_v_pp_fine = self.N_v_pp_fine[self.index]
        boundaryID_coarse = self.boundaryID_coarse[self.index]
        self.Xcoarse = self.vertices_coarse[dofC, :]
        self.Xfine = self.vertices_fine[dofF, :]

        print("Patch domain: ", self.Xfine[0, :], ",", self.Xfine[-1, :])
        if self.index == int(np.prod(self.N_coarse+1)//2 + 0.5) or self.index == int(np.prod(self.N_coarse+1) - 2*self.N_coarse[0]-1) \
                or self.index == int(np.prod(self.N_coarse+1)-2-self.N_coarse[0]//2-self.N_coarse[0]):
            self.forward_pass = self.forward_pass_wLoc
            print("Localization Function used!")
        else:
            self.forward_pass = self.forward_pass_wLoc
            print("Localization Function used!")

        if self.variableCoeffParam:
            self.Cfine = self.coeff_parameters[:, dofF]
        else:
            self.Cfine = self.coeff_parameters[:, None]
        if True:
            fig = plt.figure(figsize=(20, 25))
            self.plotCoeff(self.coeff_parameters, np.floor(
                np.sqrt(self.N_coeff)), fig, "Train")

        I = util.enforce_BC_interp(
            self.I, self.N_coarse, self.N_fine_per_element, boundaryID_coarse)

        if self.elementwise:
            shapeFunctions = self.shapeFunctions(self.index)[dofF, :]
        else:
            shapeFunctions = self.basisFunctions(np.array(
                [self.index % (self.N_coarse[0]+1), self.index//(self.N_coarse[1]+1)]))[dofF, :]

        if mode == "train":
            if self.withData:
                trainData_index = [self.TrainData[i][dofF, :]
                                   for i in range(self.N_coeff)]
                trainData_index = np.vstack(trainData_index)[:, [self.index]]
            # input grid and system matrices
            self.Xcoarsec = np.tile(self.Xcoarse, (self.N_coeff, 1))
            self.Xfinec = np.tile(self.Xfine, (self.N_coeff, 1))
            if self.variableCoeffParam:
                self.Cfinex = np.reshape(
                    self.Cfine, [self.N_coeff*N_v_pp_fine, 1], "C")
                Sfinec = self.systemMatrixDOF(
                    self.coeff_parameters, self.index)
            else:
                self.Cfinex = np.repeat(self.Cfine, N_v_pp_fine, 0)
                Sfinec = self.systemMatrixDOF(self.Cfine, self.index)
            Ic = np.kron(np.eye(self.N_coeff, dtype=self.np_dtype),
                         I[np.ix_(dofC, dofF)])
            ShFc = np.tile(shapeFunctions, (self.N_coeff, 1))
            # convert to tensors
            self.Ic_tf = tf.convert_to_tensor(Ic, dtype=DTYPE)
            self.Labels_tf = tf.convert_to_tensor(labels, dtype=DTYPE)
            self.Cfinex_tf = tf.convert_to_tensor(self.Cfinex, dtype=DTYPE)
            self.Xfinec_tf = tf.convert_to_tensor(self.Xfinec, dtype=DTYPE)
            self.Sfinec_tf = tf.convert_to_tensor(Sfinec, dtype=DTYPE)
            self.ShFc_tf = tf.convert_to_tensor(ShFc, dtype=DTYPE)
            if self.withData:
                self.TrainData_tf = tf.constant(trainData_index, dtype=DTYPE)

            self.mse_res_log = []
            self.mse_mid_log = []
            self.mse_data_log = []
            self.iteration_log = []
            self.weights_log = []

        # test data
        if self.N_test > 0:
            if mode == "train":
                self.Xcoarset = tf.convert_to_tensor(
                    np.tile(self.Xcoarse, (self.N_test, 1)), dtype=DTYPE)
                self.Xfinet = tf.convert_to_tensor(
                    np.tile(self.Xfine, (self.N_test, 1)), dtype=DTYPE)
                if self.withData:
                    testData_index = [self.TestData[i][dofF, :]
                                      for i in range(self.N_test)]
                    testData_index = np.vstack(testData_index)[:, [self.index]]
                if True:
                    fig = plt.figure(figsize=(20, 25))
                    self.plotCoeff(self.test_parameters, np.floor(
                        np.sqrt(self.N_test)), fig, "Test")
                if self.variableCoeffParam:
                    self.Ctest = self.test_parameters[:, dofF]
                    self.Ctestx = tf.convert_to_tensor(np.reshape(
                        self.Ctest, [self.N_test*N_v_pp_fine, 1], "C"), dtype=DTYPE)
                    self.Stestt = tf.convert_to_tensor(self.systemMatrixDOF(
                        self.test_parameters, self.index), dtype=DTYPE)
                else:
                    self.Ctest = self.test_parameters[:, None]
                    self.Ctestx = tf.convert_to_tensor(
                        np.repeat(self.Ctest, N_v_pp_fine, 0), dtype=DTYPE)
                    self.Stestt = tf.convert_to_tensor(
                        self.systemMatrixDOF(self.Ctest, self.index), dtype=DTYPE)
                self.It = tf.convert_to_tensor(
                    np.kron(np.eye(self.N_test, dtype=int), I[np.ix_(dofC, dofF)]), dtype=DTYPE)
                self.ShFt = tf.convert_to_tensor(
                    np.tile(shapeFunctions, (self.N_test, 1)), dtype=DTYPE)
                if self.withData:
                    self.TestData_tf = tf.convert_to_tensor(
                        testData_index, dtype=DTYPE)
            elif mode == "eval":
                self.Xcoarset = tf.convert_to_tensor(
                    np.tile(self.Xcoarse, (self.N_test, 1)), dtype=DTYPE)
                self.Xfinet = tf.convert_to_tensor(
                    np.tile(self.Xfine, (self.N_test, 1)), dtype=DTYPE)
                if True:
                    fig = plt.figure(figsize=(20, 25))
                    self.plotCoeff(self.test_parameters, np.floor(
                        np.sqrt(self.N_test)), fig, "Eval")
                if self.variableCoeffParam:
                    self.Ctest = self.test_parameters[:, dofF]
                    self.Ctestx = tf.convert_to_tensor(np.reshape(
                        self.Ctest, [self.N_test*N_v_pp_fine, 1], "C"), dtype=DTYPE)
                    self.Stestt = tf.convert_to_tensor(self.systemMatrixDOF(
                        self.test_parameters, self.index), dtype=DTYPE)
                else:
                    self.Ctest = self.test_parameters[:, None]
                    self.Ctestx = tf.convert_to_tensor(self.Ctest, dtype=DTYPE)
                    self.Stestt = tf.convert_to_tensor(
                        self.systemMatrixDOF(self.Ctest, self.index), dtype=DTYPE)
                self.It = tf.convert_to_tensor(
                    np.kron(np.eye(self.N_test, dtype=self.np_dtype), I), dtype=DTYPE)
                self.ShFt = tf.convert_to_tensor(
                    np.tile(shapeFunctions, (self.N_test, 1)), dtype=DTYPE)

        # log and config
        if resultPath:
            current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            train_log_dir = resultPath + 'logs/train/' + \
                current_time + '/BF_' + str(index)
            self.train_summary_writer = tf.summary.create_file_writer(
                train_log_dir)
            if self.N_test > 0:
                test_log_dir = resultPath + 'logs/test/' + \
                    current_time + '/BF_' + str(index)
                self.test_summary_writer = tf.summary.create_file_writer(
                    test_log_dir)

        if len(model) > 0:
            self.model = model[0]
            self.sa_weight_tf = tf.Variable(
                model[1][:, None], dtype=DTYPE, trainable=True)
            self.sa_weightData_tf = tf.Variable(
                model[2][:, None], dtype=DTYPE, trainable=True)
        else:
            self.model = self.initialize_model(layers)
            self.sa_weight_tf = tf.Variable(tf.random.uniform(
                [N_v_pp_coarse*self.N_coeff, np.size(ShFc, 1)], dtype=DTYPE), trainable=True)
            self.sa_weightData_tf = tf.Variable(tf.random.uniform(
                [N_v_pp_fine*self.N_coeff, np.size(ShFc, 1)], dtype=DTYPE), trainable=True)

        self.trainable_parameters = self.model.trainable_variables
        self.summary = self.model.summary

    def initialize_model(self, layers):
        input = tf.keras.layers.Input(shape=(layers[0],),
                                      name="input_layer_"+str(self.index),
                                      dtype=self.dtype)
        inp = input
        for depth, width in enumerate(layers[1:-1]):
            inp = self.layerType(width,
                                 activation=self.activation,
                                 kernel_initializer=self.kernel_initializer,
                                 bias_initializer=self.bias_initializer,
                                 name="hidden_layer_" +
                                 str(depth+1)+"_"+str(self.index),
                                 dtype=self.dtype)(inp)
        out = tf.keras.layers.Dense(layers[-1],
                                    activation=None,  # self.activation,
                                    kernel_initializer=self.kernel_initializer,
                                    bias_initializer=self.bias_initializer,
                                    name="output_layer_"+str(self.index),
                                    dtype=self.dtype)(inp)
        model = tf.keras.models.Model(inputs=[input], outputs=[
                                      out], name="PINNLOD"+str(self.index))
        model.compile(optimizer=self.optimizer)
        return model

    def forward_pass_wLoc(self, X, COEFF):
        Xa = self.Xfine[0, :]
        Xl = self.Xfine[-1, :]
        H = Xl-Xa
        sigma = 0.001
        inp = (X-Xa)/H
        locFun = tf.reduce_prod(
            2*tf.atan(tf.sin(np.pi * inp)/sigma)/np.pi, axis=1)[:, None]
        u = locFun * self.model(tf.concat([X, COEFF], 1))
        return u

    def forward_pass_wtLoc(self, X, COEFF):
        u = self.model(tf.concat([X, COEFF], 1))
        return u

    def u_model(self, X, COEFF):
        u_nn = self.forward_pass(X, COEFF)
        return u_nn

    def u_model_interp(self, u, I):
        u_nn = tf.linalg.matmul(I, u)
        return u_nn

    def res_data(self, u, DATA):
        return tf.subtract(u, DATA)

    def correction_model(self, u, S, BF):
        Su = tf.linalg.matmul(S, u)
        Lhs_int = tf.linalg.matmul(u, Su, transpose_a=True)
        Rhs_int = tf.linalg.matmul(BF, Su, transpose_a=True)
        res = 0.5*Lhs_int - Rhs_int
        return res

    def loss(self, X, COEFF, I, S, BF):
        u = self.u_model(X, COEFF)
        res_pde = self.correction_model(u, S, BF)
        res_interp = self.u_model_interp(u, I)
        if self.withData:
            res_data = self.res_data(u, self.TrainData_tf)
            mse_data = tf.reduce_mean(
                tf.square(self.sa_weightData_tf*res_data), axis=0)
        else:
            res_data = tf.constant([0.])
            mse_data = tf.constant([0.])
        mse_res = tf.reduce_mean((res_pde), axis=0)
        mse_interp = tf.reduce_mean(
            tf.square(self.sa_weight_tf*res_interp), axis=0)
        loss_value = mse_res + mse_interp + mse_data
        return loss_value, mse_res, tf.reduce_mean(tf.square(res_interp), axis=0), tf.reduce_mean(tf.square(res_data), axis=0)

    def evaluateTestLoss(self, Xtest, Ctest, Itest, Stest, BFtest):
        utest = self.u_model(Xtest, Ctest)
        res_pde = self.correction_model(utest, Stest, BFtest)
        res_interp = self.u_model_interp(utest, Itest)
        if self.withData:
            res_data = self.res_data(utest, self.TestData_tf)
            mse_data = tf.reduce_mean(tf.square(res_data), axis=0)
        else:
            res_data = tf.constant([0.])
            mse_data = tf.constant([0.])
        mse_res = tf.reduce_mean(res_pde, axis=0)
        mse_interp = tf.reduce_mean(tf.square(res_interp), axis=0)
        loss_value = mse_res + mse_interp + mse_data
        return loss_value, mse_res, mse_interp, mse_data

    @tf.function
    def opt_min_max(self):
        if self.withData:
            with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
                tape.watch(self.trainable_parameters)
                tape.watch(self.sa_weight_tf)
                tape.watch(self.sa_weightData_tf)
                loss_value, mse_res, mse_mid, mse_data = self.loss(self.Xfinec_tf, self.Cfinex_tf,
                                                                   self.Ic_tf, self.Sfinec_tf, self.ShFc_tf)
            gradients_model = tape.gradient(loss_value,
                                            self.trainable_parameters)
            gradients_sa = tape.gradient(loss_value,
                                         self.sa_weight_tf)
            gradients_Data_sa = tape.gradient(loss_value,
                                              self.sa_weightData_tf)
            del tape
            self.optimizer.apply_gradients(zip(gradients_model,
                                               self.trainable_parameters))
            self.optimizer_weights.apply_gradients(zip([-gradients_sa, -gradients_Data_sa],
                                                   [self.sa_weight_tf, self.sa_weightData_tf]))
        else:
            with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
                tape.watch(self.trainable_parameters)
                tape.watch(self.sa_weight_tf)
                loss_value, mse_res, mse_mid, mse_data = self.loss(self.Xfinec_tf, self.Cfinex_tf,
                                                                   self.Ic_tf, self.Sfinec_tf, self.ShFc_tf)
            gradients_model = tape.gradient(loss_value,
                                            self.trainable_parameters)
            gradients_sa = tape.gradient(loss_value,
                                         self.sa_weight_tf)
            del tape
            self.optimizer.apply_gradients(zip(gradients_model,
                                               self.trainable_parameters))
            self.optimizer_weights.apply_gradients(zip([-gradients_sa],
                                                       [self.sa_weight_tf]))
        return loss_value, mse_res, mse_mid, mse_data, self.trainable_parameters

    def train_model(self, epochs, freq_log):
        # print("Start Adam training:")
        for epoch in range(epochs):
            mse_total, mse_res, mse_mid, mse_data, weights = self.opt_min_max()
            if epoch % freq_log == 0:
                self.mse_res_log.append(np.sum(mse_res))
                self.mse_mid_log.append(np.sum(mse_mid))
                self.mse_data_log.append(np.sum(mse_data))
                self.iteration_log.append(epoch)
                self.weights_log.append(weights)
                with self.train_summary_writer.as_default():
                    tf.summary.scalar(
                        "energy's loss", np.sum(mse_res), step=epoch)
                    tf.summary.scalar(
                        "data's loss", np.sum(mse_data), step=epoch)
                    tf.summary.scalar("interpolation condition",
                                      np.sum(mse_mid), step=epoch)
                    tf.summary.scalar("weights mean", tf.reduce_mean(
                        [np.mean(weight) for weight in weights]), step=epoch)
                    tf.summary.scalar(
                        "learning rate", self.optimizer.learning_rate, step=epoch)
                if self.N_test > 0:
                    _, mse_res_test, mse_mid_test, mse_data_test = self.evaluateTestLoss(self.Xfinet, self.Ctestx,
                                                                                         self.It, self.Stestt, self.ShFt)
                    with self.test_summary_writer.as_default():
                        tf.summary.scalar("test energy's loss",
                                          mse_res_test[0], step=epoch)
                        tf.summary.scalar("test data's loss",
                                          mse_data_test[0], step=epoch)
                        tf.summary.scalar(
                            "test interpolation condition", mse_mid_test[0], step=epoch)
                if epoch // freq_log > 10 and np.std(self.mse_res_log[-10:])/np.abs(np.mean(self.mse_res_log[-10:])) < 1E-5 \
                        and np.mean(self.mse_mid_log[-10:]) < 5E-6 and np.mean(self.mse_data_log[-10:]) < 1E-4:
                    print("-> Training for model ", self.index,
                          " stopped after convergence at epoch ", epoch)
                    break

    def plotBoundaryPoints(self, ax, index_element):
        if self.dim == 1:
            ax.plot(self.vertices_fine, np.zeros_like(
                self.vertices_fine), "-o", label="fine")
            ax.plot(self.vertices_fine[self.boundaryID_fine[index_element]], np.zeros_like(
                self.boundaryID_fine[index_element]), "o", label="bd fine")
            ax.plot(self.vertices_coarse[self.boundaryID_coarse[index_element]], np.zeros_like(
                self.boundaryID_coarse[index_element]), "x", label="bd coarse")
            ax.set_title("element Nr"+str(index_element) +
                         ", l = "+str(self.ell))
            ax.set_xlabel("x")
            ax.legend()
        elif self.dim == 2:
            if np.size(index_element) == 2:
                index_element_a = index_element[0] * \
                    self.N_coarse[1]+index_element[1]
            else:
                index_element_a = index_element
            ax.scatter(self.vertices_fine[:, 0],
                       self.vertices_fine[:, 1], label="fine")
            ax.scatter(self.vertices_fine[self.boundaryID_fine[index_element_a], 0],
                       self.vertices_fine[self.boundaryID_fine[index_element_a], 1], label="bd fine")
            ax.scatter(self.vertices_coarse[self.boundaryID_coarse[index_element_a], 0],
                       self.vertices_coarse[self.boundaryID_coarse[index_element_a], 1], label="bd coarse")
            ax.set_title("element Nr"+str(index_element) +
                         ", l = "+str(self.ell))
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.legend()

    def plotModelOutput(self, u, ax, fig, shp=None):
        if shp is None:
            shp = self.ShFc_tf
        if self.dim == 1:
            ax.plot(self.Xfinec_tf.numpy(), u, "o",
                    label="$\phi(p_{{train}})$")
            ax.plot(self.Xfinec_tf, shp, "o", label="$\Lambda$")
            ax.plot(self.Xfinec_tf, shp-u, "o",
                    label="$u^{{ms}}(p_{{train}})$")
            ax.set_xlabel("x")
            ax.legend()
        elif self.dim == 2:
            im = ax.scatter(
                self.Xfinec_tf[:, 0], self.Xfinec_tf[:, 1], c=u, label="$u^{{ms}}(p_{{train}})$")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_title("correction function")
            ax.set_xlim(0., self.a[0])
            ax.set_ylim(0., self.a[1])
            fig.colorbar(im)

    def plotEvalOutput(self, ums, ax, fig):
        if self.dim == 1:
            ax.plot(self.vertices_fine, ums, label="$u^{{eval}}(p_{{test}})$")
            ax.set_xlabel("x")
            ax.legend()
        elif self.dim == 2:
            im = ax.scatter(
                self.vertices_fine[:, 0], self.vertices_fine[:, 1], c=ums, label="$u^{{ms}}(p_{\mathrm{eval}})$")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_title("corrected Basis function")
            fig.colorbar(im)

    def plotCoeff(self, eps, cols, fig, label="Train"):
        N_coeff = np.size(eps, 0)
        x = self.Mids[self.PatchF[self.index], :]
        rows = int(math.ceil(N_coeff / cols))
        gs = gridspec.GridSpec(rows, int(cols))
        if np.size(x, axis=1) == 2:
            Nx = np.size(np.unique(x[:, 0]))
            Ny = np.size(np.unique(x[:, 1]))
            X = np.reshape(x[:, 0], [Nx, Ny], "F")
            Y = np.reshape(x[:, 1], [Nx, Ny], "F")
            for i in range(N_coeff):
                ax = fig.add_subplot(gs[i])
                coeff = self.PDEcoeff(x, [eps[i]])
                Z = np.reshape(coeff, [Nx, Ny], "F")
                im = ax.pcolormesh(X, Y, Z, cmap="RdBu_r")
                fig.colorbar(im)
            fig.supxlabel("$x$")
            fig.supylabel("$y$")
            plt.savefig(self.resultPath+label +
                        "Coeff/coeffient_Index_"+str(self.index))
            plt.close()

    def plotTrainLoss(self, ax):
        ax[0].plot(self.iteration_log, self.mse_res_log, label="energy")
        ax[0].set_xlabel("iterations")
        ax[0].set_ylabel("$\mathcal{L}_{res}$")
        ax[1].loglog(self.iteration_log, self.mse_mid_log, label="interp")
        ax[1].set_xlabel("iterations")
        ax[1].set_ylabel("$\mathcal{L}_{interp}$")

    def saveModel(self):
        self.model.save(self.resultPath+"correctedHatModel" +
                        str(self.index)+".h5")
        np.savetxt(self.resultPath+"sa_weights_"+str(self.index) +
                   ".txt", self.sa_weight_tf.numpy())
        np.savetxt(self.resultPath+"sa_weightsData_" +
                   str(self.index)+".txt", self.sa_weightData_tf.numpy())
        print("Model saved as: ", self.resultPath +
              "correctedHatModel"+str(self.index)+".h5")

    def loadModel(self, loadPath):
        model = tf.keras.models.load_model(
            loadPath+"correctedHatModel"+str(self.index)+".h5")
        sa_weights = np.loadtxt(loadPath+"sa_weights_"+str(self.index)+".txt")
        sa_weightsData = np.loadtxt(
            loadPath+"sa_weightsData_"+str(self.index)+".txt")
        return model, sa_weights, sa_weightsData

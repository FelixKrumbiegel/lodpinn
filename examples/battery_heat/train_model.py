# train_model
from pathlib import Path
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"    # which gpu to use
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.append(str(Path(__file__).parent.parent.parent))

from scipy.stats import qmc
import LOD
import util
import optimizers
import multiprocessing
import DiscreteSetting
import LODPINN
import PDEcoeff_functions
import Geo
import json
import matplotlib.pyplot as plt
import timeit
import tensorflow as tf
import numpy as np

tf.random.set_seed(73)


if __name__ == '__main__':
    # parallelization
    parallelize = True
    # Transfer learning
    transferLearning = False
    # elementwise
    elementwise = False
    # Data
    withData = False
    if withData:
        DataExt = "withData"
    else:
        DataExt = "withoutData"
    # N iterations for transfer learning
    N_iterations = 1
    if N_iterations > 1:
        parallelize = False
    # Mode
    mode = 0  # {0:train, 1:train & fineTune, 2: fineTune}

    # example class
    coeff_fun = "battery_heat"

    #########################
    # Domain, Localization, and Discretisation
    domain = np.vstack([[0, 1], [0, 1]])
    N_coarse = np.array([8, 8])  # nbr of elements
    N_fine_per_element = np.array([5, 5])
    ell = 1

    PDEcoeff_class = eval("PDEcoeff_functions.PDEcoeff_"+coeff_fun)

    epsilon_min = 5
    epsilon_max = 15
    N_coeff = 20
    N_test = 25
    np.random.seed(2024)
    test_coeff = epsilon_min + \
        (epsilon_max - epsilon_min)*np.random.random(N_test)

    PDEcoeff = PDEcoeff_class(domain, ccc=236.3, acc=398.71)
    Mesh = Geo.GEO(domain, N_coarse, N_fine_per_element, ell,
                   np_dtype=np.float32, elementwise=elementwise)
    #########################
    # configuration of the neural network
    epochs = 150001
    freq_log = 100
    width_net = 128
    depth_net = 4
    if elementwise:
        typeComputation = "element"
        output_size = 2**np.shape(domain)[0]
    else:
        typeComputation = "nodal"
        output_size = 1
    initial_lr = 5E-4
    decayRate = 0.99
    decaySteps = 1000
    # architecture of the NN
    layer_sizes = [np.size(N_coarse)+1]+depth_net*[width_net]+[output_size]
    # optimizers
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=initial_lr,
                                                                 decay_steps=decaySteps,
                                                                 decay_rate=decayRate,
                                                                 staircase=False)
    lr_sa = 1E-2  # learning rate for the self adaptive part
    lr_schedule_sa = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=lr_sa,
                                                                    decay_steps=1000,
                                                                    decay_rate=0.98,
                                                                    staircase=False)
    ###########################
    # results path and config file
    resultPath = "results/case_domain_"+str(domain.tolist())+"_w"+str(width_net)+"_d"+str(depth_net) + \
        "_p"+str(epsilon_min)+"to"+str(epsilon_max)+"_c"+str(N_coarse) +\
        "_f"+str(N_fine_per_element)+"_l"+str(ell)+"_coeff" + \
        str(N_coeff)+"_"+typeComputation+"_"+DataExt+"/"
    if mode == 2:
        loadPath = resultPath
        resultPath += "fineTuned/"
    elif transferLearning or True:
        loadPath = "results/case_domain_"+str(domain.tolist())+"_w"+str(width_net)+"_d"+str(depth_net)+"_p"+str(epsilon_min)+"to" + \
            str(epsilon_max)+"_c"+str(N_coarse)+"_f"+str(N_fine_per_element)+"_l" + \
            str(ell)+"_coeff"+str(N_coeff)+"_"+typeComputation+"_"+DataExt+"/"
    resultPath = "examples/battery_heat/"+resultPath
    if not os.path.exists(resultPath):
        os.makedirs(resultPath)
    if not os.path.exists(resultPath+"TrainCoeff"):
        os.makedirs(resultPath+"TrainCoeff")
    if not os.path.exists(resultPath+"TestCoeff"):
        os.makedirs(resultPath+"TestCoeff")
    config = {"epsilon_min": epsilon_min,
              "epsilon_max": epsilon_max,
              "domain": domain.tolist(),
              "epochs": epochs,
              "freq_log": freq_log,
              "width_net": width_net,
              "depth_net": depth_net,
              "N_fine_per_element": N_fine_per_element.tolist(),
              "N_coarse": N_coarse.tolist(),
              "N_coeff": N_coeff,
              "N_test": N_test,
              "ell": ell,
              "coeffFunction": coeff_fun,
              "initial_lr": initial_lr,
              "lr_sa": lr_sa,
              "schedule": "ExponentialDecay",
              "decay_steps": decaySteps,
              "decay_rate": decayRate}
    with open(resultPath+"config.json", 'w') as f:
        json.dump(config, f)
        # if transferLearning:
        #     json.dump({"transferLearning from": loadPath}, f)
    # np.savetxt(resultPath+"config.txt", config)
    ###########################
    # Training function

    def train_all(index):
        if transferLearning or mode == 2:
            pretrained_model = util.loadModel(loadPath, index)
            print("+ load pretrained model ", index)
        else:
            pretrained_model = []
        PINN = LODPINN.LODPINN(FE, layer_sizes, lr_schedule, lr_schedule_sa, [], [], index, withTransformInput=False,
                               model=pretrained_model, resultPath=resultPath, DTYPE=tf.float32, mode="train", variableCoeffParam=False)
        if mode != 2:
            # train
            print("  -> Start training model for correction problem " + str(index))
            PINN.train_model(epochs, freq_log)
        if mode != 0:
            lr = 1E-4
            lr_sa = 1E-4
            PINN = LODPINN.LODPINN(FE, layer_sizes, lr, lr_sa, [], [], index, withTransformInput=False,
                                   model=pretrained_model, resultPath=resultPath, DTYPE=tf.float32, mode="train", variableCoeffParam=False)
            # Fine tune
            print("  - start Fine Tuning for Model ", index)
            PINN.train_model(5000, freq_log)
            print("  - end Fine Tuning for Model ", index)
        # save model
        PINN.saveModel()
        # plot solution
        phi = PINN.u_model(PINN.Xfinec_tf, PINN.Cfinex_tf)

        N = np.size(PINN.Xfine[:, 0])
        for k, coeff in enumerate(PINN.coeff_parameters):
            for i in range(output_size):
                fig, ax = plt.subplots()
                plt.scatter(PINN.Xfine[:, 0], PINN.Xfine[:, 1],
                            c=phi[k*N:(k+1)*N, i], label="$\phi_r$")
                plt.colorbar()
                plt.xlim(domain[0][0], domain[0][1])
                plt.ylim(domain[1][0], domain[1][1])
                plt.savefig(resultPath+"train_sol_"+str(index)+"_"+str(i))

                A = PINN.PDEcoeff(PINN.Mids, [coeff]).flatten()
                G, C = LOD.lodCorrector(Mesh, A)

                plt.figure()
                plt.scatter(PINN.vertices_fine[:, 0], PINN.vertices_fine[:, 1], c=C.toarray()[
                            :, index], label="$\phi_r$")
                plt.colorbar()
                plt.savefig(resultPath+"ref_sol_"+str(index)+"_"+str(i))
    # Train models for correction problems
    start = timeit.default_timer()
    if elementwise:
        N_BF = np.prod(N_coarse)
    else:
        N_BF = np.prod(N_coarse+1)
    index_vec = range(N_BF)
    print(index_vec)

    if parallelize == True:
        param_coeff = np.linspace(
            epsilon_min, epsilon_max, N_coeff, dtype=np.float32)
        np.savetxt(resultPath+"trainParameter_Iter"+str(0)+".txt", param_coeff)
        print("Train param coeff: shape(", np.shape(
            param_coeff), ")\n", param_coeff)
        FE = DiscreteSetting.FE_Nodal(
            Mesh, PDEcoeff, param_coeff, test_coeff, withData=withData)
        pool_obj = multiprocessing.Pool()
        print("-> Start training for "+str(N_BF) +
              " correction problems with " + str(multiprocessing.cpu_count())+" cpu")
        result = pool_obj.map_async(train_all, index_vec)
        result.get()
        pool_obj.close()
    else:
        for samples_i in range(N_iterations):
            param_coeff = np.linspace(
                epsilon_min, epsilon_max, N_coeff, dtype=np.float32)
            np.savetxt(resultPath+"trainParameter_Iter" +
                       str(samples_i)+".txt", param_coeff)
            print("Train param coeff nr", str(samples_i),
                  ": shape(", np.shape(param_coeff), ")\n", param_coeff)
            FE = DiscreteSetting.FE_Nodal(
                Mesh, PDEcoeff, param_coeff, test_coeff, withData=withData)
            for i, index in enumerate(index_vec):
                start_iter = timeit.default_timer()
                print("-> Training for correction problem's type " +
                      str(i+1)+"/"+str(N_BF))
                train_all(index)
                stop_iter = timeit.default_timer()
                print("training time for problem " +
                      str(i+1), ": ", stop_iter - start_iter)
            stop = timeit.default_timer()
            print("total training time until batch index",
                  str(samples_i), ": ", stop - start)
            transferLearning = True
    print("-> Assemble Corrector Model: ")
    start = timeit.default_timer()
    inp = []
    out = []
    inputModel, *_ = util.loadModel(resultPath, index_vec[0])
    input = inputModel.layers[0].input
    for i in index_vec:
        x = input
        for depth, width in enumerate(layer_sizes[1:-1]):
            modeli, *_ = util.loadModel(resultPath, i)
            x = modeli.layers[depth+1](x)
        outputModel, *_ = util.loadModel(resultPath, i)
        output = outputModel.layers[-1](x)
        out.append(output)
        inp.append(input)
    conc_models = tf.keras.layers.concatenate(out, name="concatenated_models")
    Gmodel = tf.keras.models.Model(inputs=[inp[0]], outputs=[
                                   conc_models], name="corrector_model")
    Gmodel.compile()
    Gmodel.save(resultPath+"GModel.h5")
    tf.keras.utils.plot_model(Gmodel, to_file=resultPath +
                              "corrector_model.png", show_shapes=False, show_dtype=False)
    stop = timeit.default_timer()
    print("assembling time: ", stop - start)
    fig = plt.figure()
    PDEcoeff_functions.plot_coeff(
        PDEcoeff, Mesh.vertices_fine, param_coeff, 2, fig)
    plt.close()

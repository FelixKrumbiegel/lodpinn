# Some extra optimizer for testing and fine tuning
import numpy as np
import scipy.optimize
import tensorflow as tf


def ScipyOptimizer(self, method='L-BFGS-B', **kwargs):
    """This method provides an interface to solve the learning problem
    using a routine from scipy.optimize.minimize.
    (Tensorflow 1.xx had an interface implemented, which is not longer
    supported in Tensorflow 2.xx.)
    Type conversion is necessary since scipy-routines are written in Fortran
    which requires 64-bit floats instead of 32-bit floats."""

    def get_weight_tensor():
        """Function to return current variables of the model
        as 1d tensor as well as corresponding shapes as lists."""

        weight_list = []
        shape_list = []

        # Loop over all variables, i.e. weight matrices, bias vectors and unknown parameters
        for v in self.model.variables:
            shape_list.append(v.shape)
            weight_list.extend(v.numpy().flatten())

        for v in self.sa_weight_tf:
            shape_list.append(v.shape)
            weight_list.extend(v.numpy().flatten())

        weight_list = tf.convert_to_tensor(weight_list)
        return weight_list, shape_list

    x0, shape_list = get_weight_tensor()

    def set_weight_tensor(weight_list):
        """Function which sets list of weights
        to variables in the model."""
        idx = 0
        for v in self.model.variables:
            vs = v.shape

            # Weight matrices
            if len(vs) == 2:
                sw = vs[0]*vs[1]
                new_val = tf.reshape(weight_list[idx:idx+sw], (vs[0], vs[1]))
                idx += sw

            # Bias vectors
            elif len(vs) == 1:
                new_val = weight_list[idx:idx+vs[0]]
                idx += vs[0]

            # Variables (in case of parameter identification setting)
            elif len(vs) == 0:
                new_val = weight_list[idx]
                idx += 1

            # Assign variables (Casting necessary since scipy requires float64 type)
            v.assign(tf.cast(new_val, self.dtype))

    def get_grad():
        with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
            tape.watch(self.trainable_parameters)
            tape.watch(self.sa_weight_tf)
            loss_value, mse_res, mse_mid, mse_data = self.loss(self.Xfinec_tf, self.Cfinex_tf,
                                                               self.Ic_tf, self.Sfinec_tf, self.ShFc_tf)
        gradients_model = tape.gradient(loss_value,
                                        self.trainable_parameters)
        gradients_sa = tape.gradient(loss_value,
                                     self.sa_weight_tf)
        return loss_value, gradients_model, -gradients_sa

    def get_loss_and_grad(w):
        """Function that provides current loss and gradient
        w.r.t the trainable variables as vector. This is mandatory
        for the LBFGS minimizer from scipy."""

        # Update weights in model
        set_weight_tensor(w)
        # Determine value of \phi and gradient w.r.t. \theta at w
        loss, grad, grad_sa = get_grad()

        # Store current loss for callback function
        loss = loss.numpy().astype(np.float64)

        # Flatten gradient
        grad_flat = []
        for g in grad:
            grad_flat.extend(g.numpy().flatten())
        for g in grad_sa:
            grad_flat.extend(g.numpy().flatten())

        # Gradient list to array
        grad_flat = np.array(grad_flat, dtype=np.float64)

        # Return value and gradient of \phi as tuple
        return loss, grad_flat

    return scipy.optimize.minimize(fun=get_loss_and_grad,
                                   x0=x0,
                                   jac=True,
                                   method=method,
                                   # options={"maxiter":1000,
                                   # "ftol":1E-12, "disp":False},
                                   **kwargs)

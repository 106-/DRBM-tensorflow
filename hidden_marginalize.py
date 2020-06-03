
import tensorflow as tf
import numpy as np
import math

class original:
    @staticmethod
    @tf.function
    def activation(input):
        return tf.nn.softplus(input)

class double:
    @staticmethod
    @tf.function
    @tf.custom_gradient
    def activation(input):
        approx_factors = np.array([0., -1/12, 0., 1/2, 0., math.log(2.)]).astype(input.dtype.as_numpy_dtype())
        approx_factors_grad = np.array([2/15, 0., -1/3, 0., 1., 0.]).astype(input.dtype.as_numpy_dtype())
        ret = tf.where(
            tf.math.abs(input) < 1e-3,
            tf.math.polyval(approx_factors, input),
            tf.math.log(2*tf.math.cosh(input))
        )
        @tf.function
        def grad(dy):
            return dy * tf.where(
                tf.math.abs(input) < 1e-3,
                tf.math.polyval(approx_factors_grad, input),
                tf.math.tanh(input)
            )
        return ret, grad

class continuous:
    @staticmethod
    @tf.function
    @tf.custom_gradient
    def activation(input):
        approx_factors = np.array([0., -1/180, 0., 1/6, 0., math.log(2.)]).astype(input.dtype.as_numpy_dtype())
        approx_factors_grad = np.array([2/945, 0., -1/45, 0., 1/3, 0.]).astype(input.dtype.as_numpy_dtype())
        ret = tf.where(
            tf.math.abs(input) < 1e-3,
            tf.math.polyval(approx_factors, input),
            tf.math.log(2*tf.math.sinh(input)/input)
        )
        @tf.function
        def grad(dy):
            return dy * tf.where(
                tf.math.abs(input) < 1e-3,
                tf.math.polyval(approx_factors_grad, input),
                (1/tf.math.tanh(input) - 1/input)
            )
        return ret, grad

class continuous_sparse:
    @staticmethod
    @tf.function
    def activation(input, sparse):
        sparse = tf.nn.softplus( tf.expand_dims(sparse, 1) )
        a =  (input + sparse) / 2
        b = -(input - sparse) / 2
        ret = tf.math.log( continuous_sparse._nexp_sinch(a) + continuous_sparse._nexp_sinch(b) )
        return ret
    
    @staticmethod
    @tf.function
    @tf.custom_gradient
    def _nexp_sinch(x):
        approx_factors = np.array([2/15, -1/3, 2/3, -1., 1.]).astype(x.dtype.as_numpy_dtype())
        approx_factors_grad = np.array([2/945, 0., -1/45, 0., 1/3, -1.]).astype(x.dtype.as_numpy_dtype())
        ret = tf.where(
                0 < x,
                (1-tf.exp(-2*x))/(2*x), 
                (tf.exp(2*x)-1)/(2*x*tf.exp(2*x))
            )
        ret = tf.where(
            tf.abs(x) < 1e-3,
            tf.math.polyval(approx_factors, x),
            ret
        )
        def grad(dy):
            r = tf.where(
                tf.abs(x) < 1e-3,
                tf.math.polyval(approx_factors_grad, x),
                1/tf.tanh(x) - 1/x -1
            )
            grad_ret = dy * ret * r
            return grad_ret
        return ret, grad
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;


void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    size_t totalIters = m / batch;

    for (size_t iter = 0; iter < totalIters; ++iter)
    {
        float *X_b = new float[batch * n]();
        for (size_t i = iter * batch * n; i < (iter + 1) * batch * n; ++i)
        {
            X_b[i - iter * batch * n] = X[i];
        }

        unsigned char *y_b = new unsigned char[batch]();
        for (size_t i = iter * batch; i < (iter + 1) * batch; ++i)
        {
            y_b[i - iter * batch] = y[i];
        }

        float *Z_b = new float[batch * k]();
        for (size_t i = 0; i < batch; ++i)
        {
            for (size_t j = 0; j < k; ++j)
            {
                for (size_t l = 0; l < n; ++l)
                {
                    Z_b[i * k + j] += X_b[i * n + l] * theta[l * k + j];
                }
            }
        }

        for (size_t i = 0; i < batch; ++i)
        {
            for (size_t j = 0; j < k; ++j)
            {
                Z_b[i * k + j] = exp(Z_b[i * k + j]);
            }
        }

        float *Z_b_sum = new float[batch]();
        for (size_t i = 0; i < batch; ++i)
        {
            for (size_t j = 0; j < k; ++j)
            {
                Z_b_sum[i] += Z_b[i * k + j];
            }
        }

        for (size_t i = 0; i < batch; ++i)
        {
            for (size_t j = 0; j < k; ++j)
            {
                Z_b[i * k + j] /= Z_b_sum[i];
            }
        }

        float *Iy_b = new float[batch * k]();
        for (size_t i = 0; i < batch; ++i)
        {
            Iy_b[i * k + y_b[i]] = 1.f;
        }

        float *grad = new float[n * k]();
        for (size_t i = 0; i < n; ++i)
        {
            for (size_t j = 0; j < k; ++j)
            {
                for (size_t l = 0; l < batch; ++l)
                {
                    grad[i * k + j] = X_b[l * n + i] * (Z_b[l * k + j] - Iy_b[l * k + j]);
                }
            }
        }

        for (size_t i = 0; i < n; ++i)
        {
            for (size_t j = 0; j < k; ++j)
            {
                theta[i * k + j] -= lr / float(batch) * grad[i * k + j];
            }
        }

        delete[] X_b;
        delete[] y_b;
        delete[] Z_b;
        delete[] Z_b_sum;
        delete[] Iy_b;
        delete[] grad;
    }
    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}

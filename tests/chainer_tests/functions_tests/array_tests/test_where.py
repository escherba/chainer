import unittest

import numpy

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr


@testing.parameterize(*testing.product({
    'shape': [(3, 2, 4)],
    'dtype': [numpy.float16, numpy.float32, numpy.float32],
}))
class TestWhere(unittest.TestCase):

    def setUp(self):
        self.c_data = numpy.random.uniform(-1, 1, self.shape) > 0
        self.x_data = \
            numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        self.y_data = \
            numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)

    def check_forward(self, c_data, x_data, y_data):
        c = chainer.Variable(c_data)
        x = chainer.Variable(x_data)
        y = chainer.Variable(y_data)

        z = functions.where(c, x, y)

        self.assertEqual(x.data.shape, z.data.shape)

        for i in numpy.ndindex(c.data.shape):
            if c.data[i]:
                self.assertEqual(x.data[i], z.data[i])
            else:
                self.assertEqual(y.data[i], z.data[i])

    def test_forward_cpu(self):
        self.check_forward(self.c_data, self.x_data, self.y_data)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.c_data),
                           cuda.to_gpu(self.x_data),
                           cuda.to_gpu(self.y_data))

    def check_backward(self, c_data, x_data, y_data):
        c = chainer.Variable(c_data)
        x = chainer.Variable(x_data)
        y = chainer.Variable(y_data)
        z = functions.where(c, x, y)
        z.grad = z.data
        z.backward()
        xp = cuda.get_array_module(c_data)
        gx = xp.where(c_data, x_data, 0)
        gy = xp.where(c_data, 0, y_data)
        gradient_check.assert_allclose(gx, x.grad, atol=0, rtol=0)
        gradient_check.assert_allclose(gy, y.grad, atol=0, rtol=0)

    def test_backward_cpu(self):
        self.check_backward(self.c_data, self.x_data, self.y_data)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.c_data),
                            cuda.to_gpu(self.x_data),
                            cuda.to_gpu(self.y_data))


testing.run_module(__name__, __file__)

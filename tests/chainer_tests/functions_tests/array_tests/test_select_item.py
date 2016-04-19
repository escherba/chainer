import unittest

import numpy

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr


@testing.parameterize(
    {'in_shape': (10, 5), 'out_shape': (10,), 'dtype': numpy.float32},
    {'in_shape': (0, 5),  'out_shape': (0,), 'dtype': numpy.float32},
    {'in_shape': (1, 33), 'out_shape': (1,), 'dtype': numpy.float32},
    {'in_shape': (10, 5), 'out_shape': (10,), 'dtype': numpy.float16},
    {'in_shape': (10, 5), 'out_shape': (10,), 'dtype': numpy.float64},
)
class TestSelectItem(unittest.TestCase):

    def setUp(self):
        self.x_data = numpy.random.uniform(
            -1, 1, self.in_shape).astype(self.dtype)
        self.t_data = numpy.random.randint(
            0, 2, self.out_shape).astype(numpy.int32)
        self.gy_data = numpy.random.uniform(
            -1, 1, self.out_shape).astype(self.dtype)

    def check_forward(self, x_data, t_data):
        x = chainer.Variable(x_data)
        t = chainer.Variable(t_data)
        y = functions.select_item(x, t)
        y_exp = cuda.to_cpu(x_data)[range(t_data.size), cuda.to_cpu(t_data)]

        self.assertEqual(y.data.dtype, self.dtype)
        numpy.testing.assert_equal(cuda.to_cpu(y.data), y_exp)

    def test_forward_cpu(self):
        self.check_forward(self.x_data, self.t_data)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x_data),
                           cuda.to_gpu(self.t_data))

    def check_backward(self, x_data, t_data, gpu):
        x = chainer.Variable(x_data)
        t = chainer.Variable(t_data)
        y = functions.select_item(x, t)
        y.grad = y.data
        y.backward()
        gx = numpy.zeros_like(cuda.to_cpu(x_data))
        ind = numpy.arange(t_data.size), cuda.to_cpu(t_data)
        gx[ind] = cuda.to_cpu(x_data)[ind]
        if gpu:
            gx = cuda.to_gpu(gx)
        gradient_check.assert_allclose(gx, x.grad, atol=0, rtol=0)

    def test_backward_cpu(self):
        self.check_backward(self.x_data, self.t_data, gpu=False)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x_data),
                            cuda.to_gpu(self.t_data),
                            gpu=True)


@testing.parameterize(
    {'t_value': -1, 'valid': False},
    {'t_value': 3,  'valid': False},
    {'t_value': 0,  'valid': True},
)
class TestSelectItemValueCheck(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, (1, 2)).astype(numpy.float32)
        self.t = numpy.array([self.t_value], dtype=numpy.int32)
        self.original_debug = chainer.is_debug()
        chainer.set_debug(True)

    def tearDown(self):
        chainer.set_debug(self.original_debug)

    def check_value_check(self, x_data, t_data):
        x = chainer.Variable(x_data)
        t = chainer.Variable(t_data)

        if self.valid:
            # Check if it throws nothing
            functions.select_item(x, t)
        else:
            with self.assertRaises(ValueError):
                functions.select_item(x, t)

    def test_value_check_cpu(self):
        self.check_value_check(self.x, self.t)

    @attr.gpu
    def test_value_check_gpu(self):
        self.check_value_check(self.x, self.t)


testing.run_module(__name__, __file__)

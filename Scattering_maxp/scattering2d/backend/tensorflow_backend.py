# Authors: Edouard Oyallon, Sergey Zagoruyko, Muawiz Chaudhary

import tensorflow as tf
from collections import namedtuple

BACKEND_NAME = 'tensorflow'


from ...backend.tensorflow_backend import Modulus, cdgmm, concatenate
from ...backend.base_backend import FFT

class Pad(object):
    def __init__(self, pad_size, input_size, pre_pad=False):
        """
            Padding which allows to simultaneously pad in a reflection fashion
            and map to complex.
            Parameters
            ----------
            pad_size : list of 4 integers
                size of padding to apply.
            input_size : list of 2 integers
                size of the original signal
            pre_pad : boolean
                if set to true, then there is no padding, one simply adds the imaginarty part.
        """
        self.pre_pad = pre_pad
        self.pad_size = pad_size

    def __call__(self, x):
        if self.pre_pad:
            return x
        else:
            paddings = [[0, 0]] * len(x.shape[:-2])

            paddings += [[self.pad_size[0], self.pad_size[1]], [self.pad_size[2], self.pad_size[3]]]
            return tf.cast(tf.pad(x, paddings, mode="REFLECT"), tf.complex64)

def unpad(in_):
    """
        Slices the input tensor at indices between 1::-1
        Parameters
        ----------
        in_ : tensor_like
            input tensor
        Returns
        -------
        in_[..., 1:-1, 1:-1]
    """ 
    return in_[..., 1:-1, 1:-1]

class SubsampleFourier(object):
    """ Subsampling of a 2D image performed in the Fourier domain.

        Subsampling in the spatial domain amounts to periodization
        in the Fourier domain, hence the formula.

        Parameters
        ----------
        x : tensor_like
            input tensor with at least three dimensions.
        k : int
            integer such that x is subsampled by k along the spatial variables.

        Returns
        -------
        out : tensor_like
            Tensor such that its Fourier transform is the Fourier
            transform of a subsampled version of x, i.e. in
            F^{-1}(out)[u1, u2] = F^{-1}(x)[u1 * k, u2 * k]

    """
    def __call__(self, x, k):
        M = x.get_shape().as_list()[1]
        N = x.get_shape().as_list()[2]

        if M%k == 0 and N%k == 0  :
            y = tf.reshape(x, (-1, k, M // k, k, N // k ))

            out = tf.reduce_mean(y, axis=(1, 3))
        else :
            paddings = tf.constant([ [0, 0,], [1, k - M%k - 1 ], [1 ,k-N%k -1 ]])
            y = tf.pad(x, paddings, "CONSTANT", constant_values = 0)
            y = tf.reshape(y, (-1, k, y.get_shape().as_list()[1]//k, k, y.get_shape().as_list()[2]//k))
            out = tf.reduce_mean(y,axis=(1,3))

        return out



backend = namedtuple('backend', ['name', 'cdgmm', 'modulus', 'subsample_fourier', 'fft', 'Pad', 'unpad', 'concatenate'])

backend.name = 'tensorflow'
backend.cdgmm = cdgmm
backend.modulus = Modulus()
backend.subsample_fourier = SubsampleFourier()
backend.fft = FFT(lambda x: tf.signal.fft2d(x, name='fft2d'),
                  lambda x: tf.signal.ifft2d(x, name='ifft2d'),
                  lambda x: tf.math.real(tf.signal.ifft2d(x, name='irfft2d')),
                  lambda x: None)

backend.Pad = Pad
backend.unpad = unpad
backend.concatenate = lambda x: concatenate(x, -3)

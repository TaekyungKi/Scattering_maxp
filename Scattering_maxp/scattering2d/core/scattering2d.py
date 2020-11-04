# Authors: Edouard Oyallon, Muawiz Chaudhary
# Scientific Ancestry: Edouard Oyallon, Laurent Sifre, Joan Bruna

# Some part of this code is edited by Taekyung Ki

import tensorflow as tf
import numpy as np

def scattering2d(x, pad, unpad, backend, J, L, phi, psi, max_order, out_type='array'):

    subsample_fourier = backend.subsample_fourier
    modulus = backend.modulus
    fft = backend.fft
    cdgmm = backend.cdgmm
    concatenate = backend.concatenate

    out_S_0, out_S_1, out_S_2 = [], [], []

    U_r = pad(x)
    U_0_c = fft(U_r, 'C2C') 
                            
    U_1_c = cdgmm(U_0_c, phi[1][0]) 

    U_1_c = subsample_fourier(U_1_c, k=2 ** J) 
    S_0 = fft(U_1_c, 'C2R', inverse=True) 
    S_0 = unpad(S_0)

    out_S_0.append({'coef': S_0,
                    'j': (),
                    'theta': ()}) 

    for n1 in range(len(psi)):
        j1 = psi[n1]['j']
        theta1 = psi[n1]['theta']

        U_1_c = cdgmm(U_0_c, psi[n1][0])
        
        if j1 > 0 : 
            U_1_c = subsample_fourier(  U_1_c, k=2 **(j1) )

        U_1_c = fft(U_1_c, 'C2C', inverse=True) 
        U_1_c = modulus(U_1_c) 
        U_1_c_m = tf.expand_dims(U_1_c, 3)

        U_1_c_m = tf.nn.max_pool2d(U_1_c_m, ksize=(1,2,2,1), strides = (1,2,2,1), padding = 'VALID')
        U_1_c = tf.reshape(U_1_c_m, tf.shape(U_1_c_m)[:-1])
        U_1_c = tf.cast(U_1_c, dtype = tf.complex64)
        
        U_1_c = fft(U_1_c, 'C2C') 
        U_1_c = tf.cast(U_1_c, dtype = tf.complex64)       
                                   
        S_1_c = cdgmm(U_1_c, phi[j1][1]) 
        S_1_c = subsample_fourier(S_1_c, k=2 ** (J - j1))
        S_1_r = fft(S_1_c, 'C2R', inverse=True)
        
        out_S_1.append({'coef': S_1_r,
                        'j': (j1,),
                        'theta': (theta1,)})

        if max_order < 2:
            continue
        for n2 in range(len(psi)): 
            j2 = psi[n2]['j']
            theta2 = psi[n2]['theta']

            if j2 <= j1: 
                continue
            if j1 == 0 :
                U_2_c = cdgmm(U_1_c, psi[n2][2])
            else : 
                U_2_c = cdgmm(U_1_c, psi[n2][1])
            
            U_2_c = subsample_fourier(U_2_c, k=2 ** (j2 - j1)) 
            U_2_c = fft(U_2_c, 'C2C', inverse=True)
            U_2_c = modulus(U_2_c)

            U_2_c_m = tf.expand_dims(U_2_c, 3)
            U_2_c_m = tf.nn.max_pool2d(U_2_c_m, ksize=(1,2,2,1), strides = (1,2,2,1), padding = 'VALID')
            U_2_c = tf.reshape(U_2_c_m, tf.shape(U_2_c_m)[:-1])
            U_2_c = tf.cast(U_2_c, dtype = tf.complex64)
            
            U_2_c = fft(U_2_c, 'C2C')
            
            S_2_c = cdgmm(U_2_c, phi[j2][2])
            
            
            S_2_c = subsample_fourier(S_2_c, k=2 ** (J - j2)) 

            S_2_r = fft(S_2_c, 'C2R', inverse=True) 

            out_S_2.append({'coef': S_2_r,
                            'j': (j1, j2),
                            'theta': (theta1, theta2)})

    out_S = []
    out_S.extend(out_S_0)

    out_S.extend(out_S_1)
    out_S.extend(out_S_2)

    if out_type == 'array':
        for x in out_S :

            m = x['coef'].get_shape().as_list()[-2]
            n = x['coef'].get_shape().as_list()[-1]
            x['coef']= tf.reshape(x['coef'], (-1, m*n))
        
        out_S = tf.concat([x['coef'] for x in out_S],axis = 1 )
        print(out_S.get_shape().as_list())

    return out_S 

__all__ = ['scattering2d']
import numpy as np
import keras.backend as K

def log10(x):
    log = K.log(x)
    denominator = K.log(K.constant(10, dtype=log.dtype))
    return log / denominator

def psnr_loss(y_true, y_pred):
    loss = 0
    for idx in range(2):
        loss += 10 * log10(K.mean(K.square((y_pred[:,:,:,3*(idx+1):3*(idx+2)]+y_pred[:,:,:,:3]) - y_true), axis=-1) + 1e-6)
    return loss

def mean_squared_error(y_true, y_pred):
    loss = 0
    for idx in range(2):
        loss += K.mean(K.square(y_pred - y_true), axis=-1)
    return loss
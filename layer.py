from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
import keras
import tensorflow as tf
import keras.backend as K

def conv_bn(inputs, filters, kernel_size, activation="relu"):
    layer = keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=1, padding="same", use_bias=False)(inputs)
    layer = keras.layers.BatchNormalization()(layer)
    if activation == "leaky":
        layer = keras.layers.LeakyReLU()(layer)
    else:
        layer = keras.layers.Activation(activation)(layer)
    return layer

def hinBlock(inputs, filters, kernel_size):
    residual = inputs

    layer1 = keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, padding="same", use_bias=False)(inputs)

    '''half instance'''
    layer_mid = list(keras.layers.Lambda(lambda x : tf.split(x, 2, axis=-1))(layer1))
    instance_layer = keras.layers.Lambda(lambda x : InstanceNormalization()(x))(layer_mid[0])
    identity_layer = layer_mid[1]
    layer1 = keras.layers.Concatenate()([instance_layer, identity_layer])
    layer1 = keras.layers.LeakyReLU(alpha=0.2)(layer1)

    layer2 = leakyBlock(layer1, filters=filters, kernel_size=kernel_size, strides=1)

    residual = keras.layers.Conv2D(filters=filters, kernel_size=(1, 1), strides=1, padding="same")(residual)
    out = keras.layers.Add()([residual, layer2])
    return out

def hinDownSample(inputs, filters, kernel_size):
    block_down = leakyBlock(inputs, filters=filters, kernel_size=kernel_size, strides=2)
    return block_down

def hinUpSample(inputs, filters, kernel_size):
    block_up = keras.layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=2, padding="same")(inputs)
    block_up = keras.layers.LeakyReLU(alpha=0.2)(block_up)
    return block_up

def leakyBlock(inputs, filters, kernel_size, strides=1, use_bias=True):
    layer = keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding="same", use_bias=use_bias)(inputs)
    out = keras.layers.LeakyReLU(alpha=0.2)(layer)
    return out

def resBlock(inputs, filters, kernel_size):
    residual = inputs

    layer1 = leakyBlock(inputs, filters=filters, kernel_size=kernel_size)
    layer2 = leakyBlock(layer1, filters=filters, kernel_size=kernel_size)

    residual = keras.layers.Conv2D(filters=filters, kernel_size=(1, 1), strides=1, padding="same")(residual)
    out = keras.layers.Add()([residual, layer2])
    return out

def samBlock(inputs, degraded_inputs, block_name):
    input_channels = K.int_shape(inputs)[-1]
    residual_img = keras.layers.Conv2D(filters=3, kernel_size=(3, 3), strides=1, padding="same")(inputs)
    pred_restored_img = keras.layers.Add(name=block_name)([degraded_inputs, residual_img])

    attention_mask = keras.layers.Conv2D(filters=input_channels, kernel_size=(3, 3), strides=1,
                                         padding="same", activation="sigmoid", use_bias=True)(pred_restored_img)

    '''re-calibrate'''
    recal_layer = keras.layers.Conv2D(filters=input_channels, kernel_size=(3, 3), strides=1, padding="same")(inputs)
    recal_layer = keras.layers.Multiply()([recal_layer, attention_mask])

    out = keras.layers.Add()([inputs, recal_layer])
    return pred_restored_img, out

def csffBlock(input1, input2):
    input_channels = K.int_shape(input1)[-1]

    layer1 = keras.layers.Conv2D(filters=input_channels, kernel_size=(3, 3), padding="same")(input1)
    layer2 = keras.layers.Conv2D(filters=input_channels, kernel_size=(3, 3), padding="same")(input2)

    return keras.layers.Add()([layer1, layer2])
from layer import *
import keras

class HiNet:
    def __init__(self,input_shape, weight_decay=0.0001):
        self.input_shape = input_shape
        self.l2_reg = keras.regularizers.l2(weight_decay)

    def hinet_stem(self, inputs):
        '''32 x 32'''
        block = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=1,
                                    padding="same", kernel_regularizer=self.l2_reg)(inputs)

        return block

    def hinet_body(self, inputs, csff_list=None):
        '''encoder'''
        block1 = hinBlock(inputs, filters=64, kernel_size=(3, 3))
        if csff_list:
            block1 = keras.layers.Add()([block1, csff_list[0]])
        block_down1 = hinDownSample(block1, filters=64, kernel_size=(4, 4))

        block2 = hinBlock(block_down1, filters=128, kernel_size=(3, 3))
        if csff_list:
            block2 = keras.layers.Add()([block2, csff_list[1]])
        block_down2 = hinDownSample(block2, filters=128, kernel_size=(4, 4))

        '''decoder'''
        block3 = hinBlock(block_down2, filters=256, kernel_size=(3, 3))
        block_up1 = hinUpSample(block3, filters=128, kernel_size=(2, 2))
        block_up1 = keras.layers.Add()([block2, block_up1])

        block4 = resBlock(block_up1, filters=128, kernel_size=(3, 3))
        block_up2 = hinUpSample(block4, filters=64, kernel_size=(2, 2))
        block_up2 = keras.layers.Add()([block1, block_up2])

        block5 = resBlock(block_up2, filters=64, kernel_size=(3, 3))
        return block1, block2, block4, block5

    def hinet(self):
        inputs = keras.layers.Input(shape=self.input_shape)

        # ==================================stage 1=====================================
        '''stem'''
        stem1 = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same")(inputs)

        '''body'''
        csff_1, csff_2, csff_3, csff_4 = self.hinet_body(stem1)
        pred_img, sam_out = samBlock(csff_4, degraded_inputs=inputs, block_name="pred1")

        # ==================================stage 2=====================================
        '''stage2'''
        csff_list = []
        csff_list.append(csffBlock(csff_1, csff_4))
        csff_list.append(csffBlock(csff_2, csff_3))

        '''stem'''
        stem2 = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same")(inputs)
        stem2 = keras.layers.Multiply()([sam_out, stem2])
        stem2 = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same")(stem2)

        '''body'''
        _, _, _, out = self.hinet_body(stem2, csff_list=csff_list)
        pred_img2 = keras.layers.Conv2D(filters=3, kernel_size=(3, 3), padding="same", name="pred2")(out)
        return keras.models.Model(inputs, [pred_img, pred_img2])
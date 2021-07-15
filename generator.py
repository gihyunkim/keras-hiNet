import keras
import glob
import os
import numpy as np
from keras.utils import to_categorical
import cv2
from imgaug import augmenters as iaa
from imgaug.augmentables.batches import UnnormalizedBatch

class Denoise_Generator(keras.utils.Sequence):
    def __init__(self, src_path, input_shape, batch_size, augs=None, is_train=False):
        self.is_train = is_train
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.augs = iaa.Sequential(augs)
        self.x, self.y = [], []

        self.x = glob.glob(src_path+"train_input_img/*.png")
        self.y = glob.glob(src_path+"train_label_img/*.png")

        self.on_epoch_end()

    def __getitem__(self, idx):
        batch_x, batch_y = [], []
        batch_index = self.index[idx*self.batch_size:(idx+1)*self.batch_size]
        for i in batch_index:
            batch_x.append(self.x[i])
            batch_y.append(self.y[i])
        out_x, out_y = self.data_gen(batch_x, batch_y)
        return out_x, out_y

    def __len__(self):
        return round(len(self.x) / self.batch_size)

    def on_epoch_end(self):
        self.index = np.arange(len(self.x))
        if self.is_train:
            np.random.shuffle(self.index)

    def data_gen(self, x, y):
        input_x = np.zeros((self.batch_size, self.input_shape[0], self.input_shape[1], self.input_shape[2]), dtype=np.float32)
        input_y = np.zeros((self.batch_size, self.input_shape[0], self.input_shape[1], self.input_shape[2]), dtype=np.float32)
        x_imgs = []
        y_imgs = []
        for idx in range(len(x)):
            x_img = cv2.imread(x[idx])
            x_imgs.append(cv2.resize(x_img, (self.input_shape[0], self.input_shape[1])))
            y_img = cv2.imread(y[idx])
            y_imgs.append(cv2.resize(y_img, (self.input_shape[0], self.input_shape[1])))

        # batch_imgs = UnnormalizedBatch(images=x_imgs, data=y)
        # batch_aug_imgs = list(self.augs.augment_batches(batches=batch_imgs))

        for idx in range(len(x)):
            # aug_img = batch_aug_imgs[0].images_aug[idx]
            # input_x[idx] = aug_img.astype(np.float) / 255.0
            input_x[idx] = x_imgs[idx] / 255.0
            input_y[idx] = y_imgs[idx] / 255.0
        # input_y = to_categorical(y, num_classes=self.class_num)
        return input_x, input_y

if __name__ == "__main__":
    src_path = "./datasets/train/"
    input_shape = (64, 64, 3)
    class_num = 200
    epochs = 1000
    batch_size = 16
    weight_decay = 1e-4
    lr = 1e-4

    augs = [
        iaa.Fliplr(0.5),

        iaa.SomeOf((1,2),[
            iaa.MultiplyAndAddToBrightness(),
            iaa.GammaContrast()
        ]),

        iaa.SomeOf((0,2), [
            iaa.Sometimes(0.7, iaa.AdditiveGaussianNoise()),
            iaa.Sometimes(0.7, iaa.GaussianBlur())
        ]),


        iaa.SomeOf((0,6),[
            iaa.ShearX(),
            iaa.ShearY(),
            iaa.ScaleX(),
            iaa.ScaleY(),
            iaa.Sometimes(0.5, iaa.Affine()),
            iaa.Sometimes(0.5, iaa.PerspectiveTransform()),
        ]),
        iaa.Sometimes(0.9, iaa.Dropout()),
        iaa.Sometimes(0.9, iaa.CoarseDropout()),
        iaa.Sometimes(0.9, iaa.Cutout()),
        iaa.SomeOf((0,1),[
            iaa.Sometimes(0.9, iaa.Dropout()),
            iaa.Sometimes(0.9, iaa.CoarseDropout()),
            iaa.Sometimes(0.9, iaa.Cutout())
        ])

    ]

    cg = Class_Generator(src_path, input_shape=input_shape, class_num=class_num, augs= augs, batch_size=1)
    step_size = cg.__len__()
    for i in range(step_size):
        cg.__getitem__(i)
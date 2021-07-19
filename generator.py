import keras
import glob
import os
import numpy as np
from keras.utils import to_categorical
import cv2
import albumentations as alb

class Denoise_Generator(keras.utils.Sequence):
    def __init__(self, src_path, input_shape, batch_size, augs=None, is_train=False):
        self.is_train = is_train
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.augs = augs
        self.x = list(set(glob.glob(src_path+"train_input_img_crop/*/*.png")))
        self.on_epoch_end()

    def __getitem__(self, idx):
        batch_x, batch_y = [], []
        batch_index = self.index[idx*self.batch_size:(idx+1)*self.batch_size]
        for i in batch_index:
            batch_x.append(self.x[i])
        out_x, out_y = self.data_gen(batch_x)
        return out_x, out_y

    def __len__(self):
        return round(len(self.x) / self.batch_size)

    def on_epoch_end(self):
        self.index = np.arange(len(self.x))
        if self.is_train:
            np.random.shuffle(self.index)

    def data_gen(self, x):
        input_x = np.zeros((self.batch_size, self.input_shape[0], self.input_shape[1], self.input_shape[2]), dtype=np.float32)
        input_y = np.zeros((self.batch_size, self.input_shape[0], self.input_shape[1], self.input_shape[2]), dtype=np.float32)
        x_imgs = []
        y_imgs = []
        for idx in range(len(x)):
            x_img = cv2.imread(x[idx])
            x_imgs.append(cv2.resize(x_img, (self.input_shape[0], self.input_shape[1])))
            y_img = cv2.imread(x[idx].replace("input", "label"))
            y_imgs.append(cv2.resize(y_img, (self.input_shape[0], self.input_shape[1])))
        for idx in range(len(x)):
            aug_imgs = self.augs(image=x_imgs[idx], image0=y_imgs[idx])
            aug_x = aug_imgs['image']
            aug_y = aug_imgs['image0']
            input_x[idx] = aug_x / 255.0
            input_y[idx] = aug_y / 255.0
        return input_x, input_y

if __name__ == "__main__":
    src_path = "./datasets/"
    input_shape = (256, 256, 3)
    class_num = 200
    epochs = 1000
    batch_size = 16
    weight_decay = 1e-4
    lr = 1e-4

    '''Augmentation'''
    augs = alb.Compose([
        alb.HorizontalFlip(p=0.5),
        alb.Rotate(limit=35, p=0.5),
    ], additional_targets={'image0':'image'})


    cg = Denoise_Generator(src_path, input_shape=input_shape, augs= augs, batch_size=1)
    step_size = cg.__len__()
    for i in range(step_size):
        cg.__getitem__(i)